#!/usr/bin/env node
// server.js — the worlds server (spec section 2). Node.js standard library only; no framework, no npm packages, NO AUTHENTICATION.
//   node server.js --root <DATA_ROOT> [--port 8080] [--host 127.0.0.1] [--sop <path to the sop entry script>]
// JSON over HTTP, file bodies raw. Binds to localhost by default. Paths are normalized and must stay inside the world folder; every write is
// atomic (temp file + rename); the server never trains and runs nothing but `sop check` and `sop formalize`. Errors: JSON {error, detail}.
// One log line per request on stdout.
'use strict';
const http = require('http'), fs = require('fs'), path = require('path'), url = require('url'), cp = require('child_process'), crypto = require('crypto');

function arg(name, def) { const i = process.argv.indexOf('--' + name); return i > 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def; }
const ROOT = path.resolve(arg('root', './data')), PORT = parseInt(arg('port', '8080'), 10), HOST = arg('host', '127.0.0.1');
const SOP = path.resolve(arg('sop', path.join(__dirname, 'bin', 'sop')));
const MAX_BODY = 64 * 1024 * 1024;
const WORLD_RE = /^[A-Za-z0-9_-]{1,64}$/;
const LAYOUT = ['examples', 'raw', 'audit', 'model', 'runs', 'versions', 'requests'];
const WRITABLE_DIRS = ['examples', 'raw', 'audit'];                  // subfolders may be created under these only
const WRITABLE_TOP = /^(legend\.json|world\.json|split_hints\.json|[A-Za-z0-9_-]+\.tsv)$/;   // top-level files a person may upload
const WORLD_DEFAULTS = {   // spec v2 section 1: models per step (small for pipeline / annotate / second_look, large for evolve), limits
  models: { pipeline: 'claude-haiku-5-5', annotate: 'claude-haiku-5-5', second_look: 'claude-haiku-5-5', evolve: 'claude-opus-5-5' },
  min_examples: 500, max_versions_per_run: 3, second_look_escalate: 0.05 };

class HttpError extends Error { constructor(status, error, detail) { super(error); this.status = status; this.error = error; this.detail = detail || ''; } }

function send(res, status, body, type) {
  const buf = Buffer.isBuffer(body) ? body : Buffer.from(typeof body === 'string' ? body : JSON.stringify(body, null, 1));
  res.writeHead(status, { 'Content-Type': type || (typeof body === 'string' || Buffer.isBuffer(body) ? 'application/octet-stream' : 'application/json'),
                          'Content-Length': buf.length, 'X-Content-Type-Options': 'nosniff' });
  res.end(buf);
}
function sendJSON(res, status, obj) { send(res, status, Buffer.from(JSON.stringify(obj, null, 1)), 'application/json; charset=utf-8'); }

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = []; let n = 0;
    req.on('data', c => { n += c.length; if (n > MAX_BODY) { reject(new HttpError(413, 'body too large', `limit ${MAX_BODY} bytes`)); req.destroy(); } else chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks))); req.on('error', reject);
  });
}

function atomicWrite(file, data) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const tmp = path.join(path.dirname(file), `.${path.basename(file)}.tmp-${process.pid}-${crypto.randomBytes(4).toString('hex')}`);
  fs.writeFileSync(tmp, data); fs.renameSync(tmp, file);
}

function worldDir(w) {
  if (!WORLD_RE.test(w)) throw new HttpError(400, 'bad world name', 'allowed: [A-Za-z0-9_-], 1-64 characters');
  return path.join(ROOT, w);
}
function needWorld(w) { const d = worldDir(w); if (!fs.existsSync(path.join(d, 'world.json'))) throw new HttpError(404, 'no such world', w); return d; }

// a relative path inside the world folder: decoded, normalized, no '..', no absolute, no NUL, no hidden segment; returns {abs, rel, parts}
function safePath(wd, rel) {
  let r;
  try { r = decodeURIComponent(rel || ''); } catch (e) { throw new HttpError(400, 'bad path encoding', rel); }
  if (r.includes('\0') || r.includes('\\')) throw new HttpError(400, 'bad path', 'NUL or backslash');
  const parts = r.split('/').filter(p => p !== '' && p !== '.');
  if (parts.some(p => p === '..')) throw new HttpError(400, 'path traversal rejected', r);
  if (parts.some(p => p.startsWith('.'))) throw new HttpError(400, 'hidden path segments are not allowed', r);
  const norm = path.posix.normalize(parts.join('/'));
  const abs = path.resolve(wd, norm);
  if (abs !== wd && !abs.startsWith(wd + path.sep)) throw new HttpError(400, 'path escapes the world folder', r);
  // refuse symlinks that lead outside the world
  let p = wd;
  for (const seg of parts) { p = path.join(p, seg); try { if (fs.lstatSync(p).isSymbolicLink()) throw new HttpError(400, 'symlinks are not followed', seg); } catch (e) { if (e instanceof HttpError) throw e; break; } }
  return { abs, rel: norm, parts };
}

function checkWritable(sp) {
  const top = sp.parts[0];
  if (sp.parts.length === 0) throw new HttpError(400, 'empty path', '');
  if (top === 'model' || top === 'runs') throw new HttpError(403, `${top}/ is written by training only`, sp.rel);
  if (sp.parts.length === 1) {
    if (!WRITABLE_TOP.test(top)) throw new HttpError(403, 'this top-level file is not writable', `${sp.rel} (writable: legend.json, world.json, split_hints.json, *.tsv)`);
    return;
  }
  if (!WRITABLE_DIRS.includes(top)) throw new HttpError(403, 'not writable through /files', `${sp.rel} (writable folders: ${WRITABLE_DIRS.join(', ')}; requests through /requests)`);
}

function readJSON(p, def) { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { return def; } }

function inventory(wd) {
  const out = [];
  (function walk(d, rel) {
    let ents; try { ents = fs.readdirSync(d, { withFileTypes: true }); } catch (e) { return; }
    for (const e of ents.sort((a, b) => a.name < b.name ? -1 : 1)) {
      if (e.name.startsWith('.')) continue;
      const p = path.join(d, e.name), r = rel ? rel + '/' + e.name : e.name;
      if (e.isDirectory()) walk(p, r);
      else if (e.isFile()) { const st = fs.statSync(p); out.push({ name: r, size: st.size, mtime: st.mtime.toISOString() }); }
    }
  })(wd, '');
  return out;
}

function listExamples(wd) { return inventory(path.join(wd, 'examples')).filter(f => f.name.endsWith('.sop')).map(f => path.join(wd, 'examples', f.name)); }

function runSop(args, input) {
  return new Promise(resolve => {
    const ch = cp.execFile(SOP, args, { maxBuffer: 256 * 1024 * 1024, timeout: 600000 }, (err, stdout, stderr) => {
      resolve({ code: err ? (typeof err.code === 'number' ? err.code : -1) : 0, stdout, stderr });
    });
    if (input !== undefined) { ch.stdin.on('error', () => {}); ch.stdin.end(input); }
  });
}

function versions(wd) {
  const vd = path.join(wd, 'versions'); let names = [];
  try { names = fs.readdirSync(vd).filter(n => /^v\d+$/.test(n)).sort((a, b) => parseInt(a.slice(1)) - parseInt(b.slice(1))); } catch (e) {}
  return names.map(n => {
    const d = path.join(vd, n), metrics = readJSON(path.join(d, 'metrics.json'), null), acc = readJSON(path.join(d, 'acceptance.json'), null);
    return { version: parseInt(n.slice(1)), mark: readJSON(path.join(d, 'mark.json'), {}).mark || null,
             legend_sha256: fs.existsSync(path.join(d, 'legend.json')) ? 'sha256:' + crypto.createHash('sha256').update(fs.readFileSync(path.join(d, 'legend.json'))).digest('hex') : null,
             acceptance: acc, metrics_summary: metrics && metrics.mean_over_seeds ? Object.fromEntries(Object.entries(metrics.mean_over_seeds).map(([s, v]) => [s, (v['catalog candidates'] || {}).FULL])) : null,
             files: inventory(d).map(f => f.name) };
  });
}

async function handle(req, res) {
  const u = url.parse(req.url, true), p = u.pathname.split('/').filter(Boolean), m = req.method;
  if (m === 'GET' && p.length === 0) return send(res, 200, PAGE, 'text/html; charset=utf-8');
  if (p[0] !== 'worlds') throw new HttpError(404, 'not found', u.pathname);
  if (p.length === 1) {
    if (m !== 'GET') throw new HttpError(405, 'method not allowed', m);
    fs.mkdirSync(ROOT, { recursive: true });
    const ws = fs.readdirSync(ROOT, { withFileTypes: true }).filter(e => e.isDirectory() && WORLD_RE.test(e.name) && fs.existsSync(path.join(ROOT, e.name, 'world.json')));
    return sendJSON(res, 200, ws.map(e => ({ name: e.name, status: readJSON(path.join(ROOT, e.name, 'status.json'), {}) })));
  }
  const w = p[1], wd = worldDir(w);
  if (p.length === 2) {
    if (m === 'POST') {
      if (fs.existsSync(wd)) throw new HttpError(409, 'world exists', w);
      const body = await readBody(req); let meta = {};
      if (body.length) { try { meta = JSON.parse(body.toString('utf8')); } catch (e) { throw new HttpError(400, 'body must be JSON (optional: language, description)', e.message); } }
      fs.mkdirSync(wd, { recursive: true }); for (const d of LAYOUT) fs.mkdirSync(path.join(wd, d), { recursive: true });
      const wj = Object.assign({ name: w, language: meta.language || 'en', description: meta.description || '', created: new Date().toISOString(), version: 0 },
                               JSON.parse(JSON.stringify(WORLD_DEFAULTS)), meta.models ? { models: Object.assign({}, WORLD_DEFAULTS.models, meta.models) } : {});
      for (const k of ['min_examples', 'max_versions_per_run', 'second_look_escalate']) if (meta[k] !== undefined) wj[k] = meta[k];
      atomicWrite(path.join(wd, 'world.json'), JSON.stringify(wj, null, 1) + '\n');
      atomicWrite(path.join(wd, 'status.json'), JSON.stringify({ label: null, note: 'never trained' }) + '\n');
      return sendJSON(res, 201, { created: w, world: wj });
    }
    needWorld(w);
    if (m === 'GET') return sendJSON(res, 200, { world: readJSON(path.join(wd, 'world.json'), null), legend: readJSON(path.join(wd, 'legend.json'), null),
                                                files: inventory(wd), status: readJSON(path.join(wd, 'status.json'), null) });
    if (m === 'DELETE') {
      if (u.query.confirm !== w) throw new HttpError(400, 'confirmation required', `DELETE /worlds/${w}?confirm=${w}`);
      const trash = path.join(ROOT, `.deleted-${w}-${Date.now()}`); fs.renameSync(wd, trash); fs.rmSync(trash, { recursive: true, force: true });
      return sendJSON(res, 200, { deleted: w });
    }
    throw new HttpError(405, 'method not allowed', m);
  }
  needWorld(w);
  const what = p[2];
  if (what === 'files') {
    if (p.length === 3) {
      if (m !== 'GET') throw new HttpError(405, 'method not allowed', m);
      const sp = safePath(wd, u.query.dir || '');
      if (!fs.existsSync(sp.abs) || !fs.statSync(sp.abs).isDirectory()) throw new HttpError(404, 'no such folder', sp.rel);
      return sendJSON(res, 200, { dir: sp.rel, files: inventory(sp.abs) });
    }
    const rawRel = u.pathname.split('/').slice(4).join('/');
    const sp = safePath(wd, rawRel);
    if (m === 'GET') {
      if (!fs.existsSync(sp.abs) || !fs.statSync(sp.abs).isFile()) throw new HttpError(404, 'no such file', sp.rel);
      return send(res, 200, fs.readFileSync(sp.abs), sp.rel.endsWith('.json') ? 'application/json' : 'application/octet-stream');
    }
    if (m === 'PUT') {
      checkWritable(sp); const body = await readBody(req); const existed = fs.existsSync(sp.abs);
      if (existed && fs.statSync(sp.abs).isDirectory()) throw new HttpError(409, 'a folder exists at this path', sp.rel);
      atomicWrite(sp.abs, body); return sendJSON(res, existed ? 200 : 201, { written: sp.rel, size: body.length });
    }
    if (m === 'DELETE') {
      checkWritable(sp);
      if (sp.rel === 'world.json') throw new HttpError(403, 'world.json is the manifest; DELETE /worlds/{w}?confirm={w} removes the world', sp.rel);
      if (!fs.existsSync(sp.abs) || !fs.statSync(sp.abs).isFile()) throw new HttpError(404, 'no such file', sp.rel);
      fs.unlinkSync(sp.abs); return sendJSON(res, 200, { deleted: sp.rel });
    }
    throw new HttpError(405, 'method not allowed', m);
  }
  if (what === 'check' && p.length === 3 && m === 'POST') {
    const legend = path.join(wd, 'legend.json');
    if (!fs.existsSync(legend)) throw new HttpError(409, 'no legend.json in this world', 'upload one first');
    const files = listExamples(wd);
    if (!files.length) return sendJSON(res, 200, { ok: true, exit_code: 0, report: { files: [] }, note: 'no examples' });
    const r = await runSop(['check', '--json', '--legend', `${w}=${legend}`, '--legends', ROOT, ...files]);
    let report = null; try { report = JSON.parse(r.stdout); } catch (e) {}
    if (report === null) throw new HttpError(500, 'sop check failed', (r.stderr || r.stdout).slice(0, 4000));
    for (const f of report.files) f.file = path.relative(wd, f.file);
    return sendJSON(res, 200, { ok: r.code === 0, exit_code: r.code, report });
  }
  if (what === 'formalize' && p.length === 3 && m === 'POST') {
    if (!fs.existsSync(path.join(wd, 'model', 'config.json'))) throw new HttpError(503, 'no model', `world ${w} has not been trained (model/ is empty)`);
    const body = await readBody(req);
    const r = await runSop(['formalize', '--model', path.join(wd, 'model'), '-'], body);
    if (r.code === 2) throw new HttpError(409, 'model unusable or legend mismatch', r.stderr.slice(0, 4000));
    if (r.code !== 0) throw new HttpError(r.code === 1 ? 400 : 500, 'sop formalize failed', r.stderr.slice(0, 4000));
    return send(res, 200, r.stdout, 'text/plain; charset=utf-8');
  }
  if (what === 'requests' && p.length === 4 && m === 'PUT') {
    const name = decodeURIComponent(p[3]);
    if (!/^[A-Za-z0-9_-]{1,100}\.md$/.test(name)) throw new HttpError(400, 'bad request name', 'expected <name>.md with [A-Za-z0-9_-]');
    const body = await readBody(req); const f = path.join(wd, 'requests', name); const existed = fs.existsSync(f);
    atomicWrite(f, body); return sendJSON(res, existed ? 200 : 201, { written: `requests/${name}`, size: body.length });
  }
  if (what === 'versions') {
    if (p.length === 3 && m === 'GET') return sendJSON(res, 200, { versions: versions(wd), pinned: versions(wd).filter(v => v.mark === 'pinned').map(v => v.version) });
    if (p.length === 5 && m === 'POST' && (p[4] === 'reject' || p[4] === 'pin')) {
      if (!/^\d+$/.test(p[3])) throw new HttpError(400, 'bad version', p[3]);
      const vd = path.join(wd, 'versions', 'v' + p[3]);
      if (!fs.existsSync(vd)) throw new HttpError(404, 'no such version', 'v' + p[3]);
      const mark = p[4] === 'pin' ? 'pinned' : 'rejected';
      if (mark === 'pinned') for (const v of versions(wd)) if (v.mark === 'pinned') atomicWrite(path.join(wd, 'versions', 'v' + v.version, 'mark.json'), JSON.stringify({ mark: null, unpinned: new Date().toISOString() }) + '\n');
      atomicWrite(path.join(vd, 'mark.json'), JSON.stringify({ mark, when: new Date().toISOString(), by: 'person (server)' }) + '\n');
      return sendJSON(res, 200, { version: parseInt(p[3]), mark });
    }
  }
  throw new HttpError(404, 'not found', `${m} ${u.pathname}`);
}

const PAGE = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Worlds server</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--line:#ddd;--acc:#2b5d8a}
@media (prefers-color-scheme:dark){:root{--bg:#1b1c1e;--fg:#e8e8e3;--mut:#9a9a93;--line:#3a3b3e;--acc:#7fb0dc}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:960px;padding:16px}
h1{font-size:20px}h2{font-size:17px;margin-top:28px;border-bottom:1px solid var(--line)}small,.mut{color:var(--mut)}
pre{background:rgba(127,127,127,.1);padding:8px;overflow:auto;max-height:420px;font-size:13px}textarea{width:100%;box-sizing:border-box;font:13px monospace;background:transparent;color:var(--fg)}
button,select,input{font:inherit}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;font-size:13px}
a{color:var(--acc)}</style></head><body>
<h1>Worlds server</h1><p class="mut">No authentication. Worlds, uploads, version history, and a box that formalizes sentences with a world's reader.</p>
<div><input id="nw" placeholder="new world name"> <button onclick="createWorld()">create world</button></div>
<div id="worlds"></div>
<script>
const $=s=>document.querySelector(s); const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
async function j(u,o){const r=await fetch(u,o);const t=await r.text();try{return {ok:r.ok,status:r.status,body:JSON.parse(t)}}catch(e){return {ok:r.ok,status:r.status,body:t}}}
async function createWorld(){const n=$("#nw").value.trim();const r=await j("/worlds/"+encodeURIComponent(n),{method:"POST"});alert(r.ok?"created "+n:JSON.stringify(r.body));load()}
async function upload(w){const f=document.getElementById("f_"+w).files[0];if(!f)return;const dir=document.getElementById("d_"+w).value;
 const p=(dir?dir+"/":"")+f.name;const r=await j("/worlds/"+w+"/files/"+p.split("/").map(encodeURIComponent).join("/"),{method:"PUT",body:await f.arrayBuffer()});alert(JSON.stringify(r.body));load()}
async function check(w){const r=await j("/worlds/"+w+"/check",{method:"POST"});document.getElementById("o_"+w).textContent=JSON.stringify(r.body,null,1)}
async function formalize(w){const t=document.getElementById("t_"+w).value;const r=await fetch("/worlds/"+w+"/formalize",{method:"POST",body:t});
 document.getElementById("o_"+w).textContent=r.status+"\\n"+await r.text()}
async function load(){const r=await j("/worlds");const box=$("#worlds");box.innerHTML="";
 for(const w of r.body){const d=await j("/worlds/"+w.name);const v=await j("/worlds/"+w.name+"/versions");const st=d.body.status||{};
  const files=(d.body.files||[]).map(f=>"<tr><td><a href='/worlds/"+w.name+"/files/"+f.name.split("/").map(encodeURIComponent).join("/")+"'>"+esc(f.name)+"</a></td><td>"+f.size+"</td><td>"+esc(f.mtime)+"</td></tr>").join("");
  const vers=(v.body.versions||[]).map(x=>"<tr><td>v"+x.version+"</td><td>"+esc(x.mark||"")+"</td><td>"+esc(JSON.stringify(x.acceptance||x.metrics_summary||""))+"</td></tr>").join("")||"<tr><td colspan=3 class=mut>no versions yet</td></tr>";
  box.insertAdjacentHTML("beforeend","<h2>"+esc(w.name)+" <small>"+esc(st.label||"not trained")+"</small></h2>"+
  "<div><select id='d_"+w.name+"'><option value=''>(top: legend.json, kinds.tsv, world.json)</option><option>examples</option><option>raw</option><option>audit</option></select> "+
  "<input type=file id='f_"+w.name+"'> <button onclick=\\"upload('"+w.name+"')\\">upload</button> <button onclick=\\"check('"+w.name+"')\\">check examples</button></div>"+
  "<details><summary>files ("+(d.body.files||[]).length+")</summary><table>"+files+"</table></details>"+
  "<details><summary>version history</summary><table><tr><th>version</th><th>mark</th><th>acceptance / FULL</th></tr>"+vers+"</table></details>"+
  "<p><textarea id='t_"+w.name+"' rows=3 placeholder='one sentence per line'></textarea><button onclick=\\"formalize('"+w.name+"')\\">formalize</button></p><pre id='o_"+w.name+"'></pre>")}}
load();
</script></body></html>`;

const server = http.createServer(async (req, res) => {
  const t0 = Date.now();
  res.on('finish', () => console.log(`${new Date().toISOString()} ${req.socket.remoteAddress} ${req.method} ${req.url} ${res.statusCode} ${Date.now() - t0}ms`));
  try { await handle(req, res); }
  catch (e) {
    if (e instanceof HttpError) sendJSON(res, e.status, { error: e.error, detail: e.detail });
    else { console.error(e); sendJSON(res, 500, { error: 'internal error', detail: String(e && e.message || e) }); }
  }
});
fs.mkdirSync(ROOT, { recursive: true });
server.listen(PORT, HOST, () => {
  console.log(`worlds server on http://${HOST}:${PORT}/  root ${ROOT}  sop ${SOP}`);
  console.log(`WARNING: there is NO authentication${HOST === '127.0.0.1' || HOST === 'localhost' ? ' (bound to localhost only)' : ` and the server is EXPOSED on ${HOST}`}.`);
});
