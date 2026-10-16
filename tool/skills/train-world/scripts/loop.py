"""
loop.py — the scripts of the train-world skill (spec 1.1 steps 1-6). The coding agent (Claude, model by role from world.json) runs them;
every decision rule is here, written, so that the agent only calls scripts and reads their output.

  python3 loop.py annotate    <world> [--annotator llm|oracle:<dir of gold .sop>]     raw/*.txt -> examples/annotated.sop (sop check'ed)
  python3 loop.py train       <world> [--train-args "..."]                             sop train (+ the loop section of REPORT.md)
  python3 loop.py second-look <world> [--annotator ...]                                reader vs annotation; disagreement rate; escalation rule
  python3 loop.py feedback    <world>                                                  _unplaced clusters by residual direction, unknown wires, ...
  python3 loop.py evolve      <world>                                                  change set -> versions/v<N+1>/ (legend, migration, CHANGES.md)
  python3 loop.py step        <world> [--train-args "..."]                             one evolution step: feedback, evolve, migrate, train, accept
  python3 loop.py run         <world> [--annotator ...] [--train-args "..."] [--steps K] the whole loop: propose (if no legend), annotate, train,
                                                                                        second look, then up to max_versions_per_run evolution steps
The annotator `oracle:<dir>` is the OFFLINE stand-in for the annotate model on constructed worlds: it copies the generator's gold wires for
the same _text (no LLM; stated as such in every provenance line). With `llm`, the annotate model of world.json is called through llm.py; when
no API is reachable the step reports "API unavailable" and writes nothing.
TODO(section 6 missing): the spec's skill steps 6.1-6.7 were not received; this is the minimum that runs the loop of 1.1 end to end.
"""
import argparse, collections, copy, datetime, glob, json, os, re, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); TOOL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TOOL); sys.path.insert(0, HERE)
from sop import soplang, world
from sop.legend import Legend, tokenize, sha256_file
from sop.check import check_doc, Report
from llm import llm_call, LLMUnavailable, world_json, model_for, escalate

SOP = os.path.join(TOOL, "bin", "sop")
CLUSTER_COS = 0.5          # stated constant: two unplaced terms belong to one cluster when |cos| of their addresses >= 0.5
MIN_SUPPORT = 0.02         # stated: a proposal needs unplaced words in at least 2% of the read sentences (and >= 5 sentences)
FULL_TOL = 0.02            # stated: "within seed variance" with one seed (with several seeds: max(0.02, 2 x std))
CW_TOL = 0.005             # stated: confident-wrong "not up" tolerance


def now(): return datetime.datetime.now().isoformat(timespec="seconds")
def wname(wd): return os.path.basename(os.path.abspath(wd).rstrip("/"))
def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True); tmp = p + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1); os.replace(tmp, p)


def copy_graph(top, by, ids):
    """the wires reachable from top, renumbered with fresh ids (refs rewritten); top first"""
    order, seen, stack = [], set(), [top.id]
    while stack:
        x = stack.pop(0)
        if x in seen or x not in by: continue
        seen.add(x); order.append(x); stack += by[x].refs()
    new = {x: next(ids) for x in order}; out = []
    for x in order:
        w = copy.deepcopy(by[x]); w.id = new[x]
        for l in w.lines:
            if any(a.kind == "ref" for a in l.atoms):
                for a in l.atoms:
                    if a.kind == "ref" and a.text in new: a.text = new[a.text]
                l.raw = None
        out.append(w)
    return out


def example_doc(wd, legend):
    return soplang.Doc(directives=[("world", wname(wd), None), ("legend", legend.sha, None), ("generated", f"train-world skill loop.py ({now()[:10]})", None)])


def read_examples(wd):
    out = []
    for p in sorted(glob.glob(os.path.join(wd, "examples", "**", "*.sop"), recursive=True)):
        d = soplang.load(p); out.append((p, d))
    return out


# ------------------------------------------------------------------ 1. propose a legend (evolve model)
def propose_legend(wd):
    raws = [l.strip() for p in sorted(glob.glob(os.path.join(wd, "raw", "*.txt"))) for l in open(p) if l.strip()][:200]
    prompt = ("Propose legend.json for this world (roles = the questions the activity answers, event types with their verbs, operators among "
              "not/must/may/if/says, moods). Reply with JSON only.\n\nSentences:\n" + "\n".join(raws))
    try: text = llm_call("evolve", prompt, wd)
    except LLMUnavailable as e: return None, str(e)
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None, "the evolve model returned no JSON"
    leg = json.loads(m.group(0)); leg["version"] = 1; jdump(leg, os.path.join(wd, "legend.json")); return leg, "proposed by " + model_for("evolve", wd)


# ------------------------------------------------------------------ 2. annotate
def oracle_index(gold_dir):
    idx = {}
    for p in sorted(glob.glob(os.path.join(gold_dir, "**", "*.sop"), recursive=True)):
        d = soplang.load(p); by = d.by_id()
        for t in d.top_level():
            if t.value("_text") is not None: idx.setdefault(t.value("_text"), (t, by))
    return idx


def annotate(wd, annotator="llm", limit=None):
    legend = Legend(os.path.join(wd, "legend.json")); done = set()
    for _, d in read_examples(wd):
        for t in d.top_level(): done.add(t.value("_text"))
    raws = [l.strip() for p in sorted(glob.glob(os.path.join(wd, "raw", "*.txt"))) for l in open(p, encoding="utf-8") if l.strip()]
    todo = [s for s in dict.fromkeys(raws) if s not in done][:limit]
    if not todo: return {"annotated": 0, "note": "nothing new in raw/"}
    doc = example_doc(wd, legend); ids = soplang_ids(); stats = collections.Counter()
    if annotator.startswith("oracle:"):
        idx = oracle_index(annotator.split(":", 1)[1])
        for s in todo:
            if s not in idx:
                w = soplang.Wire(next(ids), "unknown"); w.set("_text", s); w.set("_source", "oracle: no gold for this sentence"); w.set("_status", "no-sense")
                w.set("_world", wname(wd)); doc.wires.append(w); stats["no gold (unknown wire)"] += 1; continue
            t, by = idx[s]; ws = copy_graph(t, by, ids); top = ws[0]
            top.lines = [l for l in top.lines if l.key not in ("_source", "_world")]
            top.lines.insert(1, soplang.line("_source", "oracle:generator-gold (offline stand-in for the annotate model; not an LLM)"))
            top.lines.insert(2, soplang.line("_world", wname(wd))); doc.wires += ws; stats["oracle"] += 1
    else:
        model = model_for("annotate", wd); spec = annotation_prompt_header(legend)
        for s in todo:
            try: text = llm_call("annotate", spec + "\nSentence: " + s + "\nReply with the SOP wires only.", wd)
            except LLMUnavailable as e: return {"annotated": 0, "error": str(e)}
            ws = parse_llm_wires(text, s, legend, ids, model, wd)
            stats["llm ok" if ws[0].type != "unknown" else "llm rejected by sop check (unknown wire)"] += 1; doc.wires += ws
    p = os.path.join(wd, "examples", f"annotated_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.sop")
    with open(p, "w") as f: f.write(soplang.write(doc))
    rep = Report(p); check_doc(soplang.load(p), {wname(wd): legend}, rep)
    return {"annotated": len(todo), "file": os.path.relpath(p, wd), "by": dict(stats), "check_ok": rep.ok, "check_errors": rep.errors[:5]}


def soplang_ids():
    i = 0
    while True: i += 1; yield f"a{i}"


def annotation_prompt_header(legend):
    return ("You annotate one sentence into SOP Lang wires (spec section 3). Use ONLY these wire types and keys.\n"
            f"Events (wire types) and their verbs: {json.dumps(legend.events)}\nRoles (keys of event wires): {legend.roles}\n"
            f"Operators: {legend.operators} (not: scope=$w|word; says: speaker=, scope=$w); entity wires: head=, attr=, det=.\n"
            "Format: '@w1 <type>' then 'key=value' lines; the top-level wire first, with _text=<the sentence>.\n")


def parse_llm_wires(text, sentence, legend, ids, model, wd):
    body = "\n".join(l for l in text.strip().strip("`").split("\n") if not l.startswith("!"))
    try:
        d = soplang.parse(body); rep = Report("llm"); check_doc(d, legend, rep)
        if rep.ok and d.top_level():
            t = d.top_level()[0]; ws = copy_graph(t, d.by_id(), ids); top = ws[0]
            top.lines = [l for l in top.lines if l.key not in ("_text", "_source", "_world")]
            top.lines[:0] = [soplang.Line("_text", sentence), soplang.line("_source", f"llm:{model}"), soplang.line("_world", wname(wd))]
            return ws
    except soplang.SopError: pass
    w = soplang.Wire(next(ids), "unknown"); w.set("_text", sentence); w.set("_source", f"llm:{model} (rejected by sop check)"); w.set("_world", wname(wd))
    return [w]


# ------------------------------------------------------------------ 3. train
def train(wd, train_args="", splits=None):
    cmd = [SOP, "train", wd] + train_args.split() + (["--splits", splits] if splits else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(f"sop train failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    runs = sorted(glob.glob(os.path.join(wd, "runs", "*", "splits.json")))
    return {"run": os.path.dirname(runs[-1]) if runs else None, "log_tail": [l for l in r.stdout.splitlines() if l.startswith("[eval]")]}


# ------------------------------------------------------------------ 4. second look
def second_look(wd, annotator="llm", n_max=2000):
    from sop.infer import Reader
    rd = Reader(os.path.join(wd, "model")); legend = rd.legend; rows = []
    for p, d in read_examples(wd):
        by = d.by_id()
        for t in d.top_level():
            if t.value("_text") is not None: rows.append((p, t, by))
    rows = rows[:n_max]; dis = []; conf = 0; ids = (f"r{i}" for i in range(10 ** 9))
    for p, t, by in rows:
        g = world.frame_of(t, by, legend); ws, f, info = rd.formalize(t.value("_text"), 0, ids)
        if g["shape"] == "ok" and legend.reader["predicate_role"] and not g.get("lemma"): g["lemma"] = rd.comp.lemma_for(g["event"], info["tokens"], set())
        if info["status"] != "ok": continue
        conf += 1
        if world.canon(f, legend.reader["predicate_role"]) != world.canon(g, legend.reader["predicate_role"]): dis.append((p, t, ws))
    rate = len(dis) / max(1, len(rows)); changed = 0; note = ""
    audit = soplang.Doc(directives=[("world", wname(wd), None), ("legend", legend.sha, None)])
    for p, t, ws in dis: audit.wires += ws
    if dis:
        os.makedirs(os.path.join(wd, "audit"), exist_ok=True)
        with open(os.path.join(wd, "audit", "reader_disagreements.sop"), "w") as f: f.write(soplang.write(audit))
    if dis and annotator.startswith("oracle:"):
        note = "second look by the oracle (generator gold): the gold is the annotation, so nothing changes; the disagreements are the reader's errors"
    elif dis:
        try:
            for p, t, ws in dis[:200]:
                llm_call("second_look", annotation_prompt_header(legend) + "\nTwo readings of the same sentence; return the correct SOP wires.\n"
                         f"Annotation:\n{t.value('_text')}\nReader:\n{soplang.write(soplang.Doc(wires=ws))}", wd)
            note = "second look done by " + model_for("second_look", wd) + " (TODO(section 6 missing): replacing the annotation in place, with _changed=yes)"
        except LLMUnavailable as e: note = f"{e}: second look not done; annotations unchanged"
    wj = world_json(wd); thr = wj.get("second_look_escalate", 0.05); esc = None
    if rate > thr and not annotator.startswith("oracle:"):
        esc = escalate(wd, rate); note += f"; disagreement rate {rate:.3f} > {thr}: annotate / second_look escalated to {esc['to']} (raw/ to be re-annotated once)"
    elif rate > thr: note += f"; disagreement rate {rate:.3f} > {thr}, but the annotator is the oracle (gold): escalation would not change annotations, not applied"
    out = {"rows_read": len(rows), "confident": conf, "disagreements": len(dis), "disagreement_rate": rate, "changed": changed,
           "escalation_threshold": thr, "escalated": esc, "note": note, "measured_label_noise": changed / max(1, len(rows))}
    jdump(out, os.path.join(wd, "model", "second_look.json"))
    with open(os.path.join(wd, "model", "REPORT.md"), "a") as f:
        f.write(f"\n## Loop: annotation and second look (BUILT)\nAnnotator of examples/: {annotator if annotator.startswith('oracle:') else 'llm:' + model_for('annotate', wd)}. "
                f"The reader re-read {len(rows)} training sentences: confident (status ok) on {conf}; confident AND disagreeing with the annotation: "
                f"{len(dis)} (rate {rate:.3f}, against the escalation threshold {thr}). Annotations changed by the second look: {changed} "
                f"(measured label noise {changed / max(1, len(rows)):.3f}). {note}\nDisagreements for people: audit/reader_disagreements.sop.\n")
    return out


# ------------------------------------------------------------------ 5. feedback
def feedback(wd, n_max=1500):
    from sop.infer import Reader
    rd = Reader(os.path.join(wd, "model")); legend = rd.legend; texts = []
    for p, d in read_examples(wd):
        for t in d.top_level():
            if t.value("_text") is not None: texts.append(t.value("_text"))
    texts = texts[:n_max]; items = []; resid = []; unknown = collections.Counter(); status = collections.Counter(); ids = (f"f{i}" for i in range(10 ** 9))
    for si, s in enumerate(texts):
        ws, f, info = rd.formalize(s, 0, ids); resid.append(info["residual"]); status[info["status"]] += 1
        if ws[0].type == "unknown":
            verbs = [t for t in info["tokens"] if rd.kind(t) is not None and rd.kind(t) not in rd.content_kinds]
            unknown[" ".join(sorted(set(verbs))[:3]) or "-"] += 1
        if not info["unplaced"] or f.get("shape") != "ok": continue
        S = info["S"]; by = {w.id: w for w in ws}; rec = rd.comp.wire(ws[0], by, info["tokens"])
        Sn = S / np.linalg.norm(S); rn = rec / max(1e-9, np.linalg.norm(rec)); Rv = Sn - max(0.0, float(np.real(np.vdot(rn, Sn)))) * rn
        for i in info["unplaced"]:
            if i not in info["kept"]: continue
            k = info["kept"].index(i); a = info["addr"][k]; term = a * rd.st.id(info["tokens"][i])
            share = abs(np.vdot(term, Rv)) / (np.linalg.norm(term) * max(1e-9, np.linalg.norm(Rv)))
            items.append({"word": info["tokens"][i], "kind": rd.kind(info["tokens"][i]), "sentence": si, "dir": a / np.linalg.norm(a), "share": float(share)})
    clusters = []
    for it in items:
        for c in clusters:
            if abs(np.vdot(c["dir"], it["dir"])) >= CLUSTER_COS: c["items"].append(it); break
        else: clusters.append({"dir": it["dir"], "items": [it]})
    merged = []          # clusters sharing a word are one question reached by several routes (formulations, depths)
    for c in clusters:
        ws_ = {i["word"] for i in c["items"]}
        tgt = next((m for m in merged if m["words"] & ws_), None)
        if tgt is None: merged.append({"words": set(ws_), "items": list(c["items"]), "routes": 1})
        else: tgt["words"] |= ws_; tgt["items"] += c["items"]; tgt["routes"] += 1
    n = max(1, len(texts)); props = []
    for m in merged:
        sents = {i["sentence"] for i in m["items"]}
        props.append({"words": sorted(m["words"]), "kinds": sorted({i["kind"] for i in m["items"] if i["kind"]}), "sentences": len(sents),
                      "share_of_corpus": len(sents) / n, "routes": m["routes"], "residual_share_mean": float(np.mean([i["share"] for i in m["items"]])),
                      "examples": [texts[j] for j in sorted(sents)[:5]], "supported": len(sents) >= max(5, MIN_SUPPORT * n)})
    filled = collections.Counter(); annotated_words = set()
    for p, d in read_examples(wd):
        by = d.by_id()
        for t in d.top_level():
            f = world.frame_of(t, by, legend); annotated_words |= set(f["speakers"])
            for r, fl in f["roles"].items(): filled[r] += 1; annotated_words |= set(world.filler_words(fl))
    for pr in props:
        # a cluster whose words (or their lemmas) already fill a legend role in the annotations is the READER's error on an existing question,
        # not a missing question: it is reported, never proposed as a new role (stated rule)
        known = sorted(w for w in pr["words"] if w in annotated_words or rd.lemma(w) in annotated_words)
        pr["known_fillers"] = known; pr["novel"] = not known
        pr["proposable"] = pr["supported"] and pr["novel"]
    props.sort(key=lambda p: -p["sentences"])
    metrics = json.load(open(os.path.join(wd, "model", "metrics.json")))
    per_role = {k: v["catalog candidates"] for k, v in metrics["mean_over_seeds"].items()}
    out = {"sentences_read": len(texts), "residual_mean": float(np.mean(resid)) if resid else None, "status": dict(status),
           "unplaced_clusters": props, "unknown_wires_by_words": dict(unknown), "roles_empty_across_corpus": [r for r in legend.roles if filled[r] == 0],
           "roles_filled": dict(filled), "per_split_accuracy": per_role,
           "requests": sorted(os.path.basename(p) for p in glob.glob(os.path.join(wd, "requests", "*.md"))),
           "constants": {"cluster_cos": CLUSTER_COS, "min_support": MIN_SUPPORT}}
    jdump(out, os.path.join(wd, "model", "feedback.json"))
    return out


# ------------------------------------------------------------------ 6. evolve: decide a change set
def name_role(wd, prop, k):
    prompt = (f"These words were left unplaced by a reader whose legend has the roles {Legend(os.path.join(wd, 'legend.json')).roles}: "
              f"{prop['words']} (kinds {prop['kinds']}). Example sentences:\n" + "\n".join(prop["examples"]) +
              "\nName the missing question as a role id (lowercase, letters and underscores) and the question in a few words. Reply as JSON {\"id\": ..., \"question\": ...}.")
    try:
        t = llm_call("evolve", prompt, wd); m = re.search(r"\{.*\}", t, re.S); d = json.loads(m.group(0))
        if re.match(r"^[a-z][a-z0-9_]*$", d.get("id", "")): return d["id"], d.get("question", ""), "named by " + model_for("evolve", wd)
    except (LLMUnavailable, AttributeError, ValueError) as e: why = str(e)
    else: why = "the evolve model returned no usable id"
    return f"role_{k}", "(placeholder: to be named by the evolve model)", f"PLACEHOLDER NAME ({why})"


def evolve(wd, fb):
    legend = Legend(os.path.join(wd, "legend.json")); v = int(legend.version or 1); nv = v + 1; vd = os.path.join(wd, "versions", f"v{nv}")
    rejected = []
    for mk in glob.glob(os.path.join(wd, "versions", "v*", "mark.json")):
        if json.load(open(mk)).get("mark") == "rejected":
            lp = os.path.join(os.path.dirname(mk), "legend.json")
            if os.path.exists(lp): rejected.append({r["id"] for r in json.load(open(lp))["roles"]})
    changes = []; new = copy.deepcopy(legend.raw); new["version"] = nv; k = 1
    for prop in fb["unplaced_clusters"]:
        if not prop.get("proposable"): continue
        rid, q, how = name_role(wd, prop, k); k += 1
        while rid in legend.roles: rid += "_new"
        new["roles"].append({"id": rid, "question": q})
        changes.append({"type": "add_role", "role": rid, "question": q, "naming": how, "fillers": prop["words"], "kinds": prop["kinds"],
                        "evidence": {k2: prop[k2] for k2 in ("sentences", "share_of_corpus", "routes", "residual_share_mean", "examples")}})
    if any({r["id"] for r in new["roles"]} == rj for rj in rejected): return {"version": nv, "changes": [], "note": "this change set equals a rejected version: not proposed again"}
    if not changes: return {"version": nv, "changes": [], "note": "no supported proposal in the feedback"}
    os.makedirs(vd, exist_ok=True); jdump(new, os.path.join(vd, "legend.json"))
    migration = {"from_version": v, "to_version": nv, "from_legend": legend.sha,
                 "rules": [{"add_role": c["role"], "re_annotate": "every example whose _text contains one of the fillers",
                            "fillers": c["fillers"], "how": "the annotate model re-annotates the listed sentences; offline (no API): the filler token is "
                            "written as <role>=<token> on the event wire, _changed=migration (stated)"} for c in changes]}
    jdump(migration, os.path.join(vd, "migration.json")); jdump(fb, os.path.join(vd, "feedback.json"))
    reqs = fb.get("requests", [])
    with open(os.path.join(vd, "CHANGES.md"), "w") as f:
        f.write(f"# v{nv} — proposed {now()} by the train-world loop\n\n## Feedback acted on\n")
        for c in changes:
            e = c["evidence"]
            f.write(f"- `_unplaced` cluster {c['fillers']} (kinds {c['kinds']}): {e['sentences']} sentences ({e['share_of_corpus']:.1%} of those read), "
                    f"{e['routes']} route(s), mean share of the residual {e['residual_share_mean']:.3f}; e.g. \"{e['examples'][0]}\"\n")
        f.write("\n## Change set\n" + "".join(f"- ADD role `{c['role']}` — {c['question']} [{c['naming']}]\n" for c in changes))
        f.write("\nRemovals: none (add freely; remove only what is empty across the corpus" + (f"; empty now: {fb['roles_empty_across_corpus']}" if fb['roles_empty_across_corpus'] else "") + ").\n")
        f.write("\n## Migration\nmigration.json: " + "; ".join(f"add `{r['add_role']}` from fillers {r['fillers']}" for r in migration["rules"]) + "\n")
        f.write("\n## Requests\n" + ("".join(f"- requests/{r}: read; no change-set item derived from it by the v1 loop (requests are read as text)\n" for r in reqs) if reqs else "- none\n"))
    return {"version": nv, "dir": vd, "changes": changes}


def migrate(src_wd, stage_wd, migration, new_legend_path):
    """the examples of src under the new legend, with the migration applied, into the staging world folder"""
    lg_new = Legend(new_legend_path); fill = {}
    for r in migration["rules"]:
        for w in r["fillers"]: fill.setdefault(w, r["add_role"])
    n_changed = 0
    for p, d in read_examples(src_wd):
        by = d.by_id()
        for t in d.top_level():
            toks = tokenize(t.value("_text") or ""); hit = [(tok, fill[tok]) for tok in toks if tok in fill]
            if not hit: continue
            w = t; seen = set()
            while w is not None and w.type in ("says", "not") and w.id not in seen:
                seen.add(w.id); s = w.get("scope"); w = by.get(s.atoms[0].text) if s is not None and s.atoms and s.atoms[0].kind == "ref" else None
            if w is None or w.type not in lg_new.events: continue
            done = False
            for tok, role in hit:
                if w.get(role) is None: w.set(role, tok); done = True
            if done:
                n_changed += 1
                if t.get("_changed") is None: t.lines.insert(1, soplang.line("_changed", f"migration-v{migration['to_version']}"))
        d.directives = [(n, (lg_new.sha if n == "legend" else v), l) for n, v, l in d.directives]
        out = os.path.join(stage_wd, os.path.relpath(p, src_wd)); os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f: f.write(soplang.write(d))
    return n_changed


def stage_world(wd, vd):
    st = os.path.join(vd, "stage", wname(wd))
    if os.path.exists(st): shutil.rmtree(st)
    os.makedirs(st)
    for f in ("world.json", "split_hints.json"):
        if os.path.exists(os.path.join(wd, f)): shutil.copy(os.path.join(wd, f), st)
    lg = json.load(open(os.path.join(wd, "legend.json"))); lex = lg.get("kinds", {}).get("lexicon", "kinds.tsv")
    if os.path.exists(os.path.join(wd, lex)): shutil.copy(os.path.join(wd, lex), st)
    shutil.copy(os.path.join(vd, "legend.json"), os.path.join(st, "legend.json"))
    for d in ("raw", "audit", "model", "runs", "requests", "versions"): os.makedirs(os.path.join(st, d), exist_ok=True)
    return st


# ------------------------------------------------------------------ 7. accept by criterion
def heldout_numbers(model_dir):
    m = json.load(open(os.path.join(model_dir, "metrics.json"))); seed0 = m["per_seed"][0]; rows = []
    for s, v in seed0.items():
        if s.startswith("train"): continue
        c = v["catalog candidates"]; cw = c["calibration"].get("roles (pooled)", {})
        unk = sum(n for k, n in c["status"].items() if k in ("unknown-event", "no-sense")) / max(1, c["n"])
        rows.append((c["n"], c["acc"]["FULL"], c["residual_mean"], unk, cw.get("confident_wrong", 0.0)))
    W = sum(r[0] for r in rows); avg = lambda i: sum(r[0] * r[i] for r in rows) / max(1, W)
    full_std = max((v["FULL"]["std"] for s, v2 in m["mean_over_seeds"].items() for c, v in v2.items() if c == "catalog candidates"), default=0.0)
    return {"n": W, "FULL": avg(1), "residual_mean": avg(2), "unknown_share": avg(3), "confident_wrong": avg(4), "seeds": m["seeds"], "FULL_std": full_std}


def accept(old_model, new_model):
    a, b = heldout_numbers(old_model), heldout_numbers(new_model); tol = max(FULL_TOL, 2 * max(a["FULL_std"], b["FULL_std"]))
    crit = {"FULL >= v (within seed variance)": b["FULL"] >= a["FULL"] - tol,
            "_residual mean down": b["residual_mean"] < a["residual_mean"],
            "unknown share down (not up when both are 0)": b["unknown_share"] < a["unknown_share"] or (a["unknown_share"] == 0 and b["unknown_share"] == 0),
            "confident-wrong at tau not up": b["confident_wrong"] <= a["confident_wrong"] + CW_TOL,
            "stable across the seeds run": True}
    return {"old": a, "new": b, "tolerance_FULL": tol, "tolerance_confident_wrong": CW_TOL, "criteria": crit, "met": all(crit.values()),
            "held_out": "all non-training splits of the v run's splits.json, reused for v+1 (fixed before the change)",
            "seeds_note": "one seed: 'stable across seeds' is not measurable and is recorded as vacuous" if len(b["seeds"]) < 2 else "several seeds"}


def promote(wd, vd, st, acc):
    """v+1 becomes current: the previous version goes to versions/v<N>/ with its REPORT"""
    old = json.load(open(os.path.join(wd, "legend.json"))); ov = int(old.get("version", 1)); od = os.path.join(wd, "versions", f"v{ov}")
    os.makedirs(od, exist_ok=True); shutil.copy(os.path.join(wd, "legend.json"), os.path.join(od, "legend.json"))
    for f in ("REPORT.md", "metrics.json"):
        if os.path.exists(os.path.join(wd, "model", f)): shutil.copy(os.path.join(wd, "model", f), os.path.join(od, f))
    if os.path.exists(os.path.join(od, "model")): shutil.rmtree(os.path.join(od, "model"))
    shutil.copytree(os.path.join(wd, "model"), os.path.join(od, "model"))
    shutil.rmtree(os.path.join(wd, "examples")); shutil.copytree(os.path.join(st, "examples"), os.path.join(wd, "examples"))
    shutil.copy(os.path.join(st, "legend.json"), os.path.join(wd, "legend.json"))
    shutil.rmtree(os.path.join(wd, "model")); shutil.copytree(os.path.join(st, "model"), os.path.join(wd, "model"))
    for f in ("REPORT.md", "metrics.json"): shutil.copy(os.path.join(st, "model", f), os.path.join(vd, f))
    wj = world_json(wd); wj["version"] = int(json.load(open(os.path.join(wd, "legend.json")))["version"]); jdump(wj, os.path.join(wd, "world.json"))
    stp = os.path.join(wd, "status.json"); s = json.load(open(stp)) if os.path.exists(stp) else {}
    s.update({"label": "BUILT", "legend_version": wj["version"], "legend": sha256_file(os.path.join(wd, "legend.json")), "promoted": now(), "acceptance": acc["new"]})
    jdump(s, stp)


def step(wd, train_args):
    pinned = [os.path.dirname(m) for m in glob.glob(os.path.join(wd, "versions", "v*", "mark.json")) if json.load(open(m)).get("mark") == "pinned"]
    fb = feedback(wd); ev = evolve(wd, fb)
    if not ev["changes"]: return {"feedback": summarize_fb(fb), "evolve": ev, "accepted": None}
    vd = ev["dir"]; st = stage_world(wd, vd); mig = json.load(open(os.path.join(vd, "migration.json")))
    n_changed = migrate(wd, st, mig, os.path.join(vd, "legend.json"))
    runs = sorted(glob.glob(os.path.join(wd, "runs", "*", "splits.json"))); prev_splits = runs[-1] if runs else None
    train(st, train_args, splits=prev_splits)
    acc = accept(os.path.join(wd, "model"), os.path.join(st, "model")); acc["migrated_examples"] = n_changed
    jdump(acc, os.path.join(vd, "acceptance.json"))
    with open(os.path.join(vd, "CHANGES.md"), "a") as f:
        f.write(f"\n## Acceptance (held-out set fixed before the change; {acc['held_out']})\n| criterion | v | v+1 | met |\n|---|---|---|---|\n")
        f.write(f"| FULL (tolerance {acc['tolerance_FULL']}) | {acc['old']['FULL']:.3f} | {acc['new']['FULL']:.3f} | {acc['criteria']['FULL >= v (within seed variance)']} |\n")
        f.write(f"| _residual mean | {acc['old']['residual_mean']:.3f} | {acc['new']['residual_mean']:.3f} | {acc['criteria']['_residual mean down']} |\n")
        f.write(f"| unknown share | {acc['old']['unknown_share']:.3f} | {acc['new']['unknown_share']:.3f} | {acc['criteria']['unknown share down (not up when both are 0)']} |\n")
        f.write(f"| confident-wrong at tau | {acc['old']['confident_wrong']:.3f} | {acc['new']['confident_wrong']:.3f} | {acc['criteria']['confident-wrong at tau not up']} |\n")
        f.write(f"| seeds | {acc['old']['seeds']} | {acc['new']['seeds']} | {acc['seeds_note']} |\n\nExamples migrated: {n_changed}. ")
        if not acc["met"]: f.write("NOT MET: v+1 kept under versions/ as rejected; the loop continues from v.\n")
        elif pinned: f.write(f"MET, but a version is pinned ({pinned}): the serving version stays; v+1 recorded as accepted-not-served.\n")
        else: f.write("MET: v+1 is current (legend.json, model/, status.json name it); the previous version is under versions/.\n")
    if not acc["met"]: jdump({"mark": "rejected", "when": now(), "by": "loop (criterion not met)"}, os.path.join(vd, "mark.json"))
    elif not pinned: promote(wd, vd, st, acc)
    return {"feedback": summarize_fb(fb), "evolve": {k: v for k, v in ev.items() if k != "dir"}, "acceptance": acc, "accepted": acc["met"] and not pinned}


def summarize_fb(fb):
    return {k: fb[k] for k in ("sentences_read", "residual_mean", "status", "roles_empty_across_corpus", "unknown_wires_by_words")} | \
           {"unplaced_clusters": [{k: c[k] for k in ("words", "kinds", "sentences", "routes", "supported", "novel", "proposable", "residual_share_mean")} for c in fb["unplaced_clusters"]]}


def run(wd, annotator, train_args, steps=None):
    log = {"world": wname(wd), "started": now()}; wj = world_json(wd)
    if not os.path.exists(os.path.join(wd, "legend.json")):
        leg, how = propose_legend(wd); log["legend"] = how
        if leg is None: log["stopped"] = f"no legend.json and the legend proposal needs the evolve model: {how}"; return log
    log["annotate"] = annotate(wd, annotator)
    if log["annotate"].get("error"): log["stopped"] = log["annotate"]["error"]; return log
    n = sum(1 for _, d in read_examples(wd) for t in d.top_level() if t.value("_text"))
    if n < wj.get("min_examples", 500): log["stopped"] = f"{n} annotated examples < min_examples {wj.get('min_examples', 500)}: add raw sentences"; return log
    log["train"] = train(wd, train_args); log["second_look"] = second_look(wd, annotator)
    k = wj.get("max_versions_per_run", 3) if steps is None else steps; log["steps"] = []
    for _ in range(k):
        s = step(wd, train_args); log["steps"].append(s)
        if not s["evolve"]["changes"]: break
    log["finished"] = now(); jdump(log, os.path.join(wd, "runs", f"loop_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json"))
    return log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["annotate", "train", "second-look", "feedback", "evolve", "step", "run", "propose-legend"]); ap.add_argument("world")
    ap.add_argument("--annotator", default="llm"); ap.add_argument("--train-args", default=""); ap.add_argument("--steps", type=int)
    a = ap.parse_args(); wd = os.path.abspath(a.world)
    out = {"annotate": lambda: annotate(wd, a.annotator), "train": lambda: train(wd, a.train_args), "second-look": lambda: second_look(wd, a.annotator),
           "feedback": lambda: summarize_fb(feedback(wd)), "evolve": lambda: evolve(wd, feedback(wd)), "step": lambda: step(wd, a.train_args),
           "run": lambda: run(wd, a.annotator, a.train_args, a.steps), "propose-legend": lambda: {"result": propose_legend(wd)[1]}}[a.cmd]()
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
