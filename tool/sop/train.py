"""
train.py — `sop train <world>`: legend.json + examples/*.sop -> <world>/model/ (spec 4), with torch.

What is GIVEN and what is LEARNED (rule 20): the legend's roles, the identities (stars regenerated from the seed), the kinds (lexicon) and
the event synonym table are inputs; the reader learns the route book only — per-kind displacements of the path, bigram and chain codes
and the gates (which kinds emit), by the single generative objective max cos(S, S_world) (structured per-sentence targets with pre-set
roles; the model learns the assignment). The model and the loop are the repository's: experiment/v8_gates.V8 (path + chainL + bigram,
gates learned from an all-closed start) and experiment/trainer.train_loop (memory logged, checkpoint, finite-loss assertion; rule 21).

Steps: check every example against the legend (abort on errors, or on a legend-hash mismatch unless --force); build the rows (one per
top-level wire with _text); SPLITS written to runs/<ts>/splits.json BEFORE training (seen formulations = train + held-out sentences;
held-out formulations; new identities; held-out depth); trace of one row (rule 11); training; export of the route book; numpy/torch
agreement check; thresholds set on CALIBRATION rows = a sample of the training split (rule 25); evaluation of every split with the CLI's
own numpy reader; metrics.json, REPORT.md, background.npz, status.json.
TODO(section 6 missing): the spec's training skill (6.1-6.7) and its split definitions were not received; the splits here are the minimum
stated in the task (and in split_hints.json when a world provides one).
"""
import argparse, collections, datetime, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(HERE)); EXP = os.path.join(REPO, "experiment")

from . import soplang, check as checkmod, world
from .legend import Legend, tokenize

TAUS = (0.0, 0.02, 0.05, 0.1, 0.15)


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x):
        for s in self.s: s.write(x)
    def flush(self):
        for s in self.s: s.flush()


def h01(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little") / 2 ** 64


def git_commit():
    try:
        c = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", "tool/sop", "experiment"], capture_output=True, text=True).stdout.strip()
        return c + ("-dirty" if dirty else "")
    except Exception: return "unknown"


# ------------------------------------------------------------------ rows
def load_rows(world_dir, legend, force=False):
    rows, errors = [], []
    exdir = os.path.join(world_dir, "examples")
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(exdir) for f in fs if f.endswith(".sop"))
    comp = world.Compiler(legend, world.Stars(0, 8))
    for p in files:
        rep, doc = checkmod.check_file(p, {os.path.basename(os.path.abspath(world_dir)): legend}, require_text=True, force=force)
        if not rep.ok: errors += rep.lines(); continue
        by = doc.by_id(); rel = os.path.relpath(p, world_dir)
        for w in doc.top_level():
            text = w.value("_text"); wname = w.value("_world") or doc.directive("world")
            if text is None: continue
            if wname is not None and wname != os.path.basename(os.path.abspath(world_dir)):
                print(f"[rows] {rel}: wire {w.id} has _world={wname}, not this world ({os.path.basename(os.path.abspath(world_dir))}): skipped"); continue
            toks = tokenize(text); f = world.frame_of(w, by, legend)
            if f["shape"] == "ok" and legend.reader["predicate_role"] and not f.get("lemma"):
                f["lemma"] = comp.lemma_for(f["event"], toks, set())
            rows.append({"id": f"{rel}#{w.id}", "file": rel, "text": text, "tokens": toks, "wire": w, "by": by, "frame": f, "template": w.value("_template"),
                         "depth": len(f["speakers"]), "source": w.value("_source", "?")})
    return rows, errors, files


def content_kinds_of(rows, legend):
    ks = set()
    for r in rows:
        for fl in r["frame"]["roles"].values():
            for wd in world.filler_words(fl):
                k = legend.kind(wd)
                if k: ks.add(k)
        for sp in r["frame"]["speakers"]:
            k = legend.kind(sp or "")
            if k: ks.add(k)
    per = collections.Counter(legend.lexicon.values())
    return ks | {k for k, n in per.items() if n >= 2}


# ------------------------------------------------------------------ splits
def make_splits(rows, legend, hints, seed):
    share = hints.get("heldout_sentence_share", 0.1)
    t2split = {t: name for name, ts in hints.get("template_splits", {}).items() for t in ts}
    new_ids = set(hints.get("new_identities", [])); defs = {"seed": seed, "heldout_sentence_share": share,
                                                          "template_splits": hints.get("template_splits", {}), "source": "split_hints.json" if hints else "automatic"}
    if not hints:      # automatic: TODO(section 6 missing)
        by_kind = collections.defaultdict(set)
        for r in rows:
            for fl in r["frame"]["roles"].values():
                for wd in world.filler_words(fl):
                    if legend.kind(wd): by_kind[legend.kind(wd)].add(wd)
        for k, ws in by_kind.items():
            if len(ws) >= 6:
                new_ids |= set(sorted(ws, key=lambda w: h01(seed, "id", w))[:max(1, len(ws) // 9)])
        sigs = sorted({signature(r, legend) for r in rows})
        if len(sigs) >= 7:
            held = set(sorted(sigs, key=lambda s: h01(seed, "sig", s))[:max(1, len(sigs) * 15 // 100)])
            for s in held: t2split["sig:" + s] = "heldout_formulations"
            defs["heldout_signatures"] = sorted(held)
    defs["new_identities"] = sorted(new_ids); dmin = hints.get("heldout_depth_min"); defs["heldout_depth_min"] = dmin
    fsplit = hints.get("file_splits", {}); defs["file_splits"] = fsplit
    out = collections.defaultdict(list)
    for r in rows:
        words = {w for fl in r["frame"]["roles"].values() for w in world.filler_words(fl)} | set(r["frame"]["speakers"])
        key = r["template"] if r["template"] in t2split else "sig:" + signature(r, legend)
        if r["file"] in fsplit: s = fsplit[r["file"]]
        elif words & new_ids: s = "new_identities"
        elif key in t2split: s = t2split[key]
        elif dmin is not None and r["depth"] >= dmin: s = "heldout_depth"
        elif h01(seed, "sent", r["text"]) < share: s = "heldout_sentences"
        else: s = "train"
        r["split"] = s; out[s].append(r["id"])
    return defs, out


def signature(r, legend):
    """the formulation of a sentence: tokens with the filler words replaced by their kind"""
    fill = {w for fl in r["frame"]["roles"].values() for w in world.filler_words(fl)} | set(r["frame"]["speakers"])
    return " ".join((legend.kind(t) or "?") if t in fill or legend.lemma_of(t) in fill else t for t in r["tokens"])


# ------------------------------------------------------------------ schema
def build_schema(rows, legend):
    ok = [r["frame"] for r in rows if r["frame"]["shape"] == "ok"]; n = max(1, len(ok)); roles = {}
    for r in legend.roles:
        fs = [f["roles"][r] for f in ok if r in f["roles"]]
        if not fs: continue
        shapes = collections.Counter(x[0] for x in fs); ents = [x for x in fs if x[0] == "entity"]
        kinds = sorted({legend.kind(x[1]) for x in fs if x[0] in ("word", "negword") and legend.kind(x[1])})
        rs = {"rate": len(fs) / n, "shapes": dict(shapes), "main": "entity" if shapes["entity"] >= shapes["word"] + shapes["negword"] else "word",
              "kinds": kinds, "negword": shapes["negword"] > 0}
        if ents:
            rs["head_kinds"] = sorted({legend.kind(x[1]) for x in ents if legend.kind(x[1])})
            at = [x[2] for x in ents if x[2] is not None]; dt = [x[3] for x in ents if x[3] is not None]
            rs["attr_kinds"] = sorted({legend.kind(a[1]) for a in at if a[0] in ("word", "negword") and legend.kind(a[1])})
            rs["attr_neg"] = any(a[0] == "negword" for a in at); rs["attr_rate"] = len(at) / len(ents)
            rs["det_kinds"] = sorted({legend.kind(d) for d in dt if legend.kind(d)}); rs["det_rate"] = len(dt) / len(ents)
        roles[r] = rs
    core = [r for r, rs in roles.items() if rs["rate"] >= 0.95] or ([max(roles, key=lambda r: roles[r]["rate"])] if roles else [])
    sk = sorted({legend.kind(s) for f in ok for s in f["speakers"] if s and legend.kind(s)})
    return {"roles": roles, "role_order": [r for r in legend.roles if r in roles], "core_roles": core, "speaker_kinds": sk,
            "max_depth": max([len(f["speakers"]) for f in ok] or [0]), "neg_event": any(f["neg"] for f in ok),
            "single_event": next(iter(legend.events)) if len(legend.events) == 1 else None,
            "unsupported_shapes": dict(collections.Counter(r["frame"]["shape"] for r in rows if r["frame"]["shape"] != "ok"))}


# ------------------------------------------------------------------ evaluation
def head_word(fl):
    if fl is None: return None
    if fl[0] in ("word", "negword", "entity"): return fl[1]
    return json.dumps(fl)


class Calib:
    """(correct, margin) per axis -> accuracy, margin distribution, confident-wrong at tau, coverage, abstention split, acc-vs-coverage"""
    def __init__(self): self.rows = collections.defaultdict(list)
    def add(self, axis, ok, margin): self.rows[axis].append((bool(ok), float(margin)))
    def report(self, axis, tau):
        v = np.array(self.rows[axis]); ok, mg = v[:, 0].astype(bool), v[:, 1]; conf = mg >= tau
        return {"n": int(len(v)), "acc": float(ok.mean()), "margin_mean": float(mg.mean()), "margin_p10": float(np.percentile(mg, 10)),
                "tau": tau, "coverage": float(conf.mean()), "confident_wrong": float((conf & ~ok).mean()),
                "abstain_right": float((~conf & ok).mean()), "abstain_wrong": float((~conf & ~ok).mean()),
                "acc_at_coverage": {f"{t:.2f}": [float(ok[mg >= t].mean()) if (mg >= t).any() else None, float((mg >= t).mean())] for t in TAUS}}


def lexical_baseline(train_rows, legend, schema):
    """order baseline (rule 9): for each role, the filler is the token at the rank (among tokens of the role's head kinds) that held it
    most often in training; absent if the sentence has no token at that rank"""
    rank = {}
    for r, rs in schema["roles"].items():
        kinds = set(rs.get("head_kinds") or rs["kinds"]); c = collections.Counter()
        for row in train_rows:
            fl = row["frame"]["roles"].get(r); hw = head_word(fl)
            cand = [t for t in row["tokens"] if legend.kind(t) in kinds]
            c[cand.index(hw) if hw in cand else -1] += 1
        rank[r] = (c.most_common(1)[0][0] if c else -1, sorted(kinds))
    def predict(tokens):
        out = {}
        for r, (k, kinds) in rank.items():
            cand = [t for t in tokens if legend.kind(t) in set(kinds)]
            out[r] = cand[k] if 0 <= k < len(cand) else None
        return out
    return predict, {r: v[0] for r, v in rank.items()}


def evaluate(reader, rows, legend, schema, tau, filt, lexical, n_max, rng_seed=0):
    rs = rows if len(rows) <= n_max else [rows[i] for i in sorted(np.random.default_rng(rng_seed).choice(len(rows), n_max, replace=False))]
    cal = Calib(); acc = collections.defaultdict(list); status = collections.Counter(); resid = []; ids = (f"e{i}" for i in range(10 ** 9))
    for row in rs:
        g = row["frame"]; wires, f, info = reader.formalize(row["text"], 0, ids, filt); status[info["status"]] += 1; resid.append(info["residual"])
        f_ok = f["shape"] == "ok"
        acc["FULL"].append(f_ok and g["shape"] == "ok" and world.canon(f, legend.reader["predicate_role"]) == world.canon(g, legend.reader["predicate_role"]))
        acc["structure (depth, negation)"].append(f_ok and len(f["speakers"]) == len(g["speakers"]) and f["neg"] == g["neg"])
        mg = f.get("margins", {})
        if legend.reader["predicate_role"]:
            ok = f_ok and f.get("lemma") == g.get("lemma"); acc["predicate (surface lemma)"].append(ok); cal.add("predicate", ok, mg.get("event", 0))
            acc["event type (synonym table)"].append(f_ok and f.get("event") == g.get("event"))
        lex = lexical(row["tokens"]); heads_m, heads_l = [], []
        for r in schema["role_order"]:
            gf, pf = g["roles"].get(r), f["roles"].get(r) if f_ok else None
            ok = json.dumps(gf) == json.dumps(pf); acc[f"role {r}"].append(ok)
            hm, hl = head_word(gf) == head_word(pf), head_word(gf) == lex.get(r); acc[f"head {r}"].append(hm); acc[f"lexical head {r}"].append(hl)
            heads_m.append(hm); heads_l.append(hl)
            if gf is not None: cal.add(f"role {r}", ok, mg.get(r, 0)); cal.add("roles (pooled)", ok, mg.get(r, 0))
        for k, sp in enumerate(g["speakers"]):
            ok = f_ok and k < len(f["speakers"]) and f["speakers"][k] == sp; acc["speakers"].append(ok); cal.add("speakers", ok, mg.get(f"speaker.{k}", 0))
        acc["heads FULL (model)"].append(all(heads_m)); acc["heads FULL (lexical order baseline)"].append(all(heads_l))
    m = {"n": len(rs), "of": len(rows), "acc": {k: float(np.mean(v)) for k, v in acc.items()},
         "calibration": {ax: cal.report(ax, tau) for ax in cal.rows}, "residual_mean": float(np.mean(resid)) if resid else None,
         "status": dict(status)}
    return m


def chance(reader, schema):
    out = {}
    for r, rs in schema["roles"].items():
        n = len(reader.cands(rs.get("head_kinds") or rs["kinds"])[0]); out[r] = 1 / n if n else None
    if reader.legend.reader["predicate_role"]: out["predicate"] = 1 / max(1, len(reader.legend.all_verbs()))
    return out


# ------------------------------------------------------------------ export
def export_routebook(m, path):
    import torch
    th = (m.theta * m.root_mask).detach().cpu().numpy()
    wrap = lambda x: np.angle(np.exp(1j * x)).astype(np.float32)
    z = {"cats": np.array(m.cats), "codes": np.array(m.codes), "path": wrap(th[m.ipath.cpu().numpy()]), "bg0": wrap(th[m.ibg0.cpu().numpy()]),
         "bg": wrap(th[m.ibg.cpu().numpy()]), "A": wrap(th[m.iA.cpu().numpy()]), "L": wrap(th[m.iL.cpu().numpy()]),
         "logits": m.logits.detach().cpu().numpy().astype(np.float64), "gates": torch.sigmoid(m.logits).detach().cpu().numpy()}
    if getattr(m, "rel", None) is not None:
        z.update(rel=wrap(m.rel.detach().cpu().numpy()), rel_mask=m.rel_mask.cpu().numpy(), taker_mask=m.taker_mask.cpu().numpy())
    np.savez_compressed(path, **z)


class Shim:
    """the minimal world object experiment/v8_gates.V8 needs: vocab, idx, cat (word -> kind)"""
    def __init__(self, words_kinds):
        self.vocab = sorted(words_kinds); self.idx = {x: i for i, x in enumerate(self.vocab)}; self.cat = dict(words_kinds)


def train_one(a, legend, rows, splits, schema, seed, model_dir, run_dir):
    os.environ.setdefault("VSA_MIN_AVAILABLE_GB", "4")
    if a.device == "cpu": os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, EXP)
    import torch
    import trainer, v8_gates as v8
    from .infer import Reader
    torch.manual_seed(seed)
    st = world.Stars(seed, a.D, a.star_seed); comp = world.Compiler(legend, st)
    wk = dict(legend.lexicon)
    for r in rows:
        for t in r["tokens"]:
            k = legend.kind(t)
            if k is not None: wk.setdefault(t, k)
    shim = Shim(wk)
    train = [r for r in rows if r.get("split") == "train" and r["frame"]["shape"] != "unknown"]
    order = []
    rng = np.random.default_rng(seed)
    for e in range(a.epochs): order += list(rng.permutation(len(train)) if e > 0 or a.shuffle else range(len(train)))
    if a.max_train: order = order[:a.max_train]
    sents = [[t for t in train[i]["tokens"] if legend.kind(t) is not None] for i in order]
    targets = np.array([comp.wire(train[i]["wire"], train[i]["by"], train[i]["tokens"]) for i in order])
    print(f"[train] seed {seed}: {len(train)} training rows x {a.epochs} epoch(s) -> {len(sents)} steps' sentences; D={a.D}; kinds {len(set(wk.values()))}; "
          f"vocab {len(shim.vocab)}; predicate lemma not found in {comp.missing_lemma} targets", flush=True)
    F = np.array([st.id(w) for w in shim.vocab])
    m = v8.V8(shim, a.D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)
    params = [m.theta, m.logits]
    # trace (rule 11): one row, its kinds, and the terms of its target
    r0 = train[order[0]] if order else None
    if r0 is not None:
        print(f"[trace] text: {r0['text']}\n[trace] tokens/kinds: " + " ".join(f"{t}/{legend.kind(t)}" for t in r0['tokens']))
        print(f"[trace] gold frame: {world.canon(r0['frame'])}")
        print(f"[trace] target |S_world| = {np.linalg.norm(targets[0]):.2f} (sqrt(D) per unit term = {np.sqrt(a.D):.1f}); initial gates all {float(torch.sigmoid(m.logits[0])):.2f} (closed start)")
    cats, ids, mask = m.encode(sents); tgt = torch.tensor(targets, dtype=m.F.dtype, device=m.device)
    t0 = time.time()
    trainer.train_loop(m, params, cats, ids, mask, tgt, a.batch, a.lr, log_every=a.log_every, ckpt=os.path.join(run_dir, f"ckpt_seed{seed}.pt"),
                       resume=a.resume, tag=f" seed {seed}")
    t_train = time.time() - t0
    os.makedirs(model_dir, exist_ok=True)
    export_routebook(m, os.path.join(model_dir, "routebook.npz"))
    shutil.copyfile(legend.path, os.path.join(model_dir, "legend.json"))
    catalog = {"seed": seed, "star_seed": a.star_seed, "star": "identity of a word = sha256(star_seed + ':id:' + word)[:8] little-endian -> numpy default_rng -> "
               "D uniform phases in [-pi, pi); role vectors the same with the world seed and 'role:<id>' / 'op:<name>'",
               "words": {w: ({"kind": k, "lemma": legend.lemma[w]} if w in legend.lemma else {"kind": k}) for w, k in sorted(wk.items())},
               "schema": schema}
    with open(os.path.join(model_dir, "catalog.json"), "w") as f: json.dump(catalog, f, indent=1, sort_keys=True)
    cfg = {"world": a.world_name, "D": a.D, "seed": seed, "language": a.language, "legend_sha256": legend.sha, "legend_version": legend.version,
           "training_code_commit": a.commit, "trained": datetime.datetime.now().isoformat(timespec="seconds"), "codes": ["path", "chainL", "bigram"],
           "gate_init": "closed", "batch": a.batch, "lr": a.lr, "epochs": a.epochs, "n_train_steps_sentences": len(sents),
           "thresholds": {"tau": a.tau, "tau_source": "stated (the repository's v16 value), not fitted on any rows", "no_sense_share": 0.5,
                          "no_sense_source": "stated", "presence": {}, "attr_presence": {}, "det_presence": {}}}
    with open(os.path.join(model_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=1)
    # numpy (inference) vs torch (training) agreement on three rows
    rd = Reader(model_dir); worst = 0.0
    with torch.no_grad():
        for s in sents[:3]:
            St = m.states(*m.encode([s]))[0].cpu().numpy(); Sn, _, _, _ = rd.chart(s)
            worst = max(worst, float(np.linalg.norm(St - Sn) / np.linalg.norm(St)))
    print(f"[export] numpy reader vs torch states: max relative difference {worst:.2e} (float32 phases in the route book)", flush=True)
    assert worst < 1e-3, "numpy inference does not reproduce the trained model"
    # calibration rows (rule 25): a sample of the TRAINING split, never the scored rows
    cal_rows = [train[i] for i in sorted(np.random.default_rng(1234 + seed).choice(len(train), min(a.n_cal, len(train)), replace=False))]
    stats = collections.defaultdict(lambda: {"present": [], "absent": []}); resid = []
    for row in cal_rows:
        g = row["frame"]
        if g["shape"] != "ok": continue
        S, kept, _, _ = rd.chart(row["tokens"])
        f = rd.read(S, row["tokens"], force=(len(g["speakers"]), g["neg"]))
        for r in schema["role_order"]:
            gf = g["roles"].get(r); stats[("presence", r)]["present" if gf is not None else "absent"].append(f["presence"].get(r, 0))
            if gf is not None and gf[0] == "entity":
                for part, idx in (("attr", 2), ("det", 3)):
                    if r + "." + part in f["presence"]: stats[(part + "_presence", r)]["present" if gf[idx] is not None else "absent"].append(f["presence"][r + "." + part])
    th = cfg["thresholds"]; calrep = {}
    for (kind, r), v in stats.items():
        if len(v["present"]) >= 5 and len(v["absent"]) >= 5:
            mp, ma = float(np.mean(v["present"])), float(np.mean(v["absent"])); th[kind][r] = 0.5 * (mp + ma)
            calrep[f"{kind}:{r}"] = {"mean_present": mp, "mean_absent": ma, "threshold": th[kind][r], "n_present": len(v["present"]), "n_absent": len(v["absent"])}
        else:
            th[kind][r] = None; calrep[f"{kind}:{r}"] = {"threshold": None, "n_present": len(v["present"]), "n_absent": len(v["absent"]),
                                                         "rule": "one class missing in the calibration rows: read by the training rate (always / never)"}
    th["calibration_split"] = f"train (a sample of {len(cal_rows)} training sentences, rng 1234+seed); never the scored rows (rule 25)"
    with open(os.path.join(model_dir, "config.json"), "w") as f: json.dump(cfg, f, indent=1)
    rd = Reader(model_dir)
    ids = (f"c{i}" for i in range(10 ** 9))
    for row in cal_rows: resid.append(rd.formalize(row["text"], 0, ids)[2]["residual"])
    pres = {k: np.array(v["present"] + v["absent"]) for (kk, r), v in stats.items() for k in [f"{kk}:{r}"]}
    np.savez(os.path.join(model_dir, "background.npz"), residual=np.array(resid), residual_mean=np.mean(resid) if resid else 1.0,
             residual_std=np.std(resid) if resid else 0.0, **{f"presence_{k}": v for k, v in pres.items()})
    gates = {c: float(g) for c, g in zip(m.cats, torch.sigmoid(m.logits).detach().cpu().numpy())}
    return rd, {"t_train_s": round(t_train, 1), "calibration": calrep, "gates": gates, "numpy_torch_diff": worst}


def run_eval(rd, rows, legend, schema, a, lexical, seen_words, new_ids):
    by = collections.defaultdict(list)
    for r in rows: by[r["split"]].append(r)
    out = {}
    for s in sorted(by):
        if s == "train": continue
        filt = None
        if s == "new_identities":
            new_kinds = {legend.kind(w) for w in new_ids}
            filt = set(new_ids) | {w for w in seen_words if legend.kind(w) not in new_kinds}
        else: filt = set(seen_words)
        out[s] = {"catalog candidates": evaluate(rd, by[s], legend, schema, a.tau, None, lexical, a.n_eval),
                  "experiment candidates": evaluate(rd, by[s], legend, schema, a.tau, filt, lexical, a.n_eval)}
    trs = by.get("train", [])
    out["train (sample)"] = {"catalog candidates": evaluate(rd, trs, legend, schema, a.tau, None, lexical, min(a.n_eval, 300), 7)}
    return out


def write_report(path, a, legend, rows, splits, defs, schema, metrics, extra, chance_, n_files):
    L = []; P = L.append
    P(f"# REPORT — world `{a.world_name}`, legend {legend.sha[:19]}… (version {legend.version})\n")
    P("Labels (rule 22): every number below is BUILT (code and measurement exist, computed by this training run). Nothing here is PLANNED "
      "or HYPOTHESIS except where a line says so.\n")
    P("## Given and learned (rule 20)\nGIVEN: the legend's roles and their vectors, the identities (stars regenerated from the seed), the kinds "
      "(lexicon), the event synonym table, the reader block of the legend. LEARNED: the route book only — per-kind displacements of the "
      "path / bigram / chain codes and the gates (sigmoid per kind, from an all-closed start), by the single objective max cos(S, S_world) "
      "with structured per-sentence targets with pre-set roles; the model learns the assignment.\n")
    srcs = collections.Counter(r["source"] for r in rows)
    P(f"## Data\n{len(rows)} annotated sentences from {n_files} files; provenance: {dict(srcs)}. Audit: none in this world folder "
      "(audit/ empty) — agreement not measured. Shapes the v1 decoder cannot read (trained as targets, scored as FULL-wrong): "
      f"{schema['unsupported_shapes'] or 'none'}.\n")
    P(f"## Splits (written to runs/{os.path.basename(a.run_dir)}/splits.json before training; source: {defs['source']})")
    for s, ids in sorted(splits.items()): P(f"- {s}: {len(ids)} sentences")
    P(f"\nNew identities: {defs['new_identities']}. Held-out depth from {defs.get('heldout_depth_min')}. "
      "TODO(section 6 missing): the spec's split definitions (section 6) were not received; these are the minimum stated in the task.\n")
    P(f"## Training\nD={a.D}, seeds {a.seeds_list}, batch {a.batch}, lr {a.lr}, epochs {a.epochs}, device {a.device}, codes path+chainL+bigram, "
      f"gates learned (closed start). Code commit {a.commit}. Memory curve: runs/{os.path.basename(a.run_dir)}/train.log (trainer.train_loop, rule 21).\n")
    P("## Thresholds (rule 25)\nAbstention tau = %.2f: stated, not fitted. Presence / attribute / determiner thresholds: midpoint of the mean "
      "present / absent ratio on CALIBRATION rows = a sample of the TRAINING split (never the scored rows):" % a.tau)
    for k, v in extra[0]["calibration"].items():
        P(f"- {k}: " + (f"threshold {v['threshold']:.3f} (present {v['mean_present']:.3f}, n {v['n_present']} / absent {v['mean_absent']:.3f}, n {v['n_absent']})"
                       if v.get("threshold") is not None else f"no threshold ({v['rule']}; n present {v['n_present']}, absent {v['n_absent']})"))
    P(f"\n## Results (seed {a.seeds_list[0]} model in model/; mean ± std over seeds when more than one)\n")
    P("Chance per role (1 / catalog candidates of the role's kinds): " + ", ".join(f"{k} {v:.3f}" for k, v in chance_.items() if v) + "\n")
    P("Two candidate sets: `catalog` = every catalog word of the role's kinds (what `sop formalize` does); `experiment` = as the repository "
      "scripts (seen splits: the words seen in training; new identities: the held-out words), for the P0 comparison with RESULTS.\n")
    def ms(xs): return f"{np.mean(xs):.3f}" + (f"±{np.std(xs):.3f}" if len(xs) > 1 else "")
    for s in metrics[0]:
        for cs in metrics[0][s]:
            accs = metrics[0][s][cs]["acc"]
            P(f"### {s} — {cs} (n = {metrics[0][s][cs]['n']} of {metrics[0][s][cs]['of']})")
            P("| axis | accuracy |\n|---|---|")
            for k in accs: P(f"| {k} | {ms([mm[s][cs]['acc'].get(k, float('nan')) for mm in metrics])} |")
            P(f"\nresidual mean {ms([mm[s][cs]['residual_mean'] for mm in metrics])}; status {metrics[0][s][cs]['status']}")
            cal = metrics[0][s][cs]["calibration"]
            P("\n| calibration axis (seed 0) | acc | margin mean / p10 | coverage at tau | confident-wrong | abstain right / wrong |\n|---|---|---|---|---|---|")
            for ax, c in cal.items():
                P(f"| {ax} | {c['acc']:.3f} | {c['margin_mean']:.3f} / {c['margin_p10']:.3f} | {c['coverage']:.2f} | {c['confident_wrong']:.3f} | {c['abstain_right']:.3f} / {c['abstain_wrong']:.3f} |")
            P("")
    P("## Gates (seed 0; OPEN > 0.5)\n" + " ".join(f"{c}={g:.2f}" for c, g in sorted(extra[0]["gates"].items(), key=lambda x: -x[1])) + "\n")
    weak = []
    for s in metrics[0]:
        for k, v in metrics[0][s]["catalog candidates"]["acc"].items():
            if v < 0.9 and not k.startswith("lexical"): weak.append(f"{s}: {k} {v:.3f}")
    P("## What is weak, what to add\n" + ("\n".join(f"- {w}" for w in weak) if weak else "- nothing under 0.9 on these splits") + "\n")
    if len(a.seeds_list) < 3: P("- fewer than 3 seeds (rule 6): the numbers above carry no seed variance; run with --seeds 3 or more for a result.\n")
    P("## Checklist\n[x] addresses from the world [x] shared identities (one star function) [x] chance printed [x] seen / unseen axes separated "
      "[x] margin [%s] >=3 seeds [x] trivial control (lexical order baseline) [x] calibration (rule 18) [x] memory logged (rule 21) "
      "[x] thresholds on calibration rows (rule 25) [x] given vs learned stated (rule 20)" % ("x" if len(a.seeds_list) >= 3 else " "))
    with open(path, "w") as f: f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sop train", description="train the reader on a world folder (legend.json + examples/*.sop -> model/)")
    ap.add_argument("world"); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true", help="shuffle the first epoch too (default: file order, as the repository's generators)")
    ap.add_argument("--max_train", type=int, default=0); ap.add_argument("--device", default="cpu"); ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=600); ap.add_argument("--n_cal", type=int, default=400); ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--splits", help="reuse the split assignment of an earlier run (held-out set fixed before a legend change)")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--force", action="store_true", help="accept examples whose !legend hash differs (inspection only)")
    ap.add_argument("--out", help="model folder (default <world>/model)")
    ap.add_argument("--eval_only", action="store_true", help="re-score the saved model/ (no training): metrics.json and REPORT.md rewritten")
    ap.add_argument("--star_seed", default=world.GLOBAL_STAR_SEED, help="seed of the word-star catalog; default the GLOBAL one shared by all worlds (spec v2 s.0)")
    ap.add_argument("--vary_stars", action="store_true", help="rule 6 / research reproduction: identities drawn per seed (star seed 'seed<k>'), as the repository scripts")
    a = ap.parse_args(argv)
    wd = os.path.abspath(a.world); a.world_name = os.path.basename(wd.rstrip("/"))
    legend = Legend(os.path.join(wd, "legend.json"))
    if legend.errors: print("legend errors:\n  " + "\n  ".join(legend.errors)); return 1
    wj = os.path.join(wd, "world.json"); a.language = json.load(open(wj)).get("language", "en") if os.path.exists(wj) else "en"
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S"); a.run_dir = os.path.join(wd, "runs", ts); os.makedirs(a.run_dir, exist_ok=True)
    log = open(os.path.join(a.run_dir, "train.log"), "w"); sys.stdout = Tee(sys.__stdout__, log)
    a.commit = git_commit()
    with open(os.path.join(a.run_dir, "command.txt"), "w") as f: f.write("sop train " + " ".join(argv if argv is not None else sys.argv[1:]) + f"\ncommit {a.commit}\n")
    status_path = os.path.join(wd, "status.json")
    def status(label, **kw):
        st = {"label": label, "when": datetime.datetime.now().isoformat(timespec="seconds"), "legend": legend.sha, "legend_version": legend.version, "run": ts, **kw}
        tmp = status_path + ".tmp"; json.dump(st, open(tmp, "w"), indent=1); os.replace(tmp, status_path)
    rows, errors, files = load_rows(wd, legend, a.force)
    if errors:
        print("examples rejected by sop check:\n" + "\n".join(errors[:50])); status("FAILED", reason="examples failed sop check"); return 1
    if not rows: print("no annotated sentences (examples/*.sop with _text)"); status("FAILED", reason="no examples"); return 1
    exhash = hashlib.sha256("".join(open(p, "rb").read().hex() for p in files).encode()).hexdigest()
    status("TRAINING", examples_sha256=exhash)
    hp = os.path.join(wd, "split_hints.json"); hints = json.load(open(hp)) if os.path.exists(hp) else {}
    defs, splits = make_splits(rows, legend, hints, a.seed)
    if a.splits:
        prev = json.load(open(a.splits)); text2 = {}
        for s, ids in prev["rows"].items():
            for t in prev.get("texts", {}).get(s, []): text2[t] = s
        for r in rows: r["split"] = text2.get(r["text"], r["split"])
        splits = collections.defaultdict(list)
        for r in rows: splits[r["split"]].append(r["id"])
        defs["reused_from"] = a.splits
    texts = collections.defaultdict(list)
    for r in rows: texts[r["split"]].append(r["text"])
    with open(os.path.join(a.run_dir, "splits.json"), "w") as f: json.dump({"definitions": defs, "rows": splits, "texts": texts}, f, indent=1)
    print(f"[splits] written to {os.path.join(a.run_dir, 'splits.json')} before training: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(splits.items())))
    train_rows = [r for r in rows if r["split"] == "train"]; schema = build_schema(train_rows, legend)
    schema["content_kinds"] = sorted(content_kinds_of(train_rows, legend))
    print(f"[schema] roles {schema['role_order']} (core {schema['core_roles']}), max depth {schema['max_depth']}, event negation {schema['neg_event']}")
    lexical, ranks = lexical_baseline(train_rows, legend, schema)
    seen_words = {w for r in train_rows for fl in r["frame"]["roles"].values() for w in world.filler_words(fl)} | {s for r in train_rows for s in r["frame"]["speakers"]}
    model_dir = os.path.abspath(a.out or os.path.join(wd, "model")); a.seeds_list = [a.seed + k for k in range(a.seeds)]
    metrics, extra = [], []
    if a.eval_only:     # re-score a saved model with the current reader code; no training, thresholds kept from its config (rule 25)
        from .infer import Reader
        prev = json.load(open(os.path.join(model_dir, "metrics.json"))); rd = Reader(model_dir); a.seeds_list = [rd.cfg["seed"]]
        mt = run_eval(rd, rows, legend, schema, a, lexical, seen_words, set(defs["new_identities"])); metrics, extra = [mt], [prev["training"][0]]
        chance_ = chance(rd, schema); a.D = rd.D; a.commit = f"{a.commit} (re-scored; trained at {rd.cfg.get('training_code_commit')})"
        for s, v in mt.items(): print(f"[eval] {s}: FULL " + " | ".join(f"{cs} {vv['acc']['FULL']:.3f}" for cs, vv in v.items()), flush=True)
    try:
        for k, seed in enumerate([] if a.eval_only else a.seeds_list):
            md = model_dir if k == 0 else os.path.join(a.run_dir, f"seed{seed}", "model")
            if k == 0 and os.path.exists(md + ".building"): shutil.rmtree(md + ".building")
            build = md + ".building" if k == 0 else md
            if a.vary_stars: a.star_seed = f"seed{seed}"
            rd, ex = train_one(a, legend, rows, splits, schema, seed, build, a.run_dir)
            mt = run_eval(rd, rows, legend, schema, a, lexical, seen_words, set(defs["new_identities"]))
            metrics.append(mt); extra.append(ex)
            if k == 0: chance_ = chance(rd, schema)
            for s, v in mt.items():
                print(f"[eval] seed {seed} {s}: FULL " + " | ".join(f"{cs} {vv['acc']['FULL']:.3f}" for cs, vv in v.items()) + f" | residual {v['catalog candidates']['residual_mean']:.3f}", flush=True)
            sys.path.insert(0, EXP); import trainer; print(f"[mem] after seed {seed}: {trainer.mem_line()}")
            if k == 0:
                if os.path.exists(model_dir): shutil.rmtree(model_dir)
                os.replace(build, model_dir)
    except Exception as e:
        status("FAILED", reason=f"{type(e).__name__}: {e}"); raise
    summary = {"world": a.world_name, "legend": legend.sha, "legend_version": legend.version, "seeds": a.seeds_list, "label": "BUILT",
               "splits": {k: len(v) for k, v in splits.items()}, "chance": chance_, "lexical_ranks": ranks,
               "per_seed": metrics, "training": extra, "labels_rule22": "every number BUILT by this run"}
    agg = {}
    for s in metrics[0]:
        for cs in metrics[0][s]:
            for kk in metrics[0][s][cs]["acc"]:
                xs = [m[s][cs]["acc"][kk] for m in metrics if kk in m[s][cs]["acc"]]
                agg.setdefault(s, {}).setdefault(cs, {})[kk] = {"mean": float(np.mean(xs)), "std": float(np.std(xs))}
    summary["mean_over_seeds"] = agg
    with open(os.path.join(model_dir, "metrics.json"), "w") as f: json.dump(summary, f, indent=1)
    write_report(os.path.join(model_dir, "REPORT.md"), a, legend, rows, splits, defs, schema, metrics, extra, chance_, len(files))
    shutil.copyfile(os.path.join(model_dir, "metrics.json"), os.path.join(a.run_dir, "metrics.json"))
    full = {s: agg[s]["catalog candidates"]["FULL"]["mean"] for s in agg}
    status("BUILT", examples_sha256=exhash, metrics_summary={"FULL (catalog candidates)": full})
    print(f"[done] model written to {model_dir}; REPORT.md and metrics.json there; run log {a.run_dir}/train.log")
    print("checklist: [x] addresses from the world [x] shared identities [x] chance printed [x] seen/unseen axes separated [x] margin "
          f"[{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] trivial control (lexical order baseline) [x] calibration (18) [x] memory logged (21) "
          "[x] thresholds on calibration rows (25) [x] given vs learned stated (20)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
