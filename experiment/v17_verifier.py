"""
v17_verifier — round 10 B4 (E6): the VERIFIER with injected errors, on WikiSQL, in its minimal measurable form.

Setting. A formalization of a question is proposed (in the product: by an LLM; here, with no LLM API on the machine, the CONTROLLED
version: the gold SQL frame, into which we INJECT one error per instance — the detection rates are then exact, not confounded by the
LLM's own errors; the LLM version is a follow-up and is said so). The verifier reads the question with the trained reader (v11: full
model, learned gates, no table at the input), decodes its frame with the WORLD's addresses, and diffs it against the proposed frame:
    verdict = "ok" if diff is empty, else "error in <field>"; "unverifiable" if the reader's own margin on that field is below tau.
Error types injected (one per instance): DROP a condition; SWAP the selected column with a condition column; CHANGE the selected column
to another column of the table; FLIP the aggregator; ADD a spurious condition (a column not in the gold query); CHANGE a condition value
(a value the reader cannot see: it decodes column choices and the aggregator; value changes are expected to be INVISIBLE — stated).
Reported, per error type: precision / recall of the "error" verdict, the rate of "unverifiable" verdicts, and the confident-wrong rate
(rule 18); a LEXICAL baseline on the same rows: the bag-of-words frame of v12's BoW+LR (selected column by overlap features, aggregator by
BoW) diffed against the proposed frame. Criterion (judgment): detection > 0.9 on ROLE errors (drop / swap / change column / add
condition / flip aggregator); state plainly which error types it cannot see (value changes).
Run: python3 v17_verifier.py [--epochs 3] [--n_dev 8421] [--tau 0.05]
"""
import argparse, json, re, time, collections, random
import numpy as np, torch
import guard, trainer, v11_wikisql as v11, v10_controls as v10

def propose_with_error(sql, n_cols, rng, kind):
    """returns (proposed sql dict, injected error kind) — one error per instance; None if the error cannot be injected here"""
    p = {"sel": sql["sel"], "agg": sql["agg"], "conds": [list(c) for c in sql["conds"]]}
    others = [c for c in range(n_cols) if c != sql["sel"] and c not in [x[0] for x in sql["conds"]]]
    if kind == "drop_cond":
        if not p["conds"]: return None
        p["conds"].pop(rng.integers(len(p["conds"])))
    elif kind == "swap_sel_cond":
        if not p["conds"]: return None
        i = int(rng.integers(len(p["conds"]))); p["sel"], p["conds"][i][0] = p["conds"][i][0], p["sel"]
    elif kind == "change_sel":
        if not others: return None
        p["sel"] = int(rng.choice(others))
    elif kind == "flip_agg":
        p["agg"] = int((sql["agg"] + 1 + rng.integers(5)) % 6)
    elif kind == "add_cond":
        if not others: return None
        p["conds"].append([int(rng.choice(others)), 0, "x"])
    elif kind == "change_value":
        if not p["conds"]: return None
        i = int(rng.integers(len(p["conds"]))); p["conds"][i][2] = str(p["conds"][i][2]) + " zz"
    return p

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--D", type=int, default=512); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--n_dev", type=int, default=8421)
    ap.add_argument("--n_train", type=int, default=56355); ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--device", default="cuda"); ap.add_argument("--k_clusters", type=int, default=48); ap.add_argument("--n_freq", type=int, default=40)
    ap.add_argument("--log_every", type=int, default=500)
    a = ap.parse_args(); dev_ = a.device; tok, load = v11.tok, v11.load
    train, ttabs = load("train"); dev, dtabs = load("dev"); train = train[:a.n_train]; dev = dev[:a.n_dev]
    vocab = {}; vid = lambda w: vocab.setdefault(w, len(vocab))
    for rows, tabs in ((train, ttabs), (dev, dtabs)):
        for r in rows:
            for w in tok(r["question"]): vid(w)
            for h in tabs[r["table_id"]]:
                for w in tok(h): vid(w)
            for c in r["sql"]["conds"]:
                for w in tok(str(c[2])): vid(w)
    V = len(vocab); print(f"v17_verifier (B4): device={dev_}, D={a.D}; train {len(train)}, dev {len(dev)}, vocabulary {V}; tau={a.tau}; controlled version (gold frame + one injected error; no LLM)")
    hid = lambda tabs, tid: [[vid(w) for w in tok(h)] for h in tabs[tid]]; qids = lambda r: [vid(w) for w in tok(r["question"])]
    # categories (train only, as in v11/v12)
    freq = collections.Counter(w for r in train for w in tok(r["question"])); freq.update(w for r in train for h in ttabs[r["table_id"]] for w in tok(h))
    singles = [w for w, _ in freq.most_common(a.n_freq)]; sents = [tok(r["question"]) for r in train] + [tok(h) for r in train[:20000] for h in ttabs[r["table_id"]]]
    print(f"clustering data: {len(sents)} TRAIN sequences only; dev not used.")
    emb = v10.skipgram([s for s in sents if len(s) > 1], vocab, dim=64, window=2, epochs=3, device=dev_, seed=0)
    X = torch.tensor(emb, device=dev_); X = X / (X.norm(dim=1, keepdim=True) + 1e-8); gk = torch.Generator().manual_seed(0); C = X[torch.randperm(V, generator=gk)[:a.k_clusters].to(dev_)].clone()
    for it in range(25):
        asg = (X @ C.T).argmax(1)
        for c in range(a.k_clusters):
            mem = X[asg == c]
            if len(mem): C[c] = mem.mean(0) / (mem.mean(0).norm() + 1e-8)
    asg = (X @ C.T).argmax(1).cpu().numpy(); cat_of = torch.tensor(np.array([a.k_clusters + singles.index(w) if w in singles else asg[i] for w, i in sorted(vocab.items(), key=lambda x: x[1])])); K = a.k_clusters + len(singles)
    world = v11.World(vocab, a.D, dev_, 0)
    def target(r, tabs): return world.sql_state({"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [[c, o, [vid(w) for w in tok(str(v))]] for c, o, v in r["sql"]["conds"]]}, hid(tabs, r["table_id"]))
    tgt = torch.stack([target(r, ttabs) for r in train])
    # ---- train the reader (v11 RealModel, learned gates) with the trainer loop (memory logged)
    m = v11.RealModel(K, a.D, dev_, 0); seqs = [qids(r) for r in train]; order = np.argsort([len(s) for s in seqs])
    class Wrap:   # trainer expects .states(cats, ids, mask) and .theta
        theta = m.th_path
        def states(self, cats, ids, mask): return m.states(world.F, cats, ids, mask)
    t0 = time.time()
    for ep in range(a.epochs):
        g = np.random.default_rng(ep); buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]; g.shuffle(buckets); idx = np.concatenate(buckets)
        ids, mask = v11.pad([seqs[i] for i in idx], dev_); cats = cat_of[ids.cpu()].to(dev_)
        trainer.train_loop(Wrap(), m.params, cats, ids, mask, tgt[torch.tensor(idx, device=dev_)], a.batch, 0.02, log_every=a.log_every, tag=f" ep {ep}")
    print(f"reader trained ({time.time()-t0:.0f}s)")
    # ---- reader frames on dev, with margins
    def read_frame(S, cols):
        cb = torch.stack([world.bundle(c) for c in cols]); out = {}
        sc = v11.cosb((S * torch.conj(world.R["SEL"])).unsqueeze(0).expand(len(cols), -1), cb); t2 = torch.topk(sc, min(2, len(cols))); out["sel"] = (int(t2.indices[0]), float(t2.values[0] - (t2.values[1] if len(cols) > 1 else 0)))
        sa = v11.cosb((S * torch.conj(world.R["AGG"])).unsqueeze(0).expand(6, -1), world.F[torch.tensor(world.AGG, device=dev_)]); t2 = torch.topk(sa, 2); out["agg"] = (int(t2.indices[0]), float(t2.values[0] - t2.values[1]))
        conds = []
        for k in range(4):   # condition slots: a column is "present" if its best cosine clears a presence threshold relative to the sel readout
            s = v11.cosb((S * torch.conj(world.R["COND"][k] * world.R["COL"])).unsqueeze(0).expand(len(cols), -1), cb); t2 = torch.topk(s, min(2, len(cols)))
            conds.append((int(t2.indices[0]), float(t2.values[0]), float(t2.values[0] - (t2.values[1] if len(cols) > 1 else 0))))
        out["conds"] = conds; return out
    frames = []; thr_pres = []
    for i in range(0, len(dev), 256):
        rows = dev[i:i+256]; ids, mask = v11.pad([qids(r) for r in rows], dev_); cats = cat_of[ids.cpu()].to(dev_)
        with torch.no_grad(): S = m.states(world.F, cats, ids, mask)
        for s_, r in zip(S, rows):
            f = read_frame(s_, hid(dtabs, r["table_id"])); frames.append(f)
            for k, (c, top, mg) in enumerate(f["conds"]): thr_pres.append((top, k < len(r["sql"]["conds"])))
    # presence threshold set on the CALIBRATION half of dev (even rows), reported on the other half (odd rows) — judgment a8a8fab C2
    thr_cal = []
    for i, r in enumerate(dev):
        if i % 2 == 0:
            for k, (c, top, mg) in enumerate(frames[i]["conds"]): thr_cal.append((top, k < len(r["sql"]["conds"])))
    tp = np.array(thr_cal); pres_thr = 0.5 * (tp[tp[:, 1] == 1, 0].mean() + tp[tp[:, 1] == 0, 0].mean())
    tpa = np.array(thr_pres); thr_same = 0.5 * (tpa[tpa[:, 1] == 1, 0].mean() + tpa[tpa[:, 1] == 0, 0].mean())
    print(f"condition-slot presence threshold set on the calibration half of dev (even rows; midpoint of present {tp[tp[:,1]==1,0].mean():.3f} / absent {tp[tp[:,1]==0,0].mean():.3f}): {pres_thr:.3f}  (the same-rows midpoint of 10c would be {thr_same:.3f})")
    sfx = "" if a.n_train >= 56355 else f"_n{a.n_train}"
    torch.save({"params": [p_.detach().cpu() for p_ in m.params], "frames": frames, "dev_ids": [r.get("table_id") for r in dev], "pres_thr": float(pres_thr), "n_train": a.n_train}, f"logs/.ckpt_v17_reader_frames{sfx}.pt")
    print(f"reader params and dev frames saved to logs/.ckpt_v17_reader_frames{sfx}.pt (re-scoring without retraining: C4 / C5 / 4c)")
    # ---- lexical baseline frame (BoW+LR of v12: sel by overlap features, agg by hashed BoW), trained on train
    H = 4096; hb = lambda r: torch.bincount(torch.tensor([hash(w) % H for w in tok(r["question"])]), minlength=H).float().clamp(max=1)
    Xa = torch.stack([hb(r) for r in train]).to(dev_); ya = torch.tensor([r["sql"]["agg"] for r in train], device=dev_); Wa = torch.nn.Linear(H, 6).to(dev_); opt = torch.optim.Adam(Wa.parameters(), lr=1e-2, weight_decay=1e-5)
    for ep in range(8):
        p_ = torch.randperm(len(train), device=dev_)
        for i in range(0, len(train), 256): sl = p_[i:i+256]; loss = torch.nn.functional.cross_entropy(Wa(Xa[sl]), ya[sl]); opt.zero_grad(); loss.backward(); opt.step()
    def col_feats(r, tabs):
        q = set(tok(r["question"])); fs = []
        for j, h in enumerate(tabs[r["table_id"]]):
            ht = tok(h); ov = len(q & set(ht)); fs.append([ov, ov / (len(ht) + 1e-9), len(ht), float(any(w in q for w in ht))] + [1.0 if j == p else 0.0 for p in range(20)])
        return torch.tensor(fs)
    Xs = torch.cat([col_feats(r, ttabs) for r in train]).to(dev_); ys = torch.cat([torch.tensor([1.0 if j == r["sql"]["sel"] else 0.0 for j in range(len(ttabs[r["table_id"]]))]) for r in train]).to(dev_)
    Ws = torch.nn.Linear(Xs.shape[1], 1).to(dev_); opt = torch.optim.Adam(Ws.parameters(), lr=1e-2)
    for ep in range(6):
        p_ = torch.randperm(len(Xs), device=dev_)
        for i in range(0, len(Xs), 1024): sl = p_[i:i+1024]; loss = torch.nn.functional.binary_cross_entropy_with_logits(Ws(Xs[sl]).squeeze(1), ys[sl]); opt.zero_grad(); loss.backward(); opt.step()
    def lex_frame(r):
        sc = Ws(col_feats(r, dtabs).to(dev_)).squeeze(1); t2 = torch.topk(torch.softmax(sc, 0), min(2, len(sc))); pa = torch.softmax(Wa(hb(r).to(dev_).unsqueeze(0)), 1)[0]; ta = torch.topk(pa, 2)
        return {"sel": (int(t2.indices[0]), float(t2.values[0] - (t2.values[1] if len(sc) > 1 else 0))), "agg": (int(ta.indices[0]), float(ta.values[0] - ta.values[1])), "conds": []}
    # ---- verdicts
    def verdict(frame, proposed, kind, is_reader):
        """compare the proposed frame with the read frame; return ('error'|'ok'|'unverifiable', field)"""
        unv = []
        if proposed["sel"] != frame["sel"][0]:
            return ("error", "sel") if frame["sel"][1] >= a.tau else ("unverifiable", "sel")
        if proposed["agg"] != frame["agg"][0]:
            return ("error", "agg") if frame["agg"][1] >= a.tau else ("unverifiable", "agg")
        if is_reader:
            read_cols = [c for (c, top, mg) in frame["conds"] if top >= pres_thr]; prop_cols = [c[0] for c in proposed["conds"]]
            if sorted(read_cols) != sorted(prop_cols):
                mg = min([mg for (c, top, mg) in frame["conds"]] + [1.0]); return ("error", "conds") if mg >= a.tau else ("unverifiable", "conds")
        return ("ok", None)
    kinds = ["drop_cond", "swap_sel_cond", "change_sel", "flip_agg", "add_cond", "change_value"]; rng = np.random.default_rng(0)
    stats = {k: {"reader": collections.Counter(), "lex": collections.Counter()} for k in kinds + ["no_error"]}
    lex_frames = []
    for i, r in enumerate(dev):
        cols = dtabs[r["table_id"]]; lf = lex_frame(r); lex_frames.append(lf)
        if i % 2 == 0: continue                                   # calibration half: not scored (its rows set the presence threshold)
        for kind in kinds + ["no_error"]:
            prop = {"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [list(c) for c in r["sql"]["conds"]]} if kind == "no_error" else propose_with_error(r["sql"], len(cols), rng, kind)
            if prop is None: continue
            for name, fr, isr in (("reader", frames[i], True), ("lex", lf, False)):
                v, field = verdict(fr, prop, kind, isr); stats[kind][name][v] += 1
    torch.save(lex_frames, f"logs/.ckpt_v17_lex_frames{sfx}.pt")
    print(f"\n== VERIFIER on the REPORT half of dev (odd rows, {len(dev)//2} questions; the even half set the presence threshold), one injected error per instance; reader = v11 full model (no table), lex = BoW+LR frame (no condition model)")
    print("error type | n | reader: detected (recall) / unverifiable / missed(confident-wrong) | lex: detected / unverifiable / missed")
    for kind in kinds:
        for name in ("reader", "lex"):
            c = stats[kind][name]; n = sum(c.values())
            if name == "reader": line = f"  {kind:14s} | {n:5d} | reader {c['error']/max(n,1):.3f} / {c['unverifiable']/max(n,1):.3f} / {c['ok']/max(n,1):.3f}"
            else: line += f" | lex {c['error']/max(n,1):.3f} / {c['unverifiable']/max(n,1):.3f} / {c['ok']/max(n,1):.3f}"
        print(line)
    c = stats["no_error"]["reader"]; n = sum(c.values()); cl = stats["no_error"]["lex"]
    print(f"  {'no_error (gold)':14s} | {n:5d} | reader FALSE-ALARM {c['error']/n:.3f} / unverifiable {c['unverifiable']/n:.3f} / ok {c['ok']/n:.3f} | lex false-alarm {cl['error']/n:.3f} / {cl['unverifiable']/n:.3f} / {cl['ok']/n:.3f}")
    fa = c["error"] / n; tot_err = sum(stats[k]["reader"]["error"] for k in kinds[:5]); tot_n = sum(sum(stats[k]["reader"].values()) for k in kinds[:5])
    print(f"  precision of the 'error' verdict on ROLE errors vs gold (reader): {tot_err / (tot_err + fa * tot_n / max(1, len(kinds[:5]))):.3f}  (false-alarm rate on gold {fa:.3f}); recall on role errors {tot_err / max(tot_n,1):.3f}")
    print("  stated: change_value is INVISIBLE to this reader (it decodes columns and the aggregator, not values); conditions are detected by column set only.")
    print(f"  stated: on GOLD frames the verifier confirms {c['ok']/n:.3f} and says 'unverifiable' on {c['unverifiable']/n:.3f} — it is a filter on the role part of a formalization, not a judge of it.")
    print(f"\nchecklist: [x] world addresses [x] fixed identities [x] no table at the input [x] one error per instance, per-type rates [x] lexical baseline on the same rows [x] unverifiable = margin < tau (18) [x] false-alarm rate on gold [x] invisible error types stated [x] presence threshold set on the calibration half, reported on the other half [x] reader and frames saved for re-scoring [x] memory logged (21) [ ] LLM-produced formalizations (follow-up)")
