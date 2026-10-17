"""12-quater per-seed readout (judgment 45ccd1d A1 / rule 27): the readout of the B5.1 E1b-protocol run re-run from the SAVED parameters
(logs/.ckpt_v20_b51e1b_<seed>.pt), printing the PER-SEED value of every cell, same worlds, seeds, forms and evaluation rngs as v20_operators.py
(world rng seed, identities 300+seed, roles 100+seed, eval rng 900+depth). No retraining. If the means reproduce the digits of
logs/2026-10-17_0840_v20_operators_ec0aa03.txt the readout is deterministic (said in the log); if not, both are reported.
Run: python3 rescore_v20.py [--tag b51e1b]"""
import argparse, numpy as np, torch, guard, v5_scale as v5, v8_gates as v8, v20_operators as v20

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="b51e1b"); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
D = a.D; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
print(f"rescore_v20: per-seed readout of the saved B5.1 models (tag {a.tag}), forms {list(v20.FORMS)}, depths 0 / 1 / 2 (UNSEEN for the operator forms, rule 29 (B)); roles under the GOLD structure, FULL under the decoded one")
cols = ["A", "B", "C", "pred", "structure", "FULL"]; res = {}
for seed in range(a.seeds):
    w = v20.OpWorld(18, np.random.default_rng(seed)); trP, trO = w.persons[:12], w.objects[:12]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v20.OpRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)
    st = torch.load(f"logs/.ckpt_v20_{a.tag}_{seed}.pt", map_location=a.device)
    with torch.no_grad():
        for p_, v in zip([m.theta, m.logits], st["params"]): p_.copy_(v)
    for depth in (0, 1, 2):
        erng = np.random.default_rng(900 + depth)
        for form in v20.FORMS:
            acc = {c: [] for c in cols}
            for k in range(40):
                ev = w.oevent(erng, trP, trO, form, depth=depth); toks = w.orender(ev, erng, t=k % 2); c, i_, kk = m.encode([toks])
                with torch.no_grad(): h, out = v20.decode(m.states(c, i_, kk)[0], m, world, w, depth, trP, trO)
                g = v20.GOLD_H[form]; fill = out[g][1]; truth = {"A": ev["A"], "B": ev["B"], "C": ev["C"], "pred": v20.BASE[ev["kind"]]}
                for r in ("A", "B", "C", "pred"): acc[r].append(fill[r] == truth[r])
                acc["structure"].append(h == g); acc["FULL"].append(h == g and all(out[h][1][r] == truth[r] for r in truth))
            for c_ in cols: res.setdefault((depth, form, c_), []).append(float(np.mean(acc[c_])))
    print(f"  seed {seed} read", flush=True)
print("\n== per-seed values (seeds 0-4), then mean ± sd; depth | form | column")
for depth in (0, 1, 2):
    for form in v20.FORMS:
        for c_ in ("A", "FULL"):
            v = res[(depth, form, c_)]; print(f"   {depth} {'(UNSEEN)' if depth == 2 else '        '} | {form:9s} | {c_:4s} | " + ", ".join(f"{x:.3f}" for x in v) + f" | {np.mean(v):.3f}±{np.std(v):.3f}" + ("   <- BELOW the criterion 0.9 on the mean" if depth == 2 and np.mean(v) <= 0.9 else "") + (f"   ({sum(x > 0.9 for x in v)}/5 seeds above 0.9)" if depth == 2 else ""))
print("== 12-quater log (means): d2 A — aff 1.000, must 0.965, may 1.000, neg 1.000, must_neg 1.000, neg_must 0.990, may_neg 0.855; compare the means above: identical digits = deterministic readout")
print("\nchecklist: [x] saved params, no retraining [x] same worlds, seeds and evaluation rngs as the run [x] per-seed list for every cell (27) [x] the cell below the criterion named [x] rule 29 label stated (B)")
