"""C2 strict columns (judgment 3b4f837 C2) on the SAVED C2 models (logs/.ckpt_v14_q_<target>_open_<seed>.pt): depth-2 PRESENCE / FULL and
affirmative ABSENT with the DEPTH-1 threshold (no depth-2 information at all) next to the depth-2 threshold of 9c-quater. Same worlds, seeds,
calibration and scored rows as v14 (same rngs). No retraining."""
import argparse, glob, numpy as np, torch, guard, v5_scale as v5, v14_quantifiers as q
ap = argparse.ArgumentParser(); ap.add_argument("--targets", default="slot,noun"); ap.add_argument("--tag", default="q"); ap.add_argument("--device", default="cuda"); a = ap.parse_args()
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; D = 1024
for target in a.targets.split(","):
    q.SCOPE_TARGET = target; rows = []
    for seed in range(5):
        path = f"logs/.ckpt_v14_{a.tag}_{target}_open_{seed}.pt"
        rng = np.random.default_rng(seed); w = q.QWorld(rng); trNP, trNO = q.NOUNS_P[:12], q.NOUNS_O[:12]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = q.QRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = q.v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)
        st = torch.load(path, map_location=a.device)
        with torch.no_grad():
            for p_, v in zip([m.theta, m.logits], st["params"]): p_.copy_(v)
        def pres_threshold(depth, n=60):
            crng = np.random.default_rng(1234 + depth); qq, af = [], []
            for t in range(len(q.PATTERNS)):
                for nq, store in ((None, qq), (0, af)):
                    evs = [w.qevent(crng, depth=depth, nouns_p=trNP, nouns_o=trNO, n_quant=nq) for _ in range(n // len(q.PATTERNS) + 1)]; toks = [w.qrender(ev, t, crng) for ev in evs]
                    c, i_, k = m.encode(toks)
                    with torch.no_grad(): S = m.states(c, i_, k)
                    store.extend(q.decode_q(S[j], m, world, w, trNP, trNO, depth, 0.25)["pres_score"] for j in range(len(evs)))
            return 0.5 * (float(np.mean(qq)) + float(np.mean(af)))
        thr = {d: pres_threshold(d) for d in (1, 2)}
        def run(n_quant, thr_v, n=30):
            erng = np.random.default_rng(900 + 2 + (0 if n_quant is None else 100 * (n_quant + 1))); pres, full = [], []
            for t in range(len(q.PATTERNS)):
                evs = [w.qevent(erng, depth=2, nouns_p=trNP, nouns_o=trNO, n_quant=n_quant) for _ in range(n)]; toks = [w.qrender(ev, t, erng) for ev in evs]
                c, i_, k = m.encode(toks)
                with torch.no_grad(): S = m.states(c, i_, k)
                for j, ev in enumerate(evs):
                    o = q.decode_q(S[j], m, world, w, trNP, trNO, 2, 0.25); ok_n = all(o[r] == ev[i][1] for r, i in (("A", 1), ("B", 2), ("C", 3))); ok_d = all(o["d" + r] == ev[i][0] for r, i in (("A", 1), ("B", 2), ("C", 3)))
                    present = o["pres_score"] >= thr_v
                    if n_quant == 0: pres.append(not present); full.append(ok_n and ok_d and not present)
                    else: pres.append(present); full.append(ok_n and ok_d and present and o["scope_top"] == sorted(ev[-1]))
            return float(np.mean(pres)), float(np.mean(full))
        r = run(None, thr[2]) + (run(0, thr[2])[0],) + run(None, thr[1]) + (run(0, thr[1])[0],); rows.append(r)
        print(f"  {target} seed {seed}: thresholds d1 {thr[1]:.3f} d2 {thr[2]:.3f} | d2 threshold: PRESENCE {r[0]:.3f} FULL {r[1]:.3f} aff ABSENT {r[2]:.3f} | STRICT d1 threshold: PRESENCE {r[3]:.3f} FULL {r[4]:.3f} aff ABSENT {r[5]:.3f}", flush=True)
    r = np.array(rows); print(f"== {target} target, 5 seeds: depth-2 threshold PRESENCE | FULL | aff ABSENT || STRICT (depth-1 threshold) PRESENCE | FULL | aff ABSENT\n  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(6)))
print("\nchecklist: [x] saved params, no retraining [x] strict column uses no depth-2 information (threshold from depth 1) [x] same calibration and scored rows as 9c-quater (rngs) [x] both targets")
