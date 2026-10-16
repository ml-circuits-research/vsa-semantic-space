"""9c-septies (hand-over 78d36e8 Q2, rule 16): FULL on the quantifier world REDEFINED over the legend's questions — nouns, presence,
scope-or-both — with the determiner IDENTITY reported as its own axis OUTSIDE FULL (the legend asks which phrase is wide, not which word was
used). Both definitions are printed once, side by side, on the SAVED C2c models (logs/.ckpt_v14_run_c2c_learned_rel_<seed>.pt, 9c-sexies):
same worlds, seeds, calibration rows (rng 1234+depth) and scored rows (rng 900+) as v14_quantifiers.py. No retraining.
  FULL-old = nouns + determiners + presence + scope (the 9c-* columns up to 9c-sexies); FULL-new = nouns + presence + scope (unambiguous) or
  nouns + presence + 'both' (ambiguous, by the C2c readout rule); on affirmative rows presence means ABSENT."""
import numpy as np, torch, guard, v5_scale as v5, v14_quantifiers as q

D = 1024; delta = 0.25; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; q.SCOPE_TARGET = "c2c"; q.set_c2c(); dev = "cuda"
print("rescore_v14_c2c_full: C2c world (slot target, learned gates, beacon-relative emitters), saved models of 9c-sexies; decoder as in v14 (presence threshold and C2c term threshold per depth from calibration rows; STRICT = depth-1 threshold at depth 2)")
cols = ["unamb d0: dets (own axis)", "FULL-old", "FULL-new", "unamb d1: FULL-old", "FULL-new", "unamb d2 UNSEEN: FULL-old", "FULL-new", "STRICT d2: FULL-old", "FULL-new", "aff d0: FULL-old", "FULL-new", "aff d2: FULL-old", "FULL-new", "new nouns d0: FULL-old", "FULL-new", "AMBIG d0: both", "FULL-new (nouns + presence + both)"]
res = []
for seed in range(5):
    rng = np.random.default_rng(seed); w = q.QWorld(rng); trNP, teNP, trNO, teNO = q.NOUNS_P[:12], q.NOUNS_P[12:], q.NOUNS_O[:12], q.NOUNS_O[12:]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = q.QRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = q.v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], dev, gate_init="closed"); m.set_F(F); m.set_rel_emitters(["every", "each", "some", "certain"], ["NP", "NO"])
    st = torch.load(f"logs/.ckpt_v14_run_c2c_learned_rel_{seed}.pt", map_location=dev)
    with torch.no_grad():
        for p_, v in zip([m.theta, m.logits, m.rel], st["params"]): p_.copy_(v)
    def pres_threshold(depth, n=60):
        crng = np.random.default_rng(1234 + depth); qq, af, tin, tout = [], [], [], []
        for t in range(len(q.PATTERNS)):
            for nq, store in ((None, qq), (0, af)):
                evs = [w.qevent(crng, depth=depth, nouns_p=trNP, nouns_o=trNO, n_quant=nq) for _ in range(n // len(q.PATTERNS) + 1)]; toks = [w.qrender(ev, t, crng) for ev in evs]
                c, i_, k = m.encode(toks)
                with torch.no_grad(): S = m.states(c, i_, k)
                for j, ev in enumerate(evs):
                    o = q.decode_q(S[j], m, world, w, trNP, trNO, depth, delta); store.append(o["pres_score"])
                    for r, (det, _) in zip(("A", "B", "C"), ev[1:4]):
                        if det in q.WIDE_MARKERS: tin.append(o["wide_scores"][r]); tout.append(o["q_scores"][r])
                        elif det in ("every", "some"): tin.append(o["q_scores"][r]); tout.append(o["wide_scores"][r])
                        else: tout.extend([o["wide_scores"][r], o["q_scores"][r]])
        return 0.5 * (float(np.mean(qq)) + float(np.mean(af))), 0.5 * (float(np.mean(tin)) + float(np.mean(tout)))
    thr = {d: pres_threshold(d) for d in (0, 1, 2)}
    def run(depth, ambiguous, NP, NO, n=60, n_quant=None, thr_depth=None):
        thr_d = depth if thr_depth is None else thr_depth
        erng = np.random.default_rng(900 + depth + 10 * ambiguous + (0 if n_quant is None else 100 * (n_quant + 1))); dets, old, new, both = [], [], [], []
        for t in range(len(q.PATTERNS)):
            evs = [w.qevent(erng, depth=depth, nouns_p=NP, nouns_o=NO, ambiguous=ambiguous, n_quant=n_quant) for _ in range(n)]; toks = [w.qrender(ev, t, erng) for ev in evs]
            c, i_, k = m.encode(toks)
            with torch.no_grad(): S = m.states(c, i_, k)
            for j, ev in enumerate(evs):
                o = q.decode_q(S[j], m, world, w, NP, NO, depth, delta, tau_term=thr[thr_d][1]); ok_n = all(o[r] == ev[i][1] for r, i in (("A", 1), ("B", 2), ("C", 3))); ok_d = all(o["d" + r] == ev[i][0] for r, i in (("A", 1), ("B", 2), ("C", 3)))
                dets.append(ok_d); present = o["pres_score"] >= thr[thr_d][0]; truth = sorted(ev[-1])
                if n_quant == 0: old.append(ok_n and ok_d and not present); new.append(ok_n and not present); continue
                ok_s = o["scope_top"] == truth
                if ambiguous: both.append(ok_s); new.append(ok_n and present and ok_s); continue
                old.append(ok_n and ok_d and present and ok_s); new.append(ok_n and present and ok_s)
        mean = lambda x: float(np.mean(x)) if x else float("nan")
        return mean(dets), mean(old), mean(new), mean(both)
    u0 = run(0, False, trNP, trNO); u1 = run(1, False, trNP, trNO, n=40); u2 = run(2, False, trNP, trNO, n=30); s2 = run(2, False, trNP, trNO, n=30, thr_depth=1)
    a0 = run(0, False, trNP, trNO, n=30, n_quant=0); a2 = run(2, False, trNP, trNO, n=30, n_quant=0); nn_ = run(0, False, teNP, teNO, n=40); am = run(0, True, trNP, trNO)
    r = [u0[0], u0[1], u0[2], u1[1], u1[2], u2[1], u2[2], s2[1], s2[2], a0[1], a0[2], a2[1], a2[2], nn_[1], nn_[2], am[3], am[2]]; res.append(r)
    print(f"  seed {seed} " + " | ".join(f"{x:.3f}" for x in r), flush=True)
r = np.array(res); print("\n== MEANS over 5 seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:, i].mean():.3f}±{r[:, i].std():.3f}" for i in range(len(cols))))
print("\nchecklist: [x] saved params, no retraining [x] same calibration and scored rows as 9c-sexies (rngs) [x] both FULL definitions printed once (rule 16) [x] determiner identity reported as its own axis [x] strict column uses no depth-2 information [x] 5 seeds")
