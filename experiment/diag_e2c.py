"""Diagnostic for E2c (rule 10: evaluation before data/optimization/architecture). The 'none' (no-quantifier) hypothesis added to decode_q
after 9c-bis uses an ABSOLUTE threshold (tau_none = 0.08); scope scores shrink with the nesting depth, so at depth 2 the decoder may say
'none' for every sentence => scope 0.03 (below chance) in E2c for both corrected targets. This script trains ONE model per world target
(role / slot / noun; seed 0; same regime as E2c) and reports, per depth and case: nouns, determiners, the RAW scope-score profile (true wide
role vs the others vs affirmative sentences), and scope accuracy under three 'none' rules: absolute 0.08 (as in E2c), none (as in 9c-bis),
and RELATIVE (none if the best scope score < 0.25 x the best noun score at that depth)."""
import argparse, numpy as np, torch, guard, trainer, v5_scale as v5, v14_quantifiers as q

ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--targets", default="role,slot,noun"); a = ap.parse_args()
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; D = 1024; seed = a.seed
for target in a.targets.split(","):
    q.SCOPE_TARGET = target
    rng = np.random.default_rng(seed); w = q.QWorld(rng); trNP, trNO = q.NOUNS_P[:12], q.NOUNS_O[:12]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = q.QRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = q.v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], "cuda", gate_init="closed"); m.set_F(F)
    trng = np.random.default_rng(200 + seed); sents, targets = [], []
    for n in range(a.N):
        if trng.random() < 0.4: ev = w.qevent(trng, depth=int(trng.integers(3)), nouns_p=trNP, nouns_o=trNO, n_quant=0)
        else: ev = w.qevent(trng, depth=int(trng.integers(2)), nouns_p=trNP, nouns_o=trNO)
        sents.append(w.qrender(ev, int(trng.integers(len(q.PATTERNS))), trng)); targets.append(world.state(ev))
    cats, ids, mask = m.encode(sents); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
    trainer.train_loop(m, [m.theta, m.logits], cats, ids, mask, tgt, 8, 0.02, log_every=2000, tag=f" {target}")
    print(f"\n==== target {target}: case | depth | nouns | dets | scope(abs 0.08) | scope(no none) | scope(relative) | raw: true-wide | other | affirmative-max")
    for depth in (0, 1, 2):
        for name, nq, amb in (("affirmative", 0, False), ("one quantifier", 1, False), ("two, marked", 2, False), ("ambiguous", None, True)):
            erng = np.random.default_rng(500 + depth); rows = []
            for t in range(len(q.PATTERNS)):
                evs = [w.qevent(erng, depth=depth, nouns_p=trNP, nouns_o=trNO, ambiguous=amb, n_quant=nq) for _ in range(30)]; toks = [w.qrender(ev, t, erng) for ev in evs]
                c, i_, k = m.encode(toks)
                with torch.no_grad(): S = m.states(c, i_, k)
                for j, ev in enumerate(evs):
                    o_abs = q.decode_q(S[j], m, world, w, trNP, trNO, depth, tau_none=0.08); o_no = q.decode_q(S[j], m, world, w, trNP, trNO, depth, tau_none=-1)
                    # relative rule: best scope score vs best noun score at this depth
                    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}; Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S[j] / S[j].abs().pow(2).sum().sqrt()
                    rep = torch.conj(R["rep"]) ** depth; cid = torch.tensor([w.idx[x] for x in trNP], device=m.device)
                    noun_best = float(torch.real((Sn * torch.conj(R["A"]) * rep) @ torch.conj(Fn[cid]).T).max())
                    ss = o_no["scope_scores"]; best = max(ss.values()); o_rel_scope = [] if best < 0.25 * noun_best else o_no["scope"]
                    truth = sorted(ev[-1]); ok_n = all(o_no[r] == ev[i][1] for r, i in (("A", 1), ("B", 2), ("C", 3))); ok_d = all(o_no["d" + r] == ev[i][0] for r, i in (("A", 1), ("B", 2), ("C", 3)))
                    wide = ev[-1]; raw_t = np.mean([ss[r] for r in wide]) if wide else float("nan"); raw_o = np.mean([ss[r] for r in ss if r not in wide]) if len(wide) < 3 else float("nan")
                    rows.append((ok_n, ok_d, o_abs["scope"] == truth, o_no["scope"] == truth, o_rel_scope == truth, raw_t, raw_o, best if not wide else float("nan")))
            r = np.array(rows, dtype=float); print(f"  {name:14s} | {depth} | " + " | ".join(f"{np.nanmean(r[:, i]):.3f}" for i in range(8)))
    print("  gates OPEN:", sorted(c for c, v in m.gates().items() if v > 0.5))
