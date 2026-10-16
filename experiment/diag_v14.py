"""Diagnostic for E2 (rule 10: evaluation -> data -> optimization -> architecture; rule 15: per case before interpreting).
Trains one E2 model (seed 0, same regime) and reports, per depth 0/1/2 and per case (one quantifier | two with a marker | ambiguous):
noun accuracy, determiner accuracy, scope accuracy, FULL; plus the scope-score profile (mean score of the true wide role vs the others)
and the determiner-threshold check (is the 'the' heuristic what fails at depth 2?). Also: per pattern at depth 0."""
import guard  # process-level memory/time protection (see guard.py; the machine hung on 2026-10-15)
import argparse, numpy as np, torch, v5_scale as v5, v14_quantifiers as q

ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; D = 1024; seed = a.seed
rng = np.random.default_rng(seed); w = q.QWorld(rng); trNP, trNO = q.NOUNS_P[:12], q.NOUNS_O[:12]
F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = q.QRoleWorld(F, w, np.random.default_rng(100 + seed))
m = q.v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], "cuda", gate_init="closed"); m.set_F(F)
trng = np.random.default_rng(200 + seed); sents, targets = [], []
for n in range(a.N):
    d = int(trng.integers(2)); ev = w.qevent(trng, depth=d, nouns_p=trNP, nouns_o=trNO); sents.append(w.qrender(ev, int(trng.integers(len(q.PATTERNS))), trng)); targets.append(world.state(ev))
cats, ids, mask = m.encode(sents); tgt_all = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device); opt = torch.optim.Adam([m.theta, m.logits], lr=0.02)
for i in range(0, a.N, 8):
    sl = slice(i, i + 8); S = m.states(cats[sl], ids[sl], mask[sl]); tgt = tgt_all[sl]
    cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt()); opt.zero_grad(); (-cos.mean()).backward(); opt.step()
    with torch.no_grad(): m.theta[0] = 0
print("trained. gates OPEN:", sorted(c for c, v in m.gates().items() if v > 0.5))
def case(depth, n_quant, ambiguous, n=40, patterns=None):
    erng = np.random.default_rng(500 + depth); nouns, dets, scope, full, prof, det_scores = [], [], [], [], [], []
    for t in (patterns or range(len(q.PATTERNS))):
        evs = [w.qevent(erng, depth=depth, nouns_p=trNP, nouns_o=trNO, ambiguous=ambiguous, n_quant=n_quant) for _ in range(n)]; toks = [w.qrender(ev, t, erng) for ev in evs]
        c, i_, k = m.encode(toks)
        with torch.no_grad(): S = m.states(c, i_, k)
        for j, ev in enumerate(evs):
            o = q.decode_q(S[j], m, world, w, trNP, trNO, depth)
            nouns.append(all(o[r] == ev[i][1] for r, i in (("A", 1), ("B", 2), ("C", 3)))); dets.append(all(o["d" + r] == ev[i][0] for r, i in (("A", 1), ("B", 2), ("C", 3))))
            scope.append(o["scope"] == sorted(ev[-1]) if ambiguous else o["scope"] == ev[-1]); full.append(nouns[-1] and dets[-1] and scope[-1])
            ss = o["scope_scores"]; wide = ev[-1]; prof.append((np.mean([ss[r] for r in wide]), np.mean([ss[r] for r in ss if r not in wide]) if len(wide) < 3 else 0))
    p = np.array(prof); return np.mean(nouns), np.mean(dets), np.mean(scope), np.mean(full), p[:, 0].mean(), p[:, 1].mean()
print("\ncase | depth | nouns | dets | scope | FULL | scope score: true wide role | other roles")
for depth in (0, 1, 2):
    for name, nq, amb in (("one quantifier", 1, False), ("two, marked", 2, False), ("ambiguous (two unmarked)", None, True)):
        r = case(depth, nq, amb, n=30); print(f"  {name:26s} | {depth} | " + " | ".join(f"{x:.3f}" for x in r))
print("\nper pattern, depth 0, unambiguous (nouns | dets | scope | FULL):")
for t, pat in enumerate(q.PATTERNS):
    r = case(0, None, False, n=40, patterns=[t]); print(f"  '{pat}': " + " | ".join(f"{x:.3f}" for x in r[:4]))
print("\nper determiner (one-quantifier sentences, depth 0): scope accuracy")
erng = np.random.default_rng(9); acc = {d: [] for d in q.DETS_Q}
for _ in range(300):
    ev = w.qevent(erng, depth=0, nouns_p=trNP, nouns_o=trNO, n_quant=1); det = [x[0] for x in ev[1:4] if x[0] != "the"][0]; toks = w.qrender(ev, int(erng.integers(len(q.PATTERNS))), erng)
    c, i_, k = m.encode([toks])
    with torch.no_grad(): o = q.decode_q(m.states(c, i_, k)[0], m, world, w, trNP, trNO, 0)
    acc[det].append(o["scope"] == ev[-1])
print("  " + "  ".join(f"{d}: {np.mean(v):.2f} (n={len(v)})" for d, v in acc.items()))
