"""trace_b51 — rule 11 trace AFTER the B5.1 run (judgment e9274a9 Q1: "if A does NOT fail, the hard-gate assumption of the trace was wrong —
report HOW"), on the TRAINED seed-0 reader of v20_operators (logs/.ckpt_v20_b51_0.pt). CPU, no training.
(1) The subject's ADDRESS (the reader's state with the subject as the only non-zero identity) in "p0 must give o0 to p1" vs "p0 must not give
    o0 to p1" and in "p0 does not give" vs "p0 does not have to give": cos between the two addresses, and cos of the subject's term with the two
    TARGET terms (R_must R_A f vs R_must R_neg R_A f) — with the LEARNED soft gates, and with the same gates HARDENED (logit +-8): the 12-bis
    prediction holds for hard gates; the learned gates are the 'how'.
(2) The unseen depth 2, where the subject failed for EVERY form (0.07-0.18) while B, C and the predicate were read: per CODE (path / bigram /
    chainL alone) the cosine of the subject's term with its target at depths 1 and 2, affirmative form, and the same for B — which code carries
    which word, and which one does not compose to the unseen depth."""
import numpy as np, torch, v5_scale as v5, v8_gates as v8, v20_operators as v20
D = 1024; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; seed = 0
w = v20.OpWorld(18, np.random.default_rng(seed)); F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v20.OpRoleWorld(F, w, np.random.default_rng(100 + seed))
m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], "cpu", gate_init="closed"); m.set_F(F)
st = torch.load("logs/.ckpt_v20_b51_0.pt", map_location="cpu")
with torch.no_grad():
    for p_, v in zip([m.theta, m.logits], st["params"]): p_.copy_(v)
Fn = torch.tensor(F, dtype=m.F.dtype); cos = lambda x, y: float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt() + 1e-12))
def address(sent, tok):
    cats, ids, mask = m.encode([sent]); F0 = torch.zeros(len(w.vocab), D, dtype=m.F.dtype); F0[w.idx[tok]] = 1.0; keep = m.F; m.F = F0
    with torch.no_grad(): a = m.states(cats, ids, mask)[0]
    m.F = keep; return a
R = {k: torch.tensor(v, dtype=m.F.dtype) for k, v in world.R.items()}
g = m.gates(); print("trace_b51 on the trained seed-0 reader. Learned gates of the operator words: " + ", ".join(f"{c} {g[c]:.2f}" for c in ("must", "may", "might", "has", "have", "need", "does", "not", "never", "to", "that", "P", "O")))
print("\n(1) the subject's address under two stacked operators vs one, learned soft gates then hardened gates")
pairs = [("p0 must give o0 to p1 .", "p0 must not give o0 to p1 .", ["must"], ["must", "neg"]), ("p0 does not give o0 to p1 .", "p0 does not have to give o0 to p1 .", ["neg"], ["neg", "must"]), ("p0 may give o0 to p1 .", "p0 may not give o0 to p1 .", ["may"], ["may", "neg"])]
def tgt(ops):
    k = R["A"].clone()
    for op in ops: k = k * R[op]
    return k
logits0 = m.logits.detach().clone()
for mode in ("learned", "hardened"):
    if mode == "hardened":
        with torch.no_grad(): m.logits.copy_(torch.where(logits0 > 0, torch.full_like(logits0, 8.0), torch.full_like(logits0, -8.0)))
    print(f"  -- gates {mode}" + ("" if mode == "learned" else " (every learned gate > 0.5 set to 1, the rest to 0)"))
    for s1, s2, o1, o2 in pairs:
        a1, a2 = address(s1.split(), "p0"), address(s2.split(), "p0"); t1, t2 = tgt(o1), tgt(o2)
        print(f"     '{s1[:-2]}' vs '{s2[:-2]}': cos(addresses) {cos(a1, a2):+.3f} | sentence 1: cos with its target {cos(a1, t1):+.3f}, with the other {cos(a1, t2):+.3f} | sentence 2: with its target {cos(a2, t2):+.3f}, with the other {cos(a2, t1):+.3f}")
with torch.no_grad(): m.logits.copy_(logits0)
print("  reading: with the learned gates the subject's two addresses differ (the modal's gate is partly open, so a share of the subject re-seats PAST it to the negation when there is one, and to the next emitter when there is not); hardened, the addresses coincide and the reading cannot separate the targets — the 12-bis prediction.")
print("\n(2) unseen depth 2: which code carries the subject, per code alone (cos of the word's term with its target), affirmative form")
rng = np.random.default_rng(5)
for depth in (1, 2):
    ev = w.oevent(rng, w.persons[:12], w.objects[:12], "aff", depth=depth); toks = w.orender(ev, rng, 0); rep = R["rep"] ** depth
    for codes in (["path", "chainL", "bigram"], ["path"], ["bigram"], ["chainL"]):
        m.codes = codes; line = []
        for tok, role in ((ev["A"], "A"), (ev["B"], "B"), (ev["C"], "C")):
            a = address(toks, tok); line.append(f"{role} {cos(a, rep * R[role]):+.3f}")
        print(f"   depth {depth} | codes {'+'.join(codes):18s} | " + " | ".join(line) + f"   ({' '.join(toks)})")
m.codes = ["path", "chainL", "bigram"]
print("  reading: the code whose cosine for A falls between depth 1 and depth 2 is the one that does not compose for the subject at the unseen depth; B and C are compared on the same rows.")

print("\n(3) the subject under the GOLD key, per form, depth 0, 40 events: mean cos of the subject's term with its target term, mean cleanup margin (best - second candidate), accuracy")
erng = np.random.default_rng(900); Fn_ = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); cid = torch.tensor([w.idx[x] for x in w.persons[:12]])
for form in v20.FORMS:
    cs, mg, ok = [], [], []
    for k in range(40):
        ev = w.oevent(erng, w.persons[:12], w.objects[:12], form, depth=0); toks = w.orender(ev, erng, t=k % 2); cats, ids, mask = m.encode([toks])
        with torch.no_grad(): S = m.states(cats, ids, mask)[0]
        key = torch.conj(R["A"]).clone()
        for op in v20.OPS[form]: key = key * torch.conj(R[op])
        Sn = S / S.abs().pow(2).sum().sqrt(); sc = torch.real((Sn * key) @ torch.conj(Fn_[cid]).T); top = torch.topk(sc, 2)
        cs.append(float(top.values[0])); mg.append(float(top.values[0] - top.values[1])); ok.append(w.persons[:12][int(top.indices[0])] == ev["A"])
    print(f"   {form:9s} ({len(v20.OPS[form])} operator(s) after the subject) | cos {np.mean(cs):.3f} | margin {np.mean(mg):.3f} | accuracy {np.mean(ok):.3f}")
print("  reading: the identity cleanup picks the right filler as long as the blended address keeps a positive cosine with the gold key and the other 11 candidates stay near zero; the cosine and the margin, not the accuracy, show how much of the subject's address the stacked operators cost.")
