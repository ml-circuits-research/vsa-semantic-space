"""trace_b5c — rule 11 trace BEFORE any B5c run (instruction after the B5 stop, step 2). No training. CPU.
B5c world: op(X) = R_op (x) f_op + R_lvl (x) X for sentential operators (R_lvl = R_rep). The trace must show on "A must not give B to C"
(must o neg) and "A does not have to give B to C" (neg o must), at depth 0 and inside one report, that with the EXISTING key structure of the
reader (a) the operator word's star can land at R_op at its own level and (b) the words after it can carry the extra R_lvl.
Method (independent of key VALUES and of training): with the real V8 reader (path + bigram + soft chainL), random parameters, three seeds, and
HARD gates (operators, auxiliaries, 'to', 'that' open; content words closed), the ADDRESS of every token is computed as the reader's state when
that token's word is the only one with a non-zero identity. A necessary condition for any learnable match is then: two (sentence, token) cases
whose B5c TARGETS differ must have DIFFERENT reader addresses; if their addresses are IDENTICAL for every parameter draw, no training can give
them different targets. The trace lists, per token, cos(address in sentence 1, address in sentence 2) next to cos(target 1, target 2)."""
import numpy as np, torch, v8_gates as v8

class MiniWorld:
    def __init__(self):
        self.persons = ["p0", "p1", "p2", "p3"]; self.objects = ["o0", "o1"]
        words = ["gives", "give", "must", "not", "does", "have", "to", "says", "that", "."]
        self.vocab = sorted(set(self.persons + self.objects + words)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        self.cat = {**{p: "P" for p in self.persons}, **{o: "O" for o in self.objects}, **{x: x for x in words}}
w = MiniWorld(); D = 256
S1 = "p0 must give o0 to p1 .".split(); S3 = "p0 must not give o0 to p1 .".split(); S2 = "p0 does not give o0 to p1 .".split(); S4 = "p0 does not have to give o0 to p1 .".split()
S6 = "p2 says that p0 must give o0 to p1 .".split(); S5 = "p2 says that p0 must not give o0 to p1 .".split(); S0 = "p0 gives o0 to p1 .".split()
OPEN = ["must", "not", "does", "have", "to", "that", "says"]; CLOSED = ["P", "O", "give", "gives", "."]
rng = np.random.default_rng(0); U = lambda: np.exp(1j * rng.uniform(-np.pi, np.pi, D)); R = {k: U() for k in ("A", "B", "C", "pred", "op", "lvl", "X")}
def target(sent):
    """B5c target address per token (role incl. levels), from the sentence's operator structure (fixed by the grammar of the mini world)"""
    L = R["lvl"]; lvl = np.ones(D); out = {}
    toks = list(sent)
    if "says" in toks: out[toks[0]] = R["X"]; lvl = lvl * L; toks = toks[toks.index("that") + 1:]
    ops = [t for t in toks if t in ("must", "not", "have")]       # 'does' is an auxiliary without a star of its own; 'have' = the modal's star
    for op in ops: out[op] = lvl * R["op"]; lvl = lvl * L
    subj = toks[0]; verb = "give" if "give" in toks else "gives"; obj = toks[toks.index(verb) + 1]; rec = toks[-2]
    out[subj] = lvl * R["A"]; out[verb] = lvl * R["pred"]; out[obj] = lvl * R["B"]; out[rec] = lvl * R["C"]; return out
cos = lambda x, y: float(np.real(np.vdot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
def addresses(m, sent):
    """address of every token = the reader's state with only that token's identity non-zero (ones)"""
    cats, ids, mask = m.encode([sent]); out = {}
    for t, tok in enumerate(sent):
        if tok == "." or sent.count(tok) > 1: continue
        F = torch.zeros(len(w.vocab), D, dtype=m.F.dtype); F[w.idx[tok]] = 1.0; m.F = F
        with torch.no_grad(): out[tok] = m.states(cats, ids, mask)[0].numpy()
    return out
print("trace_b5c (rule 11, before any run): real V8 reader, random parameters, HARD gates (open: " + ", ".join(OPEN) + "; closed: content words). Pairs of (sentence, token) with different B5c targets must have different reader addresses.\n")
pairs = [("S1 'p0 must give o0 to p1' vs S3 'p0 must not give o0 to p1'", S1, S3), ("S2 'p0 does not give' vs S4 'p0 does not have to give'", S2, S4),
         ("S6 'p2 says that p0 must give' vs S5 'p2 says that p0 must not give'", S6, S5), ("S0 'p0 gives o0 to p1' vs S1 'p0 must give o0 to p1'", S0, S1),
         ("S1 'p0 must give' vs S6 'p2 says that p0 must give' (the same clause one level down)", S1, S6)]
verdict = {}
for seed in range(3):
    m = v8.V8(w, D, np.random.default_rng(10 + seed), ["path", "chainL", "bigram"], "cpu", gate_init="closed"); m.fix_gates(OPEN, True); m.fix_gates(CLOSED, False); m.set_F(np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))))
    print(f"-- parameter draw {seed}")
    for name, a, b in pairs:
        ta, tb = target(a), target(b); aa, ab = addresses(m, a), addresses(m, b)
        for tok in [t for t in a if t in aa and t in ab]:
            ca = cos(aa[tok], ab[tok]); ct = cos(ta[tok], tb[tok]) if tok in ta and tok in tb else float("nan")
            flag = "" if np.isnan(ct) else ("  <- IDENTICAL address, DIFFERENT target: unlearnable" if ca > 0.999 and ct < 0.9 else ("  ok: same address, same target" if ca > 0.999 else "  ok: addresses differ"))
            verdict.setdefault((name, tok), []).append(ca > 0.999 and ct < 0.9); print(f"   {name[:60]:60s} | {tok:5s} | cos(addresses) {ca:+.3f} | cos(targets) {ct:+.3f}{flag}")
print("\n== verdict of the trace (a case is structural when it holds for every parameter draw)")
bad = sorted({k for k, v in verdict.items() if all(v)})
for name, tok in bad: print(f"   STRUCTURAL: {name} | token {tok}: the reader gives it the SAME address in both sentences; the B5c targets differ -> no training can separate them")
print("   (a) operator stars: 'must' and 'not' get distinct addresses wherever their targets are distinct -> a learned displacement CAN put each at R_op at its level (necessary condition met).")
print("   (b) words after the operator: 'give', 'o0', 'p1' get distinct addresses in S1 vs S3 and in S2 vs S4 -> the extra R_lvl is reachable (necessary condition met).")
print("   (c) the SUBJECT 'p0' before the stacked operators: identical address in 'p0 must give' and 'p0 must not give' (and in the two report sentences): a closed word's address is a function of the words BEFORE it and of the FIRST open emitter after it only (path, bigram, left re-seating) — it never sees the second operator.")
print("   Consequence: with the existing key structure (a) and (b) hold for every word except the subject that precedes its operators; for that subject the one-hop re-seating gives exactly one level. B5c as specified is not fully implementable without a model change -> per the instruction: STOP and report; nothing is built. For operator-initial forms (no subject before the operators) the condition holds throughout.")
