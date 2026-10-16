"""b5_0_collisions — B5.0 (instruction after the B5 stop, step 1): MEASURE the operator-ORDER collision of commutative binding as it is, in the
E1 (negation, v13), E3 (modality / conditional, v15) and B3 (commands, v18) worlds. No remedy is built; nothing is added to the reader.

Part A (algebra, no reader). For every world: the operator stackings its grammar RENDERS, and every PAIR of sentential operators whose two
orders are expressible as world states (the second order may be unrenderable by the grammar: stated). Per pair, over fresh events: cos between
the two world states; the classification 'COLLIDING' (every term coincides: the same frame) or 'SEPARATED by a witness term' (a term under only
one of the two operators: the speaker, the predicate, a filler) with the cosine of the witness terms. Rule 11 trace on one colliding pair
("must not P" / "not must P": no world of the repository has both operators, so the pair is built from E3's R_must and a R_neg drawn like E1's)
and on one witnessed pair, term by term, marking the terms that coincide. Word-level operators (constituent negation, the negated attribute)
are not order operators and are listed as such.
Part B (reader charts). Seed 0 of each world's reader is RETRAINED with the regime of the original run (E1b 9b-bis: negation at depths 0-1;
E3b 9d-bis: p_mixed 0.15; B3 10d: p_depth1 0.3) — no checkpoints were saved by those runs; this is a reproduction, not a remedy. For each
rendered sentence whose order-reversed counterpart can be built from the world's OWN vocabulary:
   E1: "X says that A never gives B to C"            / "X never says that A gives B to C"
   E3: "if P , then A may give B to C"               / "perhaps if P , then A gives B to C"
   B3: "the admin says that do not delete the saved files" / "the admin never says that delete the saved files"
cos between the two reader charts; cos between the two bags of identities (control: the bag sees the same words); and whether the two decode
to DIFFERENT frames under the two order hypotheses (the witness term read from the chart), with the margin. The reversed sentences were never
in training (stated). Run: python3 b5_0_collisions.py [--N 40000] [--device cuda]"""
import argparse, time, numpy as np, torch
import guard, trainer, v5_scale as v5, v8_gates as v8, v13_negation as v13, v15_modality as v15, v18_commands as v18

ap = argparse.ArgumentParser(); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=100); a = ap.parse_args()
D, dev = a.D, a.device; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
cosn = lambda x, y: float(np.real(np.vdot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
def tcos(x, y): return float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt() + 1e-12))
print("B5.0: the operator-order collision of commutative binding, measured as it is (instruction after the B5 stop, step 1). No remedy.\n")

# ---------------- Part A: algebra ----------------
print("== Part A: operator algebra of the three worlds (world states only, no reader)")
rngA = np.random.default_rng(0)
w1 = v13.NegWorld(18, np.random.default_rng(0)); F1 = np.exp(1j * np.random.default_rng(300).uniform(-np.pi, np.pi, (len(w1.vocab), D))); W1 = v13.NegRoleWorld(F1, w1, np.random.default_rng(100))
w3 = v15.MWorld(18, np.random.default_rng(0)); F3 = np.exp(1j * np.random.default_rng(300).uniform(-np.pi, np.pi, (len(w3.vocab), D))); W3 = v15.MRoleWorld(F3, w3, np.random.default_rng(100))
wB = v18.CmdWorld(np.random.default_rng(0)); FB = np.exp(1j * np.random.default_rng(300).uniform(-np.pi, np.pi, (len(wB.vocab), D))); WB = v18.CmdRoleWorld(FB, wB, np.random.default_rng(100))
R_neg_for_E3 = np.exp(1j * np.random.default_rng(101).uniform(-np.pi, np.pi, D))   # E3 has no negation: drawn like E1's R_neg, for the colliding-pair trace only
print("rendered stackings — E1 (v13): rep^d o neg_S (d 0-1 in E1b training, 2 tested), rep^d o neg_A/B/C (word-level, not an order operator); the order neg o rep ('X never says that ...') is NOT rendered.")
print("rendered stackings — E3 (v15): rep^d o may / must, rep^d o if-then, rep^d o if-then(may|must on Q) (the mixed construction); the orders may o rep, may o if-then, must o may are NOT rendered.")
print("rendered stackings — B3 (v18): rep^d o neg_S (prohibition), attr-negation (word-level, inside the patient), neg_S o attr-neg (both); the order neg_S o rep is NOT rendered.\n")
def clause1(rng):
    ev = w1.event(w1.persons[:12], w1.objects[:12], depth=0); kind, A, B, C = ev[:4]
    return W1.R["A"] * F1[w1.idx[A]] + W1.R["B"] * F1[w1.idx[B]] + W1.R["C"] * F1[w1.idx[C]], rng.choice(w1.persons[12:])
def clause3(rng):
    cl = w3.clause(rng, w3.persons[:12], w3.objects[:12]); return W3.clause_state(cl)
def clauseB(rng):
    ev = wB.event(rng, v18.PATIENTS[:6], "aff"); f = lambda x: FB[wB.idx[x]]
    return WB.R["act"] * f(ev["act"]) + WB.R["pat"] * (f(ev["pat"]) + WB.R["attr"] * f(ev["attr"]))
pairs = []   # (world, name, op1 name, op2 name, rendered(op1 o op2), rendered(op2 o op1), S1 builder, S2 builder, witness description)
def rep1(S, x): return W1.R["X"] * F1[w1.idx[x]] + W1.R["rep"] * S
def rep3(S, rng): x = rng.choice(w3.persons[:12]); return W3.R["X"] * F3[w3.idx[x]] + W3.R["rep"] * S
def repB(S, rng): x = rng.choice(v18.PRINCIPALS); return WB.R["X"] * FB[wB.idx[x]] + WB.R["rep"] * S
rows = []
for name, world, builder in (
    ("E1", "rep o neg_S vs neg_S o rep", lambda rng: (lambda S, x: (rep1(W1.R["neg"] * S, x), W1.R["neg"] * rep1(S, x)))(*clause1(rng))),
    ("E3", "rep o may vs may o rep", lambda rng: (lambda S: (rep3(W3.R["may"] * S, np.random.default_rng(int(rng.integers(1 << 30)))), W3.R["may"] * rep3(S, np.random.default_rng(int(rng.integers(1 << 30))))))(clause3(rng))),
    ("E3", "must o may vs may o must", lambda rng: (lambda S: (W3.R["must"] * W3.R["may"] * S, W3.R["may"] * W3.R["must"] * S))(clause3(rng))),
    ("E3", "if-then(may on Q) vs may o if-then", lambda rng: (lambda P, Q: (W3.R["if"] * P + W3.R["then"] * W3.R["may"] * Q, W3.R["may"] * (W3.R["if"] * P + W3.R["then"] * Q)))(clause3(rng), clause3(rng))),
    ("E3", "rep o if-then vs if-then o rep (both clauses reported)", lambda rng: (lambda P, Q, r: (rep3(W3.R["if"] * P + W3.R["then"] * Q, r), W3.R["if"] * rep3(P, r) + W3.R["then"] * rep3(Q, r)))(clause3(rng), clause3(rng), np.random.default_rng(int(rng.integers(1 << 30))))),
    ("E3+neg", "must o neg vs neg o must (constructed: no world has both)", lambda rng: (lambda S: (W3.R["must"] * R_neg_for_E3 * S, R_neg_for_E3 * W3.R["must"] * S))(clause3(rng))),
    ("B3", "rep o neg_S vs neg_S o rep", lambda rng: (lambda S: (repB(WB.R["neg"] * S, rng), WB.R["neg"] * repB(S, rng)))(clauseB(rng))),
):
    rendered = {"E1": "first rendered, second NOT", "E3": "first rendered (modal / if-then inside a report; mixed), second NOT", "E3+neg": "neither (constructed)", "B3": "first rendered, second NOT"}[name]
    if world == "must o may vs may o must": rendered = "neither rendered (the grammar never stacks two modals)"
    cs = [cosn(*builder(np.random.default_rng(1000 + k))) for k in range(a.n)]; mean = float(np.mean(cs))
    cls = "COLLIDING (identical state, same frame)" if mean > 0.999 else "SEPARATED by a witness term"
    rows.append((name, world, mean, cls, rendered)); print(f"  {name:6s} | {world:58s} | cos(world states) {mean:.3f} ± {np.std(cs):.3f} | {cls} | {rendered}")
print(f"  counts: colliding pairs {sum(1 for r in rows if r[3].startswith('COLL'))}, witnessed pairs {sum(1 for r in rows if r[3].startswith('SEP'))}; rendered sentence pairs differing ONLY in operator order: 0 in all three grammars (every grammar nests its operators in one fixed order)")
# rule 11 traces, term by term
print("\n== rule 11 trace, colliding pair: 'A must not give B to C' (must o neg) vs 'A does not have to give B to C' (neg o must), one event")
rng = np.random.default_rng(7); cl = w3.clause(rng, w3.persons[:12], w3.objects[:12]); kind, A, B, C = cl
for r, x in (("A", A), ("B", B), ("C", C)):
    t1 = W3.R["must"] * R_neg_for_E3 * W3.R[r] * F3[w3.idx[x]]; t2 = R_neg_for_E3 * W3.R["must"] * W3.R[r] * F3[w3.idx[x]]
    print(f"   term {r} ({x}): R_must R_neg R_{r} f  vs  R_neg R_must R_{r} f : cos {cosn(t1, t2):.3f}  -> COINCIDES (commutative binding)")
print("   every term coincides: the two states are identical; the bag of identities is identical too (same words up to the auxiliary); the frame is the same -> colliding.")
print("\n== rule 11 trace, witnessed pair: 'X says that A never gives B to C' (rep o neg) vs 'X never says that A gives B to C' (neg o rep), one event")
S, X = clause1(np.random.default_rng(8)); ev = w1.event(w1.persons[:12], w1.objects[:12], depth=0); kind, A, B, C = ev[:4]
for r, x in (("A", A), ("B", B), ("C", C)):
    t1 = W1.R["rep"] * W1.R["neg"] * W1.R[r] * F1[w1.idx[x]]; t2 = W1.R["neg"] * W1.R["rep"] * W1.R[r] * F1[w1.idx[x]]; print(f"   term {r} ({x}): R_rep R_neg R_{r} f  vs  R_neg R_rep R_{r} f : cos {cosn(t1, t2):.3f}  -> COINCIDES")
t1 = W1.R["X"] * F1[w1.idx[X]]; t2 = W1.R["neg"] * W1.R["X"] * F1[w1.idx[X]]; print(f"   term X ({X}): R_X f_X  vs  R_neg R_X f_X : cos {cosn(t1, t2):.3f}  -> DIFFERS: the speaker term is the witness; whole-state cos {cosn(rep1(W1.R['neg'] * S, X), W1.R['neg'] * rep1(S, X)):.3f} (3 of 4 terms shared)")

# ---------------- Part B: reader charts (seed 0 retrained per world, original regimes) ----------------
print("\n== Part B: reader charts on the three constructible pairs (seed 0 of each world retrained with the original regime; the reversed sentence was never in training)")
def train(w, world, sents_targets, tag):
    sents, targets = sents_targets; m = v8.V8(w, D, np.random.default_rng(200), ["path", "chainL", "bigram"], dev, gate_init="closed"); m.set_F(world.F)
    cats, ids, mask = m.encode(sents); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device); t0 = time.time()
    trainer.train_loop(m, [m.theta, m.logits], cats, ids, mask, tgt, 8, 0.02, log_every=2000, tag=f" {tag}")
    print(f"  {tag}: reader trained ({time.time() - t0:.0f}s); gates OPEN: {sorted(c for c, v in m.gates().items() if v > 0.5)}"); return m
def Rt(world, m): return {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
def read_mean(Sn, m, keys_cands):
    """mean best cosine over (key, candidate ids) pairs"""
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); tot = 0.0
    for key, cid in keys_cands: tot += float(torch.real((Sn * key) @ torch.conj(Fn[cid]).T).max())
    return tot / len(keys_cands)
def report_pair(name, m, world, w, sent_pairs, hyps):
    """sent_pairs: list of (toks1, toks2, S1_world, S2_world); hyps: function(Sn, m) -> {h: score}; H1 = the rendered order, H2 = the reversed"""
    cc, cb, cw, h1ok, h2ok, differ, mg1, mg2 = [], [], [], [], [], [], [], []
    for t1, t2, Sw1, Sw2 in sent_pairs:
        c, i_, k = m.encode([t1, t2])
        with torch.no_grad(): S = m.states(c, i_, k)
        cc.append(tcos(S[0], S[1])); cb.append(tcos(m.F[i_[0]].sum(0), m.F[i_[1]].sum(0))); cw.append(cosn(Sw1, Sw2))
        s1 = hyps(S[0] / S[0].abs().pow(2).sum().sqrt(), m); s2 = hyps(S[1] / S[1].abs().pow(2).sum().sqrt(), m)
        a1 = max(s1, key=s1.get); a2 = max(s2, key=s2.get); h1ok.append(a1 == "H1"); h2ok.append(a2 == "H2"); differ.append(a1 != a2)
        mg1.append(s1["H1"] - s1["H2"]); mg2.append(s2["H2"] - s2["H1"])
    print(f"  {name}: n {len(sent_pairs)} | cos world states {np.mean(cw):.3f} | cos reader charts {np.mean(cc):.3f} | cos bags {np.mean(cb):.3f} | order hypothesis right on the rendered sentence {np.mean(h1ok):.3f} (margin {np.mean(mg1):+.3f}), on the reversed (unseen) sentence {np.mean(h2ok):.3f} (margin {np.mean(mg2):+.3f}) | frames DIFFER {np.mean(differ):.3f}")
n = a.n
# E1
w = v13.NegWorld(18, np.random.default_rng(0)); trP, trO = w.persons[:12], w.objects[:12]; F = np.exp(1j * np.random.default_rng(300).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v13.NegRoleWorld(F, w, np.random.default_rng(100))
rng = np.random.default_rng(200); sents, targets = [], []
for _ in range(a.N):
    if rng.random() < 0.4: sc = ["S", "A", "B", "C"][int(rng.integers(4))]; ev = w.event(trP, trO, depth=int(rng.integers(2)), neg=sc); toks = w.render(ev, int(rng.integers(2)))
    else: ev = w.event(trP, trO, depth=int(rng.integers(3))); toks = w.render(ev, int(rng.choice(v5.TRAIN_T)))
    sents.append(toks); targets.append(world.state(ev, toks))
m1 = train(w, world, (sents, targets), "E1 (E1b regime: negation at depths 0-1, p_neg 0.4)")
erng = np.random.default_rng(900); prs = []
for _ in range(n):
    ev = w.event(trP, trO, depth=1, neg="S"); t1 = w.render(ev, 1, outer=[0]); kind, A, B, C = ev[:4]; X = ev[4]
    ev0 = ev[:4] + (None,) * (len(ev) - 5) + (None,); inner = w.frames[kind][0].format(A=A, B=B, C=C).split()   # "{A} gives {B} to {C} ."
    t2 = [X, "never", "says", "that"] + inner; S1 = world.state(ev, t1); Sc = world.R["A"] * F[w.idx[A]] + world.R["B"] * F[w.idx[B]] + world.R["C"] * F[w.idx[C]]; S2 = world.R["neg"] * (world.R["X"] * F[w.idx[X]] + world.R["rep"] * Sc)
    prs.append((t1, t2, S1, S2))
R = Rt(world, m1); cidP = torch.tensor([w.idx[x] for x in trP], device=dev); cidO = torch.tensor([w.idx[x] for x in trO], device=dev)
def hyps1(Sn, m):
    inner = [(torch.conj(R["rep"] * R["neg"] * R[r]), cidO if r == "B" else cidP) for r in ("A", "B", "C")]
    return {"H1": read_mean(Sn, m, inner + [(torch.conj(R["X"]), cidP)]), "H2": read_mean(Sn, m, inner + [(torch.conj(R["neg"] * R["X"]), cidP)])}
print(f"  example pair: {' '.join(prs[0][0])}  /  {' '.join(prs[0][1])}"); report_pair("E1 rep o neg vs neg o rep", m1, world, w, prs, hyps1)
# E3
w = v15.MWorld(18, np.random.default_rng(0)); trP, trO = w.persons[:12], w.objects[:12]; F = np.exp(1j * np.random.default_rng(300).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v15.MRoleWorld(F, w, np.random.default_rng(100))
rng = np.random.default_rng(200); sents, targets = [], []
for _ in range(a.N):
    u = rng.random()
    if u < 0.15: ev = w.mevent(rng, trP, trO, depth=0, op="if", cond_form=int(rng.integers(2)), inner_modal=["may", "must"][int(rng.integers(2))])
    elif u < 0.4: ev = w.mevent(rng, trP, trO, depth=int(rng.integers(3)))
    elif u < 0.7: ev = w.mevent(rng, trP, trO, depth=int(rng.integers(2)), op=["may", "must"][int(rng.integers(2))])
    else: ev = w.mevent(rng, trP, trO, depth=int(rng.integers(2)), op="if", cond_form=int(rng.integers(3)))
    sents.append(w.mrender(ev, rng, t=int(rng.integers(2)))); targets.append(world.state(ev))
m3 = train(w, world, (sents, targets), "E3 (E3b regime: p_mixed 0.15)")
erng = np.random.default_rng(900); prs = []
for _ in range(n):
    ev = w.mevent(erng, trP, trO, depth=0, op="if", cond_form=0, inner_modal="may"); t1 = w.mrender(ev, erng, t=0)
    aff = lambda cl: w.aff[cl[0]].format(A=cl[1], B=cl[2], C=cl[3]).split(); t2 = ["perhaps", "if"] + aff(ev["P"]) + [",", "then"] + aff(ev["Q"]) + ["."]
    S1 = world.state(ev); S2 = world.R["may"] * (world.R["if"] * world.clause_state(ev["P"]) + world.R["then"] * world.clause_state(ev["Q"])); prs.append((t1, t2, S1, S2))
R = Rt(world, m3); cidP = torch.tensor([w.idx[x] for x in trP], device=dev); cidO = torch.tensor([w.idx[x] for x in trO], device=dev)
def hyps3(Sn, m):
    Q = [(torch.conj(R["then"] * R["may"] * R[r]), cidO if r == "B" else cidP) for r in ("A", "B", "C")]
    P1 = [(torch.conj(R["if"] * R[r]), cidO if r == "B" else cidP) for r in ("A", "B", "C")]; P2 = [(torch.conj(R["may"] * R["if"] * R[r]), cidO if r == "B" else cidP) for r in ("A", "B", "C")]
    return {"H1": read_mean(Sn, m, Q + P1), "H2": read_mean(Sn, m, Q + P2)}
print(f"  example pair: {' '.join(prs[0][0])}  /  {' '.join(prs[0][1])}"); report_pair("E3 if-then(may on Q) vs may o if-then", m3, world, w, prs, hyps3)
# B3
rng = np.random.default_rng(0); w = v18.CmdWorld(rng); F = np.exp(1j * np.random.default_rng(300).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v18.CmdRoleWorld(F, w, np.random.default_rng(100)); trPat = v18.PATIENTS[:6]
rng = np.random.default_rng(200); sents, targets = [], []
for _ in range(a.N):
    form = list(v18.FORMS)[int(rng.integers(len(v18.FORMS)))]; d = 1 if rng.random() < 0.3 else 0; ev = w.event(rng, trPat, form, depth=d); sents.append(w.render(ev, rng)); targets.append(world.state(ev))
mB = train(w, world, (sents, targets), "B3 (10d regime: p_depth1 0.3)")
erng = np.random.default_rng(900); prs = []; f = lambda x: F[w.idx[x]]
for _ in range(n):
    ev = w.event(erng, trPat, "prohib", depth=1); X = ev["speakers"][0]; cmd = ["the", X, "says", "that", "do", "not", ev["act"], "the", ev["attr"], ev["pat"], "."]
    t2 = ["the", X, "never", "says", "that", ev["act"], "the", ev["attr"], ev["pat"], "."]
    Sc = world.R["act"] * f(ev["act"]) + world.R["pat"] * (f(ev["pat"]) + world.R["attr"] * f(ev["attr"])); S1 = world.R["X"] * f(X) + world.R["rep"] * world.R["neg"] * Sc; S2 = world.R["neg"] * (world.R["X"] * f(X) + world.R["rep"] * Sc)
    prs.append((cmd, t2, S1, S2))
R = Rt(world, mB); cidX = torch.tensor([w.idx[x] for x in v18.PRINCIPALS], device=dev); cidA = torch.tensor([w.idx[x] for x in v18.ACTIONS], device=dev); cidPt = torch.tensor([w.idx[x] for x in trPat], device=dev)
def hypsB(Sn, m):
    inner = [(torch.conj(R["rep"] * R["neg"] * R["act"]), cidA), (torch.conj(R["rep"] * R["neg"] * R["pat"]), cidPt)]
    return {"H1": read_mean(Sn, m, inner + [(torch.conj(R["X"]), cidX)]), "H2": read_mean(Sn, m, inner + [(torch.conj(R["neg"] * R["X"]), cidX)])}
print(f"  example pair: {' '.join(prs[0][0])}  /  {' '.join(prs[0][1])}"); report_pair("B3 rep o neg vs neg o rep", mB, world, w, prs, hypsB)
print("\nreading aid: with commutative binding the inner clause is identical in both orders by construction; a pair is decodable only through its witness term (the speaker, or the clause under one operator only). 'frames DIFFER' measures that witness on the reader's chart; the 'must not / not must' pair has no witness and is not decodable by any reader of these worlds.")
print("\nchecklist: [x] measurement only, no remedy, nothing added to the reader [x] algebra per world with rendered / not-rendered stated [x] colliding vs witnessed classified with counts [x] rule 11 traces term by term [x] reader charts: seed 0 retrained with the original regimes (stated) [x] bag control on the same rows [x] reversed sentences stated as never trained")
