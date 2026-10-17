"""B7 settlement (judgment d8f8a98 Q3 a-c), on the SAVED models — the three-code reference (logs/.ckpt_v16_relg_...) and the four-code B7 run
(logs/.ckpt_v16_b7_...), 5 seeds each, same worlds, same evaluation rngs as v16; no training.
(a) TRACE of the +0.051 on unseen transitions (held-out templates with unseen category transitions): per template and per role, the accuracy of the
    three-code and the four-code reader; and, for the four-code reader on those sentences, the cosine of each word's PER-CODE term (path / bigram /
    chainL / position, each alone) with the word's target address — which code votes for the right address where the three-code reader fails.
(b) OUT-OF-RANGE RULE: the position term is OMITTED for a position beyond the trained range (the longest training sentence at depths 0-2) instead of
    carrying an untrained displacement. Trained range computed from the training sentences; depth 3 and every axis re-scored with the rule.
    Expectation written before the run (judgment): no loss, possibly a small gain at depth 3.
(c) TERMS PER SENTENCE against the B6 capacity curve (32 terms clean at D = 1024, 64 begin to fail): tokens per sentence by depth in the reference
    world, x3 and x4 codes; and at P1 lengths (20 / 30 tokens).
Run: python3 rescore_v16_b7.py"""
import argparse, numpy as np, torch, guard, v5_scale as v5, v8_gates as v8, v16_reference as v16

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--device", default="cuda"); ap.add_argument("--n_scale", type=float, default=1.0); a = ap.parse_args()
D, dev = a.D, a.device; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w0 = v16.RefWorld(18, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); verbs = sorted(set(v16.LEMMA.values()))
print(f"\nrescore_v16_b7 (B7 settlement): three-code reference vs four-code B7 models, {a.seeds} seeds; unseen-transition templates {held_new}")
tcos = lambda x, y: float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt() + 1e-12))
# trained position range: the longest training sentence (depths 0-2, TRAIN_T templates and recipient-less forms, prefixed reporting)
rng0 = np.random.default_rng(0); L = 0
for n in range(3000):
    d = int(rng0.integers(3)); ev = w0.event(w0.persons[:12], w0.objects[:12], depth=d); L = max(L, len(w0.render(ev, int(rng0.choice(v5.TRAIN_T)))), len(w0.render_omit(ev, 0)))
POS_MAX = L; print(f"trained position range: positions 0-{POS_MAX - 1} (longest training sentence {POS_MAX} tokens); depth-3 sentences reach {max(len(w0.render(w0.event(w0.persons[:12], w0.objects[:12], depth=3), t)) for t in v5.TRAIN_T)} tokens")
def load(seed, codes, tag):
    w = v16.RefWorld(18, np.random.default_rng(seed)); F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v16.RefRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = v8.V8(w, D, np.random.default_rng(200 + seed), codes, dev, gate_init="closed"); m.fix_gates(["to", "by", "from"], True); m.set_F(F)
    st = torch.load(f"logs/.ckpt_v16_{tag}_D1024_omit0.2_learned_learned_open_{seed}.pt", map_location=dev)
    with torch.no_grad():
        for p_, v in zip([m.theta, m.logits], st["params"]): p_.copy_(v)
    return w, world, m
def run(m, w, world, templs, P, O, depth, n):
    erng = np.random.default_rng(900 + depth + 7 * len(templs)); per = {}
    for t in templs:
        evs = [w.event(P, O, depth=depth) for _ in range(max(1, int(n * a.n_scale)))]; toks = [w.render(ev, t) for ev in evs]; c, i_, k = m.encode(toks)
        with torch.no_grad(): S = m.states(c, i_, k)
        for j, ev in enumerate(evs):
            o = v16.readout(S[j], m, world, w, depth, P, O, verbs); d = per.setdefault(t, {"A": [], "B": [], "C": [], "FULL": []})
            oks = [o["A"]["id"] == ev[1], o["B"]["id"] == ev[2], o["C"]["id"] == ev[3]]
            for r, ok in zip(("A", "B", "C"), oks): d[r].append(ok)
            d["FULL"].append(all(oks))
    return per
print("\n== (a) unseen-transition templates, per template and role: three codes | four codes (means over seeds)")
acc3, acc4, votes = {}, {}, {}
for seed in range(a.seeds):
    w3, world3, m3 = load(seed, ["path", "chainL", "bigram"], "relg"); w4, world4, m4 = load(seed, ["path", "chainL", "bigram", "pos"], "b7"); trP, teP, trO, teO = w3.persons[:12], w3.persons[12:], w3.objects[:12], w3.objects[12:]
    p3 = run(m3, w3, world3, held_new, trP, trO, 0, 20); p4 = run(m4, w4, world4, held_new, trP, trO, 0, 20)
    for t in held_new:
        for k in ("A", "B", "C", "FULL"): acc3.setdefault((t, k), []).append(np.mean(p3[t][k])); acc4.setdefault((t, k), []).append(np.mean(p4[t][k]))
    # per-code votes of the four-code reader on the unseen-transition sentences: cosine of each word's single-code term with its target address
    R = {k: torch.tensor(v, dtype=m4.F.dtype, device=dev) for k, v in world4.R.items()}; mk = torch.tensor(world4.m, dtype=m4.F.dtype, device=dev); erng = np.random.default_rng(77)
    for t in held_new:
        for _ in range(int(10 * a.n_scale) or 1):
            ev = w4.event(trP, trO, depth=0); toks = w4.render(ev, t); cats, ids, mask = m4.encode([toks])
            for role, word in (("A", ev[1]), ("B", ev[2]), ("C", ev[3])):
                tgt = R[role]; pos = toks.index(word)
                for code in ("path", "bigram", "chainL", "pos", "all"):
                    m4.codes = ["path", "chainL", "bigram", "pos"] if code == "all" else ([code] if code != "chainL" else ["chainL"])
                    F0 = torch.zeros(len(w4.vocab), D, dtype=m4.F.dtype, device=dev); F0[w4.idx[word]] = 1.0; keep = m4.F; m4.F = F0
                    with torch.no_grad(): addr = m4.states(cats, ids, mask)[0]
                    m4.F = keep; votes.setdefault((t, role, code), []).append(tcos(addr, tgt))
            m4.codes = ["path", "chainL", "bigram", "pos"]
    print(f"  seed {seed} read", flush=True)
for t in held_new:
    print(f"   T{t} '{v5.PATTERNS[t]}': " + " | ".join(f"{k} {np.mean(acc3[(t, k)]):.3f} -> {np.mean(acc4[(t, k)]):.3f}" for k in ("A", "B", "C", "FULL")))
print("   per-code vote (four-code reader; cos of the word's single-code address with its role address, mean over seeds and events): template | role | path | bigram | chainL | position | all four")
for t in held_new:
    for role in ("A", "B", "C"): print(f"   T{t} | {role} | " + " | ".join(f"{np.mean(votes[(t, role, c)]):+.3f}" for c in ("path", "bigram", "chainL", "pos", "all")))
print("   reading aid: the code whose single-code cosine is high where the three-code reader fails is the one that carries the gain; a position cosine near 0 means the position code does not address that word (its displacement is per absolute position: a word that moves between templates gets a different one).")
print("\n== (b) out-of-range rule: the position term omitted beyond the trained range (positions >= %d) — four-code models re-scored, every axis; seen / new ids / unseen trans. / depth 2 / depth 3 UNSEEN (means over seeds)" % POS_MAX)
rows = {"four codes, as trained (untrained displacements beyond the range)": [], "four codes, position term OMITTED beyond the range": [], "three codes (10b-quinquies models)": []}
for seed in range(a.seeds):
    for name, codes, tag, pm in (("three codes (10b-quinquies models)", ["path", "chainL", "bigram"], "relg", None), ("four codes, as trained (untrained displacements beyond the range)", ["path", "chainL", "bigram", "pos"], "b7", None), ("four codes, position term OMITTED beyond the range", ["path", "chainL", "bigram", "pos"], "b7", POS_MAX)):
        w, world, m = load(seed, codes, tag); m.pos_max = pm; trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        f = lambda per: float(np.mean([x for t in per for x in per[t]["FULL"]]))
        rows[name].append([f(run(m, w, world, v5.TRAIN_T, trP, trO, 0, 30)), f(run(m, w, world, v5.TRAIN_T, teP, teO, 0, 20)), f(run(m, w, world, held_new, trP, trO, 0, 20)), f(run(m, w, world, v5.TRAIN_T, trP, trO, 2, 15)), f(run(m, w, world, v5.TRAIN_T, trP, trO, 3, 15))])
for name, r in rows.items():
    r = np.array(r); print(f"   {name:64s} | " + " | ".join(f"{x:.3f}±{s:.3f}" for x, s in zip(r.mean(0), r.std(0))))
print("   expectation (written before): omitting the untrained positions costs nothing and may gain a little at depth 3.")
print("\n== (c) terms per sentence against the B6 capacity curve (32 terms clean at D = 1024, 64 begin to fail): depth | tokens (mean, max) | three codes | four codes")
for depth in (0, 1, 2, 3):
    ls = [len(w0.render(w0.event(w0.persons[:12], w0.objects[:12], depth=depth), t)) for t in v5.TRAIN_T for _ in range(20)]; print(f"   {depth} | {np.mean(ls):.1f}, {max(ls)} | {3 * np.mean(ls):.0f} (max {3 * max(ls)}) | {4 * np.mean(ls):.0f} (max {4 * max(ls)})")
for n in (20, 30): print(f"   P1 length {n} tokens | three codes {3 * n} terms | four codes {4 * n} terms")
print("   note: the chain code contributes one term per token like the others; the bundle's term count is tokens x codes, and the B6 curve was measured with role-filler terms over 1,000 candidates.")
print("\nchecklist: [x] saved params, no retraining [x] same evaluation rngs as v16 [x] trace per template, role and code (11) [x] out-of-range rule declared and measured once, expectation written before [x] terms per sentence vs the B6 curve [x] 5 seeds")
