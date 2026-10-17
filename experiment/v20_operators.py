"""v20_operators — B5.1 (judgment e9274a9 Q1): subject-initial declaratives with a MODAL and/or NEGATION AFTER the subject, CURRENT operator
definition (operators as roles around their scope; nothing new in the reader). A MEASUREMENT, no criterion; the decision rule is stated below.

World (compositional, English; predicate role as in 10b, no occupancy marker — stated):
    S(clause) = R_pred (x) f_lemma + R_A f_A + R_B f_B + R_C f_C;  op(S) = R_op (x) S for op in {must, may, neg};  reporting R_X f_X + R_rep (x) S.
    Forms (one clause template, subject first):
      aff        "{A} V3 {B} to {C}"
      must       "{A} must V {B} to {C}" | "{A} has to V {B} to {C}"
      may        "{A} may V {B} to {C}"
      neg        "{A} does not V {B} to {C}" | "{A} never V3 {B} to {C}"
      must_neg   "{A} must not V {B} to {C}" | "{A} must never V {B} to {C}"            target R_must R_neg S
      neg_must   "{A} does not have to V {B} to {C}" | "{A} need not V {B} to {C}"     target R_neg R_must S  (= R_must R_neg S: the colliding pair, by construction)
      may_neg    "{A} may not V {B} to {C}" | "{A} might not V {B} to {C}"             target R_may R_neg S
    Training: every form at depth 0 and inside one report (depth 1); tested at depths 0, 1 (seen) and 2 (UNSEEN combination).
Rule 28 (address check, printed before training): with hard gates the SUBJECT has the identical address in "A must give" and "A must not give"
(12-bis) — this run is the DECLARED measurement of that limit on rendered sentences, so the pair is named and the run proceeds.
Readout (rule 17): operator STRUCTURE = argmax over the hypotheses {none, must, may, neg, must+neg, may+neg} of the mean best cosine of the
four roles read under the hypothesis' keys at the given depth; ROLES reported under the GOLD structure (so the role axis is not confounded by
the structure decode) and under the decoded one (FULL). Per form: A | B | C | pred | structure | FULL. The colliding pair must_neg / neg_must:
share of event pairs whose decoded frames DIFFER (0 by construction if the reader matches its targets; reported).
Decision rule (judgment): the subject A fails where two operators follow it while B, C, pred are read -> limit (ii) recorded with this extent,
B5d designed with a trace; A does NOT fail -> the hard-gate assumption of 12-bis was wrong (soft gates leak enough), report how, B5d dropped.
Run: python3 v20_operators.py [--seeds 5] [--N 40000] [--D 1024]"""
import argparse, time, numpy as np, torch
import guard, trainer, v5_scale as v5, v8_gates as v8

BASE = {"GIVE": "give", "SELL": "sell", "SEND": "send", "LEND": "lend", "SHOW": "show", "OFFER": "offer"}
FORMS = {"aff": ["{A} V3 {B} to {C} ."], "must": ["{A} must V {B} to {C} .", "{A} has to V {B} to {C} ."], "may": ["{A} may V {B} to {C} ."],
         "neg": ["{A} does not V {B} to {C} .", "{A} never V3 {B} to {C} ."], "must_neg": ["{A} must not V {B} to {C} .", "{A} must never V {B} to {C} ."],
         "neg_must": ["{A} does not have to V {B} to {C} .", "{A} need not V {B} to {C} ."], "may_neg": ["{A} may not V {B} to {C} .", "{A} might not V {B} to {C} ."]}
OPS = {"aff": [], "must": ["must"], "may": ["may"], "neg": ["neg"], "must_neg": ["must", "neg"], "neg_must": ["neg", "must"], "may_neg": ["may", "neg"]}
HYP = {"none": [], "must": ["must"], "may": ["may"], "neg": ["neg"], "must+neg": ["must", "neg"], "may+neg": ["may", "neg"]}
GOLD_H = {"aff": "none", "must": "must", "may": "may", "neg": "neg", "must_neg": "must+neg", "neg_must": "must+neg", "may_neg": "may+neg"}

class OpWorld(v5.BigWorld):
    def __init__(self, n_id, rng):
        super().__init__(n_id, rng)
        self.oframes = {k: {f: [p.replace("V3", v5.VERBS[k][0]).replace("V", BASE[k]) for p in ps] for f, ps in FORMS.items()} for k in self.frames}
        words = {x for fs in self.oframes.values() for ps in fs.values() for t in ps for x in t.split() if not x.startswith("{")}
        for x in words | set(BASE.values()):
            if x not in self.idx: self.vocab.append(x)
        self.vocab = sorted(set(self.vocab)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        for x in words: self.cat.setdefault(x, x)
        for v in v5.COP: self.cat[v] = "COP"
    def oevent(self, rng, persons, objects, form, depth=0):
        kind = rng.choice(list(self.frames)); ids = rng.choice(persons, 2 + depth, replace=False); B = rng.choice(objects)
        return {"kind": kind, "A": ids[0], "B": B, "C": ids[1], "speakers": list(ids[2:]), "form": form}
    def orender(self, ev, rng, t=0):
        ps = self.oframes[ev["kind"]][ev["form"]]; s = ps[t % len(ps)].format(A=ev["A"], B=ev["B"], C=ev["C"]).rsplit(" .", 1)[0]
        for x in reversed(ev["speakers"]): s = v5.OUTER[int(rng.integers(len(v5.OUTER)))].replace("<P>", s).format(X=x)
        return (s + " .").split()

class OpRoleWorld:
    def __init__(self, F, world, rng):
        self.F, self.w = F, world; D = F.shape[1]; u = lambda: np.exp(1j * rng.uniform(-np.pi, np.pi, D))
        self.R = {r: u() for r in ("A", "B", "C", "pred", "X", "rep", "must", "may", "neg")}
    def state(self, ev):
        w = self.w; f = lambda x: self.F[w.idx[x]]
        S = self.R["pred"] * f(BASE[ev["kind"]]) + self.R["A"] * f(ev["A"]) + self.R["B"] * f(ev["B"]) + self.R["C"] * f(ev["C"])
        for op in reversed(OPS[ev["form"]]): S = self.R[op] * S
        for x in reversed(ev["speakers"]): S = self.R["X"] * f(x) + self.R["rep"] * S
        return S

def decode(S, m, world, w, depth, persons, objects):
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}; Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum().sqrt()
    rep = torch.conj(R["rep"]) ** depth; lem = sorted(set(BASE.values()))
    cands = {"A": persons, "B": objects, "C": persons, "pred": lem}; cid = {r: torch.tensor([w.idx[x] for x in c], device=m.device) for r, c in cands.items()}
    out = {}
    for h, ops in HYP.items():
        key = rep
        for op in ops: key = key * torch.conj(R[op])
        tot, fill = 0.0, {}
        for r in ("A", "B", "C", "pred"):
            sc = torch.real((Sn * key * torch.conj(R[r])) @ torch.conj(Fn[cid[r]]).T); j = int(sc.argmax()); tot += float(sc[j]); fill[r] = cands[r][j]
        out[h] = (tot / 4, fill)
    best = max(out, key=lambda k: out[k][0]); return best, out

def address_check(w, D, seeds=2):
    """rule 28: token addresses across the pairs whose targets differ, hard gates (operators / auxiliaries / 'to' / 'that' open; content closed)"""
    OPEN = ["must", "not", "does", "has", "have", "need", "never", "may", "might", "to", "that", "says", "claims"]; CLOSED = ["P", "O", "COP", "."] + [c for c in set(w.cat.values()) if c in BASE.values() or c in [v5.VERBS[k][0] for k in v5.VERBS]]
    pairs = [("p0 must give o0 to p1 .", "p0 must not give o0 to p1 ."), ("p0 does not give o0 to p1 .", "p0 does not have to give o0 to p1 ."), ("p0 may give o0 to p1 .", "p0 may not give o0 to p1 ."), ("p0 gives o0 to p1 .", "p0 must give o0 to p1 .")]
    print("rule 28 address check (hard gates; cos between a token's addresses in the two sentences; the targets of A, B, C, pred differ between the two sentences of every pair):")
    for seed in range(seeds):
        m = v8.V8(w, D, np.random.default_rng(10 + seed), ["path", "chainL", "bigram"], "cpu", gate_init="closed"); m.fix_gates([c for c in OPEN if c in m.cat_id], True); m.fix_gates([c for c in CLOSED if c in m.cat_id], False)
        m.set_F(np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))))
        for a, b in pairs:
            ta, tb = a.split(), b.split(); line = []
            for tok in ("p0", "o0", "p1", "give", "gives"):
                if tok in ta and tok in tb:
                    ad = []
                    for s_ in (ta, tb):
                        cats, ids, mask = m.encode([s_]); F = torch.zeros(len(w.vocab), D, dtype=m.F.dtype); F[w.idx[tok]] = 1.0; m.F = F
                        with torch.no_grad(): ad.append(m.states(cats, ids, mask)[0])
                    c = float(torch.real((torch.conj(ad[0]) * ad[1]).sum()) / (ad[0].abs().pow(2).sum().sqrt() * ad[1].abs().pow(2).sum().sqrt())); line.append(f"{tok} {c:+.3f}" + (" IDENTICAL" if c > 0.999 else ""))
                m.set_F(np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))))
            print(f"   draw {seed} | '{a[:-2]}' vs '{b[:-2]}': " + " | ".join(line))
    print("   -> the subject p0 is IDENTICAL in 'must give' / 'must not give' and in 'does not give' / 'does not have to give': the named pair the reader cannot reach; this run is the declared measurement of that limit (rule 28 exception).\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--n_id", type=int, default=18); ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--p_depth1", type=float, default=0.4); ap.add_argument("--tag", default="b51"); ap.add_argument("--p_aff2", type=float, default=0.0, help="E1b protocol (rule 4, after the first B5.1 run): share of AFFIRMATIVE sentences drawn at depths 0-2, so that the depth-2 transitions (that -> that) are SEEN and depth 2 is an unseen depth for the OPERATOR forms only, not an unseen transition")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v20_operators (B5.1): device={a.device}, D={D}, N={N}, seeds={a.seeds}; operators as roles around their scope (current definition); forms {list(FORMS)}; training depth 0 and 1 (share of depth 1 {a.p_depth1}), test depths 0, 1, 2 (UNSEEN)" + (f"; E1b protocol: affirmatives at depths 0-2 with share {a.p_aff2} (depth 2 is then an unseen depth for the operator forms, not an unseen transition)" if a.p_aff2 > 0 else "; NOTE: no sentence at depth 2 in training — depth 2 is also an UNSEEN TRANSITION (that -> that), rule 4"))
    print(f"chance: fillers 1/12, predicate 1/6, structure 1/6; the pair must_neg / neg_must has the IDENTICAL target by construction (commutative binding)\n")
    w0 = OpWorld(a.n_id, np.random.default_rng(0)); address_check(w0, 256)
    cols = ["A", "B", "C", "pred", "structure", "FULL"]; res = {}; gates_all = []; differ_all = []
    for seed in range(a.seeds):
        w = OpWorld(a.n_id, np.random.default_rng(seed)); trP, trO = w.persons[:12], w.objects[:12]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = OpRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F); assert m.F_np is world.F
        trng = np.random.default_rng(200 + seed); sents, targets = [], []; forms = list(FORMS)
        for n in range(N):
            if trng.random() < a.p_aff2: form = "aff"; d = int(trng.integers(3))                                   # E1b protocol: affirmatives at depths 0-2
            else: form = forms[int(trng.integers(len(forms)))]; d = 1 if trng.random() < a.p_depth1 else 0
            ev = w.oevent(trng, trP, trO, form, depth=d)
            sents.append(w.orender(ev, trng, t=int(trng.integers(2)))); targets.append(world.state(ev))
        t0 = time.time(); cats, ids, mask = m.encode(sents); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
        trainer.train_loop(m, [m.theta, m.logits], cats, ids, mask, tgt, a.batch, a.lr, log_every=a.log_every, ckpt=f"logs/.ckpt_v20_{a.tag}_{seed}.pt", tag=f" seed {seed}")
        if a.device == "cuda": torch.cuda.synchronize()
        t_train = time.time() - t0; gt = m.gates(); gates_all.append(gt)
        print(f"  seed {seed} ({t_train:.0f}s) gates OPEN: {sorted(c for c, v in gt.items() if v > 0.5)}")
        for depth in (0, 1, 2):
            erng = np.random.default_rng(900 + depth)
            for form in forms:
                acc = {c: [] for c in cols}
                for k in range(40):
                    ev = w.oevent(erng, trP, trO, form, depth=depth); toks = w.orender(ev, erng, t=k % 2); c, i_, kk = m.encode([toks])
                    with torch.no_grad(): h, out = decode(m.states(c, i_, kk)[0], m, world, w, depth, trP, trO)
                    g = GOLD_H[form]; fill = out[g][1]; truth = {"A": ev["A"], "B": ev["B"], "C": ev["C"], "pred": BASE[ev["kind"]]}
                    for r in ("A", "B", "C", "pred"): acc[r].append(fill[r] == truth[r])
                    acc["structure"].append(h == g); acc["FULL"].append(h == g and all(out[h][1][r] == truth[r] for r in truth))
                for c_ in cols: res.setdefault((depth, form, c_), []).append(float(np.mean(acc[c_])))
            # the colliding pair: same event rendered as must_neg and as neg_must — do the decoded frames differ?
            prng = np.random.default_rng(700 + depth); dif = []
            for k in range(60):
                ev = w.oevent(prng, trP, trO, "must_neg", depth=depth); ev2 = dict(ev, form="neg_must"); fr = []
                for e in (ev, ev2):
                    c, i_, kk = m.encode([w.orender(e, prng, t=k % 2)])
                    with torch.no_grad(): h, out = decode(m.states(c, i_, kk)[0], m, world, w, depth, trP, trO)
                    fr.append((h, tuple(sorted(out[h][1].items()))))
                dif.append(fr[0] != fr[1])
            differ_all.append((seed, depth, float(np.mean(dif))))
    print(f"\n== MEANS over {a.seeds} seeds, per depth and form (roles read under the GOLD structure; structure = decoded hypothesis; FULL = structure + roles under it)")
    print("   depth | form | " + " | ".join(cols))
    for depth in (0, 1, 2):
        for form in FORMS:
            print(f"   {depth} {'(UNSEEN)' if depth == 2 else '        '} | {form:9s} | " + " | ".join(f"{np.mean(res[(depth, form, c)]):.3f}±{np.std(res[(depth, form, c)]):.3f}" for c in cols))
    print("== the colliding pair must_neg / neg_must (same event): share of pairs whose decoded frames DIFFER, per depth: " + "; ".join(f"d{d} {np.mean([x[2] for x in differ_all if x[1] == d]):.3f}" for d in (0, 1, 2)))
    print("== subject A by number of operators after it (mean over forms and seeds): " + "; ".join(f"d{d}: 0 ops {np.mean([np.mean(res[(d, 'aff', 'A')])]):.3f}, 1 op {np.mean([np.mean(res[(d, f, 'A')]) for f in ('must', 'may', 'neg')]):.3f}, 2 ops {np.mean([np.mean(res[(d, f, 'A')]) for f in ('must_neg', 'neg_must', 'may_neg')]):.3f}" for d in (0, 1, 2)))
    cats0 = sorted(gates_all[0]); g = {c: np.mean([gt[c] for gt in gates_all]) for c in cats0}
    print("== learned gates (mean sigmoid; OPEN > 0.5): " + " ".join(f"{c}={g[c]:.2f}" for c in sorted(cats0, key=lambda c: -g[c])[:16]))
    print("\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] rule 28 address check printed before training, the unreachable pair named (declared measurement) [x] roles under the gold structure and FULL under the decoded one (17) [x] per form and per depth, unseen depth separate [x] colliding pair as a frame-difference axis [x] gates learned, stated (20) [x] params saved [x] memory logged (21) [x] >= 3 seeds")
