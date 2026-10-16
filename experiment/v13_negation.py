"""
v13_negation — E1: NEGATION AS A ROLE in the world (requirement R3), model unchanged (v8 bundle: path + bigram + soft chainL, gates learned).

World (compositional, reference): S_world(P) = R_A f_A + R_B f_B + R_C f_C; reporting R_X f_X + R_rep (x) S_world(P) as in v7.
Negation = a fixed role vector R_neg bound to its SCOPE:
    sentential  (scope S):  S_world(not P)      = R_neg (x) S_world(P)                     "X does not give B to C", "X never gives B to C"
    constituent (scope A):  A-slot = R_A (x) R_neg (x) f_A                                 "it is not X who gives B to C"
    neg. quantifier (A):    A-slot = R_A (x) R_neg (x) f_someone                           "nobody gives B to C"      (f_someone: identity of the indefinite)
    constituent (scope C):  C-slot = R_C (x) R_neg (x) f_C / f_someone                     "X gives B not to C", "X gives B to nobody"
    constituent (scope B):  B-slot = R_B (x) R_neg (x) f_something                         "X gives nothing to C"
Every operator word (does, not, never, nobody, nothing, it, who) is an ordinary category; no rule in the model. If the gates open 'not'
like 'that' (a fibre), that is a finding.
Decoding (rule 17: scope separately from presence): hypotheses H = {none, S, A, B, C}; for each, unbind every role with the keys the
hypothesis implies and take the mean best cosine over A, B, C; the argmax hypothesis gives presence and scope; the inner roles are read
under it. Reported: role accuracy (A, B, C) per form | negation PRESENCE accuracy | SCOPE accuracy | FULL frame (roles + scope + depth) |
pair separation cos(S_affirmed, S_negated) for the same event (world: ~0 by construction; criterion for the model: gap > 0.5) with the
bag-of-identities control (sees 'not' as any word) on the same row | UNSEEN combination: negation inside reported speech (depth 1-2), when
training has negation only at depth 0.
Criteria (judgment): negated frame decoded > 0.95; separation gap > 0.5; scope > 0.9.
Postposed/right-announced negation (e.g. "X gives B to C , or not") is NOT in this world: it falls under the declared limit (RESULTS 6c).
Run: python3 v13_negation.py [--seeds 5] [--N 40000] [--D 1024] [--batch 8]
"""
import argparse, time, copy
import numpy as np, torch
import v2, v5_scale as v5, v7_nested as v7, v8_gates as v8

BASE = {"GIVE": "give", "SELL": "sell", "SEND": "send", "LEND": "lend", "SHOW": "show", "OFFER": "offer"}
# negated formulation patterns per scope; V = base form (after 'does not' / 'never'), V3 = 3rd person (gives), VP participle
NEG_PATTERNS = {
    "S": ["{A} does not V {B} to {C} .", "{A} never V3 {B} to {C} ."],
    "A": ["it is not {A} who V3 {B} to {C} .", "nobody V3 {B} to {C} ."],
    "C": ["{A} V3 {B} not to {C} .", "{A} V3 {B} to nobody ."],
    "B": ["{A} V3 nothing to {C} .", "{A} does not V anything to {C} ."],     # 'anything' = the indefinite under negation (scope B)
}
INDEF = {"A": "someone", "B": "something", "C": "someone"}

class NegWorld(v5.BigWorld):
    def __init__(self, n_id, rng, depth_max=3):
        super().__init__(n_id, rng, depth_max)
        self.neg_frames = {k: {sc: [p.replace("V3", v5.VERBS[k][0]).replace("VP", v5.VERBS[k][1]).replace("V", BASE[k]) for p in ps] for sc, ps in NEG_PATTERNS.items()} for k in self.frames}
        words = {x for fs in self.neg_frames.values() for ps in fs.values() for t in ps for x in t.split() if not x.startswith("{")} | set(INDEF.values())
        for x in words:
            if x not in self.idx: self.vocab.append(x)
        self.vocab = sorted(set(self.vocab)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        for x in words: self.cat.setdefault(x, x)
        for x in INDEF.values(): self.cat[x] = "P" if x == "someone" else "O"      # indefinites behave as identities of their class
        for v in v5.COP: self.cat[v] = "COP"
    def event(self, persons=None, objects=None, depth=0, neg=None):
        ev = super().event(persons, objects, depth); return ev + (neg,)
    def render(self, ev, t, outer=None):
        neg = ev[-1]; ev0 = ev[:-1]
        if neg is None: return super().render(ev0, t, outer)
        kind, A, B, C = ev0[:4]; d = self.depth(ev0); outer = outer if outer is not None else [self.rng.integers(len(v5.OUTER)) for _ in range(d)]
        s = self.neg_frames[kind][neg][t % 2].format(A=A, B=B, C=C).rsplit(" .", 1)[0]
        for lvl in range(d - 1, -1, -1): s = v5.OUTER[outer[lvl]].replace("<P>", s).format(X=ev0[4 + lvl])
        return (s + " .").split()
    def depth(self, ev): return sum(x is not None for x in ev[4:7])

class NegRoleWorld(v7.NestedRoleWorld):
    def __init__(self, F, world, rng):
        super().__init__(F, world, rng); self.R["neg"] = np.exp(1j * rng.uniform(-np.pi, np.pi, F.shape[1]))
    def state(self, ev, toks=None):
        neg = ev[-1]; kind, A, B, C = ev[:4]; speakers = [x for x in ev[4:7] if x is not None]
        toks = toks or []
        fA, fB, fC = self.F[self.w.idx[A]], self.F[self.w.idx[B]], self.F[self.w.idx[C]]
        if neg == "A" and "nobody" in toks: fA = self.F[self.w.idx["someone"]]
        if neg == "C" and "nobody" in toks: fC = self.F[self.w.idx["someone"]]
        if neg == "B": fB = self.F[self.w.idx["something"]]
        if neg == "A": fA = self.R["neg"] * fA
        if neg == "B": fB = self.R["neg"] * fB
        if neg == "C": fC = self.R["neg"] * fC
        S = self.R["A"] * fA + self.R["B"] * fB + self.R["C"] * fC
        if neg == "S": S = self.R["neg"] * S
        for x in reversed(speakers): S = self.R["X"] * self.F[self.w.idx[x]] + self.R["rep"] * S
        return S

def decode_neg(S, m, world, w, persons, objects, depth):
    """hypotheses {none, S, A, B, C}: unbind each role with the implied keys at the given depth; argmax mean best cosine"""
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum(-1).sqrt()
    P = persons + ["someone"]; O = objects + ["something"]
    cid = {"P": torch.tensor([w.idx[x] for x in P], device=m.device), "O": torch.tensor([w.idx[x] for x in O], device=m.device)}
    rep = torch.conj(R["rep"]) ** depth; best = {}
    for h in ("none", "S", "A", "B", "C"):
        tot, fill = 0.0, {}
        for r in ("A", "B", "C"):
            key = torch.conj(R[r]) * rep
            if h == "S" or h == r: key = key * torch.conj(R["neg"])
            kind = "O" if r == "B" else "P"; sc = torch.real((Sn * key) @ torch.conj(Fn[cid[kind]]).T); j = int(sc.argmax())
            tot += float(sc[j]); fill[r] = (P if kind == "P" else O)[j]
        best[h] = (tot / 3, fill)
    h = max(best, key=lambda k: best[k][0]); return h, best[h][1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--p_neg", type=float, default=0.4); ap.add_argument("--neg_depth_train", type=int, default=0, help="E1b: negation seen in training up to this depth (0 = depth 0 only)")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v13_negation (E1): device={a.device}, D={D}, N={N}, batch={a.batch}, seeds={a.seeds}; negation as a role R_neg bound to its scope; model = v8 bundle, learned gates (start closed)")
    print(f"training: affirmative at depths 0-2 (formulations {v5.TRAIN_T}) + negated forms (scopes S, A, B, C; both patterns each) at depths 0-{a.neg_depth_train}, p_neg={a.p_neg}")
    print("test: negation at depth 0 (seen forms; fresh events), affirmative depths 0-3, and the UNSEEN combination negation inside reported speech (depth 1, 2)")
    for sc, ps in NEG_PATTERNS.items(): print(f"  scope {sc}: {ps}")
    print(f"chance: fillers 1/13 (12 + the indefinite) = {1/13:.3f}; scope among 5 hypotheses = 0.200\n")
    cols = ["aff d0: roles", "neg d0: roles", "neg d0: presence", "neg d0: SCOPE", "neg d0: FULL frame", "aff d3 UNSEEN: roles", "neg d1 UNSEEN: scope", "neg d1 UNSEEN: FULL", "neg d2 UNSEEN: FULL"]
    res = []; seps = []; gates_all = []
    for seed in range(a.seeds):
        w = NegWorld(a.n_id, np.random.default_rng(seed)); trP, trO = w.persons[:12], w.objects[:12]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = NegRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F); assert m.F_np is world.F
        # training data: own loop (world.state needs the tokens for the indefinite forms)
        rng = m.rng; sents, targets = [], []
        for n in range(N):
            if rng.random() < a.p_neg:
                sc = ["S", "A", "B", "C"][int(rng.integers(4))]; ev = w.event(trP, trO, depth=int(rng.integers(a.neg_depth_train + 1)), neg=sc); toks = w.render(ev, int(rng.integers(2)))
            else:
                d = int(rng.integers(3)); ev = w.event(trP, trO, depth=d); toks = w.render(ev, int(rng.choice(v5.TRAIN_T)))
            sents.append(toks); targets.append(world.state(ev, toks))
        t0 = time.time(); cats, ids, mask = m.encode(sents); tgt_all = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
        opt = torch.optim.Adam([m.theta, m.logits], lr=a.lr)
        for i in range(0, N, a.batch):
            sl = slice(i, i + a.batch); S = m.states(cats[sl], ids[sl], mask[sl]); tgt = tgt_all[sl]
            cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
            assert torch.isfinite(cos).all(), f"non-finite loss at step {i}: a target or state is NaN (check the world)"
            opt.zero_grad(); (-cos.mean()).backward(); opt.step()
            with torch.no_grad(): m.theta[0] = 0
        torch.cuda.synchronize(); t_train = time.time() - t0
        per_form = {}
        def run(depth, neg, n=60):
            roles, pres, scope, full = [], [], [], []
            scs = ["S", "A", "B", "C"] if neg else [None]
            for sc in scs:
                for t in range(2 if neg else len(v5.TRAIN_T)):
                    evs = [w.event(trP, trO, depth=depth, neg=sc) for _ in range(n)]; toks = [w.render(ev, t if neg else v5.TRAIN_T[t]) for ev in evs]
                    cats, ids, mask = m.encode(toks)
                    with torch.no_grad(): S = m.states(cats, ids, mask)
                    for k, ev in enumerate(evs):
                        h, fill = decode_neg(S[k], m, world, w, trP, trO, depth)
                        A, B, C = ev[1], ev[2], ev[3]
                        if sc == "A" and "nobody" in toks[k]: A = "someone"
                        if sc == "C" and "nobody" in toks[k]: C = "someone"
                        if sc == "B": B = "something"
                        ok_roles = [fill["A"] == A, fill["B"] == B, fill["C"] == C]; roles.extend(ok_roles)
                        pres.append((h != "none") == (sc is not None)); scope.append(h == (sc or "none")); full.append(all(ok_roles) and h == (sc or "none"))
                        if neg: per_form.setdefault((depth, sc, t), []).append((h == sc, h))
            return float(np.mean(roles)), float(np.mean(pres)), float(np.mean(scope)), float(np.mean(full))
        aff0 = run(0, False); neg0 = run(0, True); aff3 = run(3, False, n=30); neg1 = run(1, True, n=40); neg2 = run(2, True, n=30)
        if seed == 0:
            for (d, sc, t), v in sorted(per_form.items()):
                hs = dict(zip(*[list(x) for x in np.unique([h for _, h in v], return_counts=True)]))
                print(f"     depth {d} scope {sc} form {t} '{NEG_PATTERNS[sc][t]}': scope acc {np.mean([o for o, _ in v]):.2f}  decoded as {hs}")
        # pair separation: same event, affirmative vs negated (scope S), model vs bag of identities
        evs = [w.event(trP, trO, depth=0) for _ in range(80)]; cs_m, cs_bag, cs_w = [], [], []
        for ev in evs:
            ta = w.render(ev, 0); tn = w.render(ev + ("S",), 0)
            c1, i1, k1 = m.encode([ta]); c2, i2, k2 = m.encode([tn])
            with torch.no_grad(): Sa = m.states(c1, i1, k1)[0]; Sn_ = m.states(c2, i2, k2)[0]
            cosf = lambda x, y: float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt()))
            cs_m.append(cosf(Sa, Sn_)); cs_bag.append(cosf(m.F[i1[0]].sum(0), m.F[i2[0]].sum(0)))
            cs_w.append(float(abs(np.vdot(world.state(ev, ta), world.state(ev + ("S",), tn))) / (np.linalg.norm(world.state(ev, ta)) * np.linalg.norm(world.state(ev + ("S",), tn)))))
        seps.append((np.mean(cs_m), np.mean(cs_bag), np.mean(cs_w)))
        r = [aff0[0], neg0[0], neg0[1], neg0[2], neg0[3], aff3[0], neg1[2], neg1[3], neg2[3]]; res.append(r)
        gt = m.gates(); gates_all.append(gt); op = sorted([c for c, v in gt.items() if v > 0.5])
        print(f"  seed {seed} ({t_train:.0f}s) " + " | ".join(f"{x:.3f}" for x in r) + f"  || cos(aff, neg): model {np.mean(cs_m):.2f}, bag {np.mean(cs_bag):.2f}, world {np.mean(cs_w):.2f}  || gates OPEN: {op}", flush=True)
    r = np.array(res); s = np.array(seps)
    print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols))
    print("  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(len(cols))))
    print(f"== pair separation cos(affirmed, negated), same event: model {s[:,0].mean():.3f}±{s[:,0].std():.3f} | bag of identities (control) {s[:,1].mean():.3f} | world {s[:,2].mean():.3f}   (criterion: gap 1 - cos > 0.5 for the model)")
    cats0 = sorted(gates_all[0]); g = {c: np.mean([gt[c] for gt in gates_all]) for c in cats0}
    print("== learned gates (mean sigmoid; OPEN > 0.5): " + " ".join(f"{c}={g[c]:.2f}" for c in sorted(cats0, key=lambda c: -g[c])[:16]))
    print(f"\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] scope separately from presence (17) [x] FULL frame (R1) [x] pair separation with lexical control [x] unseen combination (negation inside reporting) [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] model unchanged, operator = role")
