"""
v15_modality — E3: MODALITY and CONDITIONALS as roles (R3), scope separately (rule 17), composition rule under test (reserve of RESULTS 9b-bis).
Model unchanged (v8 bundle, learned gates, start closed).

World (compositional, reference):
    modality:    S_world(may P)  = R_may (x) S_world(P)       forms: "{A} may V {B} to {C}", "{A} must V {B} to {C}", "perhaps {A} V3 {B} to {C}"
                 S_world(must P) = R_must (x) S_world(P)      (two modal roles; 'perhaps' = may)
    conditional: S_world(if P then Q) = R_if (x) S_world(P) + R_then (x) S_world(Q)      forms: "if P , then Q", "if P , Q", "Q if P"
                 NOTE: "Q if P" announces the condition AFTER the consequent: Q is already written when 'if' arrives. This form is RIGHT-
                 announced structure (RESULTS 6c): included on purpose and reported separately, as a limit, not a failure.
    reporting:   R_X f_X + R_rep (x) S  as before.
Training: modal sentences and conditionals at depths 0-1 (inside reporting), affirmative at 0-2; the MIXED nesting "X says that if P then
Q may ..." (a modal inside a conditional inside a report) is NEVER trained: it is the unseen construction at depth 2.
Decoding: hypotheses over the operator structure {none, may, must, if-then} x depth; the frame is read under the argmax hypothesis;
for conditionals both clauses are decoded (A, B, C of P under R_if, of Q under R_then). SCOPE here = which clause the modal applies to in
"if P then Q may ..." (consequent) — reported separately from presence.
Reported: FULL frame on modals (depth 0 / 1 / 2 unseen) | FULL frame on conditionals per form, 'Q if P' separately | pair separation
cos(P, may P), cos(P, must P), cos(may P, must P) with the lexical bag on the same row | the mixed unseen nesting: presence / scope / FULL.
Criteria: FULL > 0.95 at depth 0; separation gap > 0.5; scope > 0.9; unseen depth 2 > 0.9 (the composition rule; reserve applies).
Run: python3 v15_modality.py [--seeds 5] [--N 40000] [--D 1024]
"""
import argparse, time
import numpy as np, torch
import v2, v5_scale as v5, v7_nested as v7, v8_gates as v8

BASE = {"GIVE": "give", "SELL": "sell", "SEND": "send", "LEND": "lend", "SHOW": "show", "OFFER": "offer"}
MODAL_PATTERNS = {"may": ["{A} may V {B} to {C} .", "perhaps {A} V3 {B} to {C} ."], "must": ["{A} must V {B} to {C} .", "{A} has to V {B} to {C} ."]}
COND_FORMS = ["if <P> , then <Q> .", "if <P> , <Q> .", "<Q> if <P> ."]        # the third is right-announced
AFF = "{A} V3 {B} to {C}"

class MWorld(v5.BigWorld):
    def __init__(self, n_id, rng, depth_max=3):
        super().__init__(n_id, rng, depth_max)
        self.mframes = {k: {md: [p.replace("V3", v5.VERBS[k][0]).replace("V", BASE[k]) for p in ps] for md, ps in MODAL_PATTERNS.items()} for k in self.frames}
        self.aff = {k: AFF.replace("V3", v5.VERBS[k][0]) for k in self.frames}
        words = {x for fs in self.mframes.values() for ps in fs.values() for t in ps for x in t.split() if not x.startswith("{")} | {"if", "then", ","}
        for x in words:
            if x not in self.idx: self.vocab.append(x)
        self.vocab = sorted(set(self.vocab)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        for x in words: self.cat.setdefault(x, x)
        for v in v5.COP: self.cat[v] = "COP"
    def clause(self, rng, persons, objects):
        kind = rng.choice(list(self.frames)); A, C = rng.choice(persons, 2, replace=False); B = rng.choice(objects); return (kind, A, B, C)
    def mevent(self, rng, persons, objects, depth=0, op=None, cond_form=None, inner_modal=None):
        """op in {None, 'may', 'must', 'if'}; for 'if': two clauses (P, Q), cond_form index; inner_modal applies to Q (the mixed construction)"""
        speakers = list(rng.choice(persons, depth, replace=False)) if depth else []
        P = self.clause(rng, persons, objects); Q = self.clause(rng, persons, objects) if op == "if" else None
        return {"op": op, "P": P, "Q": Q, "cond_form": cond_form, "inner_modal": inner_modal, "speakers": speakers}
    def mrender(self, ev, rng, t=0):
        def aff(cl): return self.aff[cl[0]].format(A=cl[1], B=cl[2], C=cl[3])
        def modal(cl, md, j): return self.mframes[cl[0]][md][j].format(A=cl[1], B=cl[2], C=cl[3]).rsplit(" .", 1)[0]
        if ev["op"] is None: s = aff(ev["P"])
        elif ev["op"] in ("may", "must"): s = modal(ev["P"], ev["op"], t % 2)
        else:
            q = modal(ev["Q"], ev["inner_modal"], 0) if ev["inner_modal"] else aff(ev["Q"])
            s = COND_FORMS[ev["cond_form"]].replace("<P>", aff(ev["P"])).replace("<Q>", q).rsplit(" .", 1)[0]
        for x in reversed(ev["speakers"]): s = v5.OUTER[int(rng.integers(len(v5.OUTER)))].replace("<P>", s).format(X=x)
        return (s + " .").split()

class MRoleWorld(v7.NestedRoleWorld):
    def __init__(self, F, world, rng):
        super().__init__(F, world, rng); D = F.shape[1]
        for r in ("may", "must", "if", "then"): self.R[r] = np.exp(1j * rng.uniform(-np.pi, np.pi, D))
    def clause_state(self, cl):
        kind, A, B, C = cl; w = self.w; return self.R["A"] * self.F[w.idx[A]] + self.R["B"] * self.F[w.idx[B]] + self.R["C"] * self.F[w.idx[C]]
    def state(self, ev):
        w = self.w
        if ev["op"] is None: S = self.clause_state(ev["P"])
        elif ev["op"] in ("may", "must"): S = self.R[ev["op"]] * self.clause_state(ev["P"])
        else:
            Q = self.clause_state(ev["Q"]); Q = self.R[ev["inner_modal"]] * Q if ev["inner_modal"] else Q
            S = self.R["if"] * self.clause_state(ev["P"]) + self.R["then"] * Q
        for x in reversed(ev["speakers"]): S = self.R["X"] * self.F[w.idx[x]] + self.R["rep"] * S
        return S

def decode_m(S, m, world, w, persons, objects, depth):
    """hypotheses: none | may | must | if-then (consequent plain) | if-then with may | if-then with must; argmax of mean best cosine over the roles read"""
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum(-1).sqrt(); rep = torch.conj(R["rep"]) ** depth
    cid = {"P": torch.tensor([w.idx[x] for x in persons], device=m.device), "O": torch.tensor([w.idx[x] for x in objects], device=m.device)}
    def read(prefix):
        tot, fill = 0.0, {}
        for r in ("A", "B", "C"):
            kind = "O" if r == "B" else "P"; sc = torch.real((Sn * prefix * torch.conj(R[r])) @ torch.conj(Fn[cid[kind]]).T); j = int(sc.argmax()); tot += float(sc[j]); fill[r] = (persons if kind == "P" else objects)[j]
        return tot / 3, fill
    H = {}
    H["none"] = read(rep); H["may"] = read(rep * torch.conj(R["may"])); H["must"] = read(rep * torch.conj(R["must"]))
    for md in (None, "may", "must"):
        p = read(rep * torch.conj(R["if"])); qpre = rep * torch.conj(R["then"]) * (torch.conj(R[md]) if md else 1)
        q = read(qpre); H[("if", md)] = ((p[0] + q[0]) / 2, {"P": p[1], "Q": q[1]})
    h = max(H, key=lambda k: H[k][0]); return h, H[h][1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda"); ap.add_argument("--n_id", type=int, default=18)
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v15_modality (E3): device={a.device}, D={D}, N={N}, seeds={a.seeds}; modality R_may/R_must (x) S, conditional R_if (x) P + R_then (x) Q; model = v8 bundle, learned gates")
    print(f"training: affirmative depths 0-2; modals and conditionals (forms {COND_FORMS}) at depths 0-1; the MIXED 'if P then Q may' and the mixed inside reporting are NEVER trained")
    print("test: modals d0 / d1 / d2 (unseen depth); conditionals per form d0 (form 2 'Q if P' is right-announced: reported separately); mixed d0 (unseen construction) and mixed d1 (unseen construction inside reporting)")
    print(f"chance: fillers 1/12; operator hypothesis 1/6\n")
    cols = ["aff d0 FULL", "modal d0 FULL", "modal d1 FULL", "modal d2 UNSEEN FULL", "cond d0 'if P , then Q' FULL", "cond d0 'if P , Q' FULL", "cond d0 'Q if P' (right-announced) FULL", "cond d1 FULL", "cond d2 UNSEEN FULL", "MIXED d0 (unseen): op+scope", "MIXED d0 FULL", "MIXED d1 FULL"]
    res = []; seps = []; gates_all = []
    for seed in range(a.seeds):
        w = MWorld(a.n_id, np.random.default_rng(seed)); trP, trO = w.persons[:12], w.objects[:12]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = MRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F); assert m.F_np is world.F
        trng = np.random.default_rng(200 + seed); sents, targets = [], []
        for n in range(N):
            u = trng.random()
            if u < 0.4: ev = w.mevent(trng, trP, trO, depth=int(trng.integers(3)))
            elif u < 0.7: ev = w.mevent(trng, trP, trO, depth=int(trng.integers(2)), op=["may", "must"][int(trng.integers(2))])
            else: ev = w.mevent(trng, trP, trO, depth=int(trng.integers(2)), op="if", cond_form=int(trng.integers(3)))
            sents.append(w.mrender(ev, trng, t=int(trng.integers(2)))); targets.append(world.state(ev))
        t0 = time.time(); cats, ids, mask = m.encode(sents); tgt_all = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
        opt = torch.optim.Adam([m.theta, m.logits], lr=a.lr)
        for i in range(0, N, a.batch):
            sl = slice(i, i + a.batch); S = m.states(cats[sl], ids[sl], mask[sl]); tgt = tgt_all[sl]
            cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
            assert torch.isfinite(cos).all(), f"non-finite loss at step {i}: a target or state is NaN (check the world)"
            opt.zero_grad(); (-cos.mean()).backward(); opt.step()
            with torch.no_grad(): m.theta[0] = 0
        torch.cuda.synchronize(); t_train = time.time() - t0
        def run(depth, op, cond_form=None, inner=None, n=60, seed_e=0):
            erng = np.random.default_rng(900 + seed_e); full, opok = [], []
            for k in range(n):
                ev = w.mevent(erng, trP, trO, depth=depth, op=op, cond_form=cond_form, inner_modal=inner); toks = w.mrender(ev, erng, t=k % 2)
                cats, ids, mask = m.encode([toks])
                with torch.no_grad(): S = m.states(cats, ids, mask)[0]
                h, fill = decode_m(S, m, world, w, trP, trO, depth)
                truth = ev["op"] if ev["op"] != "if" else ("if", ev["inner_modal"]); truth = truth or "none"
                ok_op = h == truth; opok.append(ok_op)
                if ev["op"] == "if": ok = ok_op and fill["P"] == {"A": ev["P"][1], "B": ev["P"][2], "C": ev["P"][3]} and fill["Q"] == {"A": ev["Q"][1], "B": ev["Q"][2], "C": ev["Q"][3]}
                else: ok = ok_op and fill == {"A": ev["P"][1], "B": ev["P"][2], "C": ev["P"][3]}
                full.append(ok)
            return float(np.mean(full)), float(np.mean(opok))
        r = [run(0, None)[0], run(0, "may")[0], run(1, "may", seed_e=1)[0], run(2, "may", n=40, seed_e=2)[0],
             run(0, "if", 0)[0], run(0, "if", 1, seed_e=3)[0], run(0, "if", 2, seed_e=4)[0], run(1, "if", 0, n=40, seed_e=5)[0], run(2, "if", 0, n=30, seed_e=6)[0]]
        mx0 = run(0, "if", 0, inner="may", n=40, seed_e=7); mx1 = run(1, "if", 0, inner="may", n=30, seed_e=8); r += [mx0[1], mx0[0], mx1[0]]
        # separation
        erng = np.random.default_rng(77); cm, cb = [], []
        for k in range(60):
            ev = w.mevent(erng, trP, trO, depth=0); ta = w.mrender(ev, erng); evm = dict(ev, op="may"); evu = dict(ev, op="must"); tm = w.mrender(evm, erng, 0); tu = w.mrender(evu, erng, 0)
            st = lambda toks: m.states(*m.encode([toks]))[0]
            with torch.no_grad(): Sa, Sm, Su = st(ta), st(tm), st(tu)
            cosf = lambda x, y: float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt()))
            bag = lambda toks: m.F[m.encode([toks])[1][0]].sum(0)
            cm.append((cosf(Sa, Sm), cosf(Sa, Su), cosf(Sm, Su))); cb.append((cosf(bag(ta), bag(tm)), cosf(bag(ta), bag(tu)), cosf(bag(tm), bag(tu))))
        seps.append((np.mean(cm, 0), np.mean(cb, 0))); res.append(r); gt = m.gates(); gates_all.append(gt)
        print(f"  seed {seed} ({t_train:.0f}s) " + " | ".join(f"{x:.3f}" for x in r) + f"  || cos(P,may P) {cm and np.mean(cm,0)[0]:.2f} cos(P,must P) {np.mean(cm,0)[1]:.2f} cos(may,must) {np.mean(cm,0)[2]:.2f} ; bag {np.mean(cb,0)[0]:.2f}/{np.mean(cb,0)[1]:.2f}/{np.mean(cb,0)[2]:.2f} || gates OPEN: {sorted(c for c, v in gt.items() if v > 0.5)}", flush=True)
    r = np.array(res); print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(len(cols))))
    s = np.array([x[0] for x in seps]); b = np.array([x[1] for x in seps])
    print(f"== separation (model | bag): cos(P, may P) {s[:,0].mean():.3f} | {b[:,0].mean():.3f} ; cos(P, must P) {s[:,1].mean():.3f} | {b[:,1].mean():.3f} ; cos(may P, must P) {s[:,2].mean():.3f} | {b[:,2].mean():.3f}   (criterion: gaps > 0.5 for the model)")
    cats0 = sorted(gates_all[0]); g = {c: np.mean([gt[c] for gt in gates_all]) for c in cats0}
    print("== learned gates (mean sigmoid; OPEN > 0.5): " + " ".join(f"{c}={g[c]:.2f}" for c in sorted(cats0, key=lambda c: -g[c])[:16]))
    print(f"\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] operator vs scope (17) [x] FULL frame (R1) [x] right-announced form reported separately (limit) [x] unseen depth 2 and unseen mixed construction (composition rule, reserve) [x] lexical control [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] model unchanged")
