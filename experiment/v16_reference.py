"""
v16_reference — round 10 B1 + B2: the reference table re-run with the PREDICATE ROLE and the OCCUPANCY MARKER, and CALIBRATION in evaluate().

B1 world (compositional, English, prefixed reporting; v7 + two additions, rule 20: roles/identities/categories given, moves/addresses/gates learned):
    S_world(P) = R_pred (x) f_verb + sum_r R_r (x) (f_filler + m)          r in {A, B, C};  m = a fixed random OCCUPANCY MARKER vector
    reporting: R_X (x) (f_X + m) + R_rep (x) S_world(P)
  Predicate filler, option (a) (decided before this run; see JUDGMENT Part IV B1): the SURFACE verb's lemma identity (gives -> give,
  receives -> receive, is given -> give ...), so the give/receive paraphrase pair has DIFFERENT predicate fillers by construction; paraphrase
  pairs are judged equal on the non-predicate roles; the predicate is reported as "surface verb" and, separately, as "event type" through a
  per-world SYNONYM TABLE applied at readout (give, receive -> transfer; sell, buy -> sale; ...). Option (b) (event-type identity in the
  target = a learned verb-to-event map) is NOT run: it would be a change of principle (learned identities).
  Occupancy (external review A1 / X2): presence of a role = cosine with m after unbinding; identity = cleanup over candidates. Three cases
  per role: given with a KNOWN filler / given with an UNKNOWN filler (a new identity never in training) / NOT given (sentence forms that
  omit the recipient: "{A} gives {B} ." etc.). Criterion: the three cases separated > 0.95 each; no loss on the reference table; give vs
  sell vs lend decoded as distinct predicates > 0.95.
B2 calibration (rule 18, X4), in evaluate(): for every readout the margin (best - second best cosine); confident-wrong rate at a stated
  threshold tau (margin > tau and wrong); abstention rate (margin < tau) split into would-have-been-right / wrong; accuracy-vs-coverage
  curve at thresholds {0, 0.02, 0.05, 0.1, 0.15}; reconstruction residual (frame.residual-like, with the predicate and the marker).
  Criterion for the assistant use case: confident-wrong < 2% at coverage > 80%, reported, not tuned.
Model: v8 bundle (path + chainL + bigram), gates learned (start closed); trainer.train_loop (memory logged, checkpoints). 5 seeds.
Run: python3 v16_reference.py [--seeds 5] [--N 30000] [--D 1024] [--tau 0.05]
"""
import argparse, time
import numpy as np, torch
import guard, trainer, v2, v5_scale as v5, v7_nested as v7, v8_gates as v8

LEMMA = {"gives": "give", "given": "give", "receives": "receive", "sells": "sell", "sold": "sell", "buys": "buy", "sends": "send", "sent": "send",
         "gets": "get", "lends": "lend", "lent": "lend", "borrows": "borrow", "shows": "show", "shown": "show", "sees": "see", "offers": "offer", "offered": "offer", "accepts": "accept"}
EVENT = {"give": "transfer", "receive": "transfer", "sell": "sale", "buy": "sale", "send": "shipment", "get": "shipment", "lend": "loan", "borrow": "loan",
         "show": "display", "see": "display", "offer": "proposal", "accept": "proposal"}
OMIT_C = ["{A} V3 {B} .", "{B} is VP by {A} .", "it is {A} who V3 {B} ."]        # forms WITHOUT a recipient: the "not given" occupancy case

class RefWorld(v5.BigWorld):
    def __init__(self, n_id, rng):
        super().__init__(n_id, rng)
        self.omit = {k: [p.replace("V3", v5.VERBS[k][0]).replace("VP", v5.VERBS[k][1]) for p in OMIT_C] for k in self.frames}
        lemmas = sorted(set(LEMMA.values())); events = sorted(set(EVENT.values()))
        for x in lemmas + events + ["<m>"]:
            if x not in self.idx: self.vocab.append(x)
        self.vocab = sorted(set(self.vocab)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        for x in lemmas + events + ["<m>"]: self.cat.setdefault(x, "SYM")      # target-side symbols: never in a sentence; one category so the model's tables stay square
        for v in v5.COP: self.cat[v] = "COP"
    def surface_verb(self, toks):
        for t in toks:
            if t in LEMMA: return LEMMA[t]
        return None
    def render_omit(self, ev, t, outer=None):
        kind, A, B, C = ev[:4]; d = self.depth(ev); outer = outer if outer is not None else [self.rng.integers(len(v5.OUTER)) for _ in range(d)]
        s = self.omit[kind][t].format(A=A, B=B).rsplit(" .", 1)[0]
        for lvl in range(d - 1, -1, -1): s = v5.OUTER[outer[lvl]].replace("<P>", s).format(X=ev[4 + lvl])
        return (s + " .").split()

class RefRoleWorld(v7.NestedRoleWorld):
    def __init__(self, F, world, rng):
        super().__init__(F, world, rng); D = F.shape[1]
        self.R["pred"] = np.exp(1j * rng.uniform(-np.pi, np.pi, D)); self.m = self.F[world.idx["<m>"]]     # the marker is a fixed identity-like vector
    def state(self, ev, toks, has_C=True):
        w = self.w; kind, A, B, C = ev[:4]; speakers = [x for x in ev[4:7] if x is not None]
        verb = w.surface_verb(toks) or v5.VERBS[kind][0]; verb = LEMMA.get(verb, verb)
        S = self.R["pred"] * self.F[w.idx[verb]] + self.R["A"] * (self.F[w.idx[A]] + self.m) + self.R["B"] * (self.F[w.idx[B]] + self.m)
        if has_C: S = S + self.R["C"] * (self.F[w.idx[C]] + self.m)
        for x in reversed(speakers): S = self.R["X"] * (self.F[w.idx[x]] + self.m) + self.R["rep"] * S
        return S

def readout(S, m, world, w, depth, persons, objects, verbs):
    """per role: presence (cos with m after unbinding), identity + margin; predicate (surface lemma) + margin; event type via the synonym table"""
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}; mk = torch.tensor(world.m, dtype=m.F.dtype, device=m.device)
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum(-1).sqrt(); rep = torch.conj(R["rep"]) ** depth; out = {}
    def best(key, cands):
        cid = torch.tensor([w.idx[x] for x in cands], device=m.device); sc = torch.real((Sn * key) @ torch.conj(Fn[cid]).T); top = torch.topk(sc, 2)
        return cands[int(top.indices[0])], float(top.values[0] - top.values[1]), float(top.values[0])
    for r, cands in (("A", persons), ("B", objects), ("C", persons)):
        key = torch.conj(R[r]) * rep; pres = float(torch.real((Sn * key * torch.conj(mk)).sum()) / mk.abs().pow(2).sum().sqrt())
        ident, margin, top = best(key, cands); out[r] = {"present": pres, "id": ident, "margin": margin, "top": top}
    ident, margin, top = best(torch.conj(R["pred"]) * rep, verbs); out["pred"] = {"id": ident, "margin": margin, "event": EVENT.get(ident, "?")}
    return out

class Calib:
    """collects (correct, margin) pairs per axis; reports accuracy, confident-wrong at tau, abstention split, accuracy-vs-coverage"""
    def __init__(self): self.rows = {}
    def add(self, axis, correct, margin): self.rows.setdefault(axis, []).append((bool(correct), float(margin)))
    def report(self, axis, tau):
        v = np.array(self.rows[axis]); ok, mg = v[:, 0].astype(bool), v[:, 1]; n = len(v)
        conf = mg > tau; cw = float((conf & ~ok).sum() / n); cov = float(conf.mean()); abst_r = float((~conf & ok).sum() / n); abst_w = float((~conf & ~ok).sum() / n)
        curve = " ".join(f"{t:.2f}:{(ok[mg > t].mean() if (mg > t).any() else float('nan')):.3f}@{(mg > t).mean():.2f}" for t in (0, 0.02, 0.05, 0.1, 0.15))
        return f"acc {ok.mean():.3f} | margin mean {mg.mean():.3f} p10 {np.percentile(mg, 10):.3f} | tau {tau}: coverage {cov:.2f}, confident-wrong {cw:.3f}, abstain right/wrong {abst_r:.3f}/{abst_w:.3f} | acc@coverage: {curve}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda"); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--p_omit", type=float, default=0.2); ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--ckpt", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="ref")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v16_reference (B1 + B2): device={a.device}, D={D}, N={N}, seeds={a.seeds}, tau={a.tau}; predicate role (surface lemma, option a) + occupancy marker m; calibration on every readout")
    print(f"world: R_pred x f_lemma + sum_r R_r x (f + m); reporting R_X x (f_X + m) + R_rep x S; recipient-less forms (share {a.p_omit}) {OMIT_C}; synonym table {EVENT}")
    w0 = RefWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); cues_ok, _ = v5.split_templates_rolecues(w0)
    verbs = sorted(set(LEMMA.values())); print(f"predicate candidates ({len(verbs)}): {verbs}; chance 1/{len(verbs)}; roles: chance 1/12 (new ids 1/6)\n")
    cols = ["seen FULL(roles)", "pred (surface)", "pred (event)", "new ids FULL", "role cues FULL", "unseen trans. FULL", "depth 2 FULL", "depth 3 UNSEEN FULL", "occupancy: known", "unknown filler", "not given"]
    res = []; gates_all = []
    for seed in range(a.seeds):
        w = RefWorld(a.n_id, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = RefRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F); assert m.F_np is world.F
        trng = np.random.default_rng(200 + seed); sents, targets = [], []
        for n in range(N):
            d = int(trng.integers(3)); ev = w.event(trP, trO, depth=d)
            if trng.random() < a.p_omit: toks = w.render_omit(ev, int(trng.integers(len(OMIT_C)))); targets.append(world.state(ev, toks, has_C=False))
            else: toks = w.render(ev, int(trng.choice(v5.TRAIN_T))); targets.append(world.state(ev, toks))
            sents.append(toks)
        t0 = time.time(); cats, ids, mask = m.encode(sents); tgt_all = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
        trainer.train_loop(m, [m.theta, m.logits], cats, ids, mask, tgt_all, a.batch, a.lr, log_every=a.log_every, ckpt=(f"logs/.ckpt_{a.tag}_{seed}.pt" if a.ckpt else None), resume=a.resume, tag=f" seed {seed}")
        if a.device == "cuda": torch.cuda.synchronize()
        t_train = time.time() - t0; cal = Calib()
        def run(templs, P, O, depth, n=30, axis="x"):
            erng = np.random.default_rng(900 + depth + 7 * len(templs)); full, ps, pe = [], [], []
            for t in templs:
                evs = [w.event(P, O, depth=depth) for _ in range(n)]; toks = [w.render(ev, t) for ev in evs]; c, i_, k = m.encode(toks)
                with torch.no_grad(): S = m.states(c, i_, k)
                for j, ev in enumerate(evs):
                    o = readout(S[j], m, world, w, depth, P, O, verbs); lem = w.surface_verb(toks[j])   # already a lemma
                    oks = [o["A"]["id"] == ev[1], o["B"]["id"] == ev[2], o["C"]["id"] == ev[3]]
                    for r, ok in zip(("A", "B", "C"), oks): cal.add(axis + ":" + r, ok, o[r]["margin"])
                    cal.add(axis + ":pred", o["pred"]["id"] == lem, o["pred"]["margin"])
                    full.append(all(oks)); ps.append(o["pred"]["id"] == lem); pe.append(o["pred"]["event"] == EVENT[lem])
            return float(np.mean(full)), float(np.mean(ps)), float(np.mean(pe))
        seen = run(v5.TRAIN_T, trP, trO, 0, axis="seen"); newid = run(v5.TRAIN_T, teP, teO, 0, n=20, axis="newid"); cues = run(cues_ok, trP, trO, 0, axis="cues"); unseen = run(held_new, trP, trO, 0, n=20, axis="unseen")
        d2 = run(v5.TRAIN_T, trP, trO, 2, n=15, axis="d2"); d3 = run(v5.TRAIN_T, trP, trO, 3, n=15, axis="d3")
        # occupancy: the C role in three cases
        erng = np.random.default_rng(555); pres = {"known": [], "unknown": [], "absent": []}
        for k in range(120):
            ev = w.event(trP, trO, depth=0); evu = w.event(trP, trO, depth=0); evu = (evu[0], evu[1], evu[2], teP[k % len(teP)]) + evu[4:]
            for case, e, toks in (("known", ev, w.render(ev, 0)), ("unknown", evu, w.render(evu, 0)), ("absent", ev, w.render_omit(ev, k % len(OMIT_C)))):
                c, i_, kk = m.encode([toks])
                with torch.no_grad(): o = readout(m.states(c, i_, kk)[0], m, world, w, 0, trP + teP, trO, verbs)
                pres[case].append(o["C"]["present"])
        thr = 0.5 * (np.mean(pres["known"]) + np.mean(pres["absent"]))        # presence threshold: midpoint, stated
        occ = (float(np.mean(np.array(pres["known"]) > thr)), float(np.mean(np.array(pres["unknown"]) > thr)), float(np.mean(np.array(pres["absent"]) < thr)))
        r = [seen[0], seen[1], seen[2], newid[0], cues[0], unseen[0], d2[0], d3[0], occ[0], occ[1], occ[2]]; res.append(r); gt = m.gates(); gates_all.append(gt)
        print(f"  seed {seed} ({t_train:.0f}s) " + " | ".join(f"{x:.3f}" for x in r) + f"  || presence cos: known {np.mean(pres['known']):.2f} unknown {np.mean(pres['unknown']):.2f} absent {np.mean(pres['absent']):.2f} (thr {thr:.2f}) || gates OPEN: {sorted(c for c, v in gt.items() if v > 0.5)}", flush=True)
        if seed == 0:
            print("  calibration (seed 0, tau = %.2f):" % a.tau)
            for ax in ("seen:A", "seen:B", "seen:C", "seen:pred", "newid:A", "d3:A", "d3:C", "unseen:A", "unseen:C"): print(f"     {ax:10s} " + cal.report(ax, a.tau))
    r = np.array(res); print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(len(cols))))
    cats0 = sorted(gates_all[0]); g = {c: np.mean([gt[c] for gt in gates_all]) for c in cats0}
    print("== learned gates (mean sigmoid; OPEN > 0.5): " + " ".join(f"{c}={g[c]:.2f}" for c in sorted(cats0, key=lambda c: -g[c])[:14]))
    print(f"\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] predicate role present (23) [x] occupancy by marker, not norm (A1) [x] calibration: margin, confident-wrong, abstention, acc-vs-coverage (18) [x] memory logged (21) [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] given vs learned stated (20)")
