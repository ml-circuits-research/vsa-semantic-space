"""
v14_quantifiers — E2: QUANTIFIERS AS ROLES on the argument (R3), scope reported separately (rule 17), and SCOPE AMBIGUITY held as a
superposition ("knows when it does not know", judgment 2026-10-15). Model unchanged (v8 bundle, learned gates).

World. Arguments are now (determiner, common noun) phrases. An argument slot:  R_role (x) ( f_noun + R_Q (x) f_det )   for a quantified
determiner (every, each, some, a, a certain); the definite 'the' adds no R_Q term; 'no' is the E1 negative quantifier (R_role (x) R_neg (x) f_noun).
Relative SCOPE: the role R_sc is bound to the identity of the role whose quantifier takes wide scope: S += R_sc (x) f_<role>  (f_A, f_B, f_C
are fixed identities of the roles themselves). Rules of the world:
   - one quantified argument: it takes scope (R_sc (x) f_that_role).
   - two quantified arguments, one marked: 'each' forces WIDE scope on its argument; 'a certain' forces WIDE scope on its argument.
   - two unmarked quantified arguments ('every ... a ...', 'some ... every ...'): AMBIGUOUS — the world holds BOTH hypotheses:
     S += (R_sc (x) f_role1 + R_sc (x) f_role2) / sqrt(2).
Training uses ONLY unambiguous sentences (one quantifier, or two with a marker), at depths 0-1 (reporting). Ambiguous sentences are NEVER
trained on; at test the question is whether the state decodes BOTH scope hypotheses with near-equal strength (composition of the two
quantifiers' learned displacements) — the model would then know that it does not know — and ONE hypothesis on disambiguated sentences.
Decoding: nouns and determiners by unbinding (R_role, R_role R_Q) + cleanup; scope by sc_r = cos(S (x) conj(R_sc), f_role) for the quantified
roles; 'ambiguous' is reported when the top two scope scores are within delta = 0.25 of each other (relative), else the argmax is the choice.
Reported, separately: noun accuracy | determiner accuracy | scope accuracy on UNAMBIGUOUS test sentences (criterion > 0.9) | on AMBIGUOUS
sentences: rate of "both reported" (want high), forced-choice distribution (surface vs inverse), cos to the superposed world state |
FULL frame on unambiguous (nouns + determiners + scope + depth) | the unseen depth 2 (criterion > 0.9, the composition rule under test, with
the reserve of RESULTS 9b-bis) | lexical bag control on the scope question (it cannot represent scope: reported for completeness).
Run: python3 v14_quantifiers.py [--seeds 5] [--N 40000] [--D 1024]
"""
import guard  # process-level memory/time protection (see guard.py; the machine hung on 2026-10-15)
import argparse, time
import numpy as np, torch
import trainer, v2, v5_scale as v5, v7_nested as v7, v8_gates as v8

NOUNS_P = ["child", "teacher", "student", "nurse", "doctor", "farmer", "driver", "guard", "clerk", "judge", "singer", "pilot", "baker", "tailor", "miner", "sailor", "monk", "scout"]
NOUNS_O = ["book", "pen", "key", "coin", "map", "cup", "hat", "ring", "ball", "card", "lamp", "bag", "flag", "rope", "drum", "shell", "stone", "kite"]
DETS_Q = ["every", "each", "some", "a", "a certain"]           # quantified; 'each' and 'a certain' are scope markers (wide)
DET_DEF = "the"
WIDE_MARKERS = {"each", "a certain"}
PATTERNS = ["{DA} {A} V3 {DB} {B} to {DC} {C} .", "{DC} {C} VR {DB} {B} from {DA} {A} .", "{DB} {B} is VP to {DC} {C} by {DA} {A} .",
            "{DB} {B} is VP by {DA} {A} to {DC} {C} .", "to {DC} {C} , {DA} {A} V3 {DB} {B} .", "{DA} {A} V3 {DB} {B} , to {DC} {C} ."]

class QWorld(v5.BigWorld):
    def __init__(self, rng, depth_max=3):
        super().__init__(18, rng, depth_max)          # persons p0.. are kept only as speakers of reported speech
        self.nouns_p, self.nouns_o = NOUNS_P, NOUNS_O
        self.qframes = {k: [p.replace("V3", v5.VERBS[k][0]).replace("VP", v5.VERBS[k][1]).replace("VR", v5.VERBS[k][2]) for p in PATTERNS] for k in self.frames}
        words = {x for ps in self.qframes.values() for t in ps for x in t.split() if not x.startswith("{")} | set(NOUNS_P) | set(NOUNS_O) | {"every", "each", "some", "a", "certain", "the", "no"}
        for x in words:
            if x not in self.idx: self.vocab.append(x)
        self.vocab = sorted(set(self.vocab)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        for x in words: self.cat.setdefault(x, x)
        for x in NOUNS_P: self.cat[x] = "NP"
        for x in NOUNS_O: self.cat[x] = "NO"
        for v in v5.COP: self.cat[v] = "COP"
        self.role_ids = {r: len(self.vocab) + i for i, r in enumerate(("A", "B", "C"))}      # identities of the roles themselves (for R_sc)
    def qevent(self, rng, depth=0, kind=None, nouns_p=None, nouns_o=None, ambiguous=False, n_quant=None):
        """(kind, (dA, nA), (dB, nB), (dC, nC), speakers..., scope) ; scope = list of wide roles (1, or 2 if ambiguous)"""
        NP = nouns_p or self.nouns_p; NO = nouns_o or self.nouns_o
        kind = kind or rng.choice(list(self.frames)); nA, nC = rng.choice(NP, 2, replace=False); nB = rng.choice(NO)
        speakers = list(rng.choice(self.persons, depth, replace=False)) if depth else []
        roles = ["A", "B", "C"]
        if ambiguous:
            q_roles = list(rng.choice(roles, 2, replace=False)); dets = {r: rng.choice(["every", "some", "a"]) for r in q_roles}; scope = sorted(q_roles)
        else:
            nq = n_quant if n_quant is not None else int(rng.integers(1, 3))
            q_roles = list(rng.choice(roles, nq, replace=False)) if nq else []
            if nq == 0: dets = {}; scope = []                                           # affirmative, all definite: the base sentences at depths 0-2
            elif nq == 1: dets = {q_roles[0]: rng.choice(DETS_Q)}; scope = [q_roles[0]]
            else:
                wide = q_roles[0]; dets = {wide: rng.choice(["each", "a certain"]), q_roles[1]: rng.choice(["every", "some", "a"])}; scope = [wide]
        D = {r: dets.get(r, DET_DEF) for r in roles}
        return (kind, (D["A"], nA), (D["B"], nB), (D["C"], nC)) + tuple(speakers) + (None,) * (3 - len(speakers)) + (scope,)
    def qrender(self, ev, t, rng):
        kind = ev[0]; (dA, nA), (dB, nB), (dC, nC) = ev[1], ev[2], ev[3]; speakers = [x for x in ev[4:7] if x is not None]
        s = self.qframes[kind][t].format(DA=dA, A=nA, DB=dB, B=nB, DC=dC, C=nC).rsplit(" .", 1)[0]
        for lvl in range(len(speakers) - 1, -1, -1): s = v5.OUTER[int(rng.integers(len(v5.OUTER)))].replace("<P>", s).format(X=speakers[lvl])
        return (s + " .").split()

SCOPE_TARGET = "role"      # set from --scope_target: 'role' (9c-bis), 'slot' (E2c primary), 'noun' (E2c secondary)

class QRoleWorld(v7.NestedRoleWorld):
    def __init__(self, F, world, rng):
        super().__init__(F, world, rng); D = F.shape[1]
        self.R["Q"] = np.exp(1j * rng.uniform(-np.pi, np.pi, D)); self.R["sc"] = np.exp(1j * rng.uniform(-np.pi, np.pi, D)); self.R["neg"] = np.exp(1j * rng.uniform(-np.pi, np.pi, D))
        self.Frole = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, D)) for r in ("A", "B", "C")}
    def det_vec(self, det):
        w = self.w; return self.F[w.idx["certain"]] if det == "a certain" else self.F[w.idx[det]]
    def state(self, ev):
        w = self.w; speakers = [x for x in ev[4:7] if x is not None]; scope = ev[-1]; S = 0
        for r, (det, noun) in zip(("A", "B", "C"), ev[1:4]):
            slot = self.F[w.idx[noun]]
            if det != DET_DEF: slot = slot + self.R["Q"] * self.det_vec(det)
            S = S + self.R[r] * slot
        if scope:                                                   # no quantifier => no scope term (empty scope gave 0/0 = NaN on 2026-10-15)
            nouns = {r: w.idx[n] for r, (d_, n) in zip(("A", "B", "C"), ev[1:4])}
            if SCOPE_TARGET == "role": terms = [self.R["sc"] * self.Frole[r] for r in scope]                                  # 9c-bis: a ROLE identity as filler (unreachable)
            elif SCOPE_TARGET == "slot": terms = [self.R["sc"] * self.R[r] * self.F[nouns[r]] for r in scope]               # E2c primary: the scope operator wraps the quantified SLOT
            else: terms = [self.R["sc"] * self.F[nouns[r]] for r in scope]                                                   # E2c secondary: the noun's identity under R_sc
            S = S + sum(terms) / np.sqrt(len(scope))
        for x in reversed(speakers): S = self.R["X"] * self.F[w.idx[x]] + self.R["rep"] * S
        return S

def decode_q(S, m, world, w, nouns_p, nouns_o, depth, delta=0.25, tau_none="relative"):
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fr = {r: torch.tensor(world.Frole[r], dtype=m.F.dtype, device=m.device) for r in world.Frole}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum(-1).sqrt(); rep = torch.conj(R["rep"]) ** depth
    dets = ["every", "each", "some", "a", "a certain", "the"]; dvec = torch.stack([torch.tensor(world.det_vec(d), dtype=m.F.dtype, device=m.device) for d in dets[:-1]])
    dvec = dvec / dvec.abs().pow(2).sum(-1, keepdim=True).sqrt(); out = {}
    for r in ("A", "B", "C"):
        cands = nouns_o if r == "B" else nouns_p; cid = torch.tensor([w.idx[x] for x in cands], device=m.device)
        sc = torch.real((Sn * torch.conj(R[r]) * rep) @ torch.conj(Fn[cid]).T); out[r] = cands[int(sc.argmax())]
        sd = torch.real((Sn * torch.conj(R[r]) * torch.conj(R["Q"]) * rep) @ torch.conj(dvec).T); j = int(sd.argmax())
        out["d" + r] = dets[j] if float(sd[j]) > 0.5 * float(sc.max()) else "the"           # no quantifier term => definite
    if SCOPE_TARGET == "role":
        ssc = {r: float(torch.real((Sn * torch.conj(R["sc"]) * rep * torch.conj(Fr[r])).sum()) / Fr[r].abs().pow(2).sum().sqrt()) for r in ("A", "B", "C")}
    else:                                                                 # the scope term carries the NOUN: read it with the decoded noun of each role
        ssc = {}
        for r in ("A", "B", "C"):
            key = torch.conj(R["sc"]) * rep * (torch.conj(R[r]) if SCOPE_TARGET == "slot" else 1)
            ssc[r] = float(torch.real((Sn * key * torch.conj(Fn[w.idx[out[r]]])).sum()))
    top = sorted(ssc, key=lambda r: -ssc[r]); both = ssc[top[1]] >= (1 - delta) * ssc[top[0]] and ssc[top[1]] > 0.5 * ssc[top[0]]
    # "no quantifier" hypothesis (evaluation gap found 2026-10-16: affirmative sentences always got a scope). tau_none: "relative" (default from
    # 9c-ter on: none if the best scope score < 0.25 x the best noun score at this depth — scope scores shrink with depth, an absolute threshold
    # said 'none' everywhere at depth 2, RESULTS 9c-ter), a float = absolute threshold (0.08 in E2c), or a negative float = no 'none' hypothesis.
    noun_best = max(float(torch.real((Sn * torch.conj(R[r]) * rep) @ torch.conj(Fn[torch.tensor([w.idx[x] for x in (nouns_o if r == "B" else nouns_p)], device=m.device)]).T).max()) for r in ("A", "B", "C"))
    none = ssc[top[0]] < (0.25 * noun_best if tau_none == "relative" else tau_none)
    out["scope_scores"] = ssc; out["scope"] = [] if none else (sorted(top[:2]) if both else [top[0]]); out["scope_choice"] = None if none else top[0]
    # rule 17 (judgment a8a8fab C1): PRESENCE is its own axis (best scope score, thresholded on CALIBRATION sentences, never on the scored rows);
    # SCOPE is reported conditional on presence as the argmax (or 'both'), with no 'none' rule involved.
    out["pres_score"] = ssc[top[0]]; out["scope_top"] = sorted(top[:2]) if both else [top[0]]; out["scope_first"] = top[0]
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda"); ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--ckpt", action="store_true", help="checkpoint every 2000 steps to logs/.ckpt_<tag>_<seed>.pt"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="run"); ap.add_argument("--delta", type=float, default=0.25); ap.add_argument("--p_aff", type=float, default=0.4, help="share of affirmative (all-definite) sentences at depths 0-2 in training (E1b protocol); 0 = the 9c run"); ap.add_argument("--scope_target", default="role", choices=["role", "slot", "noun"], help="world definition of the wide-scope term: role = 9c-bis (R_sc x f_role), slot = E2c primary (R_sc x R_role x f_noun), noun = E2c secondary (R_sc x f_noun)")
    ap.add_argument("--det_gates", default="learned", choices=["learned", "open"], help="C2 (judgment a8a8fab E2): 'open' = determiner gates GIVEN open (rule 20), not learned")
    ap.add_argument("--rel", action="store_true", help="C2b (judgment 3b4f837 E2): the quantified determiners are BEACON-RELATIVE emitters (one learned displacement each, relative to the noun's own address); gates untouched")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    SCOPE_TARGET = a.scope_target; globals()["SCOPE_TARGET"] = a.scope_target
    print(f"scope target = {a.scope_target}")
    print(f"v14_quantifiers (E2): device={a.device}, D={D}, N={N}, seeds={a.seeds}; quantifier = role on the argument (R_Q (x) f_det inside the slot); scope = R_sc (x) f_role; ambiguity = superposition in the WORLD")
    print(f"training: affirmative (all definite) at depths 0-2 (share {a.p_aff}) + UNAMBIGUOUS quantified sentences (one quantifier, or two with a wide-scope marker) at depths 0-1; 12 nouns per class (6 new at test)")
    print(f"test: unambiguous depth 0 / 1 / 2 (UNSEEN depth), new nouns, AMBIGUOUS (never trained) depth 0; 'both' reported when the top-2 scope scores are within delta={a.delta}")
    print(f"patterns: {PATTERNS}\nchance: noun 1/12 = 0.083 (new 1/6); determiner 1/6; scope 1/3; full frame ~0\n")
    cols = ["unamb d0: nouns", "dets", "PRESENCE", "SCOPE|gold-q", "FULL (joint)", "aff d0: ABSENT", "unamb d1: FULL", "unamb d2 UNSEEN: PRESENCE", "SCOPE|gold-q", "FULL", "aff d2: ABSENT", "new nouns d0: FULL", "AMBIG d0: both reported", "ambig: cos to superposed world", "ambig: forced surface-first", "STRICT d2 (d1 threshold): PRESENCE", "FULL", "aff d2 ABSENT"]
    print("decoder (rule 17, from judgment a8a8fab C1): PRESENCE = best scope score >= a threshold set per depth on CALIBRATION sentences (midpoint of the mean best score on 60 quantified vs 60 affirmative"
          " calibration sentences per pattern, rng 1234+depth; the scored rows use rng 900+); SCOPE|gold-quantified = argmax role (or 'both') on sentences the GOLD marks as quantified (not conditioned on the predicted presence), no 'none' rule;"
          " FULL = nouns, determiners, presence and scope all right. The depth-2 threshold uses LABELLED depth-2 calibration sentences (held out, rule 25) — the STRICT depth-2 columns use the DEPTH-1 threshold, with no depth-2 information at all (judgment 3b4f837 C2).")
    res = []; gates_all = []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed); w = QWorld(rng); trNP, teNP, trNO, teNO = NOUNS_P[:12], NOUNS_P[12:], NOUNS_O[:12], NOUNS_O[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = QRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F); assert m.F_np is world.F
        if a.det_gates == "open": print(f"  gates GIVEN open (rule 20, not learned): {m.fix_gates(['every', 'each', 'some', 'a', 'certain'], True)}")
        params = [m.theta, m.logits]
        if a.rel: print(f"  C2b beacon-relative emitters (given set, learned displacement): {m.set_rel_emitters(['every', 'each', 'some', 'a', 'certain'], ['NP', 'NO'])} -> takers NP, NO"); params.append(m.rel)
        trng = np.random.default_rng(200 + seed); sents, targets = [], []
        for n in range(N):
            # E1b protocol (after the 9c protocol error): affirmative (all 'the') sentences at depths 0-2, quantified sentences at depths 0-1
            if trng.random() < a.p_aff: ev = w.qevent(trng, depth=int(trng.integers(3)), nouns_p=trNP, nouns_o=trNO, n_quant=0)
            else: ev = w.qevent(trng, depth=int(trng.integers(2)), nouns_p=trNP, nouns_o=trNO)
            sents.append(w.qrender(ev, int(trng.integers(len(PATTERNS))), trng)); targets.append(world.state(ev))
        t0 = time.time(); cats, ids, mask = m.encode(sents); tgt_all = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
        ckpt = f"logs/.ckpt_v14_{a.tag}_{a.scope_target}_{a.det_gates}{'_rel' if a.rel else ''}_{seed}.pt"   # always saved (final params): re-scoring without retraining (judgment a8a8fab C1 iv)
        trainer.train_loop(m, params, cats, ids, mask, tgt_all, a.batch, a.lr, log_every=a.log_every, ckpt=ckpt, resume=a.resume, tag=f" seed {seed}")
        if a.device == "cuda": torch.cuda.synchronize()
        t_train = time.time() - t0
        def pres_threshold(depth, n=60):
            """calibration rows (rule: a threshold is never set on the rows it is reported on): midpoint of the mean best scope score on quantified vs affirmative sentences"""
            crng = np.random.default_rng(1234 + depth); q, af = [], []
            for t in range(len(PATTERNS)):
                for nq, store in ((None, q), (0, af)):
                    evs = [w.qevent(crng, depth=depth, nouns_p=trNP, nouns_o=trNO, n_quant=nq) for _ in range(n // len(PATTERNS) + 1)]; toks = [w.qrender(ev, t, crng) for ev in evs]
                    cats, ids, mask = m.encode(toks)
                    with torch.no_grad(): S = m.states(cats, ids, mask)
                    store.extend(decode_q(S[k], m, world, w, trNP, trNO, depth, a.delta)["pres_score"] for k in range(len(evs)))
            return 0.5 * (float(np.mean(q)) + float(np.mean(af))), float(np.mean(q)), float(np.mean(af))
        thr = {d: pres_threshold(d) for d in (0, 1, 2)}
        print(f"  seed {seed} presence thresholds (calibration rows; mean quantified / affirmative): " + "; ".join(f"d{d}: {thr[d][0]:.3f} ({thr[d][1]:.3f} / {thr[d][2]:.3f})" for d in thr))
        def run(depth, ambiguous, NP, NO, n=60, n_quant=None, thr_depth=None):
            thr_d = depth if thr_depth is None else thr_depth                                  # strict column: the DEPTH-1 threshold at depth 2
            erng = np.random.default_rng(900 + depth + 10 * ambiguous + (0 if n_quant is None else 100 * (n_quant + 1))); nouns, dets, pres, scope, full, both, cosw, surf, sc1, sc2 = [], [], [], [], [], [], [], [], [], []
            for t in range(len(PATTERNS)):
                evs = [w.qevent(erng, depth=depth, nouns_p=NP, nouns_o=NO, ambiguous=ambiguous, n_quant=n_quant) for _ in range(n)]; toks = [w.qrender(ev, t, erng) for ev in evs]
                cats, ids, mask = m.encode(toks)
                with torch.no_grad(): S = m.states(cats, ids, mask)
                for k, ev in enumerate(evs):
                    o = decode_q(S[k], m, world, w, NP, NO, depth, a.delta); ok_n = [o[r] == ev[i][1] for r, i in (("A", 1), ("B", 2), ("C", 3))]; ok_d = [o["d" + r] == ev[i][0] for r, i in (("A", 1), ("B", 2), ("C", 3))]
                    nouns.extend(ok_n); dets.extend(ok_d); present = o["pres_score"] >= thr[thr_d][0]; truth = sorted(ev[-1])
                    if n_quant == 0: pres.append(not present); full.append(all(ok_n) and all(ok_d) and not present); continue       # affirmative: ABSENT is the axis
                    pres.append(present)
                    if ambiguous:
                        both.append(o["scope_top"] == truth); Sw = torch.tensor(world.state(ev), dtype=m.F.dtype, device=m.device)
                        cosw.append(float(torch.real((torch.conj(Sw) * S[k]).sum()) / (Sw.abs().pow(2).sum().sqrt() * S[k].abs().pow(2).sum().sqrt())))
                        # surface order: the quantified role that appears first in the sentence
                        first = min(ev[-1], key=lambda r: toks[k].index(ev[{"A": 1, "B": 2, "C": 3}[r]][1])); surf.append(o["scope_first"] == first)
                    else:
                        ok_s = o["scope_top"] == truth; scope.append(ok_s); full.append(all(ok_n) and all(ok_d) and present and ok_s)
                        (sc1 if sum(1 for i in (1, 2, 3) if ev[i][0] != "the") == 1 else sc2).append(ok_s)
            if ambiguous: return float(np.mean(both)), float(np.mean(cosw)), float(np.mean(surf)), float(np.mean(nouns)), float(np.mean(dets)), float(np.mean(pres))
            if n_quant == 0: return float(np.mean(nouns)), float(np.mean(dets)), float(np.mean(pres)), float(np.mean(full))
            return float(np.mean(nouns)), float(np.mean(dets)), float(np.mean(pres)), float(np.mean(scope)), float(np.mean(full)), (float(np.mean(sc1)) if sc1 else float("nan")), (float(np.mean(sc2)) if sc2 else float("nan"))
        u0 = run(0, False, trNP, trNO); u1 = run(1, False, trNP, trNO, n=40); u2 = run(2, False, trNP, trNO, n=30); nn_ = run(0, False, teNP, teNO, n=40); am = run(0, True, trNP, trNO)
        a0 = run(0, False, trNP, trNO, n=30, n_quant=0); a2 = run(2, False, trNP, trNO, n=30, n_quant=0)
        s2 = run(2, False, trNP, trNO, n=30, thr_depth=1); sa2 = run(2, False, trNP, trNO, n=30, n_quant=0, thr_depth=1)   # STRICT: depth-1 threshold at depth 2
        print(f"  seed {seed} SCOPE|gold-quantified one-quantifier / two-quantifier (marked): d0 {u0[5]:.3f} / {u0[6]:.3f}; d1 {u1[5]:.3f} / {u1[6]:.3f}; d2 UNSEEN {u2[5]:.3f} / {u2[6]:.3f}; presence on ambiguous d0 {am[5]:.3f}")
        r = [u0[0], u0[1], u0[2], u0[3], u0[4], a0[2], u1[4], u2[2], u2[3], u2[4], a2[2], nn_[4], am[0], am[1], am[2], s2[2], s2[4], sa2[2]]; res.append(r); gt = m.gates(); gates_all.append(gt)
        print(f"  seed {seed} ({t_train:.0f}s) " + " | ".join(f"{x:.3f}" for x in r) + f"  || ambiguous: nouns {am[3]:.2f} dets {am[4]:.2f} || gates OPEN: {sorted(c for c, v in gt.items() if v > 0.5)}", flush=True)
    r = np.array(res); print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(len(cols))))
    cats0 = sorted(gates_all[0]); g = {c: np.mean([gt[c] for gt in gates_all]) for c in cats0}
    print("== learned gates (mean sigmoid; OPEN > 0.5): " + " ".join(f"{c}={g[c]:.2f}" for c in sorted(cats0, key=lambda c: -g[c])[:16]))
    print(f"\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] presence and scope as separate axes, scope conditional on presence (17) [x] presence threshold set on calibration rows, not the scored rows [x] one- and two-quantifier scope reported separately [x] params saved for re-scoring [x] given vs learned gates stated (20) [x] ambiguity reported separately, never trained [x] FULL frame (R1) [x] unseen depth 2 (composition rule, with reserve) [x] new nouns [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] model unchanged")
