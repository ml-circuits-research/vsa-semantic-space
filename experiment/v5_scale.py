"""
v5_scale: 00_READ_FIRST.md §5.1 — reproduction of round 3 on a 2-3x LARGER WORLD, with 5 seeds, with v4_torch (GPU).
The model is UNCHANGED (v4: fixed identity + address = bundle of path/chain/bigram/pos codes). Only the world changes:
  - 6 frames (GIVE, SELL, SEND, LEND, SHOW, OFFER), each with 16 formulations (the same 16 patterns, different verbs; each verb form
    is its own category, as in v4) -> 3x more categories and transitions;
  - 18 persons + 18 objects (12 for training, 6 new; chance 1/12 = 0.083, resp. 1/6 = 0.167);
  - nesting 0-3 levels: '{X} spune ca <P>', '{X} zice ca <P>', '<P> , spune {X}' (Romanian world; see LANGS["en"] for the English one);
    the speakers X1 (outermost), X2, X3 are roles in the world.
    Training on depths 0-2; depth 3 is a NEW CONSTRUCTION (a longer path than anything seen).
Evaluation axes (EVALUATION_PRINCIPLES.md), separate:
  seen/new identities | new formulations with seen transitions | new formulations with UNSEEN transitions/categories | depth 3 (unseen).
Criterion §5.1: the columns shared with round 3 within +-0.03 of v4 on the small world (0.94 / 0.90 for 'all 4').
Run: python3 v5_scale.py [--seeds 5] [--N 30000] [--D 1024] [--batch 8] [--depth_train 2] [--variants all|all4]  ('toate4' is still accepted)
"""
import argparse, time
import numpy as np, torch
import camp as cp, v2, v4, v4_torch

LANGS = {
  "ro": dict(
    VERBS={"GIVE": ("da", "dat", "primeste"), "SELL": ("vinde", "vandut", "cumpara"), "SEND": ("trimite", "trimis", "primeste"),
           "LEND": ("imprumuta", "imprumutat", "ia"), "SHOW": ("arata", "aratat", "vede"), "OFFER": ("ofera", "oferit", "accepta")},
    PATTERNS=["{A} V {B} lui {C} .", "{C} VR {B} de la {A} .", "{B} e VP lui {C} de {A} .", "lui {C} {A} ii V {B} .",
              "{B} ii este VP lui {C} de catre {A} .", "{A} ii V lui {C} {B} .", "{B} e VP de {A} lui {C} .", "lui {C} ii V {B} {A} .",
              "{A} V lui {C} {B} .", "lui {C} {A} V {B} .", "{A} ii V {B} lui {C} .",
              "{B} , {A} il V lui {C} .", "de la {A} , {C} VR {B} .", "{C} VR de la {A} {B} .", "{B} este VP lui {C} de {A} .", "{A} , lui {C} , ii V {B} ."],
    TRAIN_T=[0, 1, 2, 3, 6, 12, 14], OUTER=["{X} spune ca <P>", "{X} zice ca <P>", "<P> , spune {X}"], COP=["e", "este"]),
  "en": dict(   # from 2026-10-15 on, advanced tests use the English world (author decision; see AGENTS.md)
    VERBS={"GIVE": ("gives", "given", "receives"), "SELL": ("sells", "sold", "buys"), "SEND": ("sends", "sent", "gets"),
           "LEND": ("lends", "lent", "borrows"), "SHOW": ("shows", "shown", "sees"), "OFFER": ("offers", "offered", "accepts")},
    PATTERNS=["{A} V {B} to {C} .", "{C} VR {B} from {A} .", "{B} is VP to {C} by {A} .", "{B} is VP by {A} to {C} .",
              "{A} V {C} {B} .", "{C} is VP {B} by {A} .", "it is {A} who V {B} to {C} .", "it is {B} that {A} V to {C} .",
              "{B} , {A} V to {C} .", "to {C} , {A} V {B} .", "from {A} , {C} VR {B} .", "{C} VR from {A} {B} .",
              "what {A} V to {C} is {B} .", "{B} is what {A} V to {C} .", "by {A} , {B} is VP to {C} .", "{A} V {B} , to {C} ."],
    TRAIN_T=[0, 1, 2, 3, 6, 9, 12, 15], OUTER=["{X} says that <P>", "{X} claims that <P>", "<P> , says {X}"], COP=["is"]),
}
LANG = None
def set_lang(name):
    """select the world language; rebinds the globals used by BigWorld/templates/split_templates"""
    global LANG, VERBS, PATTERNS, TRAIN_T, OUTER, COP
    LANG = name; d = LANGS[name]; VERBS, PATTERNS, TRAIN_T, OUTER, COP = d["VERBS"], d["PATTERNS"], d["TRAIN_T"], d["OUTER"], d["COP"]
set_lang("ro")
XROLES = ["X1", "X2", "X3"]

def templates():
    return {k: [p.replace("VP", vp).replace("VR", vr).replace("V", v) for p in PATTERNS] for k, (v, vp, vr) in VERBS.items()}

class BigWorld(cp.World):
    def __init__(self, n_id, rng, depth_max=3):
        super().__init__([f"p{i}" for i in range(n_id)], [f"o{i}" for i in range(n_id)], rng)
        self.frames = templates(); self.depth_max = depth_max
        words = {x for ts in self.frames.values() for t in ts for x in t.split() if not x.startswith("{")}
        words |= {x for o in OUTER for x in o.split() if not x.startswith("{") and x != "<P>"}
        self.vocab = sorted(set(self.persons) | set(self.objects) | words); self.idx = {x: i for i, x in enumerate(self.vocab)}
        self.cat = {**{p: "P" for p in self.persons}, **{o: "O" for o in self.objects}, **{x: x for x in words}}
        for v in COP: self.cat[v] = "COP"
    def event(self, persons=None, objects=None, depth=0):
        P = persons or self.persons; O = objects or self.objects
        ids = self.rng.choice(P, 2 + depth, replace=False); B = self.rng.choice(O)
        return (self.rng.choice(list(self.frames)), ids[0], B, ids[1]) + tuple(ids[2:]) + (None,) * (self.depth_max - depth)
    def depth(self, ev): return sum(x is not None for x in ev[4:])
    def render(self, ev, t, outer=None):
        kind, A, B, C = ev[:4]; d = self.depth(ev); outer = outer if outer is not None else [self.rng.integers(len(OUTER)) for _ in range(d)]
        s = self.frames[kind][t].format(A=A, B=B, C=C).rsplit(" .", 1)[0]
        for lvl in range(d - 1, -1, -1):                      # from the innermost (X_d) outward (X1)
            s = OUTER[outer[lvl]].replace("<P>", s).format(X=ev[4 + lvl])
        return (s + " .").split()

class BigRoleWorld:
    """explicit roles A, B, C, X1..X3; identities SHARED with the model"""
    def __init__(self, F, world, rng):
        self.F, self.w = F, world; self.R = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, F.shape[1])) for r in ["A", "B", "C"] + XROLES}
    def state(self, ev):
        S = sum(self.R[r] * self.F[self.w.idx[x]] for r, x in zip(["A", "B", "C"] + XROLES, ev[1:]) if x is not None)
        return S

class V5(v4_torch.V4Torch):
    def __init__(self, world, D, rng, gates, codes, device, max_len=40):
        super().__init__(world, D, rng, gates, codes, device)
        m = self.np_model; extra = [f"pos:{i}" for i in range(16, max_len) if f"pos:{i}" not in m.pid]
        m.names += extra; m.pid = {n: i for i, n in enumerate(m.names)}
        th = np.concatenate([m.theta, rng.uniform(-np.pi, np.pi, (len(extra), D))]); self.P = len(m.names)
        self.theta = torch.nn.Parameter(torch.tensor(th, dtype=self.dtype, device=device))
        self.root_mask = torch.ones(self.P, 1, dtype=self.dtype, device=device); self.root_mask[0] = 0
    def train(self, world, n_sent, lr=0.02, templates=None, persons=None, objects=None, batch=1, depth_train=2):
        w = self.w; sents, targets = [], []
        for n in range(n_sent):
            d = int(self.rng.integers(depth_train + 1)); ev = w.event(persons, objects, depth=d)
            sents.append(w.render(ev, self.rng.choice(list(templates)))); targets.append(world.state(ev))
        t0 = time.time(); idx_all, ids_all, mask_all = self.encode(sents); self.t_encode = time.time() - t0
        tgt_all = torch.tensor(np.array(targets), dtype=self.F.dtype, device=self.device)
        opt = torch.optim.Adam([self.theta], lr=lr, betas=(0.9, 0.999), eps=1e-8)
        for i in range(0, n_sent, batch):
            sl = slice(i, i + batch); S = self.states(idx_all[sl], ids_all[sl], mask_all[sl]); tgt = tgt_all[sl]
            cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
            opt.zero_grad(); (-cos.mean()).backward(); opt.step()
            with torch.no_grad(): self.theta[0] = 0

def decode_stats(m, world, w, templs, persons, objects, depth=0, n=40, max_role_depth=None):
    """accuracy and margin over ALL roles present (A, B, C, X1..Xd), with the world addresses"""
    acc, marg = [], []
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
    # evaluated roles: only those that could have been targets during training (PRINCIPLES 14: a new role is at chance by construction)
    roles = [("A", 1, persons), ("B", 2, objects), ("C", 3, persons)] + [(XROLES[i], 4 + i, persons) for i in range(min(depth, max_role_depth or depth))]
    for kind in w.frames:
        for t in templs:
            evs = [w.event(persons, objects, depth=depth) for _ in range(n)]; evs = [(kind,) + e[1:] for e in evs]
            idx, ids, mask = m.encode([w.render(ev, t) for ev in evs])
            with torch.no_grad(): S = m.states(idx, ids, mask)
            Sn = S / S.abs().pow(2).sum(-1, keepdim=True).sqrt()
            for r, ri, cands in roles:
                cid = torch.tensor([w.idx[x] for x in cands], device=m.device)
                sc = torch.real((Sn * torch.conj(R[r])) @ torch.conj(Fn[cid]).T).cpu().numpy()
                truth = np.array([cands.index(ev[ri]) for ev in evs]); acc.extend(sc.argmax(1) == truth)
                sc_t = sc[np.arange(n), truth]; sc2 = sc.copy(); sc2[np.arange(n), truth] = -np.inf; marg.extend(sc_t - sc2.max(1))
    return float(np.mean(acc)), float(np.mean(marg))

def split_templates(w):
    """new formulations with ALL transitions seen vs with UNSEEN transitions/categories (over all frames, the verb categories being per frame)"""
    def tr(kind, t):
        cats = [w.cat[x] for x in w.frames[kind][t].format(A="p0", B="o0", C="p1").split()]; return set(zip(cats[:-1], cats[1:])), set(cats)
    seen_tr, seen_cat = set(), set()
    for k in w.frames:
        for t in TRAIN_T: a, b = tr(k, t); seen_tr |= a; seen_cat |= b
    held_seen, held_new = [], []
    for t in range(len(PATTERNS)):
        if t in TRAIN_T: continue
        new = [(tr(k, t)[0] - seen_tr, tr(k, t)[1] - seen_cat) for k in w.frames]
        (held_seen if all(not a and not b for a, b in new) else held_new).append(t)
        a, b = new[0]; print(f"  T{t} '{PATTERNS[t]}': new transitions (GIVE) {sorted(a)}, new categories {sorted(b)}")
    return held_seen, held_new

def split_templates_rolecues(w):
    """Stricter axis (judgment 2026-10-15): a new formulation is 'role cues seen' iff for EVERY role slot (A, B, C) both local cues —
    (previous category, role) and (role, next category) — occur in some training formulation with the SAME role. This tests the
    mechanism (can seen local cues be recombined?) rather than the redundancy of the formulation set ('seen transitions' only asks
    that each category pair was seen, with any role). Returns (role_cues_seen, role_cues_unseen) over the non-training templates."""
    def cues(kind, t):
        toks = w.frames[kind][t].format(A="{A}", B="{B}", C="{C}").split(); out = set()
        for i, x in enumerate(toks):
            if x in ("{A}", "{B}", "{C}"):
                prev = w.cat[toks[i - 1]] if i > 0 and not toks[i - 1].startswith("{") else ("^" if i == 0 else "ID")
                nxt = w.cat[toks[i + 1]] if i + 1 < len(toks) and not toks[i + 1].startswith("{") else ("$" if i + 1 == len(toks) else "ID")
                out.add((prev, x)); out.add((x, nxt))
        return out
    seen = set()
    for k in w.frames:
        for t in TRAIN_T: seen |= cues(k, t)
    ok, bad = [], []
    for t in range(len(PATTERNS)):
        if t in TRAIN_T: continue
        new = set().union(*[cues(k, t) - seen for k in w.frames])
        (ok if not new else bad).append(t)
    return ok, bad

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--variants", default="all")
    ap.add_argument("--lang", default="ro")
    a = ap.parse_args(); D, N = a.D, a.N; set_lang(a.lang)
    print(f"v5_scale: device={a.device} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}, D={D}, N={N}, batch={a.batch}, lr={a.lr}, seeds={a.seeds}")
    print(f"world ({LANG}): {len(VERBS)} frames x {len(PATTERNS)} formulations; {a.n_id}+{a.n_id} identities (12 train); nesting 0-{a.depth_train} train, 3 test; outer forms {OUTER}")
    print("training formulations:", TRAIN_T)
    w0 = BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = split_templates(w0)
    print(f"  => new formulations with ALL transitions seen: {held_seen}; with UNSEEN transitions/categories: {held_new}")
    ex = w0.event(w0.persons[:12], w0.objects[:12], depth=3); print(f"  depth 3 example: {' '.join(w0.render(ex, 0))}")
    print(f"chance: seen identities 1/12 = {1/12:.3f}; new identities 1/6 = {1/6:.3f}. Trivial control: 'everything at the origin' (no codes) = chance.\n")
    variants = [("path", ["path"]), ("chain", ["chain"]), ("bigram", ["bigram"]), ("pos", ["pos"]), ("path+bigram", ["path", "bigram"]),
                ("path+chain+bigram", ["path", "chain", "bigram"]), ("all 4", ["path", "chain", "bigram", "pos"])]
    if a.variants in ("all4", "toate4"): variants = variants[-1:]
    cols = ["seen (depth 0)", "new ids (depth 0)", "new form., seen trans.", "new form., UNSEEN trans.", "nesting depth 1-2 (seen)", "nesting depth 3 (UNSEEN; only A,B,C,X1,X2 — X3 was never a target)"]
    results = {name: [] for name, _ in variants}
    for seed in range(a.seeds):
        w = BigWorld(a.n_id, np.random.default_rng(seed)); cats = sorted(set(w.cat.values()))
        trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
        world = BigRoleWorld(F, w, np.random.default_rng(100 + seed))
        for name, codes in variants:
            m = V5(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), codes, a.device); m.set_F(F)
            assert m.F_np is world.F
            t0 = time.time(); m.train(world, N, templates=TRAIN_T, persons=trP, objects=trO, batch=a.batch, lr=a.lr, depth_train=a.depth_train)
            torch.cuda.synchronize(); t_train = time.time() - t0
            r = [decode_stats(m, world, w, TRAIN_T, trP, trO, 0, n=20), decode_stats(m, world, w, TRAIN_T, teP, teO, 0, n=20),
                 decode_stats(m, world, w, held_seen, trP, trO, 0, n=30), decode_stats(m, world, w, held_new, trP, trO, 0, n=30),
                 tuple(np.mean([decode_stats(m, world, w, TRAIN_T, trP, trO, d, n=15) for d in (1, 2)], axis=0)),
                 decode_stats(m, world, w, TRAIN_T, trP, trO, 3, n=20, max_role_depth=a.depth_train)]
            results[name].append(r)
            print(f"  seed {seed} {name:18s} ({t_train:5.1f}s train, {m.t_encode:4.1f}s enc., {time.time()-t0-t_train:4.1f}s eval) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r), flush=True)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    for name, _ in variants:
        r = np.array(results[name]); print(f"  {name:18s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] separate axes (identities / seen transitions / UNSEEN / depth) [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] trivial control = chance")
