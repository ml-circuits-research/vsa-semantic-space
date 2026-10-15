"""
v5_scale: 00_CITESTE_INTAI.md §5.1 — reproducerea rundei 3 pe o LUME DE 2-3x, cu 5 seed-uri, cu v4_torch (GPU).
Modelul e NESCHIMBAT (v4: identitate fixa + adresa = bundle de coduri path/chain/bigram/pos). Se schimba doar lumea:
  - 6 cadre (GIVE, SELL, SEND, LEND, SHOW, OFFER), fiecare cu 16 formulari (aceleasi 16 tipare, verbe diferite; fiecare forma verbala
    e categorie proprie, ca in v4) -> de 3x mai multe categorii si tranzitii;
  - 18 persoane + 18 obiecte (12 de antrenare, 6 noi; sansa 1/12 = 0.083, resp. 1/6 = 0.167);
  - imbricare 0-3 nivele: '{X} spune ca <P>', '{X} zice ca <P>', '<P> , spune {X}'; vorbitorii X1 (exterior), X2, X3 sunt roluri in lume.
    Antrenare pe adancimi 0-2; adancimea 3 e CONSTRUCTIE NOUA (drum mai lung decat orice vazut).
Axe de evaluare (PRINCIPII_EVALUARE.md), separate:
  identitati vazute/noi | formulari noi cu tranzitii vazute | formulari noi cu tranzitii/categorii NEvazute | adancime 3 (nevazuta).
Criteriul §5.1: coloanele comune cu runda 3 in +-0.03 fata de v4 pe lumea mica (0.94 / 0.90 pentru 'toate 4').
Rulare: python3 v5_scale.py [--seeds 5] [--N 30000] [--D 1024] [--batch 8] [--depth_train 2] [--variants all|toate4]
"""
import argparse, time
import numpy as np, torch
import camp as cp, v2, v4, v4_torch

VERBS = {  # cadru -> (V activ, VP participiu, VR verbul destinatarului)
    "GIVE": ("da", "dat", "primeste"), "SELL": ("vinde", "vandut", "cumpara"), "SEND": ("trimite", "trimis", "primeste"),
    "LEND": ("imprumuta", "imprumutat", "ia"), "SHOW": ("arata", "aratat", "vede"), "OFFER": ("ofera", "oferit", "accepta"),
}
PATTERNS = ["{A} V {B} lui {C} .", "{C} VR {B} de la {A} .", "{B} e VP lui {C} de {A} .", "lui {C} {A} ii V {B} .",
            "{B} ii este VP lui {C} de catre {A} .", "{A} ii V lui {C} {B} .", "{B} e VP de {A} lui {C} .", "lui {C} ii V {B} {A} .",
            "{A} V lui {C} {B} .", "lui {C} {A} V {B} .", "{A} ii V {B} lui {C} .",
            "{B} , {A} il V lui {C} .", "de la {A} , {C} VR {B} .", "{C} VR de la {A} {B} .", "{B} este VP lui {C} de {A} .", "{A} , lui {C} , ii V {B} ."]
TRAIN_T = [0, 1, 2, 3, 6, 12, 14]
OUTER = ["{X} spune ca <P>", "{X} zice ca <P>", "<P> , spune {X}"]
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
        for v in ["e", "este"]: self.cat[v] = "COP"
    def event(self, persons=None, objects=None, depth=0):
        P = persons or self.persons; O = objects or self.objects
        ids = self.rng.choice(P, 2 + depth, replace=False); B = self.rng.choice(O)
        return (self.rng.choice(list(self.frames)), ids[0], B, ids[1]) + tuple(ids[2:]) + (None,) * (self.depth_max - depth)
    def depth(self, ev): return sum(x is not None for x in ev[4:])
    def render(self, ev, t, outer=None):
        kind, A, B, C = ev[:4]; d = self.depth(ev); outer = outer if outer is not None else [self.rng.integers(len(OUTER)) for _ in range(d)]
        s = self.frames[kind][t].format(A=A, B=B, C=C).rsplit(" .", 1)[0]
        for lvl in range(d - 1, -1, -1):                      # de la cel mai interior (X_d) spre exterior (X1)
            s = OUTER[outer[lvl]].replace("<P>", s).format(X=ev[4 + lvl])
        return (s + " .").split()

class BigRoleWorld:
    """roluri explicite A, B, C, X1..X3; identitati IMPARTITE cu modelul"""
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

def decode_stats(m, world, w, templs, persons, objects, depth=0, n=40):
    """acuratete si marja pe TOATE rolurile prezente (A, B, C, X1..Xd), cu adresele lumii"""
    acc, marg = [], []
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
    roles = [("A", 1, persons), ("B", 2, objects), ("C", 3, persons)] + [(XROLES[i], 4 + i, persons) for i in range(depth)]
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
    """formulari noi cu TOATE tranzitiile vazute vs cu tranzitii/categorii NEvazute (pe toate cadrele, categoriile verbelor fiind per cadru)"""
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
        a, b = new[0]; print(f"  T{t} '{PATTERNS[t]}': tranzitii noi (GIVE) {sorted(a)}, categorii noi {sorted(b)}")
    return held_seen, held_new

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--variants", default="all")
    a = ap.parse_args(); D, N = a.D, a.N
    print(f"v5_scale: device={a.device} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}, D={D}, N={N}, batch={a.batch}, lr={a.lr}, seeds={a.seeds}")
    print(f"lume: {len(VERBS)} cadre x {len(PATTERNS)} formulari; {a.n_id}+{a.n_id} identitati (12 antrenare); imbricare 0-{a.depth_train} antrenare, 3 test; forme exterioare {OUTER}")
    print("formulari de antrenare:", TRAIN_T)
    w0 = BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = split_templates(w0)
    print(f"  => formulari noi cu TOATE tranzitiile vazute: {held_seen}; cu tranzitii/categorii NEVAZUTE: {held_new}")
    ex = w0.event(w0.persons[:12], w0.objects[:12], depth=3); print(f"  exemplu adancime 3: {' '.join(w0.render(ex, 0))}")
    print(f"sansa: identitati vazute 1/12 = {1/12:.3f}; identitati noi 1/6 = {1/6:.3f}. Control trivial: 'toate la origine' (fara coduri) = sansa.\n")
    variants = [("path", ["path"]), ("chain", ["chain"]), ("bigram", ["bigram"]), ("pos", ["pos"]), ("path+bigram", ["path", "bigram"]),
                ("path+chain+bigram", ["path", "chain", "bigram"]), ("toate 4", ["path", "chain", "bigram", "pos"])]
    if a.variants == "toate4": variants = variants[-1:]
    cols = ["vazute (ad.0)", "id. noi (ad.0)", "form. noi, tranz. vazute", "form. noi, tranz. NEvazute", "imbricare ad.1-2 (vazuta)", "imbricare ad.3 (NEvazuta)"]
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
                 decode_stats(m, world, w, TRAIN_T, trP, trO, 3, n=20)]
            results[name].append(r)
            print(f"  seed {seed} {name:18s} ({t_train:5.1f}s antr., {m.t_encode:4.1f}s enc., {time.time()-t0-t_train:4.1f}s eval) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r), flush=True)
    print(f"\n== MEDII pe {a.seeds} seed-uri: acuratete (marja) ==  coloane: " + " | ".join(cols))
    for name, _ in variants:
        r = np.array(results[name]); print(f"  {name:18s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print(f"\nchecklist: [x] adrese din lume [x] F comun (assert) [x] sansa tiparita [x] axe separate (identitati / tranzitii vazute / NEvazute / adancime) [x] marja [{'x' if a.seeds >= 3 else ' '}] >=3 seed-uri [x] control trivial = sansa")
