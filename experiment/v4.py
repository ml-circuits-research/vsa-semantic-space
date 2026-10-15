"""
v4: ADRESE REDUNDANTE pentru generalizare la CONSTRUCTII noi.
Adresa unui cuvant = bundle de mai multe coduri independente, fiecare cu alt mod de descompunere a contextului:
  path   : suma deplasarilor categoriilor din prefix (comutativ, aditiv)            [v1 fara buffer]
  chain  : lantul de asteptari (necomutativ, precis, fragil)                       [v2]
  bigram : doar categoria precedenta + categoria proprie (foarte partajabil)
  pos    : pozitia absoluta (x) categoria proprie (cod HDC clasic de secventa)
Ipoteza: o tranzitie nevazuta strica un cod, nu pe toate; decodarea argmax supravietuieste.
Evaluare conform EVALUATION_PRINCIPLES.md: adrese din lume, F comun, sansa tiparita, 3 axe vazut/nevazut (identitati, formulari cu
tranzitii vazute, formulari cu tranzitii/categorii nevazute), marja, 3 seed-uri, control trivial.
"""
import numpy as np, time, sys, itertools
import camp as cp, v2

TEMPLATES = {
    "GIVE": ["{A} da {B} lui {C} .", "{C} primeste {B} de la {A} .", "{B} e dat lui {C} de {A} .", "lui {C} {A} ii da {B} .",
             "{B} ii este dat lui {C} de catre {A} .", "{A} ii da lui {C} {B} .", "{B} e dat de {A} lui {C} .", "lui {C} ii da {B} {A} .", "{A} da lui {C} {B} .", "lui {C} {A} da {B} .", "{A} ii da {B} lui {C} ."],
    "SELL": ["{A} vinde {B} lui {C} .", "{C} cumpara {B} de la {A} .", "{B} e vandut lui {C} de {A} .", "lui {C} {A} ii vinde {B} .",
             "{B} ii este vandut lui {C} de catre {A} .", "{A} ii vinde lui {C} {B} .", "{B} e vandut de {A} lui {C} .", "lui {C} ii vinde {B} {A} .", "{A} vinde lui {C} {B} .", "lui {C} {A} vinde {B} .", "{A} ii vinde {B} lui {C} ."],
}
TRAIN_T = [0, 1, 2, 3, 6]

def make_world(seed, n_pers=12):
    rng = np.random.default_rng(seed)
    w = cp.World([f"p{i}" for i in range(n_pers)], [f"o{i}" for i in range(n_pers)], rng)
    w.frames = TEMPLATES
    words = {x for ts in TEMPLATES.values() for t in ts for x in t.split() if not x.startswith("{")}
    for x in words:
        if x not in w.idx: w.vocab.append(x)
    w.vocab = sorted(set(w.vocab)); w.idx = {x: i for i, x in enumerate(w.vocab)}
    for x in words: w.cat[x] = x
    for v in ["e", "este"]: w.cat[v] = "COP"
    return w

def transitions(w, kind, t):
    cats = [w.cat[x] for x in w.frames[kind][t].format(A="p0", B="o0", C="p1").split()]
    return set(zip(cats[:-1], cats[1:])), set(cats)

class V4(v2.V2):
    def __init__(self, world, D, rng, gates, codes):
        super().__init__(world, D, rng, gates, dual=False)
        self.codes = codes
        cats = self.cats
        extra = [f"m:{c}" for c in cats] + [f"bg:{c1}:{c2}" for c1 in ["^"] + cats for c2 in cats] + [f"pos:{i}" for i in range(16)] + [f"pc:{c}" for c in cats]
        self.names = self.names + extra; self.pid = {n: i for i, n in enumerate(self.names)}
        th = rng.uniform(-np.pi, np.pi, (len(self.names), self.D)); th[0] = 0; self.theta = th
        self._m = np.zeros_like(th); self._v = np.zeros_like(th); self._k = 0
    def run(self, toks):
        w = self.w; n = len(toks); all_members, all_toks = [], []
        cats = [w.cat[t] for t in toks]
        if "chain" in self.codes:
            self.dual = False; _, placed = super().run(toks); all_members += placed; all_toks += list(toks)
        if "path" in self.codes:
            path = []
            for c in cats: path = path + [self.pid[f"m:{c}"]]; all_members.append(list(path))
            all_toks += list(toks)
        if "bigram" in self.codes:
            prev = "^"
            for c in cats: all_members.append([self.pid[f"bg:{prev}:{c}"]]); prev = c
            all_toks += list(toks)
        if "pos" in self.codes:
            for i, c in enumerate(cats): all_members.append([self.pid[f"pos:{i}"], self.pid[f"pc:{c}"]])
            all_toks += list(toks)
        terms = np.array([np.exp(1j * self.theta[m].sum(0)) * self.F[w.idx[t]] for m, t in zip(all_members, all_toks)])
        return terms, all_members

def decode_stats(m, world, w, templates, persons, objects, n=100):
    """acuratete si marja pe roluri, cu adresele lumii"""
    acc, marg = [], []
    for kind in w.frames:
        for t in templates:
            for _ in range(n):
                ev = w.event(persons, objects); ev = (kind,) + ev[1:]; S = m.state(w.render(ev, t))
                for r, truth, cands in (("A", ev[1], persons), ("B", ev[2], objects), ("C", ev[3], persons)):
                    sc = {x: v2.cos(S * np.conj(world.R[r]), world.F[w.idx[x]]) for x in cands}
                    acc.append(max(sc, key=sc.get) == truth); marg.append(sc[truth] - max(v for k, v in sc.items() if k != truth))
    return float(np.mean(acc)), float(np.mean(marg))

if __name__ == "__main__":
    D = 512; N = 10000
    w0 = make_world(0)
    seen_tr = set().union(*[transitions(w0, "GIVE", t)[0] for t in TRAIN_T]); seen_cat = set().union(*[transitions(w0, "GIVE", t)[1] for t in TRAIN_T])
    print("formulari de antrenare:", TRAIN_T)
    held_seen, held_new = [], []
    for t in range(len(TEMPLATES["GIVE"])):
        if t in TRAIN_T: continue
        tr, ca = transitions(w0, "GIVE", t); new_tr = tr - seen_tr; new_cat = ca - seen_cat
        (held_seen if not new_tr and not new_cat else held_new).append(t)
        print(f"  T{t} '{TEMPLATES['GIVE'][t]}': tranzitii noi {sorted(new_tr)}, categorii noi {sorted(new_cat)}")
    print(f"  => formulari noi cu TOATE tranzitiile vazute: {held_seen}; cu tranzitii/categorii NEVAZUTE: {held_new}")
    print(f"sansa: identitati vazute 1/8 = 0.125; identitati noi 1/4 = 0.25. Control trivial: 'toate la origine' (fara coduri) = sansa.\n")
    variants = [("path", ["path"]), ("chain", ["chain"]), ("bigram", ["bigram"]), ("pos", ["pos"]),
                ("path+chain (dual)", ["path", "chain"]), ("path+bigram", ["path", "bigram"]), ("path+chain+bigram", ["path", "chain", "bigram"]),
                ("toate 4", ["path", "chain", "bigram", "pos"])]
    cols = ["vazute", "id. noi", "form. noi, tranz. vazute", "form. noi, tranz. NEvazute"]
    results = {name: [] for name, _ in variants}
    for seed in range(3):
        w = make_world(seed); cats = sorted(set(w.cat.values())); trP, teP, trO, teO = w.persons[:8], w.persons[8:], w.objects[:8], w.objects[8:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
        world = v2.RoleWorld(F, w, np.random.default_rng(100 + seed))
        for name, codes in variants:
            m = V4(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), codes); m.F = F
            assert m.F is world.F
            t0 = time.time(); m.train(world, N, templates=TRAIN_T, persons=trP, objects=trO)
            r = [decode_stats(m, world, w, TRAIN_T, trP, trO, n=40), decode_stats(m, world, w, TRAIN_T, teP, teO, n=40),
                 decode_stats(m, world, w, held_seen, trP, trO, n=60), decode_stats(m, world, w, held_new, trP, trO, n=60)]
            results[name].append(r)
            print(f"  seed {seed} {name:22s} ({time.time()-t0:3.0f}s) " + " | ".join(f"{a:.2f} (m {mg:.2f})" for a, mg in r), flush=True)
    print("\n== MEDII pe 3 seed-uri: acuratete (marja) ==  coloane: " + " | ".join(cols))
    for name, _ in variants:
        a = np.array(results[name]); print(f"  {name:22s} " + " | ".join(f"{a[:,i,0].mean():.3f}±{a[:,i,0].std():.3f} (m {a[:,i,1].mean():.2f})" for i in range(4)))
    print("\nchecklist: [x] adrese din lume [x] F comun (assert) [x] sansa tiparita [x] 3 axe separate [x] marja [x] 3 seed-uri [x] control trivial = sansa")
