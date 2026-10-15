"""
v2: adresare prin ASTEPTARI ("prezice, apoi lasa cuvantul sa-si gaseasca locul").

Pasul, identic pentru orice cuvant w (categoria c):
  1. daca exista o asteptare neconsumata pentru categoria c -> cuvantul o ia (cea mai recenta); adresa lui = adresa asteptarii
     altfel -> adresa provizorie r_c si intra in buffer (superpozitie de locatii nerezolvate; poate fi mutat cat timp nimic
     nu a fost plasat relativ la el)
  2. depune: S += adresa (x) identitate(w)
  3. daca w e EMITATOR (poarta g_c = 1): emite o asteptare pentru fiecare categorie c2 la adresa(w) (x) A[w, c2]
     cuvintele din buffer de categoria c2 iau noua asteptare (se muta)
Parametri invatati: r_c (adresa provizorie pe categorie), A[w, c2] (adrese relative), portile g_c (discrete, cautare).
Nu exista: deplasari pe drum, verbe speciale, 'ca' special, stiva. Toata structura vine din asteptari.

Lumea: roluri explicite: S_lume = sum_rol R_rol (x) identitate(umplutura). Obiectiv: cos(S, S_lume) (generativ, fara T).
Evaluare: decodare cu ADRESELE LUMII (R_rol), pe toate formularile, fara calibrare pe forma activa.
"""
import numpy as np, time, sys
import camp as cp

class V2:
    def __init__(self, world, D, rng, gates, additive=False, dual=False):
        self.w, self.D, self.rng, self.additive, self.dual = world, D, rng, additive, dual
        self.gates = dict(gates)                       # categorie -> bool (emitator)
        cats = sorted(set(world.cat[x] for x in world.vocab))
        self.names = ["root"] + ([f"out:{c}" for c in cats] + [f"in:{c}" for c in cats] if additive else [f"A:{c1}:{c}" for c1 in cats for c in cats]) + [f"m:{c}" for c in cats]
        self.pid = {n: i for i, n in enumerate(self.names)}
        self.theta = rng.uniform(-np.pi, np.pi, (len(self.names), D)); self.theta[0] = 0
        self.F = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(world.vocab), D)))
        self._m = np.zeros_like(self.theta); self._v = np.zeros_like(self.theta); self._k = 0
        self.cats = cats

    def run(self, toks):
        w = self.w; pool = []; placed = [None] * len(toks); emitted = {}; buffer = []
        for p, tok in enumerate(toks):
            c = w.cat[tok]
            e = next((e for e in reversed(pool) if e["cat"] == c and not e["used"]), None)
            if e is not None:
                e["used"] = True; placed[p] = list(e["members"])
            else:
                placed[p] = [self.pid["root"]]; buffer.append(p)
            if self.gates.get(c, False):
                emitted[p] = []
                for c2 in self.cats:
                    rel = [self.pid[f"out:{c}"], self.pid[f"in:{c2}"]] if self.additive else [self.pid[f"A:{c}:{c2}"]]
                    ent = {"cat": c2, "members": placed[p] + rel, "used": False, "emitter": p}
                    pool.append(ent); emitted[p].append(ent)
                # cuvintele din buffer se muta pe noile asteptari (in ordinea sosirii)
                for q in list(buffer):
                    if q == p: continue
                    if emitted.get(q) and any(x["used"] for x in emitted[q]): continue   # ceva a fost plasat relativ la q
                    ent = next((x for x in emitted[p] if x["cat"] == w.cat[toks[q]] and not x["used"]), None)
                    if ent is None: continue
                    ent["used"] = True; placed[q] = list(ent["members"]); buffer.remove(q)
                    for x in emitted.get(q, []): x["members"] = placed[q] + x["members"][-(2 if self.additive else 1):]
        if self.dual:                                  # adresa = lant de asteptari (+) drum pe categorii (grosier, aditiv)
            path = []
            for p, tok in enumerate(toks):
                path = path + [self.pid[f"m:{w.cat[tok]}"]]; placed.append(list(path))
            toks = list(toks) + list(toks)
        terms = np.array([np.exp(1j * self.theta[m].sum(0)) * self.F[w.idx[t]] for m, t in zip(placed, toks)])
        return terms, placed

    def state(self, toks): return self.run(toks)[0].sum(0)

    def grad_cos(self, terms, members, other):
        S = terms.sum(0); na, nb = np.linalg.norm(S), np.linalg.norm(other); c = np.real(np.vdot(other, S)) / (na * nb)
        dT = 1j * terms; gk = np.real(dT * np.conj(other)[None]) / (na * nb) - c * np.real(dT * np.conj(S)[None]) / na ** 2
        g = np.zeros_like(self.theta)
        for k, mem in enumerate(members):
            for pidx in mem: g[pidx] += gk[k]
        return c, g

    def train(self, world, n_sent, lr=0.02, templates=None, persons=None, objects=None):
        w = self.w
        for n in range(n_sent):
            ev = w.event(persons, objects); ts = templates or range(len(w.frames[ev[0]]))
            toks = w.render(ev, self.rng.choice(list(ts)))
            terms, members = self.run(toks)
            c, g = self.grad_cos(terms, members, world.state(ev))
            self._k += 1; self._m = 0.9 * self._m + 0.1 * g; self._v = 0.999 * self._v + 0.001 * g * g
            self.theta += lr * (self._m / (1 - 0.9 ** self._k)) / (np.sqrt(self._v / (1 - 0.999 ** self._k)) + 1e-8); self.theta[0] = 0

class RoleWorld:
    """lumea: roluri explicite, identitati impartite cu modelul"""
    def __init__(self, F, world, rng, roles=("A", "B", "C")):
        self.F, self.w = F, world; self.R = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, F.shape[1])) for r in roles}
    def state(self, ev):
        kind, A, B, C = ev[:4]; S = self.R["A"] * self.F[self.w.idx[A]] + self.R["B"] * self.F[self.w.idx[B]] + self.R["C"] * self.F[self.w.idx[C]]
        if "X" in self.R: S = S + self.R["X"] * self.F[self.w.idx[ev[4]]]
        return S

def cos(a, b): return float(np.real(np.vdot(b, a)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def evaluate(predict, world, w, n=100, templates=None, persons=None, objects=None):
    """decodare cu adresele lumii; intoarce {(kind,t): {A,B,C}} si media"""
    P = persons or w.persons; O = objects or w.objects; out = {}
    for kind in w.frames:
        for t in (templates or range(len(w.frames[kind]))):
            acc = {"A": 0, "B": 0, "C": 0}
            for _ in range(n):
                ev = w.event(P, O); ev = (kind,) + ev[1:]
                S = predict(w.render(ev, t))
                for r, truth, cands in (("A", ev[1], P), ("B", ev[2], O), ("C", ev[3], P)):
                    acc[r] += max(cands, key=lambda x: cos(S * np.conj(world.R[r]), world.F[w.idx[x]])) == truth
            out[(kind, t)] = {k: v / n for k, v in acc.items()}
    return out, float(np.mean([v for a in out.values() for v in a.values()]))

def pred_cos(predict, world, w, n=200, templates=None):
    cs = []
    for _ in range(n):
        ev = w.event(); ts = templates or range(len(w.frames[ev[0]])); cs.append(cos(predict(w.render(ev, w.rng.choice(list(ts)))), world.state(ev)))
    return float(np.mean(cs))

def make_world(seed, n_pers=12):
    rng = np.random.default_rng(seed)
    persons = [f"p{i}" for i in range(n_pers)]; objects = [f"o{i}" for i in range(n_pers)]
    w = cp.World(persons, objects, rng)
    for v in ["e", "este"]: w.cat[v] = "COP"      # verbele si cuvintele functionale raman categorii individuale
    return w

HAND_GATES = lambda cats: {c: (c not in ("P", "O")) for c in cats}     # emitatori: tot ce nu e identitate

if __name__ == "__main__":
    D = 512; N = 12000
    print("=== v2 (asteptari) vs v1 (drum + buffer declansat de verb), lume cu roluri explicite, decodare cu adresele lumii ===")
    print("    antrenare: formularile 0-3, 8 persoane/8 obiecte; test: formulari 0-3 si 4 (nevazuta), identitati vazute/nevazute (sansa 1/8, 1/4)")
    rows = {"v1": [], "v2 porti de mana": [], "v2 toate portile deschise": [], "v2 toate inchise (sac)": []}
    for seed in range(3):
        w = make_world(seed); rng = np.random.default_rng(100 + seed)
        trP, teP, trO, teO = w.persons[:8], w.persons[8:], w.objects[:8], w.objects[8:]
        cats = sorted(set(w.cat.values()))
        # ---- v1 ----
        v1 = cp.Model(w, D, rng, deferred=True, tied=True); world = RoleWorld(v1.F, w, rng)
        t0 = time.time()
        for n in range(N):
            ev = w.event(trP, trO); toks = w.render(ev, rng.choice([0, 1, 2, 3]))
            terms, members, _ = v1.run(toks); _, g = v1.grad_cos(terms, members, world.state(ev) * np.conj(v1.T))
            if not hasattr(v1, "_m"): v1._m = np.zeros_like(v1.theta); v1._v = np.zeros_like(v1.theta); v1._k = 0
            v1._k += 1; v1._m = 0.9 * v1._m + 0.1 * g; v1._v = 0.999 * v1._v + 0.001 * g * g
            v1.theta += 0.02 * (v1._m / (1 - 0.9 ** v1._k)) / (np.sqrt(v1._v / (1 - 0.999 ** v1._k)) + 1e-8)
        p1 = lambda toks: v1.state(toks) * v1.T
        r = [evaluate(p1, world, w, templates=[0,1,2,3], persons=trP, objects=trO)[1], evaluate(p1, world, w, templates=[0,1,2,3], persons=teP, objects=teO)[1],
             evaluate(p1, world, w, templates=[4], persons=trP, objects=trO)[1], evaluate(p1, world, w, templates=[4], persons=teP, objects=teO)[1]]
        rows["v1"].append(r); print(f"  seed {seed} v1 ({time.time()-t0:.0f}s): {[round(x,3) for x in r]}", flush=True)
        # ---- v2 variants ----
        for name, gates in [("v2 porti de mana", HAND_GATES(cats)), ("v2 toate portile deschise", {c: True for c in cats}), ("v2 toate inchise (sac)", {c: False for c in cats})]:
            m = V2(w, D, np.random.default_rng(200 + seed), gates); m.F = v1.F
            t0 = time.time(); m.train(world, N, templates=[0, 1, 2, 3], persons=trP, objects=trO)
            r = [evaluate(m.state, world, w, templates=[0,1,2,3], persons=trP, objects=trO)[1], evaluate(m.state, world, w, templates=[0,1,2,3], persons=teP, objects=teO)[1],
                 evaluate(m.state, world, w, templates=[4], persons=trP, objects=trO)[1], evaluate(m.state, world, w, templates=[4], persons=teP, objects=teO)[1]]
            rows[name].append(r); print(f"  seed {seed} {name} ({time.time()-t0:.0f}s): {[round(x,3) for x in r]}", flush=True)
    print("\n  MEDII pe 3 seed-uri  [vazute | identitati noi | formulare noua | ambele noi]")
    for k, v in rows.items():
        a = np.array(v); print(f"    {k:28s} " + "  ".join(f"{m:.3f}±{s:.3f}" for m, s in zip(a.mean(0), a.std(0))))
    # detaliu pe formulari pentru v2 porti de mana, seed 0
    w = make_world(0); cats = sorted(set(w.cat.values())); rng = np.random.default_rng(100)
    m = V2(w, D, rng, HAND_GATES(cats)); world = RoleWorld(m.F, w, rng); m.train(world, N, templates=[0,1,2,3], persons=w.persons[:8], objects=w.objects[:8])
    res, mean = evaluate(m.state, world, w, n=150)
    print("\n  v2 porti de mana, seed 0, pe formulari (T4 nevazuta la antrenare):")
    for (kind, t), a in res.items(): print(f"    {kind} T{t} '{w.frames[kind][t]}': agent {a['A']:.2f} obiect {a['B']:.2f} dest {a['C']:.2f}")
