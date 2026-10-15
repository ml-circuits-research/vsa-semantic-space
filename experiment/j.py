import numpy as np, camp as cp, sys
def make(n_pers, D, seed=1, adj=False):
    rng = np.random.default_rng(seed)
    persons = [f"p{i}" for i in range(n_pers)]; objects = [f"o{i}" for i in range(n_pers)]
    w = cp.World(persons, objects, rng, extra_adjuncts=adj)
    # categoriile descoperite in I: persoane, obiecte, verbe finite, participii, copule, restul individual
    for v in ["da","vinde","primeste","cumpara"]: w.cat[v] = "VF"
    for v in ["dat","vandut"]: w.cat[v] = "VP"
    for v in ["e","este"]: w.cat[v] = "COP"
    wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
    m = cp.Model(w, D, np.random.default_rng(2), deferred=True, tied=True)
    return w, wm, m
def mean_acc(res): return np.mean([v for a in res.values() for v in a.values()])

print("J1: dimensiunea D si numarul de identitati (sansa = 1/n)")
for n_pers in [8, 24, 64]:
    for D in [256, 512, 1024]:
        w, wm, m = make(n_pers, D)
        m.train(wm, n_sent=30000, lr=0.02)
        print(f"   {n_pers} persoane/obiecte, D={D}: media decodare = {mean_acc(m.evaluate(n=80)):.3f}", flush=True)

print("\nJ2: zgomot din adjuncte (adverbe inserate aleator), 16 identitati, D=512; antrenat cu 0-2 adjuncte")
w, wm, m = make(16, 512, adj=True)
for stage in range(3):
    m.train(wm, n_sent=12000, lr=0.02, n_adj=int(stage))   # curriculum: 0, 1, 2 adjuncte
for n_adj in [0, 1, 2, 3, 5]:
    print(f"   {n_adj} adjuncte inserate: media decodare = {mean_acc(m.evaluate(n=80, n_adj=n_adj)):.3f}", flush=True)
