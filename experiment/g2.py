import numpy as np, camp as cp, sys
rng = np.random.default_rng(1); D = 512
persons = ["ion","ana","dan","maria","radu","ioana","vlad","elena"]; objects = ["carte","mar","pix","bani","ceas","tort","cheie","pahar"]
w = cp.World(persons, objects, rng)
wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
for name, deferred, tied in [("G1 factorizat, fara amanare", False, True), ("G2 factorizat + amanare", True, True), ("G3 ne-factorizat + amanare", True, False)]:
    m = cp.Model(w, D, np.random.default_rng(2), deferred=deferred, tied=tied)
    for stage in range(4):
        m.train(wm, n_sent=10000, lr=0.02)
        res = m.evaluate(n=100); allv=[v for a in res.values() for v in a.values()]
        print(f"{name}: dupa {(stage+1)*10000} propozitii, media decodare = {np.mean(allv):.3f}", flush=True)
    print(cp.summarize(m.evaluate(n=150), w), flush=True)
