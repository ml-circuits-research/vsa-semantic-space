import numpy as np, camp as cp
rng = np.random.default_rng(1); D = 512
persons = ["ion","ana","dan","maria","radu","ioana","vlad","elena"]; objects = ["carte","mar","pix","bani","ceas","tort","cheie","pahar"]
w = cp.World(persons, objects, rng)
wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
def agglom(vecs, names, thr):
    """uneste clusterele cat timp similaritatea maxima dintre centre depaseste pragul"""
    cl = [[n] for n in names]; cen = [v.copy() for v in vecs]
    while True:
        best, bi, bj = -1, -1, -1
        for i in range(len(cl)):
            for j in range(i+1, len(cl)):
                s = float(np.real(np.vdot(cen[j], cen[i]))) / (np.linalg.norm(cen[i]) * np.linalg.norm(cen[j]))
                if s > best: best, bi, bj = s, i, j
        if best < thr: break
        cl[bi] += cl[bj]; cen[bi] = cen[bi] + cen[bj]; del cl[bj]; del cen[bj]
    return cl
cur_w = w; model = cp.Model(w, D, np.random.default_rng(2), deferred=True, tied=False)
for it in range(3):
    model.train(wm, n_sent=30000, lr=0.02)
    res = model.evaluate(n=100); print(f"iteratia {it}: media decodare = {np.mean([v for a in res.values() for v in a.values()]):.3f}", flush=True)
    # deplasarea fiecarui cuvant = deplasarea categoriei lui curente
    vecs = [np.exp(1j * model.theta[model.pid[f'm:{model.cat_of(x)}']]) for x in w.vocab]
    cl = agglom(vecs, list(w.vocab), thr=0.5)
    print("   clustere:", [c for c in cl if len(c) > 1])
    new_w = cp.World(persons, objects, rng)
    for k, c in enumerate(cl):
        for x in c: new_w.cat[x] = f"K{k}"
    new = cp.Model(new_w, D, np.random.default_rng(2), deferred=True, tied=True)
    # initializeaza deplasarea clusterului cu media membrilor (continuitate), relocarile se pastreaza
    for k, c in enumerate(cl):
        new.theta[new.pid[f"m:K{k}"]] = np.angle(sum(np.exp(1j * model.theta[model.pid[f'm:{model.cat_of(x)}']]) for x in c))
    for v in sorted(w.verbs): new.theta[new.pid[f"r:{v}"]] = model.theta[model.pid[f"r:{v}"]]
    model, cur_w = new, new_w
model.train(wm, n_sent=30000, lr=0.02)
res = model.evaluate(n=150); print(f"\nFINAL (categorii descoperite iterativ): media decodare = {np.mean([v for a in res.values() for v in a.values()]):.3f}")
print(cp.summarize(res, cur_w))
