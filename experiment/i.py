import numpy as np, camp as cp
rng = np.random.default_rng(1); D = 512
persons = ["ion","ana","dan","maria","radu","ioana","vlad","elena"]; objects = ["carte","mar","pix","bani","ceas","tort","cheie","pahar"]
w = cp.World(persons, objects, rng)
wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)

# pasul 1: model NE-factorizat (deplasare per cuvant), invatat generativ
m3 = cp.Model(w, D, np.random.default_rng(2), deferred=True, tied=False)
m3.train(wm, n_sent=40000, lr=0.02)
res = m3.evaluate(n=100); print(f"I1: ne-factorizat invatat, media decodare = {np.mean([v for a in res.values() for v in a.values()]):.3f}")

# pasul 2: similaritatea deplasarilor invatate: persoanele seamana intre ele? obiectele?
def mvec(word): return np.exp(1j * m3.theta[m3.pid[f"m:{word}"]])
def sim(a, b): return float(np.real(np.vdot(mvec(b), mvec(a))) / D)
PP = np.mean([sim(a, b) for i, a in enumerate(persons) for b in persons[i+1:]])
OO = np.mean([sim(a, b) for i, a in enumerate(objects) for b in objects[i+1:]])
PO = np.mean([sim(a, b) for a in persons for b in objects])
fw = [x for x in w.function_words if x not in w.verbs]
PF = np.mean([sim(a, b) for a in persons for b in fw])
print(f"I2: similaritate medie a deplasarilor: persoana-persoana {PP:.3f}, obiect-obiect {OO:.3f}, persoana-obiect {PO:.3f}, persoana-functional {PF:.3f}")

# pasul 3: clustering nesupervizat al deplasarilor (k-means pe [cos, sin]), k = nr. de categorii reale
words = list(w.vocab); X = np.concatenate([np.cos(m3.theta[[m3.pid[f"m:{x}"] for x in words]]), np.sin(m3.theta[[m3.pid[f"m:{x}"] for x in words]])], 1)
k = len(set(w.cat.values()))
def kmeans(X, k, iters=50, seed=0):
    r = np.random.default_rng(seed); C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = np.argmax(X @ C.T, 1)
        for j in range(k):
            if (lab == j).any(): C[j] = X[lab == j].mean(0); C[j] /= np.linalg.norm(C[j]) + 1e-9
    return lab
lab = kmeans(X, k)
clusters = {}
for x, l in zip(words, lab): clusters.setdefault(int(l), []).append(x)
print("I3: clustere descoperite:"); 
for l, ws in clusters.items(): print("   ", ws)
# puritate: pentru fiecare cluster, fractia celei mai frecvente categorii reale
pur = np.mean([max(sum(w.cat[x] == c for x in ws) for c in set(w.cat[x] for x in ws)) for ws in clusters.values()])
print(f"    puritate medie = {pur:.2f}")

# pasul 4: legam deplasarile dupa clusterele descoperite si reantrenam de la zero
w2 = cp.World(persons, objects, rng)
for l, ws in clusters.items():
    for x in ws: w2.cat[x] = f"K{l}"
m4 = cp.Model(w2, D, np.random.default_rng(2), deferred=True, tied=True)
m4.train(wm, n_sent=40000, lr=0.02)
res = m4.evaluate(n=150); print(f"\nI4: factorizat cu categorii DESCOPERITE, media decodare = {np.mean([v for a in res.values() for v in a.values()]):.3f}")
print(cp.summarize(res, w2))
