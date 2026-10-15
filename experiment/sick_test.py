"""
Primul test pe TEXT REAL: SICK (Sentences Involving Compositional Knowledge), ~10k propozitii din descrieri de imagini,
perechi cu scor de inrudire 1-5 (test: 4927 perechi). Intrebare: geometria "deplasare + bundle" bate sacul de cuvinte pe propozitii reale?
Semnalul de echivalenta: perechile de antrenare cu scor >= 4 (in original provin din aceeasi imagine; aici nu avem imaginile).
Nu exista adresare amanata (nu avem lista de verbe): e varianta G1 a modelului.
"""
import numpy as np, re, time, sys
import camp as cp

def load(path, has_header=True):
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if has_header and i == 0: continue
        p = line.rstrip("\n").split("\t")
        if len(p) < 5: continue
        rows.append((p[1], p[2], float(p[3]), p[4]))
    return rows
def tok(s): return re.findall(r"[a-z]+", s.lower())

train = load("corp/sick.txt") + load("corp/SICK_trial.txt"); test = load("corp/SICK_test_annotated.txt")
sents = sorted({s for r in train + test for s in (r[0], r[1])})
vocab = sorted({w for s in sents for w in tok(s)}); idx = {w: i for i, w in enumerate(vocab)}
print(f"propozitii unice: {len(sents)}, vocabular: {len(vocab)}, perechi antrenare: {len(train)}, test: {len(test)}")

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])

rng = np.random.default_rng(0); D = 512
F = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(vocab), D)))

class Seq:
    """stare = suma_t exp(i * suma_{s<=t} theta[cat(w_s)]) * F[w_t]"""
    def __init__(self, cat_of, n_cats):
        self.cat_of = cat_of; self.theta = rng.uniform(-np.pi, np.pi, (n_cats, D))
        self._m = np.zeros_like(self.theta); self._v = np.zeros_like(self.theta); self._k = 0
    def run(self, toks):
        ids = [idx[w] for w in toks]; cats = [self.cat_of(w) for w in ids]
        cum = np.cumsum(self.theta[cats], axis=0)
        terms = np.exp(1j * cum) * F[ids]
        return terms, cats
    def state(self, toks): return self.run(toks)[0].sum(0)
    def grad_cos(self, terms, members, other):
        S = terms.sum(0); na, nb = np.linalg.norm(S), np.linalg.norm(other); c = np.real(np.vdot(other, S)) / (na * nb)
        dT = 1j * terms; gk = np.real(dT * np.conj(other)[None]) / (na * nb) - c * np.real(dT * np.conj(S)[None]) / na ** 2
        suf = np.cumsum(gk[::-1], axis=0)[::-1]        # cuvantul s participa la toti termenii k >= s
        g = np.zeros_like(self.theta); np.add.at(g, members, suf)
        return c, g
    def adam(self, g, lr):
        self._k += 1; self._m = 0.9 * self._m + 0.1 * g; self._v = 0.999 * self._v + 0.001 * g * g
        self.theta += lr * (self._m / (1 - 0.9 ** self._k)) / (np.sqrt(self._v / (1 - 0.999 ** self._k)) + 1e-8)
    def train_contrastive(self, pairs, n_steps, lr=0.02, tau=0.1, n_neg=6):
        pos = [(a, b) for a, b, s, e in pairs if s >= 4.0]
        for n in range(n_steps):
            a, b = pos[rng.integers(len(pos))]
            cands = [tok(b)] + [tok(sents[rng.integers(len(sents))]) for _ in range(n_neg)]
            ta, ma = self.run(tok(a)); Sa = ta.sum(0)
            runs = [self.run(c) for c in cands]; Ss = [r[0].sum(0) for r in runs]
            sims = np.array([cp.Model.cos(Sa, S) for S in Ss]); p = np.exp(sims / tau - (sims / tau).max()); p /= p.sum()
            g = np.zeros_like(self.theta)
            for k, ((tk, mk), S) in enumerate(zip(runs, Ss)):
                coef = (p[k] - (k == 0)) / tau
                g -= coef * self.grad_cos(ta, ma, S)[1]; g -= coef * self.grad_cos(tk, mk, Sa)[1]
            self.adam(g, lr)

def evaluate(model_state, name):
    sims = np.array([cp.Model.cos(model_state(tok(a)), model_state(tok(b))) for a, b, s, e in test])
    gold = np.array([s for a, b, s, e in test])
    contra = np.array([e == "CONTRADICTION" for a, b, s, e in test]); ent = np.array([e == "ENTAILMENT" for a, b, s, e in test])
    print(f"  {name:48s} Pearson {pearson(sims, gold):.3f}  Spearman {spearman(sims, gold):.3f}  | sim medie: entailment {sims[ent].mean():.3f}, contradictie {sims[contra].mean():.3f}")
    return sims

print("\n== TEST SICK (4927 perechi), corelatie intre cosinusul starilor si scorul uman 1-5 ==")
jac = np.array([len(set(tok(a)) & set(tok(b))) / len(set(tok(a)) | set(tok(b))) for a, b, s, e in test]); gold = np.array([s for a, b, s, e in test])
print(f"  {'Jaccard (suprapunere de cuvinte)':48s} Pearson {pearson(jac, gold):.3f}  Spearman {spearman(jac, gold):.3f}")
evaluate(lambda t: sum(F[idx[w]] for w in t), "sac de cuvinte (bundle fara deplasari)")
m_rand = Seq(lambda w: w, len(vocab)); evaluate(m_rand.state, "deplasari aleatoare per cuvant, neantrenate")

t0 = time.time(); m1 = Seq(lambda w: w, len(vocab)); m1.train_contrastive(train, 12000)
print(f"  [antrenat {time.time()-t0:.0f}s]"); evaluate(m1.state, "deplasari per cuvant, antrenate (echivalenta)")

# categorii descoperite: clustering aglomerativ al deplasarilor invatate, apoi legare si reantrenare
vec = np.exp(1j * m1.theta); counts = np.zeros(len(vocab))
for s in sents:
    for w in tok(s): counts[idx[w]] += 1
order = np.argsort(-counts)[:400]            # clustering doar pe cele mai frecvente 400 de cuvinte (restul: categorie proprie)
X = np.concatenate([np.cos(m1.theta[order]), np.sin(m1.theta[order])], 1); X /= np.linalg.norm(X, axis=1, keepdims=True)
cl = [[i] for i in range(len(order))]; cen = [X[i].copy() for i in range(len(order))]
while True:
    C = np.array([c / np.linalg.norm(c) for c in cen]); S = C @ C.T; np.fill_diagonal(S, -1)
    i, j = np.unravel_index(S.argmax(), S.shape)
    if S[i, j] < 0.35 or len(cl) <= 8: break
    cl[i] += cl[j]; cen[i] = cen[i] + cen[j]; del cl[j]; del cen[j]
cat = {}; ncat = 0
for c in cl:
    for i in c: cat[int(order[i])] = ncat
    ncat += 1
for w in range(len(vocab)):
    if w not in cat: cat[w] = ncat; ncat += 1
big = sorted([c for c in cl if len(c) > 2], key=len, reverse=True)
print(f"\n  clustere descoperite (din 400 cuvinte frecvente): {len(cl)}; cele mai mari:")
for c in big[:8]: print("    ", [vocab[order[i]] for i in c][:14])
m2 = Seq(lambda w: cat[w], ncat)
for c in cl: m2.theta[cat[int(order[c[0]])]] = np.angle(sum(vec[order[i]] for i in c))
t0 = time.time(); m2.train_contrastive(train, 12000)
print(f"  [antrenat {time.time()-t0:.0f}s]"); evaluate(m2.state, "deplasari pe categorii descoperite, antrenate")
