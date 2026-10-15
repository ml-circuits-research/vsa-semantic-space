"""
Campanie de experimente: pasul uniform + factorizare (ce este / ce face) + adresare amanata.
Model general:  starea unei propozitii = suma de termeni  term_k = exp(i * theta_k) * f_k
                 theta_k = suma fazelor parametrilor care participa la termenul k (deplasari + relocari)
Invatare: doar eroare de predictie generativa:  maximizeaza cos( S * T , S_lume(consecinta) ), T fix.
"""
import numpy as np, sys, json
from collections import defaultdict

class World:
    """lume sintetica cu cadre (verbe), roluri si formulari (template-uri)"""
    def __init__(self, persons, objects, rng, extra_adjuncts=False):
        self.persons, self.objects, self.rng = list(persons), list(objects), rng
        # cadre: nume -> (consecinta, formulari). Roluri: A agent, B tema, C destinatar
        self.frames = {
            "GIVE": ["{A} da {B} lui {C} .", "{C} primeste {B} de la {A} .", "{B} e dat lui {C} de {A} .",
                     "lui {C} {A} ii da {B} .", "{B} ii este dat lui {C} de catre {A} ."],
            "SELL": ["{A} vinde {B} lui {C} .", "{C} cumpara {B} de la {A} .", "{B} e vandut lui {C} de {A} .",
                     "lui {C} {A} ii vinde {B} .", "{B} ii este vandut lui {C} de catre {A} ."],
        }
        self.consequence = "{C} are {B} . {A} nu are {B} ."
        self.adjuncts = ["azi", "repede", "acum", "iar", "din nou"] if extra_adjuncts else []
        words = set()
        for ts in self.frames.values():
            for t in ts: words |= {w for w in t.split() if not w.startswith("{")}
        words |= {w for w in self.consequence.split() if not w.startswith("{")}
        self.function_words = sorted(words)
        self.vocab = sorted(set(self.persons) | set(self.objects) | words | set(self.adjuncts))
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.verbs = {"da", "primeste", "vinde", "cumpara", "dat", "vandut", "are"}
        self.cat = {}
        for w in self.persons: self.cat[w] = "P"
        for w in self.objects: self.cat[w] = "O"
        for w in self.adjuncts: self.cat[w] = "ADV"
        for w in words: self.cat[w] = w          # cuvintele functionale/verbele: categorie proprie

    def event(self, persons=None, objects=None):
        P = persons or self.persons; O = objects or self.objects
        A, C = self.rng.choice(P, 2, replace=False); B = self.rng.choice(O)
        return (self.rng.choice(list(self.frames)), A, B, C)

    def render(self, ev, t, n_adj=0):
        kind, A, B, C = ev
        toks = self.frames[kind][t].format(A=A, B=B, C=C).split()
        for _ in range(n_adj):                    # adjuncte inserate la pozitii aleatoare (nu la final)
            pos = self.rng.integers(1, len(toks) - 1)
            toks.insert(pos, self.rng.choice(self.adjuncts))
        return toks

    def roles(self, ev, toks):
        kind, A, B, C = ev
        return {"A": toks.index(A), "B": toks.index(B), "C": toks.index(C)}

    def cons(self, ev):
        kind, A, B, C = ev
        return self.consequence.format(A=A, B=B, C=C).split()


class Model:
    """parametri: deplasare pe categorie (ce face), trasatura pe cuvant (ce este), relocare per verb"""
    def __init__(self, world, D, rng, deferred=True, tied=True):
        self.w, self.D, self.rng, self.deferred = world, D, rng, deferred
        cats = sorted(set(world.cat.values())) if tied else list(world.vocab)
        self.tied = tied
        self.cat_of = (lambda w: world.cat[w]) if tied else (lambda w: w)
        self.names = [f"m:{c}" for c in cats] + [f"r:{v}" for v in sorted(world.verbs)]
        self.pid = {n: i for i, n in enumerate(self.names)}
        self.theta = rng.uniform(-np.pi, np.pi, (len(self.names), D))   # parametri de faza (invatati)
        self.F = np.exp(1j * rng.uniform(-np.pi, np.pi, (len(world.vocab), D)))  # trasaturi (fixe)
        self.T = np.exp(1j * rng.uniform(-np.pi, np.pi, D))             # operatia fixa a nivelului lent

    def run(self, toks, theta=None):
        """intoarce terms (K x D), members (lista de liste de param-id), si pos2term (pozitie -> indice termen)"""
        theta = self.theta if theta is None else theta
        w = self.w
        path = []          # param-ids acumulati in locatie (integrarea de drum)
        terms, members, pos2term = [], [], {}
        buffer = []        # (pozitie, cuvant) dinaintea primului verb, daca deferred
        flushed = not self.deferred
        for p, tok in enumerate(toks):
            is_verb = tok in w.verbs
            if is_verb and not flushed:
                for (bp, bw, rel) in buffer:     # cuvintele amanate se scriu la adresa verbului + drumul relativ din buffer
                    mem = [self.pid[f"r:{tok}"]] + rel
                    terms.append(np.exp(1j * theta[mem].sum(0)) * self.F[w.idx[bw]]); members.append(mem); pos2term[bp] = len(terms) - 1
                flushed = True
            if not flushed and not is_verb:
                buffer.append((p, tok, list(path)))
                path = path + [self.pid[f"m:{self.cat_of(tok)}"]]
                continue
            path = path + [self.pid[f"m:{self.cat_of(tok)}"]]
            mem = list(path)
            terms.append(np.exp(1j * theta[mem].sum(0)) * self.F[w.idx[tok]]); members.append(mem); pos2term[p] = len(terms) - 1
        if not flushed:                            # propozitie fara verb: scrie bufferul la adresa curenta
            for (bp, bw, rel) in buffer:
                mem = list(path); terms.append(np.exp(1j * theta[mem].sum(0)) * self.F[w.idx[bw]]); members.append(mem); pos2term[bp] = len(terms) - 1
        return np.array(terms), members, pos2term

    def state(self, toks, theta=None):
        return self.run(toks, theta)[0].sum(0)

    @staticmethod
    def cos(a, b):
        return float(np.real(np.vdot(b, a)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def grad_cos(self, terms, members, other):
        """d cos(S, other)/d theta pentru S = sum terms; other fixat"""
        S = terms.sum(0); na, nb = np.linalg.norm(S), np.linalg.norm(other)
        c = np.real(np.vdot(other, S)) / (na * nb)
        dT = 1j * terms
        gk = np.real(dT * np.conj(other)[None, :]) / (na * nb) - c * np.real(dT * np.conj(S)[None, :]) / na**2
        grad = np.zeros_like(self.theta)
        for k, mem in enumerate(members):
            for pidx in mem: grad[pidx] += gk[k]
        return c, grad

    def train(self, world_model, n_sent=12000, lr=0.02, n_adj=0, persons=None, objects=None, log_every=0):
        """invatare generativa: starea propozitiei * T trebuie sa fie starea (senzoriala, fixa) a consecintei"""
        w = self.w
        if not hasattr(self, "_m"): self._m = np.zeros_like(self.theta); self._v = np.zeros_like(self.theta); self._k = 0
        b1, b2 = 0.9, 0.999
        for n in range(n_sent):
            ev = w.event(persons, objects)
            t = self.rng.integers(len(w.frames[ev[0]]))
            toks = w.render(ev, t, n_adj=n_adj)
            terms, members, _ = self.run(toks)
            Sc = world_model.state(w.cons(ev))
            c, g = self.grad_cos(terms, members, Sc * np.conj(self.T))
            self._k += 1; self._m = b1 * self._m + (1 - b1) * g; self._v = b2 * self._v + (1 - b2) * g * g
            mh = self._m / (1 - b1 ** self._k); vh = self._v / (1 - b2 ** self._k)
            self.theta += lr * mh / (np.sqrt(vh) + 1e-8)

    # ---------- evaluare: decodare roluri din orice formulare, cu adresele formei active (template 0) ----------
    def role_addresses(self, kind="GIVE"):
        w = self.w
        ev = (kind, w.persons[0], w.objects[0], w.persons[1])
        toks = w.render(ev, 0); terms, members, p2t = self.run(toks)
        r = w.roles(ev, toks)
        return {role: terms[p2t[r[role]]] * np.conj(self.F[w.idx[toks[r[role]]]]) for role in "ABC"}

    def decode(self, S, addr, cands):
        read = S * np.conj(addr)
        return max(cands, key=lambda c: self.cos(read, self.F[self.w.idx[c]]))

    def evaluate(self, n=200, n_adj=0, persons=None, objects=None):
        w = self.w; P = persons or w.persons; O = objects or w.objects
        out = {}
        for kind in w.frames:
            addr = self.role_addresses(kind)
            for t in range(len(w.frames[kind])):
                acc = {"A": 0, "B": 0, "C": 0}
                for _ in range(n):
                    ev = w.event(P, O); ev = (kind,) + ev[1:]
                    S = self.state(w.render(ev, t, n_adj=n_adj))
                    acc["A"] += self.decode(S, addr["A"], P) == ev[1]
                    acc["B"] += self.decode(S, addr["B"], O) == ev[2]
                    acc["C"] += self.decode(S, addr["C"], P) == ev[3]
                out[(kind, t)] = {k: v / n for k, v in acc.items()}
        return out

def summarize(res, world):
    rows = []
    for (kind, t), acc in res.items():
        rows.append(f"  {kind} T{t} '{world.frames[kind][t]}': agent {acc['A']:.2f} obiect {acc['B']:.2f} dest {acc['C']:.2f}")
    allv = [v for acc in res.values() for v in acc.values()]
    return "\n".join(rows) + f"\n  MEDIA pe toate formularile si rolurile: {np.mean(allv):.3f}"


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    D = 512
    persons = ["ion", "ana", "dan", "maria", "radu", "ioana", "vlad", "elena"]
    objects = ["carte", "mar", "pix", "bani", "ceas", "tort", "cheie", "pahar"]
    w = World(persons, objects, rng)
    world_model = Model(w, D, np.random.default_rng(99), deferred=False, tied=True)   # codul "senzorial" al lumii, fix

    print("=== G0: factorizat, FARA adresare amanata, fara invatare (sansa agent/dest 1/8, obiect 1/8)")
    m = Model(w, D, np.random.default_rng(2), deferred=False); print(summarize(m.evaluate(), w))
    print("\n=== G1: factorizat, FARA adresare amanata, invatat")
    m.train(world_model); print(summarize(m.evaluate(), w))
    print("\n=== G2: factorizat + adresare amanata (subiectul se scrie la verb), invatat")
    m2 = Model(w, D, np.random.default_rng(2), deferred=True); m2.train(world_model); print(summarize(m2.evaluate(), w))
    print("\n=== G3: ca G2, dar NE-factorizat (deplasare per cuvant), invatat")
    m3 = Model(w, D, np.random.default_rng(2), deferred=True, tied=False); m3.train(world_model); print(summarize(m3.evaluate(), w))
