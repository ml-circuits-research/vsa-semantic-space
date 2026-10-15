"""K: compozitie ierarhica (fibrat de fibrat). "X spune ca <propozitie>": subordonatorul "ca" deschide un cadru nou:
drumul interior porneste de la zero RELATIV la ancora (adresa de la "ca"), iar bufferul se reactiveaza.
Adresa unui cuvant interior = ancora (x) drum_interior. Asta e zoom-in; starea ramane un singur bundle."""
import numpy as np, camp as cp

class NestedWorld(cp.World):
    def __init__(self, persons, objects, rng):
        super().__init__(persons, objects, rng)
        self.outer = ["{X} spune ca", "{X} zice ca", "lui {Y} {X} ii spune ca"]
        for wd in ["spune", "zice", "ca", "ii"]:
            if wd not in self.vocab: self.vocab.append(wd)
        self.vocab = sorted(set(self.vocab)); self.idx = {w: i for i, w in enumerate(self.vocab)}
        self.verbs |= {"spune", "zice"}
        for wd in ["spune", "zice", "ca"]: self.cat[wd] = wd
        self.cat["ii"] = "ii"
        for v in ["da", "vinde", "primeste", "cumpara"]: self.cat[v] = "VF"
        for v in ["dat", "vandut"]: self.cat[v] = "VP"
        for v in ["e", "este"]: self.cat[v] = "COP"
    def event(self, persons=None, objects=None):
        P = persons or self.persons; O = objects or self.objects
        X, Y, A, C = self.rng.choice(P, 4, replace=False); B = self.rng.choice(O)
        return (self.rng.choice(list(self.frames)), A, B, C, X, Y)
    def render(self, ev, t, n_adj=0, to=None):
        kind, A, B, C, X, Y = ev
        to = self.rng.integers(len(self.outer)) if to is None else to
        inner = self.frames[kind][t].format(A=A, B=B, C=C)
        return (self.outer[to].format(X=X, Y=Y) + " " + inner).split()
    def roles(self, ev, toks):
        kind, A, B, C, X, Y = ev
        return {"A": toks.index(A), "B": toks.index(B), "C": toks.index(C), "X": toks.index(X)}
    def cons(self, ev):
        kind, A, B, C, X, Y = ev
        return f"{X} spune . {C} are {B} . {A} nu are {B} .".split()

class NestedModel(cp.Model):
    def run(self, toks, theta=None):
        theta = self.theta if theta is None else theta
        w = self.w
        anchor = []; path = []; terms, members, pos2term = [], [], {}
        buffer = []; flushed = not self.deferred
        def write(p, tok, mem):
            terms.append(np.exp(1j * theta[mem].sum(0)) * self.F[w.idx[tok]]); members.append(mem); pos2term[p] = len(terms) - 1
        for p, tok in enumerate(toks):
            if tok == "ca":                        # zoom in: cadru nou, relativ la ancora
                path = path + [self.pid[f"m:{self.cat_of(tok)}"]]
                write(p, tok, anchor + path)
                anchor = anchor + path; path = []; buffer = []; flushed = not self.deferred
                continue
            is_verb = tok in w.verbs
            if is_verb and not flushed:
                for (bp, bw, rel) in buffer: write(bp, bw, anchor + [self.pid[f"r:{tok}"]] + rel)
                flushed = True
            if not flushed and not is_verb:
                buffer.append((p, tok, list(path))); path = path + [self.pid[f"m:{self.cat_of(tok)}"]]; continue
            path = path + [self.pid[f"m:{self.cat_of(tok)}"]]
            write(p, tok, anchor + path)
        if not flushed:
            for (bp, bw, rel) in buffer: write(bp, bw, anchor + path)
        return np.array(terms), members, pos2term

    def role_addresses(self, kind="GIVE"):
        w = self.w
        ev = (kind, w.persons[0], w.objects[0], w.persons[1], w.persons[2], w.persons[3])
        toks = w.render(ev, 0, to=0); terms, members, p2t = self.run(toks); r = w.roles(ev, toks)
        return {role: terms[p2t[r[role]]] * np.conj(self.F[w.idx[toks[r[role]]]]) for role in "ABCX"}

    def evaluate(self, n=100, **kw):
        w = self.w; out = {}
        for kind in w.frames:
            addr = self.role_addresses(kind)
            for to in range(len(w.outer)):
                for t in range(len(w.frames[kind])):
                    acc = {"A": 0, "B": 0, "C": 0, "X": 0}
                    for _ in range(n):
                        ev = w.event(); ev = (kind,) + ev[1:]
                        S = self.state(w.render(ev, t, to=to))
                        acc["A"] += self.decode(S, addr["A"], w.persons) == ev[1]
                        acc["B"] += self.decode(S, addr["B"], w.objects) == ev[2]
                        acc["C"] += self.decode(S, addr["C"], w.persons) == ev[3]
                        acc["X"] += self.decode(S, addr["X"], w.persons) == ev[4]
                    out[(kind, to, t)] = {k: v / n for k, v in acc.items()}
        return out

if __name__ == "__main__":
    rng = np.random.default_rng(1); D = 1024
    persons = ["ion", "ana", "dan", "maria", "radu", "ioana", "vlad", "elena"]; objects = ["carte", "mar", "pix", "bani", "ceas", "tort", "cheie", "pahar"]
    w = NestedWorld(persons, objects, rng)
    wm = NestedModel(w, D, np.random.default_rng(99), deferred=False)
    m = NestedModel(w, D, np.random.default_rng(2), deferred=True)
    for stage in range(4):
        m.train(wm, n_sent=10000, lr=0.02)
        res = m.evaluate(n=40); print(f"K: dupa {(stage+1)*10000} propozitii, media = {np.mean([v for a in res.values() for v in a.values()]):.3f}", flush=True)
    res = m.evaluate(n=80)
    print("\nK: decodare pe 2 cadre x 3 forme exterioare x 5 forme interioare (sansa 1/8):")
    byrole = {k: np.mean([a[k] for a in res.values()]) for k in "ABCX"}
    print("   medii pe rol: agent %.2f obiect %.2f destinatar %.2f vorbitor(X) %.2f" % (byrole["A"], byrole["B"], byrole["C"], byrole["X"]))
    for to in range(3):
        v = [a for (k, o, t), a in res.items() if o == to]
        print(f"   forma exterioara '{w.outer[to]}': media {np.mean([x for a in v for x in a.values()]):.2f}")
    worst = sorted(res.items(), key=lambda kv: np.mean(list(kv[1].values())))[:3]
    for (k, o, t), a in worst: print(f"   cel mai slab: {w.outer[o]} + {w.frames[k][t]} -> {a}")
