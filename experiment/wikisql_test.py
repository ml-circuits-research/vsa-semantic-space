"""
TEXT REAL cu CONSECINTA EXECUTABILA: WikiSQL (56k intrebari in engleza -> interogari SQL pe tabele Wikipedia).
Lumea: interogarea SQL codata ca structura rol-umplutura (cod fix, aleator):
   S_sql = SEL (x) bundle(tokeni coloana selectata) + AGG (x) f(agregator) + sum_k CONDk (x) [COL (x) bundle(col) + OP (x) f(op) + VAL (x) bundle(val)]
Modelul: intrebarea -> stare prin pasul uniform (deplasare per cuvant, invatata). Obiectiv generativ: cos(S_q (x) T, S_sql).
Evaluare pe dev (8421): din PREDICTIE se decodeaza coloana selectata (dintre coloanele tabelului), agregatorul (6), coloana conditiei 1.
Plus echivalenta: perechi de intrebari din acelasi tabel cu acelasi SQL-schelet (sel, agg, coloane conditii) vs cu schelet diferit.
"""
import json, re, time, sys, numpy as np
rng = np.random.default_rng(0); D = 512
P = "corp/wikisql/data/"
def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
def load(split):
    rows = [json.loads(l) for l in open(P + f"{split}.jsonl")]
    tabs = {}
    for l in open(P + f"{split}.tables.jsonl"):
        t = json.loads(l); tabs[t["id"]] = t["header"]
    return rows, tabs
train, ttabs = load("train"); dev, dtabs = load("dev")
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 56355
train = train[:N_TRAIN]
vocab = {}
def vid(w):
    if w not in vocab: vocab[w] = len(vocab)
    return vocab[w]
for rows, tabs in ((train, ttabs), (dev, dtabs)):
    for r in rows:
        for w in tok(r["question"]): vid(w)
        for h in tabs[r["table_id"]]:
            for w in tok(h): vid(w)
        for c in r["sql"]["conds"]:
            for w in tok(str(c[2])): vid(w)
V = len(vocab); print(f"intrebari antrenare {len(train)}, dev {len(dev)}, vocabular {V}")
F = np.exp(1j * rng.uniform(-np.pi, np.pi, (V + 64, D)))      # trasaturi fixe; ultimele 64 = simboluri SQL (agg, op)
AGG = [V + i for i in range(6)]; OPS = [V + 10 + i for i in range(3)]
role = lambda: np.exp(1j * rng.uniform(-np.pi, np.pi, D))
R = {"SEL": role(), "AGG": role(), "COL": role(), "OP": role(), "VAL": role(), "COND": [role() for _ in range(4)]}
def bundle(ws): return sum(F[vid(w)] for w in ws) if ws else np.zeros(D, complex)
def sql_state(sql, header):
    S = R["SEL"] * bundle(tok(header[sql["sel"]])) + R["AGG"] * F[AGG[sql["agg"]]]
    for k, (c, o, v) in enumerate(sql["conds"][:4]):
        S = S + R["COND"][k] * (R["COL"] * bundle(tok(header[c])) + R["OP"] * F[OPS[o]] + R["VAL"] * bundle(tok(str(v))))
    return S
def cos(a, b): return float(np.real(np.vdot(b, a)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

class QModel:
    def __init__(self, mode):
        self.mode = mode                       # "bow": fara deplasari; "pos": o singura deplasare pentru toate; "word": per cuvant
        n = {"bow": 1, "pos": 1, "word": V}[mode]
        self.theta = rng.uniform(-np.pi, np.pi, (n, D)); self.T = np.exp(1j * rng.uniform(-np.pi, np.pi, D))
        self._m = np.zeros_like(self.theta); self._v = np.zeros_like(self.theta); self._k = 0
    def cats(self, ids): return [0] * len(ids) if self.mode != "word" else ids
    def run(self, ids):
        cats = self.cats(ids)
        if self.mode == "bow": terms = F[ids]
        else: terms = np.exp(1j * np.cumsum(self.theta[cats], 0)) * F[ids]
        return terms, cats
    def state(self, ids): return self.run(ids)[0].sum(0)
    def step(self, ids, target, lr=0.02):
        terms, cats = self.run(ids); S = terms.sum(0); other = target * np.conj(self.T)
        na, nb = np.linalg.norm(S), np.linalg.norm(other); c = np.real(np.vdot(other, S)) / (na * nb)
        if self.mode == "bow": return c
        dT = 1j * terms; gk = np.real(dT * np.conj(other)[None]) / (na * nb) - c * np.real(dT * np.conj(S)[None]) / na ** 2
        suf = np.cumsum(gk[::-1], 0)[::-1]
        rows = np.unique(cats); g = np.zeros((len(rows), D)); np.add.at(g, np.searchsorted(rows, cats), suf)
        if not hasattr(self, "_cnt"): self._cnt = np.zeros(len(self.theta))
        self._cnt[rows] += 1; k = self._cnt[rows][:, None]          # Adam rar: doar randurile atinse
        self._m[rows] = 0.9 * self._m[rows] + 0.1 * g; self._v[rows] = 0.999 * self._v[rows] + 0.001 * g * g
        self.theta[rows] += lr * (self._m[rows] / (1 - 0.9 ** k)) / (np.sqrt(self._v[rows] / (1 - 0.999 ** k)) + 1e-8)
        return c

def decode(pred, header):
    cols = [bundle(tok(h)) for h in header]
    sel = int(np.argmax([cos(pred * np.conj(R["SEL"]), cb) for cb in cols]))
    agg = int(np.argmax([cos(pred * np.conj(R["AGG"]), F[a]) for a in AGG]))
    c1 = int(np.argmax([cos(pred * np.conj(R["COND"][0] * R["COL"]), cb) for cb in cols]))
    return sel, agg, c1

def evaluate(m, name, n=3000):
    acc = np.zeros(3); cs = []; preds = {}
    for r in dev[:n]:
        ids = [vid(w) for w in tok(r["question"])]
        if not ids: continue
        pred = m.state(ids) * m.T; header = dtabs[r["table_id"]]
        sel, agg, c1 = decode(pred, header); s = r["sql"]
        acc += [sel == s["sel"], agg == s["agg"], (len(s["conds"]) > 0 and c1 == s["conds"][0][0])]
        cs.append(cos(pred, sql_state(s, header))); preds[id(r)] = pred
    acc /= len(cs)
    # echivalenta: perechi din acelasi tabel, acelasi schelet vs schelet diferit
    bytab = {}
    for r in dev[:n]:
        if id(r) in preds: bytab.setdefault(r["table_id"], []).append(r)
    same, diff = [], []
    for rs in bytab.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]; ka = (a["sql"]["sel"], a["sql"]["agg"], tuple(c[0] for c in a["sql"]["conds"])); kb = (b["sql"]["sel"], b["sql"]["agg"], tuple(c[0] for c in b["sql"]["conds"]))
                (same if ka == kb else diff).append(cos(preds[id(a)], preds[id(b)]))
    print(f"  {name:34s} col.selectata {acc[0]:.3f}  agregator {acc[1]:.3f}  col.conditie {acc[2]:.3f} | cos(pred, sql) {np.mean(cs):.3f} | perechi acelasi tabel: acelasi schelet {np.mean(same):.3f} (n={len(same)}), schelet diferit {np.mean(diff):.3f} (n={len(diff)})", flush=True)

# baze de comparatie triviale
hdr_sizes = [len(dtabs[r["table_id"]]) for r in dev[:3000]]
print(f"sansa: col.selectata ~{np.mean([1/h for h in hdr_sizes]):.3f}, agregator majoritar {np.mean([r['sql']['agg']==0 for r in dev[:3000]]):.3f}")
for mode in ["bow", "pos", "word"]:
    m = QModel(mode); t0 = time.time()
    if mode != "bow":
        for ep in range(2):
            for i, r in enumerate(train):
                ids = [vid(w) for w in tok(r["question"])]
                if ids: m.step(ids, sql_state(r["sql"], ttabs[r["table_id"]]))
                if i % 20000 == 0: print(f"    [{mode}] epoca {ep} pas {i} ({time.time()-t0:.0f}s)", flush=True)
    evaluate(m, f"{mode} ({time.time()-t0:.0f}s antrenare)")
