"""
Simulare: cuvintele ca deplasari (pasul uniform), stare = bundle de (locatie (x) trasatura).
Codare: fractional power encoding (vectori complecsi de modul 1).
  - fiecare cuvant w: deplasare m_w = exp(i*phi_w) (invatata), trasatura f_w = exp(i*psi_w) (fixa, aleatoare)
  - pas identic pentru orice cuvant:  L <- L * m_w ;  C <- C + L * f_w
Ipoteze testate:
  H1  deplasarile devin diferentiate pe clase (verbe mari, functionale mici) fara sa le spunem clasele
  H2a doar cu predictie locala (locatie -> trasatura asteptata), parafrazele se apropie partial
  H2b cu o presiune de echivalenta (contrastiv), parafrazele converg si schimbarea rolurilor ramane separata
  H3  rolurile emerg ca locatii consistente: agentul "aterizeaza" in acelasi loc in activ si pasiv
"""
import numpy as np, json, sys
rng = np.random.default_rng(0)
D = 512

# ---------------- lume sintetica ----------------
persons = ["ion", "ana", "dan", "maria", "radu", "ioana"]
objects = ["carte", "mar", "pix", "bani", "ceas", "tort"]
# evenimente: GIVE(A,B,C) si SELL(A,B,C); fiecare are 3 template-uri (parafraze)
templates = {
    "GIVE": [
        "{A} da {B} lui {C} .",
        "{C} primeste {B} de la {A} .",
        "{B} e dat lui {C} de {A} .",
    ],
    "SELL": [
        "{A} vinde {B} lui {C} .",
        "{C} cumpara {B} de la {A} .",
        "{B} e vandut lui {C} de {A} .",
    ],
}
vocab = sorted({"are", "nu"} | set(w for t in templates.values() for s in t for w in s.split()
                   if not w.startswith("{")) | set(persons) | set(objects))
V = len(vocab); idx = {w: i for i, w in enumerate(vocab)}
cls = {}
for w in persons: cls[w] = "persoana"
for w in objects: cls[w] = "obiect"
for w in ["da", "primeste", "vinde", "cumpara", "dat", "vandut"]: cls[w] = "verb"
for w in ["lui", "de", "la", "e"]: cls[w] = "functional"
cls["."] = "sfarsit"; cls["are"] = "verb"; cls["nu"] = "functional"

def render(kind, A, B, C, t):
    return templates[kind][t].format(A=A, B=B, C=C).split()

HELD = {("ion","ana"),("dan","maria"),("radu","ioana"),("ana","dan"),("maria","radu"),("ioana","ion")}
def random_event(test=False):
    while True:
        A, C = rng.choice(persons, 2, replace=False)
        if ((A, C) in HELD) == test: break
    B = rng.choice(objects)
    return (rng.choice(["GIVE", "SELL"]), A, B, C)

# ---------------- model ----------------
phi = rng.uniform(-np.pi, np.pi, (V, D))        # deplasari (invatate)
psi = rng.uniform(-np.pi, np.pi, (V, D))        # trasaturi (fixe)
F = np.exp(1j * psi)

def run(sent, phi):
    """intoarce locatiile L_t (T x D) si starea finala C (D)"""
    ids = [idx[w] for w in sent]
    cum = np.cumsum(phi[ids], axis=0)
    L = np.exp(1j * cum)
    C = (L * F[ids]).sum(0)
    return ids, L, C

def cos(a, b):
    return float(np.real(np.vdot(b, a)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

# ---------------- obiectiv A: predictie locala  (locatie -> trasatura) ----------------
M = np.zeros(D, dtype=complex)   # memorie pe termen lung: unde a fost vazuta fiecare trasatura
def train_predict(phi, n_sent=6000, lr=0.5, decay=0.995, scale=None):
    global M
    for n in range(n_sent):
        kind, A, B, C = random_event()
        sent = render(kind, A, B, C, rng.integers(3))
        ids, L, Cst = run(sent, phi)
        if np.abs(M).sum() > 0:
            grad = np.zeros_like(phi)
            Lprev = np.ones(D, dtype=complex)
            for t, w in enumerate(ids):
                # candidat w': Z = M * conj(Lprev) * exp(-i(phi_w' + psi_w'))
                Z = M[None, :] * np.conj(Lprev)[None, :] * np.exp(-1j * (phi + psi))
                z = np.real(Z).mean(1)
                sc = scale if scale else 1.0 / (z.std() + 1e-9)
                logits = z * sc
                p = np.exp(logits - logits.max()); p /= p.sum()
                y = np.zeros(V); y[w] = 1
                G = (p - y)[:, None] * np.imag(Z) / D * sc
                grad += G                               # prin propria deplasare a candidatului
                for s in ids[:t]:                       # prin locatia curenta (cuvintele anterioare)
                    grad[s] += G.sum(0)
                Lprev = L[t]
            phi -= lr * grad
        M = decay * M + (L * F[ids]).sum(0)
    return phi

# ---------------- obiectiv B: echivalenta (contrastiv pe starea finala) ----------------
def grad_cos_wrt_phi(ids, L, Cst, other):
    """d cos(C, other) / d phi pentru cuvintele propozitiei (other tratat ca fixat)"""
    na, nb = np.linalg.norm(Cst), np.linalg.norm(other)
    c = np.real(np.vdot(other, Cst)) / (na * nb)
    terms = L * F[ids]                       # T x D
    suffix = np.cumsum(terms[::-1], 0)[::-1]  # A_s = sum_{t>=s} L_t f_t
    dA = 1j * suffix
    g = (np.real(dA * np.conj(other)[None, :]) / (na * nb)
         - c * np.real(dA * np.conj(Cst)[None, :]) / na**2)
    return c, g

def train_contrast(phi, n_pairs=4000, lr=2.0, tau=0.1, n_rand_neg=4):
    for n in range(n_pairs):
        kind, A, B, C = random_event()
        ti, tj = rng.choice(3, 2, replace=False)
        anchor = render(kind, A, B, C, ti)
        cands = [render(kind, A, B, C, tj)]                       # pozitiv
        cands.append(render(kind, C, B, A, rng.integers(3)))       # roluri schimbate (negativ greu)
        for _ in range(n_rand_neg):
            k2, A2, B2, C2 = random_event(); cands.append(render(k2, A2, B2, C2, rng.integers(3)))
        ia, La, Ca = run(anchor, phi)
        runs = [run(s, phi) for s in cands]
        sims = np.array([cos(Ca, r[2]) for r in runs])
        logits = sims / tau
        p = np.exp(logits - logits.max()); p /= p.sum()
        y = np.zeros(len(cands)); y[0] = 1
        grad = np.zeros_like(phi)
        for k, (ik, Lk, Ck) in enumerate(runs):
            coef = (p[k] - y[k]) / tau
            _, ga = grad_cos_wrt_phi(ia, La, Ca, Ck)
            _, gk = grad_cos_wrt_phi(ik, Lk, Ck, Ca)
            for s, g in zip(ia, ga): grad[s] += coef * g
            for s, g in zip(ik, gk): grad[s] += coef * g
        phi -= lr * grad
    return phi

# ---------------- evaluare ----------------
def evaluate(phi, n=400, test=False):
    para, swap, rand, obj = [], [], [], []
    ag_same, ag_vs_rec = [], []
    for _ in range(n):
        kind, A, B, C = random_event(test)
        s1 = render(kind, A, B, C, 0); s2 = render(kind, A, B, C, 1); s3 = render(kind, A, B, C, 2)
        _, L1, C1 = run(s1, phi); _, L2, C2 = run(s2, phi); _, L3, C3 = run(s3, phi)
        para += [cos(C1, C2), cos(C1, C3), cos(C2, C3)]
        _, _, Cs = run(render(kind, C, B, A, rng.integers(3)), phi); swap.append(cos(C1, Cs))
        B2 = rng.choice([o for o in objects if o != B])
        _, _, Co = run(render(kind, A, B2, C, rng.integers(3)), phi); obj.append(cos(C1, Co))
        k2, A2, B2, C2 = random_event(); _, _, Cr = run(render(k2, A2, B2, C2, rng.integers(3)), phi); rand.append(cos(C1, Cr))
        # H3: locatia agentului in T1 (pozitia 0) vs in T2 (pozitia 5) ; vs locatia destinatarului in T2 (pozitia 0)
        ag_same.append(cos(L1[0], L2[5])); ag_vs_rec.append(cos(L1[0], L2[0]))
    disp = {}
    for c in ["persoana", "obiect", "verb", "functional", "sfarsit"]:
        ws = [idx[w] for w in vocab if cls[w] == c]
        disp[c] = float(np.mean(1 - np.cos(phi[ws])))   # 0 = identitate, 1 = deplasare maxima (medie)
    ok = np.mean(np.array(para[::3]) > np.array(swap))   # parafraza mai aproape decat schimbarea rolurilor?
    return dict(parafraze=np.mean(para), roluri_schimbate=np.mean(swap), obiect_schimbat=np.mean(obj),
                aleator=np.mean(rand), parafraza_bate_schimbul=float(ok),
                H3_agent_activ_vs_pasiv=np.mean(ag_same), H3_agent_vs_destinatar=np.mean(ag_vs_rec),
                deplasari=disp)

def show(tag, r):
    print(f"\n== {tag} ==")
    for k, v in r.items():
        if isinstance(v, dict): print("  deplasare medie pe clasa:", {a: round(b, 3) for a, b in v.items()})
        else: print(f"  {k:28s} {v:6.3f}")


# ---------------- obiectiv C: predictie la ceasul lent (starea propozitiei -> consecinta) ----------------
# Nivelul lent trateaza starea finala a propozitiei ca pe un "cuvant": deplasarea lui este faza lui S (L2 = S/|S|).
# Apoi prezice, cu ACEEASI regula ca in A, ce trasaturi urmeaza la locatia L2: cuvintele propozitiei-consecinta.
# Consecinta e data de lume, nu de formulare: pentru DA(A,B,C) si VINDE(A,B,C) ea e "C are B ."
M2 = np.zeros(D, dtype=complex)
def consequence(kind, A, B, C):
    return f"{C} are {B} .".split()

def train_consequence(phi, n_sent=8000, lr=0.5, decay=0.995, clip=5.0):
    global M2
    for n in range(n_sent):
        kind, A, B, C = random_event()
        sent = render(kind, A, B, C, rng.integers(3))
        ids, L, S = run(sent, phi)
        L2 = S / (np.abs(S) + 1e-9)
        cons = [idx[w] for w in consequence(kind, A, B, C)]
        if np.abs(M2).sum() > 0:
            terms = L * F[ids]
            suffix = np.cumsum(terms[::-1], 0)[::-1]            # A_s
            dphase = np.clip(np.real(suffix / (S[None, :] + 1e-9)), -clip, clip)   # d arg(S) / d phi_s
            Zf = M2[None, :] * np.conj(L2)[None, :] * np.conj(F)  # V x D
            z = np.real(Zf).mean(1)
            sc = 1.0 / (z.std() + 1e-9)
            logits = z * sc
            p = np.exp(logits - logits.max()); p /= p.sum()
            grad = np.zeros_like(phi)
            for w in cons:
                y = np.zeros(V); y[w] = 1
                G = ((p - y)[:, None] * np.imag(Zf) / D * sc).sum(0)   # D
                for s_i, s in enumerate(ids):
                    grad[s] += G * dphase[s_i]
            phi -= lr * grad
        M2 = decay * M2 + L2 * F[cons].sum(0)
    return phi

if __name__ == "__main__":
    import copy
    phi0 = phi.copy()
    show("0. deplasari aleatoare, fara invatare", evaluate(phi))
    phi = train_predict(phi)
    show("A. dupa predictie locala la nivel de cuvant", evaluate(phi))
    phiA = phi.copy()
    phi = train_consequence(phi)
    show("A+C. apoi predictie a consecintei (ceas lent), perechi vazute", evaluate(phi))
    show("A+C. perechi (A,C) NEVAZUTE", evaluate(phi, test=True))
    M2[:] = 0
    phi = train_consequence(phi0.copy())
    show("C singur, pornind de la aleator, perechi vazute", evaluate(phi))
    show("C singur, perechi NEVAZUTE", evaluate(phi, test=True))

# ---------------- obiectiv D: echivalenta data de lume (prototip per consecinta) ----------------
# Nivelul lent tine cate un prototip (bundle) pentru fiecare consecinta distincta pe care a observat-o.
# Regula: starea propozitiei e trasa spre prototipul consecintei care a urmat si impinsa de celelalte.
# Semnalul "sunt echivalente" vine din lume (aceeasi consecinta), nu din adnotari de parafraze.
def train_prototype(phi, n_sent=8000, lr=2.0, tau=0.1, decay=0.98):
    P = {}
    for n in range(n_sent):
        kind, A, B, C = random_event()
        key = (C, B)
        ids, L, S = run(render(kind, A, B, C, rng.integers(3)), phi)
        if key in P and len(P) > 1:
            keys = list(P.keys())
            cs, gs = [], []
            for k in keys:
                c, g = grad_cos_wrt_phi(ids, L, S, P[k]); cs.append(c); gs.append(g)
            logits = np.array(cs) / tau
            p = np.exp(logits - logits.max()); p /= p.sum()
            grad = np.zeros_like(phi)
            for k, g, pk in zip(keys, gs, p):
                coef = (pk - (k == key)) / tau
                for s_i, s in enumerate(ids): grad[s] += coef * g[s_i]
            phi -= lr * grad
        P[key] = decay * P.get(key, 0) + S / np.linalg.norm(S)
    return phi

if __name__ == "__main__":
    for tag, start in []:
        phi = train_prototype(start)
        show(tag + ", perechi vazute", evaluate(phi))
        show(tag + ", perechi NEVAZUTE", evaluate(phi, test=True))

# ---------------- obiectiv E: gat de sticla — predictie GENERATIVA printr-o operatie fixa ----------------
# Nivelul lent nu mai are memorie asociativa: trebuie sa GENEREZE starea consecintei din starea propozitiei
# printr-un bind fix, S_cons ≈ S_prop (x) T, cu T aleator si inversabil. Singurul lucru care se invata sunt
# deplasarile cuvintelor (de ambele parti). Fara negative, fara etichete: doar eroare de predictie.
T_fix = np.exp(1j * rng.uniform(-np.pi, np.pi, D))
def train_fixedT(phi, n_sent=8000, lr=2.0):
    for n in range(n_sent):
        kind, A, B, C = random_event()
        ids, L, S = run(render(kind, A, B, C, rng.integers(3)), phi)
        idc, Lc, Sc = run(consequence(kind, A, B, C), phi)
        c, gs = grad_cos_wrt_phi(ids, L, S, Sc * np.conj(T_fix))   # maximizeaza cos(S*T, Sc)
        _, gc = grad_cos_wrt_phi(idc, Lc, Sc, S * T_fix)
        grad = np.zeros_like(phi)
        for s_i, s in enumerate(ids): grad[s] -= gs[s_i]
        for s_i, s in enumerate(idc): grad[s] -= gc[s_i]
        phi -= lr * grad
    return phi

def pred_quality(phi, n=300):
    cs = []
    for _ in range(n):
        kind, A, B, C = random_event()
        _, _, S = run(render(kind, A, B, C, rng.integers(3)), phi)
        _, _, Sc = run(consequence(kind, A, B, C), phi)
        cs.append(cos(S * T_fix, Sc))
    return float(np.mean(cs))

if __name__ == "__main__":
    for tag, start in [("E singur, de la aleator", phi0.copy()), ("A apoi E", phiA.copy())]:
        phi = train_fixedT(start)
        print(f"\n[{tag}] calitatea predictiei cos(S*T, S_consecinta) = {pred_quality(phi):.3f}")
        show(tag + ", perechi vazute", evaluate(phi))
        show(tag + ", perechi NEVAZUTE", evaluate(phi, test=True))
