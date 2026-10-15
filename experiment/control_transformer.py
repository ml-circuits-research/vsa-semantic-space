"""
CONTROL: un transformer mic (numpy, backprop scris de mana) antrenat pe EXACT aceeasi sarcina ca modelul VSA:
  din propozitie -> prezice starea senzoriala a consecintei (vector real de dimensiune 2D = [Re, Im]).
Comparatia (acelasi test pentru ambele): din predictie se decodeaza rolurile cu adresele lumii.
Teste: (1) identitati vazute, (2) identitati NEVAZUTE, (3) formulare NEVAZUTA (template-ul 4 exclus la antrenare).
"""
import numpy as np, sys, time
import camp as cp

# ---------------- transformer minimal cu backprop manual ----------------
class TinyTransformer:
    def __init__(self, V, d, L, out_dim, n_layers=2, n_heads=4, rng=None, feat=None):
        r = rng or np.random.default_rng(0)
        s = 1 / np.sqrt(d)
        self.V, self.d, self.L, self.nl, self.nh = V, d, L, n_layers, n_heads
        self.feat = feat     # V x Fdim, fix (identitatea senzoriala a cuvantului); None = embedding invatat
        self.p = {"P": r.normal(0, 0.1, (L, d)), "Wout": r.normal(0, s, (d, out_dim))}
        if feat is None: self.p["E"] = r.normal(0, 0.1, (V, d))
        else: self.p["Win"] = r.normal(0, 1 / np.sqrt(feat.shape[1]), (feat.shape[1], d))
        for l in range(n_layers):
            for nm in ["Wq", "Wk", "Wv", "Wo"]: self.p[f"{nm}{l}"] = r.normal(0, s, (d, d))
            self.p[f"W1{l}"] = r.normal(0, s, (d, 4 * d)); self.p[f"W2{l}"] = r.normal(0, 1 / np.sqrt(4 * d), (4 * d, d))
            self.p[f"g1{l}"] = np.ones(d); self.p[f"g2{l}"] = np.ones(d)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}; self.v = {k: np.zeros_like(v) for k, v in self.p.items()}; self.t = 0

    @staticmethod
    def ln_f(x, g):   # layer norm (fara bias)
        mu = x.mean(-1, keepdims=True); xc = x - mu; var = (xc ** 2).mean(-1, keepdims=True); inv = 1 / np.sqrt(var + 1e-5)
        xh = xc * inv; return xh * g, (xh, inv, g)
    @staticmethod
    def ln_b(dy, cache):
        xh, inv, g = cache; dg = (dy * xh).sum(axis=tuple(range(dy.ndim - 1)))
        dxh = dy * g; d = xh.shape[-1]
        dx = inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
        return dx, dg

    def forward(self, ids):            # ids: B x L (toate propozitiile din batch au aceeasi lungime)
        p = self.p; B, L = ids.shape; d, nh = self.d, self.nh; hd = d // nh
        X = (p["E"][ids] if self.feat is None else self.feat[ids] @ p["Win"]) + p["P"][:L][None]; cache = {"ids": ids, "X0": X}
        for l in range(self.nl):
            Xn, c1 = self.ln_f(X, p[f"g1{l}"])
            Q = Xn @ p[f"Wq{l}"]; K = Xn @ p[f"Wk{l}"]; Vv = Xn @ p[f"Wv{l}"]
            Qh = Q.reshape(B, L, nh, hd).transpose(0, 2, 1, 3); Kh = K.reshape(B, L, nh, hd).transpose(0, 2, 1, 3); Vh = Vv.reshape(B, L, nh, hd).transpose(0, 2, 1, 3)
            S = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(hd); S = S - S.max(-1, keepdims=True); A = np.exp(S); A /= A.sum(-1, keepdims=True)
            Hh = A @ Vh; H = Hh.transpose(0, 2, 1, 3).reshape(B, L, d); O = H @ p[f"Wo{l}"]
            X1 = X + O
            X1n, c2 = self.ln_f(X1, p[f"g2{l}"]); Z = X1n @ p[f"W1{l}"]; R = np.maximum(Z, 0); M = R @ p[f"W2{l}"]
            X2 = X1 + M
            cache[l] = (X, Xn, c1, Qh, Kh, Vh, A, H, X1, X1n, c2, Z, R)
            X = X2
        h = X.mean(1); y = h @ p["Wout"]; cache["h"] = h; cache["Xf"] = X
        return y, cache

    def backward(self, dy, cache):
        p = self.p; g = {k: np.zeros_like(v) for k, v in p.items()}
        B, L = cache["ids"].shape; d, nh = self.d, self.nh; hd = d // nh
        g["Wout"] = cache["h"].T @ dy; dh = dy @ p["Wout"].T
        dX = np.repeat(dh[:, None, :], L, axis=1) / L
        for l in reversed(range(self.nl)):
            X, Xn, c1, Qh, Kh, Vh, A, H, X1, X1n, c2, Z, R = cache[l]
            dM = dX; g[f"W2{l}"] += R.reshape(-1, 4 * d).T @ dM.reshape(-1, d); dR = dM @ p[f"W2{l}"].T; dZ = dR * (Z > 0)
            g[f"W1{l}"] += X1n.reshape(-1, d).T @ dZ.reshape(-1, 4 * d); dX1n = dZ @ p[f"W1{l}"].T
            dX1, dg2 = self.ln_b(dX1n, c2); g[f"g2{l}"] += dg2; dX1 = dX1 + dX
            dO = dX1; g[f"Wo{l}"] += H.reshape(-1, d).T @ dO.reshape(-1, d); dH = dO @ p[f"Wo{l}"].T
            dHh = dH.reshape(B, L, nh, hd).transpose(0, 2, 1, 3)
            dA = dHh @ Vh.transpose(0, 1, 3, 2); dVh = A.transpose(0, 1, 3, 2) @ dHh
            dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(hd)
            dQh = dS @ Kh; dKh = dS.transpose(0, 1, 3, 2) @ Qh
            dQ = dQh.transpose(0, 2, 1, 3).reshape(B, L, d); dK = dKh.transpose(0, 2, 1, 3).reshape(B, L, d); dV = dVh.transpose(0, 2, 1, 3).reshape(B, L, d)
            Xn2 = Xn.reshape(-1, d)
            g[f"Wq{l}"] += Xn2.T @ dQ.reshape(-1, d); g[f"Wk{l}"] += Xn2.T @ dK.reshape(-1, d); g[f"Wv{l}"] += Xn2.T @ dV.reshape(-1, d)
            dXn = dQ @ p[f"Wq{l}"].T + dK @ p[f"Wk{l}"].T + dV @ p[f"Wv{l}"].T
            dXl, dg1 = self.ln_b(dXn, c1); g[f"g1{l}"] += dg1
            dX = dXl + dX1
        if self.feat is None: np.add.at(g["E"], cache["ids"], dX)
        else: g["Win"] += self.feat[cache["ids"]].reshape(-1, self.feat.shape[1]).T @ dX.reshape(-1, d)
        g["P"][:L] += dX.sum(0)
        return g

    def step(self, g, lr=1e-3, b1=0.9, b2=0.999):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]; self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            self.p[k] -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + 1e-8)

def cos_loss_grad(y, tgt):
    """loss = -cos(y, tgt) pe fiecare rand; intoarce loss mediu si dL/dy"""
    ny = np.linalg.norm(y, axis=1, keepdims=True) + 1e-9; nt = np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-9
    c = (y * tgt).sum(1, keepdims=True) / (ny * nt)
    dy = -(tgt / (ny * nt) - c * y / ny ** 2) / len(y)
    return -c.mean(), dy

def gradcheck():
    r = np.random.default_rng(0); V, d, L = 7, 8, 5
    tt = TinyTransformer(V, d, L, 6, n_layers=1, n_heads=2, rng=r)
    ids = r.integers(0, V, (3, L)); tgt = r.normal(size=(3, 6))
    y, c = tt.forward(ids); loss, dy = cos_loss_grad(y, tgt); g = tt.backward(dy, c)
    worst = 0
    for k in ["E", "Wq0", "Wv0", "W1", "g1", "Wout", "P"]:
        k = k if k in tt.p else k + "0"
        idx = tuple(r.integers(0, s) for s in tt.p[k].shape); eps = 1e-6
        tt.p[k][idx] += eps; lp, _ = cos_loss_grad(tt.forward(ids)[0], tgt); tt.p[k][idx] -= 2 * eps; lm, _ = cos_loss_grad(tt.forward(ids)[0], tgt); tt.p[k][idx] += eps
        num = (lp - lm) / (2 * eps); worst = max(worst, abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx]) + 1e-12))
    return worst

# ---------------- sarcina comuna ----------------
def to_real(z): return np.concatenate([np.real(z), np.imag(z)])
def to_complex(y): n = len(y) // 2; return y[:n] + 1j * y[n:]

def world_addresses(wm, w, kind="GIVE"):
    ev = (kind, w.persons[0], w.objects[0], w.persons[1]); toks = w.cons(ev); terms, _, p2t = wm.run(toks)
    pos = {"C": 0, "B": 2, "A": 4}
    return {r: terms[p2t[pos[r]]] * np.conj(wm.F[w.idx[toks[pos[r]]]]) for r in "ABC"}

def decode_from_prediction(pred, addr, wm, w, P, O):
    out = {}
    for r, cands in (("A", P), ("B", O), ("C", P)):
        read = pred * np.conj(addr[r]); out[r] = max(cands, key=lambda c: wm.cos(read, wm.F[w.idx[c]]))
    return out

def eval_predictor(predict, wm, w, P, O, templates, n=100):
    acc = []
    for kind in w.frames:
        addr = world_addresses(wm, w, kind)
        for t in templates:
            a = {"A": 0, "B": 0, "C": 0}
            for _ in range(n):
                ev = w.event(P, O); ev = (kind,) + ev[1:]
                dec = decode_from_prediction(predict(w.render(ev, t)), addr, wm, w, P, O)
                a["A"] += dec["A"] == ev[1]; a["B"] += dec["B"] == ev[2]; a["C"] += dec["C"] == ev[3]
            acc.append(np.mean([v / n for v in a.values()]))
    return float(np.mean(acc))

if __name__ == "__main__":
    print("gradient check (eroare relativa maxima):", f"{gradcheck():.2e}")
    rng = np.random.default_rng(1); D = 512
    persons = ["ion", "ana", "dan", "maria", "radu", "ioana", "vlad", "elena", "mihai", "irina", "paul", "sofia"]
    objects = ["carte", "mar", "pix", "bani", "ceas", "tort", "cheie", "pahar", "lampa", "scaun", "harta", "inel"]
    trP, teP, trO, teO = persons[:8], persons[8:], objects[:8], objects[8:]
    w = cp.World(persons, objects, rng)
    for v in ["da", "vinde", "primeste", "cumpara"]: w.cat[v] = "VF"
    for v in ["dat", "vandut"]: w.cat[v] = "VP"
    for v in ["e", "este"]: w.cat[v] = "COP"
    wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
    TR_T = [0, 1, 2, 3]; TE_T = [4]          # formularea 4 nu se vede la antrenare
    N_SENT = 40000

    # ---- VSA ----
    class WorldRestricted(cp.World):
        pass
    vsa = cp.Model(w, D, np.random.default_rng(2), deferred=True); vsa.F = wm.F
    # antrenare restransa la template-urile 0-3 si identitatile de antrenare
    orig_render = w.render
    t0 = time.time()
    for n in range(N_SENT):
        ev = w.event(trP, trO); t = rng.choice(TR_T); toks = w.render(ev, t)
        terms, members, _ = vsa.run(toks); Sc = wm.state(w.cons(ev))
        _, g = vsa.grad_cos(terms, members, Sc * np.conj(vsa.T))
        if not hasattr(vsa, "_m"): vsa._m = np.zeros_like(vsa.theta); vsa._v = np.zeros_like(vsa.theta); vsa._k = 0
        vsa._k += 1; vsa._m = 0.9 * vsa._m + 0.1 * g; vsa._v = 0.999 * vsa._v + 0.001 * g * g
        vsa.theta += 0.02 * (vsa._m / (1 - 0.9 ** vsa._k)) / (np.sqrt(vsa._v / (1 - 0.999 ** vsa._k)) + 1e-8)
    vsa_pred = lambda toks: vsa.state(toks) * vsa.T
    print(f"\nVSA ({N_SENT} propozitii, {time.time()-t0:.0f}s, {vsa.theta.size} parametri):")
    print(f"  identitati vazute, formulari vazute:      {eval_predictor(vsa_pred, wm, w, trP, trO, TR_T):.3f}")
    print(f"  identitati NEVAZUTE, formulari vazute:    {eval_predictor(vsa_pred, wm, w, teP, teO, TR_T):.3f}")
    print(f"  identitati vazute, formulare NEVAZUTA:    {eval_predictor(vsa_pred, wm, w, trP, trO, TE_T):.3f}")
    print(f"  identitati NEVAZUTE, formulare NEVAZUTA:  {eval_predictor(vsa_pred, wm, w, teP, teO, TE_T):.3f}", flush=True)

    # ---- Transformer ----
    for d, nl in [(128, 3)]:
        feat = np.concatenate([np.real(wm.F), np.imag(wm.F)], 1)
        tt = TinyTransformer(len(w.vocab), d, 12, 2 * D, n_layers=nl, n_heads=4, rng=np.random.default_rng(3), feat=feat)
        nparam = sum(v.size for v in tt.p.values())
        t0 = time.time(); BATCH = 32; steps = 3 * N_SENT // BATCH      # 3 epoci-echivalent
        for s in range(steps):
            kind = rng.choice(list(w.frames)); t = rng.choice(TR_T)
            evs = [(kind,) + w.event(trP, trO)[1:] for _ in range(BATCH)]
            ids = np.array([[w.idx[x] for x in w.render(ev, t)] for ev in evs])
            tgt = np.array([to_real(wm.state(w.cons(ev))) for ev in evs])
            y, c = tt.forward(ids); loss, dy = cos_loss_grad(y, tgt); g = tt.backward(dy, c); tt.step(g, lr=1e-3)
            if s % 1000 == 0: print(f"    [d={d},L={nl}] pas {s}/{steps} loss {loss:.3f}", flush=True)
        tf_pred = lambda toks: to_complex(tt.forward(np.array([[w.idx[x] for x in toks]]))[0][0])
        print(f"\nTransformer d={d}, {nl} straturi ({steps*BATCH} propozitii, {time.time()-t0:.0f}s, {nparam} parametri):")
        print(f"  identitati vazute, formulari vazute:      {eval_predictor(tf_pred, wm, w, trP, trO, TR_T):.3f}")
        print(f"  identitati NEVAZUTE, formulari vazute:    {eval_predictor(tf_pred, wm, w, teP, teO, TR_T):.3f}")
        print(f"  identitati vazute, formulare NEVAZUTA:    {eval_predictor(tf_pred, wm, w, trP, trO, TE_T):.3f}")
        print(f"  identitati NEVAZUTE, formulare NEVAZUTA:  {eval_predictor(tf_pred, wm, w, teP, teO, TE_T):.3f}", flush=True)
