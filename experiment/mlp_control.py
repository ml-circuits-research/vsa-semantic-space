"""Control MLP: bundle (fara ordine) -> MLP -> stare prezisa. Baseline 'fara ordine, cu capacitate'. Aceeasi sarcina/evaluare ca control_transformer.py."""
import numpy as np, time
import camp as cp, control_transformer as ct
rng = np.random.default_rng(1); D = 512
persons = ["ion","ana","dan","maria","radu","ioana","vlad","elena","mihai","irina","paul","sofia"]; objects = ["carte","mar","pix","bani","ceas","tort","cheie","pahar","lampa","scaun","harta","inel"]
trP, teP, trO, teO = persons[:8], persons[8:], objects[:8], objects[8:]
w = cp.World(persons, objects, rng)
for v in ["da","vinde","primeste","cumpara"]: w.cat[v] = "VF"
for v in ["dat","vandut"]: w.cat[v] = "VP"
for v in ["e","este"]: w.cat[v] = "COP"
wm = cp.Model(w, D, np.random.default_rng(99), deferred=False)
feat = np.concatenate([np.real(wm.F), np.imag(wm.F)], 1)        # 2D
TR_T, TE_T = [0,1,2,3], [4]
H = 1024; r = np.random.default_rng(3)
W1 = r.normal(0, 1/np.sqrt(2*D), (2*D, H)); b1 = np.zeros(H); W2 = r.normal(0, 1/np.sqrt(H), (H, 2*D))
P = {"W1": W1, "b1": b1, "W2": W2}; M = {k: np.zeros_like(v) for k, v in P.items()}; Vv = {k: np.zeros_like(v) for k, v in P.items()}; t = 0
def fwd(X): Z = X @ P["W1"] + P["b1"]; R = np.maximum(Z, 0); return R @ P["W2"], (Z, R)
t0 = time.time(); BATCH = 32
for s in range(3 * 40000 // BATCH):
    kind = rng.choice(list(w.frames)); tt = rng.choice(TR_T)
    evs = [(kind,) + w.event(trP, trO)[1:] for _ in range(BATCH)]
    X = np.array([feat[[w.idx[x] for x in w.render(ev, tt)]].sum(0) for ev in evs])   # bundle fara ordine
    tgt = np.array([ct.to_real(wm.state(w.cons(ev))) for ev in evs])
    y, (Z, R) = fwd(X); loss, dy = ct.cos_loss_grad(y, tgt)
    g = {"W2": R.T @ dy}; dR = dy @ P["W2"].T; dZ = dR * (Z > 0); g["W1"] = X.T @ dZ; g["b1"] = dZ.sum(0)
    t += 1
    for k in P:
        M[k] = 0.9*M[k] + 0.1*g[k]; Vv[k] = 0.999*Vv[k] + 0.001*g[k]**2
        P[k] -= 1e-3 * (M[k]/(1-0.9**t)) / (np.sqrt(Vv[k]/(1-0.999**t)) + 1e-8)
pred = lambda toks: ct.to_complex(fwd(feat[[w.idx[x] for x in toks]].sum(0)[None])[0][0])
print(f"MLP pe bundle fara ordine ({sum(v.size for v in P.values())} parametri, {time.time()-t0:.0f}s):")
print(f"  identitati vazute, formulari vazute:      {ct.eval_predictor(pred, wm, w, trP, trO, TR_T):.3f}")
print(f"  identitati NEVAZUTE, formulari vazute:    {ct.eval_predictor(pred, wm, w, teP, teO, TR_T):.3f}")
print(f"  identitati vazute, formulare NEVAZUTA:    {ct.eval_predictor(pred, wm, w, trP, trO, TE_T):.3f}")
print(f"  identitati NEVAZUTE, formulare NEVAZUTA:  {ct.eval_predictor(pred, wm, w, teP, teO, TE_T):.3f}")
