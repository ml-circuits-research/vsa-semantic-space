import numpy as np, sim_deplasari as sm
rng = sm.rng
# F: deplasarea e o proprietate a CATEGORIEI (ce face cuvantul), identitatea ramane in trasatura (ce este).
groups = [[sm.idx[p] for p in sm.persons], [sm.idx[o] for o in sm.objects]]
def tie(phi):
    for g in groups: phi[g] = phi[g].mean(0)
    return phi
phi_world = sm.phi.copy(); tie(phi_world)
def consequence2(kind, A, B, C): return f"{C} are {B} . {A} nu are {B} .".split()
def train_F(phi, n_sent=12000, lr=2.0):
    tie(phi)
    for n in range(n_sent):
        kind, A, B, C = sm.random_event()
        ids, L, S = sm.run(sm.render(kind, A, B, C, rng.integers(3)), phi)
        _, _, Sc = sm.run(consequence2(kind, A, B, C), phi_world)
        c, gs = sm.grad_cos_wrt_phi(ids, L, S, Sc * np.conj(sm.T_fix))
        grad = np.zeros_like(phi)
        for s_i, s in enumerate(ids): grad[s] -= gs[s_i]
        phi -= lr * grad; tie(phi)
    return phi
def pq(phi, n=300):
    cs=[]
    for _ in range(n):
        kind,A,B,C = sm.random_event()
        _,_,S = sm.run(sm.render(kind,A,B,C,rng.integers(3)), phi)
        _,_,Sc = sm.run(consequence2(kind,A,B,C), phi_world)
        cs.append(sm.cos(S*sm.T_fix, Sc))
    return np.mean(cs)
# control: aceeasi legare pe categorii, dar fara invatare
phi = tie(sm.phi.copy())
sm.show("F0: deplasari pe categorii, FARA invatare", sm.evaluate(phi))
phi = train_F(sm.phi.copy())
print(f"\n[F] predictie cos(S*T, S_cons) = {pq(phi):.3f}")
sm.show("F: categorii + predictie generativa, vazute", sm.evaluate(phi))
sm.show("F: NEVAZUTE", sm.evaluate(phi, test=True))
# decodare: agent si destinatar din starea finala, prin de-bind cu locatia invatata (template 0) + cleanup
def loc(phi, sent, pos):
    _, L, _ = sm.run(sent, phi); return L[pos]
hits_A = hits_C = 0; N = 300
Lag = loc(phi, "ion da carte lui ana .".split(), 0); Lrec = loc(phi, "ion da carte lui ana .".split(), 4)
for _ in range(N):
    kind,A,B,C = sm.random_event()
    _,_,S = sm.run(sm.render(kind,A,B,C,rng.integers(3)), phi)
    ag = max(sm.persons, key=lambda p: sm.cos(S*np.conj(Lag), sm.F[sm.idx[p]]))
    rc = max(sm.persons, key=lambda p: sm.cos(S*np.conj(Lrec), sm.F[sm.idx[p]]))
    hits_A += ag == A; hits_C += rc == C
print(f"\nDecodare din orice formulare, cu locatiile din forma activa: agent {hits_A/N:.2f}, destinatar {hits_C/N:.2f} (sansa 0.17)")
