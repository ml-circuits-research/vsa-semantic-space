import numpy as np, sim_deplasari as sm
rng = sm.rng
phi_world = sm.phi.copy()   # starea consecintei e "senzoriala": codata cu deplasari fixe, neinvatate
def train_fixedT2(phi, n_sent=12000, lr=2.0):
    for n in range(n_sent):
        kind, A, B, C = sm.random_event()
        ids, L, S = sm.run(sm.render(kind, A, B, C, rng.integers(3)), phi)
        _, _, Sc = sm.run(sm.consequence(kind, A, B, C), phi_world)
        c, gs = sm.grad_cos_wrt_phi(ids, L, S, Sc * np.conj(sm.T_fix))
        grad = np.zeros_like(phi)
        for s_i, s in enumerate(ids): grad[s] -= gs[s_i]
        phi -= lr * grad
    return phi
def pq(phi, n=300):
    cs=[]
    for _ in range(n):
        kind,A,B,C = sm.random_event()
        _,_,S = sm.run(sm.render(kind,A,B,C,rng.integers(3)), phi)
        _,_,Sc = sm.run(sm.consequence(kind,A,B,C), phi_world)
        cs.append(sm.cos(S*sm.T_fix, Sc))
    return np.mean(cs)
phiA = sm.train_predict(sm.phi.copy())
for tag, start in [("E2 singur", sm.phi.copy()), ("A apoi E2", phiA)]:
    phi = train_fixedT2(start)
    print(f"\n[{tag}] predictie cos(S*T, S_cons) = {pq(phi):.3f}")
    sm.show(tag+", vazute", sm.evaluate(phi)); sm.show(tag+", NEVAZUTE", sm.evaluate(phi, test=True))
