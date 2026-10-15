import numpy as np, sim_deplasari as sm
rng = sm.rng
def pred_acc(phi, n=300):
    hit_C = hit_B = 0
    for _ in range(n):
        kind, A, B, C = sm.random_event()
        ids, L, S = sm.run(sm.render(kind, A, B, C, rng.integers(3)), phi)
        L2 = S/(np.abs(S)+1e-9)
        z = np.real(sm.M2[None,:]*np.conj(L2)[None,:]*np.conj(sm.F)).mean(1)
        hit_C += max(sm.persons, key=lambda p: z[sm.idx[p]]) == C
        hit_B += max(sm.objects, key=lambda o: z[sm.idx[o]]) == B
    return hit_C/n, hit_B/n
for lr, n in [(0.5, 8000), (3.0, 8000), (3.0, 20000)]:
    sm.M2[:] = 0
    phi = sm.train_consequence(sm.phi.copy(), n_sent=n, lr=lr)
    r = sm.evaluate(phi)
    print(f"lr={lr} n={n}: pred (C,B) acc={pred_acc(phi)}  parafraze={r['parafraze']:.3f} swap={r['roluri_schimbate']:.3f} aleator={r['aleator']:.3f} para>swap={r['parafraza_bate_schimbul']:.2f}")
