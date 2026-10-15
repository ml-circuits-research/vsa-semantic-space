import numpy as np, sim_deplasari as sm, f as F
rng = sm.rng
POS = {0: dict(A=0, B=2, C=4), 1: dict(C=0, B=2, A=5), 2: dict(B=0, C=4, A=6)}  # pozitiile rolurilor in fiecare template
def role_report(phi, tag, n=300):
    print(f"\n-- {tag} --")
    ev = ("GIVE","ion","carte","ana")
    L0 = {r: F.loc(phi, sm.render(*ev, 0), POS[0][r]) for r in "ABC"}
    for t in (0,1,2):
        Lt = {r: F.loc(phi, sm.render(*ev, t), POS[t][r]) for r in "ABC"}
        align = {r: round(sm.cos(L0[r], Lt[r]),2) for r in "ABC"}
        cross = round(sm.cos(L0["A"], Lt["C"]),2)
        acc = dict(A=0,B=0,C=0)
        for _ in range(n):
            kind,A,B,C = sm.random_event()
            _,_,S = sm.run(sm.render(kind,A,B,C,t), phi)
            for r, truth, cands in (("A",A,sm.persons),("C",C,sm.persons),("B",B,sm.objects)):
                acc[r] += max(cands, key=lambda w: sm.cos(S*np.conj(L0[r]), sm.F[sm.idx[w]])) == truth
        print(f"template {t}: aliniere locatii (agent,obiect,dest) = {align['A']},{align['B']},{align['C']} ; agent-vs-dest = {cross} ; decodare cu locatiile din T0: agent {acc['A']/n:.2f} obiect {acc['B']/n:.2f} dest {acc['C']/n:.2f}")
phi0 = F.tie(sm.phi.copy()); role_report(phi0, "F0 fara invatare")
for lr, n in [(2.0, 12000), (5.0, 30000)]:
    phi = F.train_F(sm.phi.copy(), n_sent=n, lr=lr)
    role_report(phi, f"F invatat lr={lr} n={n}, predictie={F.pq(phi):.3f}")
