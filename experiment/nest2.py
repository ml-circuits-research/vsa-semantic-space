"""Compozitie ierarhica cu v2/v3: "X spune ca <propozitie>" FARA nicio regula pentru 'ca' sau pentru verbe.
Lumea: roluri explicite X (vorbitor), A, B, C. Comparatie: drum singur (comutativ) vs dual (asteptari + drum)."""
import numpy as np, time, v2, k as kk
D = 1024
class NW(kk.NestedWorld):
    pass
def run_nested(dual_gates, label, seeds=(0, 1)):
    accs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        persons = [f"p{i}" for i in range(8)]; objects = [f"o{i}" for i in range(8)]
        w = NW(persons, objects, rng)
        for v in ["e", "este"]: w.cat[v] = "COP"
        for v in ["da", "vinde", "primeste", "cumpara", "dat", "vandut", "spune", "zice"]: w.cat[v] = v
        cats = sorted(set(w.cat.values()))
        gates = dual_gates(cats)
        m = v2.V2(w, D, np.random.default_rng(200 + seed), gates, dual=True)
        if label.startswith("drum"):
            m.run_orig = m.run; m.run = (lambda toks, m=m: (lambda r: (r[0][len(toks):], r[1][len(toks):]))(m.run_orig(toks)))
        world = v2.RoleWorld(m.F, w, np.random.default_rng(100 + seed), roles=("A", "B", "C", "X"))
        # antrenare: toate formele exterioare, formularile interioare 0-3 (T4 interioara nevazuta)
        t0 = time.time()
        for n in range(16000):
            ev = w.event(); toks = w.render(ev, m.rng.choice([0, 1, 2, 3]))
            terms, members = m.run(toks); c, g = m.grad_cos(terms, members, world.state(ev))
            m._k += 1; m._m = 0.9 * m._m + 0.1 * g; m._v = 0.999 * m._v + 0.001 * g * g
            m.theta += 0.02 * (m._m / (1 - 0.9 ** m._k)) / (np.sqrt(m._v / (1 - 0.999 ** m._k)) + 1e-8); m.theta[0] = 0
        acc = {"A": [], "B": [], "C": [], "X": []}; acc4 = []
        for to in range(3):
            for t in range(5):
                for _ in range(60):
                    ev = w.event(); S = m.state(w.render(ev, t, to=to))
                    ok = {r: max(cands, key=lambda x: v2.cos(S * np.conj(world.R[r]), world.F[w.idx[x]])) == truth
                          for r, truth, cands in (("A", ev[1], w.persons), ("B", ev[2], w.objects), ("C", ev[3], w.persons), ("X", ev[4], w.persons))}
                    if t < 4:
                        for r in ok: acc[r].append(ok[r])
                    else: acc4.append(np.mean(list(ok.values())))
        accs.append([np.mean(acc["X"]), np.mean(acc["A"]), np.mean(acc["B"]), np.mean(acc["C"]), np.mean(acc4)])
        print(f"  seed {seed} {label} ({time.time()-t0:.0f}s): vorbitor {accs[-1][0]:.2f} agent {accs[-1][1]:.2f} obiect {accs[-1][2]:.2f} dest {accs[-1][3]:.2f} | T4 interioara nevazuta {accs[-1][4]:.2f}", flush=True)
    a = np.array(accs); print(f"  MEDIE {label}: " + " ".join(f"{x:.3f}" for x in a.mean(0)))
print("=== 'X spune ca <prop>' (3 forme exterioare x 5 interioare x 2 cadre), 8 identitati (sansa 0.125), D=1024 ===")
run_nested(lambda cats: {c: False for c in cats}, "drum singur (comutativ)")
run_nested(v2.HAND_GATES, "dual: asteptari + drum (porti: tot ce nu e P/O)")
