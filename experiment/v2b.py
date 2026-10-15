import numpy as np, time, v2, camp as cp
D = 512; N = 12000; rows = {"v2 aditiv (out[c1]+in[c2]), porti de mana": []}
for seed in range(3):
    w = v2.make_world(seed); rng = np.random.default_rng(100 + seed); cats = sorted(set(w.cat.values()))
    trP, teP, trO, teO = w.persons[:8], w.persons[8:], w.objects[:8], w.objects[8:]
    m = v2.V2(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), additive=True); world = v2.RoleWorld(m.F, w, rng)
    t0 = time.time(); m.train(world, N, templates=[0,1,2,3], persons=trP, objects=trO)
    r = [v2.evaluate(m.state, world, w, templates=[0,1,2,3], persons=trP, objects=trO)[1], v2.evaluate(m.state, world, w, templates=[0,1,2,3], persons=teP, objects=teO)[1],
         v2.evaluate(m.state, world, w, templates=[4], persons=trP, objects=trO)[1], v2.evaluate(m.state, world, w, templates=[4], persons=teP, objects=teO)[1]]
    rows[list(rows)[0]].append(r); print(f"  seed {seed} ({time.time()-t0:.0f}s): {[round(x,3) for x in r]}", flush=True)
    if seed == 0:
        res, _ = v2.evaluate(m.state, world, w, n=150)
        for (kind, t), a in res.items(): print(f"    {kind} T{t}: agent {a['A']:.2f} obiect {a['B']:.2f} dest {a['C']:.2f}")
a = np.array(rows[list(rows)[0]]); print("  MEDIE:", "  ".join(f"{x:.3f}±{s:.3f}" for x, s in zip(a.mean(0), a.std(0))))
