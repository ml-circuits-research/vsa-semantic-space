import numpy as np, time, v2
D = 512; N = 12000; rows = {"v3 = lant de asteptari (+) drum pe categorii": [], "drum pe categorii singur (fara buffer, fara verb)": []}
for seed in range(3):
    w = v2.make_world(seed); rng = np.random.default_rng(100 + seed); cats = sorted(set(w.cat.values()))
    trP, teP, trO, teO = w.persons[:8], w.persons[8:], w.objects[:8], w.objects[8:]
    for name, gates in [(list(rows)[0], v2.HAND_GATES(cats)), (list(rows)[1], {c: False for c in cats})]:
        m = v2.V2(w, D, np.random.default_rng(200 + seed), gates, dual=True); world = v2.RoleWorld(m.F, w, rng)
        if name.startswith("drum"):   # fara asteptari: termenii din lant sunt toti la root; ii eliminam prin masca: folosim doar drumul
            m.run_orig = m.run
            def run_path(toks, m=m):
                terms, placed = m.run_orig(toks); k = len(toks); return terms[k:], placed[k:]
            m.run = run_path
        t0 = time.time(); m.train(world, N, templates=[0,1,2,3], persons=trP, objects=trO)
        r = [v2.evaluate(m.state, world, w, templates=[0,1,2,3], persons=trP, objects=trO)[1], v2.evaluate(m.state, world, w, templates=[0,1,2,3], persons=teP, objects=teO)[1],
             v2.evaluate(m.state, world, w, templates=[4], persons=trP, objects=trO)[1], v2.evaluate(m.state, world, w, templates=[4], persons=teP, objects=teO)[1]]
        rows[name].append(r); print(f"  seed {seed} {name} ({time.time()-t0:.0f}s): {[round(x,3) for x in r]}", flush=True)
        if seed == 0:
            res, _ = v2.evaluate(m.state, world, w, n=150)
            for (kind, t), a in res.items(): print(f"    {kind} T{t}: agent {a['A']:.2f} obiect {a['B']:.2f} dest {a['C']:.2f}")
for k, v in rows.items():
    a = np.array(v); print(f"  MEDIE {k:48s}", "  ".join(f"{x:.3f}±{s:.3f}" for x, s in zip(a.mean(0), a.std(0))))
