import numpy as np, v2
D = 512; w = v2.make_world(0); cats = sorted(set(w.cat.values()))
learned = {c: (c in {'.', 'COP', 'are', 'catre', 'cumpara', 'da', 'de', 'ii', 'vinde', 'la'}) for c in cats}   # rezultatul cautarii (log_gates.txt)
for name, gates in [("porti INVATATE", learned), ("porti de mana", v2.HAND_GATES(cats))]:
    accs = []
    for seed in range(3):
        m = v2.V2(w, D, np.random.default_rng(200 + seed), gates, dual=True); world = v2.RoleWorld(m.F, w, np.random.default_rng(100 + seed))
        m.train(world, 12000, templates=[0,1,2,3]); res, mean = v2.evaluate(m.state, world, w, n=100)
        accs.append([np.mean([v for (k,t),a in res.items() if t < 4 for v in a.values()]), np.mean([v for (k,t),a in res.items() if t == 4 for v in a.values()])])
    a = np.array(accs); print(f"{name:16s} emitatori={sorted(c for c in cats if gates[c])}\n{'':16s} decodare T0-T3 {a[:,0].mean():.3f}±{a[:,0].std():.3f}   T4 nevazuta {a[:,1].mean():.3f}±{a[:,1].std():.3f}", flush=True)
