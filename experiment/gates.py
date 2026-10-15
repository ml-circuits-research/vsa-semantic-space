"""Pot fi INVATATE declansatoarele? Cautare discreta (coordinate descent) a portilor g_c pe criteriul predictiei generative (cos held-out)."""
import numpy as np, time, v2
D = 512; N = 6000
def margin(m, world, w, templates, n=150):
    """marja medie: cos(candidat corect) - max cos(candidat gresit), pe rolurile agent/dest"""
    ms = []
    for _ in range(n):
        ev = w.event(); t = w.rng.choice(templates); S = m.state(w.render(ev, t))
        for r, truth in (("A", ev[1]), ("C", ev[3])):
            sc = {x: v2.cos(S * np.conj(world.R[r]), world.F[w.idx[x]]) for x in w.persons}
            ms.append(sc[truth] - max(v for k, v in sc.items() if k != truth))
    return float(np.mean(ms))
w = v2.make_world(0); cats = sorted(set(w.cat.values())); rng = np.random.default_rng(7)
print("categorii:", cats)
def fit(gates, seed=0, dual=True, n=N):
    m = v2.V2(w, D, np.random.default_rng(200 + seed), gates, dual=dual); world = v2.RoleWorld(m.F, w, np.random.default_rng(100 + seed))
    m.train(world, n, templates=[0,1,2,3]); return m, world
print("\n== marjele de decodare (cat de 'curata' e adresa): drum singur vs asteptari (porti de mana) vs dual ==")
for name, gates, dual in [("drum singur", {c: False for c in cats}, True), ("asteptari, porti de mana", v2.HAND_GATES(cats), False), ("dual", v2.HAND_GATES(cats), True)]:
    m, world = fit(gates, dual=dual)
    if name == "drum singur":
        m.run_orig = m.run; m.run = (lambda toks, m=m: (lambda r: (r[0][len(toks):], r[1][len(toks):]))(m.run_orig(toks)))
        m, world = fit(gates, dual=True); m.run_orig = m.run; m.run = (lambda toks, m=m: (lambda r: (r[0][len(toks):], r[1][len(toks):]))(m.run_orig(toks)))
    print(f"  {name:28s} marja T0-T3: {margin(m, world, w, [0,1,2,3]):.3f}   predictie cos: {v2.pred_cos(m.state, world, w, templates=[0,1,2,3]):.3f}")

print("\n== cautarea portilor (dual; criteriu: cos held-out al predictiei) ==")
gates = {c: False for c in cats}
m, world = fit(gates); best = v2.pred_cos(m.state, world, w, templates=[0,1,2,3]); print(f"  start (toate inchise): {best:.3f}")
for sweep in range(2):
    changed = False
    for c in cats:
        g2 = dict(gates); g2[c] = not g2[c]
        m, world = fit(g2); sc = v2.pred_cos(m.state, world, w, templates=[0,1,2,3])
        if sc > best + 0.005: gates, best, changed = g2, sc, True; print(f"  sweep {sweep}: {'deschide' if g2[c] else 'inchide'} {c:8s} -> {sc:.3f}  ACCEPTAT", flush=True)
        else: print(f"  sweep {sweep}: {'deschide' if g2[c] else 'inchide'} {c:8s} -> {sc:.3f}", flush=True)
    if not changed: break
print("\n  porti invatate (emitatori):", sorted(c for c in cats if gates[c]))
print("  porti de mana (emitatori): ", sorted(c for c in cats if v2.HAND_GATES(cats)[c]))
m, world = fit(gates, n=12000)
res, mean = v2.evaluate(m.state, world, w, n=150)
print(f"  decodare cu portile invatate: media toate formularile {mean:.3f}; T4 (nevazuta): {res[('GIVE',4)]}")
