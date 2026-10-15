"""Round-trip test (R2): text -> model state -> frame(S) -> Prolog facts + query / SMT-LIB + z3 -> answer, compared with the WORLD's answer.
Model: v8 bundle (path + chainL + bigram) with the LEARNED gates (start closed), reference world (compositional, English, prefixed
reporting), trained on depths 0-2; the round trip is run on fresh sentences at depths 0-3 (depth 3 UNSEEN), with the depth ESTIMATED from S.
Reports: full-frame accuracy (all roles at once, depth included), Prolog-query accuracy, SMT consistency, per depth. 3 seeds."""
import argparse, numpy as np, torch, v2, v5_scale as v5, v7_nested as v7, v8_gates as v8, frame as fr
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--n_test", type=int, default=100)
a = ap.parse_args(); v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; D = 1024
print(f"frame_roundtrip: reference world, learned gates, {a.seeds} seeds, {a.n_test} fresh sentences per depth 0-3 (3 unseen); depth estimated from S")
res = {d: [] for d in range(4)}
for seed in range(a.seeds):
    w = v5.BigWorld(18, np.random.default_rng(seed)); trP, trO = w.persons[:12], w.objects[:12]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v7.NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], "cuda", gate_init="closed"); m.set_F(F)
    m.train(world, a.N, templates=v5.TRAIN_T, persons=trP, objects=trO, batch=8, depth_train=2)
    for d in range(4):
        evs = [w.event(trP, trO, depth=d) for _ in range(a.n_test)]; sents = [w.render(ev, int(np.random.default_rng(7).choice(v5.TRAIN_T))) for ev in evs]
        r = fr.round_trip(m, world, w, sents, evs, trP, trO, verbose=(2 if (seed == 0 and d in (0, 3)) else 0)); res[d].append(r)
        print(f"  seed {seed} depth {d}{' (UNSEEN)' if d == 3 else ''}: full-frame {r[0]:.3f}  prolog-query {r[1]:.3f}  smt-consistent {r[2]:.3f}", flush=True)
print("\n== MEANS: depth | full-frame accuracy | Prolog query (recipient & agent) | SMT consistency (z3)")
for d in range(4):
    r = np.array(res[d]); print(f"  depth {d}{' (UNSEEN)' if d == 3 else '        '}  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(3)))
print("\nchecklist: [x] world addresses [x] shared F [x] depth estimated, not given [x] full-frame (R1) [x] round trip Prolog + SMT (R2) [x] >=3 seeds [x] unseen depth")
