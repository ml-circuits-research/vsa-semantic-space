"""
v7_nested: nesting test with a COMPOSITIONAL world (reported speech as a fibre), model unchanged.

Why. Round 5 / v6_left showed that with a FLAT world (speaker roles X1, X2, X3 indexed by depth; inner roles A, B, C depth-invariant)
no cumulative address scheme can separate X2 from X1 while keeping A depth-invariant: both are reached through the same chain prefix,
so the chain through 'says that' must sum to zero for A and to something non-zero for the speaker — a contradiction (for exact equality;
argmax decoding softens it, but v6 measures how much). The flat world, not the model, is what asks for the impossible.

The compositional world instead defines reported speech the way the model addresses: entering a reported clause binds the whole inner
state with a fixed role vector R_rep (a fibre = a copy of the space):
    S_world(X says that P) = R_X (x) f_X + R_rep (x) S_world(P)
One speaker role X (not X1..X3); depth is the number of R_rep bindings. Decoding A at depth d: argmax_x cos(S (x) conj(R_rep)^d (x)
conj(R_A), f_x); speaker at level k (0 = outermost): S (x) conj(R_rep)^k (x) conj(R_X). A new depth introduces NO new role, so the unseen
depth 3 is a fair test (EVALUATION_PRINCIPLES rule 14) of whether the learned displacements compose one level further than seen.

Model: v4 codes (path, chain, bigram) and chainL (v6_left), unchanged; no rule for 'that' or for verbs. Sentences: the English BigWorld
of v5_scale (6 frames x 16 formulations, nesting 0-3). Training depths 0-2, test depth 3.
Run: python3 v7_nested.py [--seeds 5] [--N 30000] [--D 1024] [--batch 8] [--lang en]
"""
import argparse, time
import numpy as np, torch
import v2, v5_scale as v5, v6_left as v6

class NestedRoleWorld:
    """compositional roles: A, B, C inside; X = speaker; R_rep = the fibre of a reported clause. Identities F shared with the model."""
    def __init__(self, F, world, rng):
        self.F, self.w = F, world
        self.R = {r: np.exp(1j * rng.uniform(-np.pi, np.pi, F.shape[1])) for r in ("A", "B", "C", "X", "rep")}
    def state(self, ev):
        kind, A, B, C = ev[:4]; speakers = [x for x in ev[4:] if x is not None]      # ev[4] = outermost speaker (X1), ...
        S = self.R["A"] * self.F[self.w.idx[A]] + self.R["B"] * self.F[self.w.idx[B]] + self.R["C"] * self.F[self.w.idx[C]]
        for x in reversed(speakers):                                                  # innermost speaker wraps first
            S = self.R["X"] * self.F[self.w.idx[x]] + self.R["rep"] * S
        return S

def decode_nested(m, world, w, templs, persons, objects, depth, n=20, per_role=False):
    """accuracy and margin with the WORLD's addresses, unbinding conj(R_rep)^k for the level of each role"""
    acc, marg, per = [], [], {}
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
    conj_rep = torch.conj(R["rep"])
    roles = [("A", 1, persons, depth), ("B", 2, objects, depth), ("C", 3, persons, depth)] + [(f"X{k+1}", 4 + k, persons, k) for k in range(depth)]
    for kind in w.frames:
        for t in templs:
            evs = [(kind,) + w.event(persons, objects, depth=depth)[1:] for _ in range(n)]
            idx, ids, mask = m.encode([w.render(ev, t) for ev in evs])
            with torch.no_grad(): S = m.states(idx, ids, mask)
            Sn = S / S.abs().pow(2).sum(-1, keepdim=True).sqrt()
            for r, ri, cands, level in roles:
                key = R["X"] if r.startswith("X") else R[r]
                unbind = torch.conj(key) * conj_rep ** level
                cid = torch.tensor([w.idx[x] for x in cands], device=m.device)
                sc = torch.real((Sn * unbind) @ torch.conj(Fn[cid]).T).cpu().numpy()
                truth = np.array([cands.index(ev[ri]) for ev in evs]); ok = sc.argmax(1) == truth
                acc.extend(ok); per.setdefault(r, []).extend(ok)
                sc_t = sc[np.arange(n), truth]; sc2 = sc.copy(); sc2[np.arange(n), truth] = -np.inf; marg.extend(sc_t - sc2.max(1))
    out = (float(np.mean(acc)), float(np.mean(marg)))
    return (out, {k: float(np.mean(v)) for k, v in per.items()}) if per_role else out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--variants", default="all")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang(a.lang)
    print(f"v7_nested: device={a.device} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}, D={D}, N={N}, batch={a.batch}, lr={a.lr}, seeds={a.seeds}")
    print(f"world = v5_scale ({v5.LANG}) sentences, COMPOSITIONAL roles: S = R_X f_X + R_rep (x) S_inner; one speaker role; training depths 0-{a.depth_train}, test depth 3")
    print(f"training formulations: {v5.TRAIN_T}")
    w0 = v5.BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0)
    print(f"  => new formulations with ALL transitions seen: {held_seen}; with UNSEEN transitions/categories: {held_new}")
    print(f"chance: seen identities 1/12 = {1/12:.3f}; new identities 1/6 = {1/6:.3f}. Trivial control: 'everything at the origin' = chance.\n")
    variants = [("path", ["path"]), ("chain", ["chain"]), ("chainL", ["chainL"]), ("bigram", ["bigram"]),
                ("path+chain+bigram", ["path", "chain", "bigram"]), ("path+chainL+bigram", ["path", "chainL", "bigram"])]
    if a.variants == "main": variants = variants[-2:]
    cols = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]
    results = {name: [] for name, _ in variants}; roles2 = {name: [] for name, _ in variants}; roles3 = {name: [] for name, _ in variants}
    for seed in range(a.seeds):
        w = v5.BigWorld(a.n_id, np.random.default_rng(seed)); cats = sorted(set(w.cat.values()))
        trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
        world = NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
        for name, codes in variants:
            m = v6.V6(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), codes, a.device); m.set_F(F)
            assert m.F_np is world.F
            t0 = time.time(); m.train(world, N, templates=v5.TRAIN_T, persons=trP, objects=trO, batch=a.batch, lr=a.lr, depth_train=a.depth_train)
            torch.cuda.synchronize(); t_train = time.time() - t0
            d2, pr2 = decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 2, n=15, per_role=True)
            d3, pr3 = decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 3, n=15, per_role=True)
            r = [decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 0, n=20), decode_nested(m, world, w, v5.TRAIN_T, teP, teO, 0, n=20),
                 decode_nested(m, world, w, held_seen, trP, trO, 0, n=30), decode_nested(m, world, w, held_new, trP, trO, 0, n=30),
                 decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 1, n=15), d2, d3]
            results[name].append(r); roles2[name].append([pr2[k] for k in ("A", "B", "C", "X1", "X2")]); roles3[name].append([pr3[k] for k in ("A", "B", "C", "X1", "X2", "X3")])
            print(f"  seed {seed} {name:19s} ({t_train:5.1f}s train) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r)
                  + "  || depth 2 per role: " + " ".join(f"{k} {pr2[k]:.2f}" for k in ("A", "B", "C", "X1", "X2"))
                  + "  || depth 3: " + " ".join(f"{k} {pr3[k]:.2f}" for k in ("A", "B", "C", "X1", "X2", "X3")), flush=True)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    for name, _ in variants:
        r = np.array(results[name]); print(f"  {name:19s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print("== depth 2 (seen), PER ROLE: A | B | C | X1 | X2")
    for name, _ in variants:
        r = np.array(roles2[name]); print(f"  {name:19s} " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(5)))
    print("== depth 3 (UNSEEN), PER ROLE: A | B | C | X1 | X2 | X3")
    for name, _ in variants:
        r = np.array(roles3[name]); print(f"  {name:19s} " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(6)))
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] separate axes [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] trivial control = chance [x] same regime (13) [x] no new role at new depth (14) [x] per role (15)")
