"""
v6_left: §5.1b — RETROACTIVE RE-PLACEMENT through LEFTWARD expectations (union then restriction), for identical prefixes with different roles.
Problem (round 5, diag. 2): '{X1} spune ca {A} V ..' and '{X1} spune ca {X2} zice ca ..' have an identical prefix; any address from the prefix
is identical for A and X2. The role is decided by the NEXT word.
Mechanism (the same step for every word; no rule for the verb or for 'ca'/'that'):
  - as in v2: a word takes the most recent unconsumed expectation for its category; an EMITTER (open gate) emits rightward expectations
    for every category, relative to its own address;
  - NEW: an emitter also emits LEFTWARD expectations: for every category c2, the most recent NON-emitter word of category c2 before it,
    not yet re-placed, is RE-PLACED at address(emitter) (x) L[c_emitter, c2]. Re-placement = the old term is subtracted from the state, the
    new one is added (the bundle is additive; on the final state this is equivalent to replacing the address). A word is re-placed only once,
    by the FIRST emitter that follows it; emitters anchor (they are not re-placed, so that what was placed relative to them does not move).
  Thus: 'ca X2 zice' -> X2 = left of 'zice'; 'ca A V' -> A = left of V. The identical prefix no longer matters; what follows decides.
New parameters: L[c, c2] (phases), a category x category matrix, like A[c, c2] from v2.
The code is called 'chainL' and replaces 'chain' in the bundle: path+chainL+bigram vs path+chain+bigram (same regime, PRINCIPLES 13).
Criterion (judgment 2026-10-15): X2 at depth 2 > 0.9, with no PER-ROLE losses in the round 5 table.
Run: python3 v6_left.py [--seeds 5] [--N 30000] [--D 1024] [--batch 8]   (same world as v5_scale.py)
"""
import argparse, time
import numpy as np, torch
import v2, v4, v4_torch, v5_scale as v5

def run_left(model, toks):
    """v2.V2.run + leftward expectations. Returns (None, placed): the member lists (parameter indices) per word."""
    w = model.w; pool = []; placed = [None] * len(toks); emitted = {}; buffer = []; releft = set()
    for p, tok in enumerate(toks):
        c = w.cat[tok]
        e = next((e for e in reversed(pool) if e["cat"] == c and not e["used"]), None)
        if e is not None: e["used"] = True; placed[p] = list(e["members"])
        else: placed[p] = [model.pid["root"]]; buffer.append(p)
        if model.gates.get(c, False):
            emitted[p] = []
            for c2 in model.cats:
                ent = {"cat": c2, "members": placed[p] + [model.pid[f"A:{c}:{c2}"]], "used": False, "emitter": p}
                pool.append(ent); emitted[p].append(ent)
            for q in list(buffer):                                   # v2: the buffered words move onto the new expectations
                if q == p: continue
                if emitted.get(q) and any(x["used"] for x in emitted[q]): continue
                ent = next((x for x in emitted[p] if x["cat"] == w.cat[toks[q]] and not x["used"]), None)
                if ent is None: continue
                ent["used"] = True; placed[q] = list(ent["members"]); buffer.remove(q)
                for x in emitted.get(q, []): x["members"] = placed[q] + x["members"][-1:]
            # NEW: leftward expectations — the most recent non-emitter of each category, not yet re-placed
            for c2 in model.cats:
                q = next((q for q in range(p - 1, -1, -1) if w.cat[toks[q]] == c2 and not model.gates.get(c2, False) and q not in releft), None)
                if q is None: continue
                placed[q] = placed[p] + [model.pid[f"L:{c}:{c2}"]]; releft.add(q)
                if q in buffer: buffer.remove(q)
    return None, placed

class V6(v5.V5):
    """V5 (large world) + the L[c, c2] parameters; the 'chainL' code = the expectation chain with leftward re-placement"""
    def __init__(self, world, D, rng, gates, codes, device, max_len=40):
        super().__init__(world, D, rng, gates, codes, device, max_len)
        m = self.np_model; extra = [f"L:{c1}:{c2}" for c1 in m.cats for c2 in m.cats]
        m.names += extra; m.pid = {n: i for i, n in enumerate(m.names)}
        th = np.concatenate([self.theta.detach().cpu().numpy(), rng.uniform(-np.pi, np.pi, (len(extra), D))]); self.P = len(m.names)
        self.theta = torch.nn.Parameter(torch.tensor(th, dtype=self.dtype, device=device))
        self.root_mask = torch.ones(self.P, 1, dtype=self.dtype, device=device); self.root_mask[0] = 0

# 'chainL' in build_terms: same place as 'chain', different run
_orig_build = v4_torch.build_terms
def build_terms(model, toks):
    members, ids = _orig_build(model, toks)
    if "chainL" in model.codes:
        _, placed = run_left(model, toks); members = placed + members; ids = [model.w.idx[t] for t in toks] + ids
    return members, ids
v4_torch.build_terms = build_terms

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--variants", default="all")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang(a.lang)
    print(f"v6_left: device={a.device} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}, D={D}, N={N}, batch={a.batch}, lr={a.lr}, seeds={a.seeds}")
    print(f"world = v5_scale ({v5.LANG}; 6 frames x 16 formulations, {a.n_id}+{a.n_id} identities, nesting 0-{a.depth_train} train, 3 test). training formulations: {v5.TRAIN_T}")
    w0 = v5.BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0)
    print(f"  => new formulations with ALL transitions seen: {held_seen}; with UNSEEN transitions/categories: {held_new}")
    # trace (PRINCIPLES 11): the addresses (parameter names) for every word, chain vs chainL, on the identical prefix
    for codes in (["chain"], ["chainL"]):
        m0 = V6(w0, 8, np.random.default_rng(0), v2.HAND_GATES(sorted(set(w0.cat.values()))), codes, "cpu")
        for s in (("p0 says that p1 gives o0 to p2 .", "p0 says that p1 claims that p2 gives o0 to p3 .") if v5.LANG == "en" else ("p0 spune ca p1 da o0 lui p2 .", "p0 spune ca p1 zice ca p2 da o0 lui p3 .")):
            toks = s.split()
            members, _ = (run_left(m0.np_model, toks)[1], None) if codes == ["chainL"] else (v2.V2.run(m0.np_model, toks)[1], None)
            print(f"  {codes[0]:6s} {s}\n         " + " | ".join(f"{t}:" + "+".join(m0.np_model.names[i].replace('A:', '').replace('L:', 'L') for i in mem if i) for t, mem in zip(toks, members)))
    print(f"chance: seen identities 1/12 = {1/12:.3f}; new identities 1/6 = {1/6:.3f}. Trivial control: 'everything at the origin' = chance.\n")
    variants = [("chain", ["chain"]), ("chainL", ["chainL"]), ("path+chain+bigram", ["path", "chain", "bigram"]), ("path+chainL+bigram", ["path", "chainL", "bigram"])]
    if a.variants == "L": variants = [v for v in variants if "chainL" in v[1]]
    cols = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "depth 1", "depth 2 (seen)", "depth 3 (A,B,C,X1,X2)"]
    roles_cols = ["depth 2: A", "B", "C", "X1", "X2"]
    results = {name: [] for name, _ in variants}; roles_res = {name: [] for name, _ in variants}
    for seed in range(a.seeds):
        w = v5.BigWorld(a.n_id, np.random.default_rng(seed)); cats = sorted(set(w.cat.values()))
        trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
        world = v5.BigRoleWorld(F, w, np.random.default_rng(100 + seed))
        for name, codes in variants:
            m = V6(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), codes, a.device); m.set_F(F)
            assert m.F_np is world.F
            t0 = time.time(); m.train(world, N, templates=v5.TRAIN_T, persons=trP, objects=trO, batch=a.batch, lr=a.lr, depth_train=a.depth_train)
            torch.cuda.synchronize(); t_train = time.time() - t0
            r = [v5.decode_stats(m, world, w, v5.TRAIN_T, trP, trO, 0, n=20), v5.decode_stats(m, world, w, v5.TRAIN_T, teP, teO, 0, n=20),
                 v5.decode_stats(m, world, w, held_seen, trP, trO, 0, n=30), v5.decode_stats(m, world, w, held_new, trP, trO, 0, n=30),
                 v5.decode_stats(m, world, w, v5.TRAIN_T, trP, trO, 1, n=15), v5.decode_stats(m, world, w, v5.TRAIN_T, trP, trO, 2, n=15),
                 v5.decode_stats(m, world, w, v5.TRAIN_T, trP, trO, 3, n=20, max_role_depth=a.depth_train)]
            # per role at depth 2 (criterion: X2 > 0.9)
            import diag_v5 as dg
            o = {}
            for t in v5.TRAIN_T:
                for k, v in dg.per_role(m, world, w, t, trP, trO, 2, n=15).items(): o.setdefault(k, []).extend(v)
            pr = [float(np.mean(o[r])) for r in ("A", "B", "C", "X1", "X2")]
            results[name].append(r); roles_res[name].append(pr)
            print(f"  seed {seed} {name:19s} ({t_train:5.1f}s train) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r) + "  || depth 2 per role: " + " ".join(f"{c} {x:.2f}" for c, x in zip("A B C X1 X2".split(), pr)), flush=True)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    for name, _ in variants:
        r = np.array(results[name]); print(f"  {name:19s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print("== depth 2, PER ROLE (means): A | B | C | X1 | X2")
    for name, _ in variants:
        r = np.array(roles_res[name]); print(f"  {name:19s} " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(5)))
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] separate axes [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] trivial control = chance [x] same regime (13) [x] target roles (14) [x] per role (15)")
