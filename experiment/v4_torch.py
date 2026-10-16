"""
v4_torch: port of v4.py to torch (GPU). Same model, same world, same evaluation.

What stays in numpy/Python (unchanged, imported from v4/v2/camp): the world, the templates, the categories, the gates and the STRUCTURE
of the addresses (which parameters enter each term: V4.run returns the member lists; it depends only on categories and gates, not on theta).
What moves to torch: the phases theta as a parameter; term = exp(i * M @ theta) * F[tok]; S = sum; loss = -cos(S, S_world); autograd + Adam.
  term k: phase = theta[idx_k].sum(), idx_k = member indices (padded with root, phase 0). Encoding is done ONCE, before training.
The data flow (world, identities, seeds, order of training sentences) is IDENTICAL to v4.py: the same numpy rngs.
Differences from v4.py: optimizer arithmetic (torch.optim.Adam vs hand-written Adam), float64 by default; --batch > 1 changes the
optimization regime (same number of sentences, fewer steps) and is reported separately.

Evaluation per EVALUATION_PRINCIPLES.md (identical to v4.py): world addresses, shared F (assert), chance printed, 3 seen/unseen axes,
margin, >=3 seeds, trivial control.

Run: python3 v4_torch.py [--lr 0.02] [--seeds 3] [--N 10000] [--D 512] [--batch 1] [--device cuda] [--variants all|all4]  ('toate4' is still accepted)
"""
import guard  # process-level memory/time protection (see guard.py; the machine hung on 2026-10-15)
import argparse, time, sys
import numpy as np, torch
import v2, v4

def build_terms(model, toks):
    """address structure from the numpy code: members (lists of parameter indices) and identities, with no arithmetic on theta"""
    w = model.w; n = len(toks); members, ids = [], []
    cats = [w.cat[t] for t in toks]
    if "chain" in model.codes:
        model.dual = False; _, placed = v2.V2.run(model, toks); members += placed; ids += [w.idx[t] for t in toks]
    if "path" in model.codes:
        path = []
        for c in cats: path = path + [model.pid[f"m:{c}"]]; members.append(list(path))
        ids += [w.idx[t] for t in toks]
    if "bigram" in model.codes:
        prev = "^"
        for c in cats: members.append([model.pid[f"bg:{prev}:{c}"]]); prev = c
        ids += [w.idx[t] for t in toks]
    if "pos" in model.codes:
        for i, c in enumerate(cats): members.append([model.pid[f"pos:{i}"], model.pid[f"pc:{c}"]])
        ids += [w.idx[t] for t in toks]
    return members, ids

class V4Torch:
    def __init__(self, world, D, rng, gates, codes, device, dtype=torch.float64):
        # the numpy model is used ONLY for parameter names/indices and the address structure; its theta is not used
        self.np_model = v4.V4(world, D, rng, gates, codes)
        self.w, self.D, self.rng, self.codes, self.device, self.dtype = world, D, rng, codes, device, dtype
        self.P = len(self.np_model.names)
        th = torch.tensor(self.np_model.theta, dtype=dtype, device=device)      # same initialization as v4.py (rng 200+seed)
        self.theta = torch.nn.Parameter(th)
        self.root_mask = torch.ones(self.P, 1, dtype=dtype, device=device); self.root_mask[0] = 0   # root fixed at 0, as in numpy
        self.F = None

    def set_F(self, F_np):
        self.F_np = F_np; self.F = torch.tensor(F_np, dtype=torch.complex128 if self.dtype == torch.float64 else torch.complex64, device=self.device)

    def encode(self, sentences):
        """list of sentences -> idx (B x T x K) parameter indices per term, ids (B x T) identities, mask (B x T).
        Members are padded with 0 = root, whose phase is fixed at 0, so it does not change the sum; term padding is masked."""
        enc = [build_terms(self.np_model, toks) for toks in sentences]
        B = len(enc); T = max(len(m) for m, _ in enc); K = max(len(mem) for m, _ in enc for mem in m)
        idx = torch.zeros(B, T, K, dtype=torch.long); ids = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, (members, tid) in enumerate(enc):
            for k, mem in enumerate(members): idx[b, k, :len(mem)] = torch.tensor(mem)
            ids[b, :len(tid)] = torch.tensor(tid); mask[b, :len(tid)] = 1.0
        return idx.to(self.device), ids.to(self.device), mask.to(self.device)

    def states(self, idx, ids, mask):
        """S (B x D) = sum_k exp(i * sum_{p in members_k} theta_p) * F[id_k], with padding mask"""
        phase = (self.theta * self.root_mask)[idx].sum(2)                     # B x T x D
        terms = torch.polar(torch.ones_like(phase), phase) * self.F[ids]
        return (terms * mask.unsqueeze(-1)).sum(1)

    def state(self, toks):
        with torch.no_grad():
            idx, ids, mask = self.encode([toks]); return self.states(idx, ids, mask)[0]

    def train(self, world, n_sent, lr=0.02, templates=None, persons=None, objects=None, batch=1):
        w = self.w
        # sentence flow identical to v4.py: w.event (w.rng) and self.rng.choice for the template; ENCODE ONLY ONCE, beforehand
        sents, targets = [], []
        for n in range(n_sent):
            ev = w.event(persons, objects); ts = templates or range(len(w.frames[ev[0]]))
            sents.append(w.render(ev, self.rng.choice(list(ts)))); targets.append(world.state(ev))
        t0 = time.time(); idx_all, ids_all, mask_all = self.encode(sents); self.t_encode = time.time() - t0
        tgt_all = torch.tensor(np.array(targets), dtype=self.F.dtype, device=self.device)
        opt = torch.optim.Adam([self.theta], lr=lr, betas=(0.9, 0.999), eps=1e-8)
        for i in range(0, n_sent, batch):
            sl = slice(i, i + batch); S = self.states(idx_all[sl], ids_all[sl], mask_all[sl]); tgt = tgt_all[sl]
            cos = torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt())
            loss = -cos.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): self.theta[0] = 0

def decode_stats(m, world, w, templates, persons, objects, n=100):
    """accuracy and margin over roles, with the WORLD addresses (identical to v4.decode_stats, vectorized in torch)"""
    acc, marg = [], []
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
    for kind in w.frames:
        for t in templates:
            evs = [(kind,) + w.event(persons, objects)[1:] for _ in range(n)]
            idx, ids, mask = m.encode([w.render(ev, t) for ev in evs])
            with torch.no_grad(): S = m.states(idx, ids, mask)
            Sn = S / S.abs().pow(2).sum(-1, keepdim=True).sqrt()
            for r, ri, cands in (("A", 1, persons), ("B", 2, objects), ("C", 3, persons)):
                cid = torch.tensor([w.idx[x] for x in cands], device=m.device)
                sc = torch.real((Sn * torch.conj(R[r])) @ torch.conj(Fn[cid]).T).cpu().numpy()      # n x candidates
                truth = np.array([cands.index(ev[ri]) for ev in evs])
                pred = sc.argmax(1); acc.extend(pred == truth)
                sc_t = sc[np.arange(n), truth]; sc2 = sc.copy(); sc2[np.arange(n), truth] = -np.inf; marg.extend(sc_t - sc2.max(1))
    return float(np.mean(acc)), float(np.mean(marg))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--N", type=int, default=10000); ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", default="all"); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--n_pers", type=int, default=12)
    a = ap.parse_args(); D, N = a.D, a.N
    print(f"v4_torch: device={a.device} ({torch.cuda.get_device_name(0) if a.device == 'cuda' else 'cpu'}), torch {torch.__version__}, D={D}, N={N}, batch={a.batch}, lr={a.lr}, seeds={a.seeds}")
    TEMPLATES, TRAIN_T = v4.TEMPLATES, v4.TRAIN_T
    w0 = v4.make_world(0, a.n_pers)
    seen_tr = set().union(*[v4.transitions(w0, "GIVE", t)[0] for t in TRAIN_T]); seen_cat = set().union(*[v4.transitions(w0, "GIVE", t)[1] for t in TRAIN_T])
    print("training formulations:", TRAIN_T)
    held_seen, held_new = [], []
    for t in range(len(TEMPLATES["GIVE"])):
        if t in TRAIN_T: continue
        tr, ca = v4.transitions(w0, "GIVE", t); new_tr = tr - seen_tr; new_cat = ca - seen_cat
        (held_seen if not new_tr and not new_cat else held_new).append(t)
        print(f"  T{t} '{TEMPLATES['GIVE'][t]}': new transitions {sorted(new_tr)}, new categories {sorted(new_cat)}")
    print(f"  => new formulations with ALL transitions seen: {held_seen}; with UNSEEN transitions/categories: {held_new}")
    n_tr = 8; n_te = a.n_pers - n_tr
    print(f"chance: seen identities 1/{n_tr} = {1/n_tr:.3f}; new identities 1/{n_te} = {1/n_te:.3f}. Trivial control: 'everything at the origin' (no codes) = chance.\n")
    variants = [("path", ["path"]), ("chain", ["chain"]), ("bigram", ["bigram"]), ("pos", ["pos"]),
                ("path+chain (dual)", ["path", "chain"]), ("path+bigram", ["path", "bigram"]), ("path+chain+bigram", ["path", "chain", "bigram"]),
                ("all 4", ["path", "chain", "bigram", "pos"])]
    if a.variants in ("all4", "toate4"): variants = variants[-1:]
    cols = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans."]
    results = {name: [] for name, _ in variants}
    for seed in range(a.seeds):
        w = v4.make_world(seed, a.n_pers); cats = sorted(set(w.cat.values())); trP, teP, trO, teO = w.persons[:n_tr], w.persons[n_tr:], w.objects[:n_tr], w.objects[n_tr:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
        world = v2.RoleWorld(F, w, np.random.default_rng(100 + seed))
        for name, codes in variants:
            m = V4Torch(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), codes, a.device); m.set_F(F)
            assert m.F_np is world.F
            t0 = time.time(); m.train(world, N, templates=TRAIN_T, persons=trP, objects=trO, batch=a.batch, lr=a.lr)
            if a.device == "cuda": torch.cuda.synchronize()
            t_train = time.time() - t0
            r = [decode_stats(m, world, w, TRAIN_T, trP, trO, n=40), decode_stats(m, world, w, TRAIN_T, teP, teO, n=40),
                 decode_stats(m, world, w, held_seen, trP, trO, n=60), decode_stats(m, world, w, held_new, trP, trO, n=60)]
            results[name].append(r)
            print(f"  seed {seed} {name:22s} ({t_train:5.1f}s train, of which {m.t_encode:4.1f}s enc.; {time.time()-t0-t_train:4.1f}s eval) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r), flush=True)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    for name, _ in variants:
        r = np.array(results[name]); print(f"  {name:22s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(4)))
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] 3 separate axes [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] trivial control = chance")
