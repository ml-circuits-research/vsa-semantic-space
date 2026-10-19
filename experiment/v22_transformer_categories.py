"""
v22_transformer_categories: the 7b transformer control (v9_transformer.py) with ONE change: the transformer also receives the word
CATEGORIES, as the reader does. Closes the input gap named by the external review of the paper draft (judgment after 753287f; Astra,
2026-10-19): on the constructed worlds the reader is given each word's category and the 7b transformer is not, so the 1.000 vs 0.215 on new
identities did not isolate the mechanism.

Change (one, rule 23): the input of token t is Linear(F[w_t]) + Emb(category(w_t)), with Emb a learned table over the world's categories
(32 on this world; P for every person, O for every object, every other word its own category, exactly the reader's inventory, w.cat).
Everything else is v9: same world (v7 NestedRoleWorld, prefixed reporting), same fixed identities F, same objective -cos(S, S_world), same
decoder (v7.decode_nested), same 30k sentences, same epochs, same tuning grid selected on seed 0 by training-distribution validation, same
axes. Held-out persons and objects carry their category (P / O) at the input, as for the reader.

CRITERION written before the run:
  - if new identities reach > 0.9 (5 seeds), the 1.000 vs 0.215 gap of 7b came from the category input, not from the mechanism; the paper's
    new-identity claim is restated as "with the same categories the transformer also reads new identities" and the mechanism claim rests on
    depth and on WikiSQL;
  - if new identities stay < 0.5, the gap is the mechanism: the transformer, given the categories, still does not carry an unseen vector to
    its role; the paper says the control was run and the comparison now isolates the mechanism on this axis;
  - in between: reported as is, with both readings stated.
  Depth 3 (unseen) is reported on the same rule with thresholds 0.9 / 0.5.
Run: python3 v22_transformer_categories.py [--seeds 5] [--device cpu]
"""
import argparse, time, sys
import numpy as np, torch, torch.nn as nn
import v5_scale as v5, v7_nested as v7
from v9_transformer import TransformerState, TModel


class TransformerStateCat(TransformerState):
    def __init__(self, D, n_cat, **kw):
        super().__init__(D, **kw); self.cat = nn.Embedding(n_cat, self.inp.out_features)
    def forward(self, x, mask, cats):
        h = self.drop(self.inp(x) + self.cat(cats) + self.pe[: x.shape[1]].unsqueeze(0))
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        o = self.out(pooled); D = o.shape[1] // 2
        return torch.complex(o[:, :D], o[:, D:])


class TModelCat(TModel):
    """v9's TModel with the category ids carried next to the identities; same interface for v7.decode_nested"""
    def __init__(self, world, D, device, seed, **kw):
        self.w, self.D, self.device = world, D, device; self.dtype = torch.float32
        self.cats = sorted(set(world.cat.values())); self.cat_id = {c: i for i, c in enumerate(self.cats)}
        torch.manual_seed(seed); self.net = TransformerStateCat(D, len(self.cats), **kw).to(device); self.F = None
        self.n_params = sum(p.numel() for p in self.net.parameters())
    def encode(self, sentences):
        B = len(sentences); T = max(len(s) for s in sentences)
        ids = torch.zeros(B, T, dtype=torch.long); cats = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, toks in enumerate(sentences):
            ids[b, :len(toks)] = torch.tensor([self.w.idx[t] for t in toks])
            cats[b, :len(toks)] = torch.tensor([self.cat_id[self.w.cat[t]] for t in toks]); mask[b, :len(toks)] = 1
        ids, cats, mask = ids.to(self.device), cats.to(self.device), mask.to(self.device)
        return self.Freal[ids], cats, mask                          # the "ids" slot of the interface carries the categories
    def states(self, x, cats, mask):
        self.net.eval()
        with torch.no_grad(): return self.net(x, mask, cats)
    def make_data(self, world, rng, n_sent, templates, persons, objects, depth_train):
        sents, targets = [], []
        for n in range(n_sent):
            d = int(rng.integers(depth_train + 1)); ev = self.w.event(persons, objects, depth=d)
            sents.append(self.w.render(ev, rng.choice(list(templates)))); targets.append(world.state(ev))
        ids, cats, mask = self.encode_ids(sents)            # identities gathered per batch (memory: 30k x T x 2D floats would not fit on a small host)
        return ids, cats, mask, torch.tensor(np.array(targets), dtype=torch.complex64, device=self.device)
    def encode_ids(self, sentences):
        B = len(sentences); T = max(len(s) for s in sentences)
        ids = torch.zeros(B, T, dtype=torch.long); cats = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, toks in enumerate(sentences):
            ids[b, :len(toks)] = torch.tensor([self.w.idx[t] for t in toks])
            cats[b, :len(toks)] = torch.tensor([self.cat_id[self.w.cat[t]] for t in toks]); mask[b, :len(toks)] = 1
        return ids.to(self.device), cats.to(self.device), mask.to(self.device)
    def train(self, data, lr, epochs, batch, wd=0.01):
        ids, cats, mask, tgt = data; n = ids.shape[0]; opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ((n + batch - 1) // batch)); g = torch.Generator(device="cpu").manual_seed(0)
        for ep in range(epochs):
            self.net.train(); perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, batch):
                sl = perm[i:i + batch]; S = self.net(self.Freal[ids[sl]], mask[sl], cats[sl]); t = tgt[sl]
                cos = torch.real((torch.conj(t) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * t.abs().pow(2).sum(-1).sqrt() + 1e-8)
                opt.zero_grad(); (-cos.mean()).backward(); opt.step(); sched.step()
            print(f"    epoch {ep + 1}/{epochs} done", flush=True)
    def val_cos(self, data):
        ids, cats, mask, tgt = data; S = self.states(self.Freal[ids], cats, mask)
        return float((torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt() + 1e-8)).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cpu"); ap.add_argument("--lang", default="en")
    ap.add_argument("--lrs", default="1e-3,3e-4"); ap.add_argument("--dropouts", default="0.0,0.1"); ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang(a.lang); torch.set_num_threads(a.threads)
    v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]                      # prefixed reporting, as 6c / 7b
    print(f"v22_transformer_categories: device={a.device}, torch {torch.__version__}, D={D}, N={N}, epochs={a.epochs}, batch={a.batch}, seeds={a.seeds}")
    print("the 7b transformer control with ONE change: the category of every word at the input (Linear(F[w]) + Emb(category)), as the reader has it.")
    w0 = v5.BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); cues_ok, _ = v5.split_templates_rolecues(w0)
    print(f"  categories: {len(set(w0.cat.values()))}; new formulations: seen trans. {held_seen}; UNSEEN trans. {held_new}; ROLE CUES seen {cues_ok}")
    print(f"chance: seen ids 1/12 = {1/12:.3f}; new ids 1/6 = {1/6:.3f}. Reference (7b, no categories): new ids 0.215, depth 3 0.390.\n", flush=True)
    grid = [(float(lr), float(dp)) for lr in a.lrs.split(",") for dp in a.dropouts.split(",")]
    cols = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "new form., ROLE CUES seen", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]
    results, roles3, chosen = [], [], None
    for seed in range(a.seeds):
        w = v5.BigWorld(a.n_id, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v7.NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
        configs = grid if seed == 0 else [chosen]; best = None
        for lr, dp in configs:
            m = TModelCat(w, D, a.device, seed, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers, d_ff=a.d_ff, dropout=dp); m.set_F(F)
            assert m.F_np is world.F
            data = m.make_data(world, np.random.default_rng(200 + seed), N, v5.TRAIN_T, trP, trO, a.depth_train)
            val = m.make_data(world, np.random.default_rng(900 + seed), 2000, v5.TRAIN_T, trP, trO, a.depth_train)
            t0 = time.time(); m.train(data, lr, a.epochs, a.batch); t_train = time.time() - t0; vc = m.val_cos(val)
            print(f"  seed {seed} lr={lr:g} dropout={dp:g} ({t_train:5.0f}s, {m.n_params/1e6:.2f}M params) validation cos {vc:.3f}", flush=True)
            del data, val
            if best is None or vc > best[0]: best = (vc, lr, dp, m)
            else: del m
        if seed == 0: chosen = (best[1], best[2]); print(f"  => chosen config: lr={chosen[0]:g}, dropout={chosen[1]:g} (by validation cos, seed 0)", flush=True)
        m = best[3]
        d3, pr3 = v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 3, n=15, per_role=True)
        r = [v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 0, n=20), v7.decode_nested(m, world, w, v5.TRAIN_T, teP, teO, 0, n=20),
             v7.decode_nested(m, world, w, held_seen, trP, trO, 0, n=30), v7.decode_nested(m, world, w, held_new, trP, trO, 0, n=30),
             v7.decode_nested(m, world, w, cues_ok, trP, trO, 0, n=30), v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 1, n=15),
             v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 2, n=15), d3]
        results.append(r); roles3.append([pr3[k] for k in ("A", "B", "C", "X1", "X2", "X3")])
        print(f"  seed {seed} transformer+categories " + " | ".join(f"{x:.3f} (m {mg:.2f})" for x, mg in r), flush=True)
    r = np.array(results)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    print(f"  transformer + categories (lr={chosen[0]:g}, dropout={chosen[1]:g}) " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print("  per seed, new ids: " + ", ".join(f"{x:.3f}" for x in r[:, 1, 0]) + "; depth 3: " + ", ".join(f"{x:.3f}" for x in r[:, 7, 0]))
    r3 = np.array(roles3); print("== depth 3 (UNSEEN), PER ROLE: A | B | C | X1 | X2 | X3\n  " + " | ".join(f"{r3[:,i].mean():.3f}±{r3[:,i].std():.3f}" for i in range(6)))
    print("reference: 7b transformer WITHOUT categories (log 2026-10-15_1414): new ids 0.215±0.012, depth 3 0.390±0.024, unseen trans. 0.783±0.057; "
          "reader (7c, learned gates, closed start): new ids 1.000, depth 3 0.996, unseen trans. 0.870±0.038.")
    ni, d3m = r[:, 1, 0].mean(), r[:, 7, 0].mean()
    verdict = lambda v: "> 0.9: the 7b gap on this axis came from the category input" if v > 0.9 else ("< 0.5: the gap on this axis is the mechanism" if v < 0.5 else "in between: reported as is, both readings stated")
    print(f"== decision rule (written before the run): new ids {ni:.3f} -> {verdict(ni)}; depth 3 {d3m:.3f} -> {verdict(d3m)}")
