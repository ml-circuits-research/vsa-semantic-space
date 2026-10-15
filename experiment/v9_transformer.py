"""
v9_transformer: CONTROL for test 6c (judgment 2026-10-15). A small transformer on the SAME task, data, identities and objective.

Same as the VSA model: input = the sequence of FIXED identity vectors F[w] (the same F as the world; new identities are new random
vectors never seen in training); output = a state vector in the same complex space; loss = -cos(S, S_world) with the compositional world
of v7_nested; decoding with the WORLD's addresses (unbinding conj(R_rep)^k conj(R_role)), per role, with margin. Training on depths 0-2
and formulations TRAIN_T, 30k sentences (several epochs: the transformer needs more passes; the number is printed); test: new identities,
new formulations, UNSEEN depth 3. Only the decoder is shared with the VSA model; nothing of the VSA addressing is used.

Architecture: F[w] (complex D -> real 2D) -> linear to d_model -> sinusoidal positions (so that longer depth-3 sentences are representable)
-> TransformerEncoder (n_layers, n_heads, d_ff, dropout) -> mean pooling over the mask -> linear to 2D -> complex S.
Minimal tuning (EVALUATION_PRINCIPLES 13: same regime for comparisons, no test calibration): lr in {1e-3, 3e-4}, dropout in {0.0, 0.1},
selected on seed 0 by VALIDATION = fresh training-distribution sentences (depths 0-2, training formulations, training identities), never
on the test axes; the chosen config is then run on all seeds.
Run: python3 v9_transformer.py [--seeds 5] [--N 30000] [--epochs 12] [--D 1024] [--d_model 128] [--layers 2]
"""
import argparse, time, math
import numpy as np, torch, torch.nn as nn
import v5_scale as v5, v7_nested as v7

class TransformerState(nn.Module):
    def __init__(self, D, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.1, max_len=64):
        super().__init__()
        self.inp = nn.Linear(2 * D, d_model)
        pe = torch.zeros(max_len, d_model); pos = torch.arange(max_len).unsqueeze(1); div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers); self.out = nn.Linear(d_model, 2 * D); self.drop = nn.Dropout(dropout)
    def forward(self, x, mask):                              # x: B x T x 2D real, mask: B x T (1 = token)
        h = self.drop(self.inp(x) + self.pe[: x.shape[1]].unsqueeze(0))
        h = self.enc(h, src_key_padding_mask=(mask == 0))
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        o = self.out(pooled); D = o.shape[1] // 2
        return torch.complex(o[:, :D], o[:, D:])

class TModel:
    """the same interface as the VSA models for v7.decode_nested: encode(sentences) -> (x, ids, mask); states(x, ids, mask) -> complex S"""
    def __init__(self, world, D, device, seed, **kw):
        self.w, self.D, self.device = world, D, device; self.dtype = torch.float32
        torch.manual_seed(seed); self.net = TransformerState(D, **kw).to(device); self.F = None
        self.n_params = sum(p.numel() for p in self.net.parameters())
    def set_F(self, F_np):
        self.F_np = F_np; self.F = torch.tensor(F_np, dtype=torch.complex64, device=self.device)
        self.Freal = torch.cat([self.F.real, self.F.imag], 1)                       # V x 2D, fixed
    def encode(self, sentences):
        B = len(sentences); T = max(len(s) for s in sentences)
        ids = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, toks in enumerate(sentences): ids[b, :len(toks)] = torch.tensor([self.w.idx[t] for t in toks]); mask[b, :len(toks)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        return self.Freal[ids], ids, mask
    def states(self, x, ids, mask):
        self.net.eval()
        with torch.no_grad(): return self.net(x, mask)
    def make_data(self, world, rng, n_sent, templates, persons, objects, depth_train):
        sents, targets = [], []
        for n in range(n_sent):
            d = int(rng.integers(depth_train + 1)); ev = self.w.event(persons, objects, depth=d)
            sents.append(self.w.render(ev, rng.choice(list(templates)))); targets.append(world.state(ev))
        x, ids, mask = self.encode(sents); return x, mask, torch.tensor(np.array(targets), dtype=torch.complex64, device=self.device)
    def train(self, data, lr, epochs, batch, wd=0.01):
        x, mask, tgt = data; n = x.shape[0]; opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * ((n + batch - 1) // batch)); g = torch.Generator(device="cpu").manual_seed(0)
        for ep in range(epochs):
            self.net.train(); perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, batch):
                sl = perm[i:i + batch]; S = self.net(x[sl], mask[sl]); t = tgt[sl]
                cos = torch.real((torch.conj(t) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * t.abs().pow(2).sum(-1).sqrt() + 1e-8)
                opt.zero_grad(); (-cos.mean()).backward(); opt.step(); sched.step()
    def val_cos(self, data):
        x, mask, tgt = data; S = self.states(x, None, mask)
        return float((torch.real((torch.conj(tgt) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * tgt.abs().pow(2).sum(-1).sqrt() + 1e-8)).mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--n_id", type=int, default=18)
    ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--heads", type=int, default=4); ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--depth_train", type=int, default=2); ap.add_argument("--device", default="cuda"); ap.add_argument("--lang", default="en"); ap.add_argument("--outer", default="prefixed")
    ap.add_argument("--lrs", default="1e-3,3e-4"); ap.add_argument("--dropouts", default="0.0,0.1")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang(a.lang)
    if a.outer == "prefixed": v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v9_transformer: device={a.device} ({torch.cuda.get_device_name(0)}), torch {torch.__version__}, D={D}, N={N}, epochs={a.epochs}, batch={a.batch}, seeds={a.seeds}")
    print(f"control for 6c: same compositional world ({v5.LANG}, outer {v5.OUTER}), same fixed identities at the input, same objective -cos(S, S_world), same decoder.")
    print(f"transformer: d_model={a.d_model}, layers={a.layers}, heads={a.heads}, d_ff={a.d_ff}; tuning grid lr {a.lrs} x dropout {a.dropouts}, selected on seed 0 by training-distribution validation")
    w0 = v5.BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); cues_ok, _ = v5.split_templates_rolecues(w0)
    print(f"  new formulations: seen trans. {held_seen}; UNSEEN trans. {held_new}; ROLE CUES seen {cues_ok}")
    print(f"chance: seen ids 1/12 = {1/12:.3f}; new ids 1/6 = {1/6:.3f}. Trivial control: mean-pooled bag of identities = order-free.\n")
    grid = [(float(lr), float(dp)) for lr in a.lrs.split(",") for dp in a.dropouts.split(",")]
    cols = ["seen", "new ids", "new form., seen trans.", "new form., UNSEEN trans.", "new form., ROLE CUES seen", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]
    results, roles3, roles2 = [], [], []
    chosen = None
    for seed in range(a.seeds):
        w = v5.BigWorld(a.n_id, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v7.NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
        configs = grid if seed == 0 else [chosen]
        best = None
        for lr, dp in configs:
            m = TModel(w, D, a.device, seed, d_model=a.d_model, n_heads=a.heads, n_layers=a.layers, d_ff=a.d_ff, dropout=dp); m.set_F(F)
            assert m.F_np is world.F
            rng = np.random.default_rng(200 + seed); data = m.make_data(world, rng, N, v5.TRAIN_T, trP, trO, a.depth_train)
            val = m.make_data(world, np.random.default_rng(900 + seed), 2000, v5.TRAIN_T, trP, trO, a.depth_train)
            t0 = time.time(); m.train(data, lr, a.epochs, a.batch); torch.cuda.synchronize(); t_train = time.time() - t0
            vc = m.val_cos(val)
            print(f"  seed {seed} lr={lr:g} dropout={dp:g} ({t_train:4.0f}s, {m.n_params/1e6:.2f}M params) validation cos {vc:.3f}", flush=True)
            if best is None or vc > best[0]: best = (vc, lr, dp, m)
        if seed == 0: chosen = (best[1], best[2]); print(f"  => chosen config: lr={chosen[0]:g}, dropout={chosen[1]:g} (by validation cos, seed 0)")
        m = best[3]
        d2, pr2 = v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 2, n=15, per_role=True)
        d3, pr3 = v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 3, n=15, per_role=True)
        r = [v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 0, n=20), v7.decode_nested(m, world, w, v5.TRAIN_T, teP, teO, 0, n=20),
             v7.decode_nested(m, world, w, held_seen, trP, trO, 0, n=30), v7.decode_nested(m, world, w, held_new, trP, trO, 0, n=30),
             v7.decode_nested(m, world, w, cues_ok, trP, trO, 0, n=30), v7.decode_nested(m, world, w, v5.TRAIN_T, trP, trO, 1, n=15), d2, d3]
        results.append(r); roles2.append([pr2[k] for k in ("A", "B", "C", "X1", "X2")]); roles3.append([pr3[k] for k in ("A", "B", "C", "X1", "X2", "X3")])
        print(f"  seed {seed} transformer " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r) + "  || depth 3 per role: " + " ".join(f"{k} {pr3[k]:.2f}" for k in ("A", "B", "C", "X1", "X2", "X3")), flush=True)
    print(f"\n== MEANS over {a.seeds} seeds: accuracy (margin) ==  columns: " + " | ".join(cols))
    r = np.array(results); print(f"  transformer (lr={chosen[0]:g}, dropout={chosen[1]:g}) " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(len(cols))))
    print("== depth 2 (seen), PER ROLE: A | B | C | X1 | X2"); r2 = np.array(roles2); print("  transformer " + " | ".join(f"{r2[:,i].mean():.3f}±{r2[:,i].std():.3f}" for i in range(5)))
    print("== depth 3 (UNSEEN), PER ROLE: A | B | C | X1 | X2 | X3"); r3 = np.array(roles3); print("  transformer " + " | ".join(f"{r3[:,i].mean():.3f}±{r3[:,i].std():.3f}" for i in range(6)))
    print("reference (RESULTS 6c, VSA path+chainL+bigram, same world/regime): depth 1 1.000, depth 2 1.000, depth 3 1.000 on all six roles; new ids 1.000.")
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] separate axes [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] no test calibration (validation on training distribution) [x] same data/objective (13) [x] per role (15)")
