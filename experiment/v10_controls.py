"""
v10_controls: the two cheap controls requested after round 7 (00_READ_FIRST §5.3-5.4), on the reference world (compositional, English,
prefixed reporting), same sentences as the model (30k, depths 0-2, TRAIN_T), 5 seeds.

Control A — word2vec for categories (§5.3). Claim under test: "syntactic categories emerge from the learned displacements" (round 1, i2.py:
clustering per-word displacements gave the true categories). If a plain distributional embedding trained on the SAME sentences clusters
the words equally well, the emergence is not the architecture's merit.
  - Model side: the v4 path code with UNTIED displacements (one m:w per WORD, no category knowledge; identities fixed as always), trained
    with the generative objective; representation of word w = exp(i theta[m:w]) (unit complex vector), similarity = cosine.
  - word2vec side: skip-gram with negative sampling (own torch implementation, no external package), window 2, dim 64, 5 epochs on the
    same sentences; representation = input embedding; similarity = cosine.
  - Evaluation (identical for both): agglomerative clustering (average linkage, cosine) of the 54 content words into k = 5 clusters;
    ground truth = {P (18), O (18), V active (6), V participle (6), V recipient (6)}; report purity and adjusted Rand index; plus the
    nearest-neighbour category accuracy (is the nearest other word of the same category?). Chance purity for 5 clusters ~ 0.33 (largest class).
    Function words are excluded (single-member categories).

Control B — associative memory with a bottleneck (§5.4). Claim under test: "a generative construction with a FIXED operation is necessary;
an associative map does not align states" (round 1: unlimited associative memory predicts but does not align). If a LINEAR map with a
limited rank aligns equally well, the claim narrows to "limited capacity is necessary".
  - Input: an order-aware, structure-free representation x = sum_k pos_k (x) F[w_k] (fixed random position vectors; the classic HDC
    sequence code), complex D. Output: S = U V x, with V: D -> r and U: r -> D complex matrices (r = bottleneck), trained with the same
    -cos(S, S_world). r in {8, 32, 128, 1024 (= D, no bottleneck)}.
  - Evaluation: the round-7 table (same decoder with the world's addresses): seen / new identities / new formulations / depths 1-3.
Run: python3 v10_controls.py [--seeds 5] [--N 30000] [--D 1024]
"""
import argparse, time
import numpy as np, torch
import v2, v5_scale as v5, v6_left as v6, v7_nested as v7

# ---------------- Control A ----------------
def content_words(w):
    classes = {}
    for p in w.persons: classes[p] = "P"
    for o in w.objects: classes[o] = "O"
    for k, (v, vp, vr) in v5.VERBS.items(): classes[v] = "V"; classes[vp] = "VP"; classes[vr] = "VR"
    return classes

def purity_ari(labels_true, labels_pred):
    import itertools
    n = len(labels_true); classes = sorted(set(labels_true)); clusters = sorted(set(labels_pred))
    cont = np.array([[sum(1 for a, b in zip(labels_true, labels_pred) if a == c and b == k) for k in clusters] for c in classes])
    purity = cont.max(0).sum() / n
    comb = lambda x: x * (x - 1) / 2
    sum_ij = comb(cont).sum(); a = comb(cont.sum(1)).sum(); b = comb(cont.sum(0)).sum(); exp = a * b / comb(n)
    ari = (sum_ij - exp) / (0.5 * (a + b) - exp) if 0.5 * (a + b) - exp != 0 else 0.0
    return float(purity), float(ari)

def agglomerative(X, k):
    """average-linkage agglomerative clustering on cosine distance (own implementation, no sklearn)"""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True); S = np.real(Xn @ Xn.conj().T); n = len(X)
    clusters = [[i] for i in range(n)]
    while len(clusters) > k:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = S[np.ix_(clusters[i], clusters[j])].mean()
                if best is None or s > best[0]: best = (s, i, j)
        _, i, j = best; clusters[i] += clusters[j]; del clusters[j]
    lab = np.zeros(n, dtype=int)
    for c, members in enumerate(clusters): lab[members] = c
    return lab

def nn_accuracy(X, labels):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True); S = np.real(Xn @ Xn.conj().T); np.fill_diagonal(S, -np.inf)
    return float(np.mean([labels[i] == labels[S[i].argmax()] for i in range(len(X))]))

class UntiedPath(v6.V6):
    """v4 path code with one displacement per WORD (no category knowledge); identities fixed and shared with the world"""
    def __init__(self, world, D, rng, device):
        import copy
        w2 = copy.copy(world); w2.cat = {x: x for x in world.vocab}          # every word its own 'category' => untied displacements
        super().__init__(w2, D, rng, {x: False for x in world.vocab}, ["path"], device)
        self.true_world = world
    def word_vectors(self, words):
        th = self.theta.detach().cpu().numpy(); return np.exp(1j * np.stack([th[self.np_model.pid[f"m:{x}"]] for x in words]))

def skipgram(sentences, vocab_idx, dim=64, window=2, epochs=5, neg=5, lr=0.01, device="cuda", seed=0):
    torch.manual_seed(seed); V = len(vocab_idx)
    pairs = []
    for toks in sentences:
        ids = [vocab_idx[t] for t in toks]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i: pairs.append((c, ids[j]))
    pairs = torch.tensor(pairs, device=device); emb_in = torch.nn.Embedding(V, dim).to(device); emb_out = torch.nn.Embedding(V, dim).to(device)
    torch.nn.init.normal_(emb_in.weight, 0, 0.1); torch.nn.init.zeros_(emb_out.weight)
    counts = torch.bincount(pairs[:, 1], minlength=V).float(); probs = (counts ** 0.75); probs = probs / probs.sum()
    opt = torch.optim.Adam(list(emb_in.parameters()) + list(emb_out.parameters()), lr=lr); B = 4096
    for ep in range(epochs):
        perm = torch.randperm(len(pairs), device=device)
        for i in range(0, len(pairs), B):
            p = pairs[perm[i:i + B]]; c, o = emb_in(p[:, 0]), emb_out(p[:, 1])
            negs = emb_out(torch.multinomial(probs, p.shape[0] * neg, replacement=True).view(p.shape[0], neg))
            loss = -(torch.nn.functional.logsigmoid((c * o).sum(-1)).mean() + torch.nn.functional.logsigmoid(-(negs * c.unsqueeze(1)).sum(-1)).sum(-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    return emb_in.weight.detach().cpu().numpy()

# ---------------- Control B ----------------
class Bottleneck:
    """S = U V x, x = sum_k pos_k (x) F[w_k] (fixed random positions), complex linear with rank r; same interface for v7.decode_nested"""
    def __init__(self, world, D, r, device, seed, max_len=40):
        self.w, self.D, self.device, self.dtype = world, D, device, torch.float64; g = torch.Generator().manual_seed(seed)
        self.pos = torch.polar(torch.ones(max_len, D, dtype=torch.float64), (torch.rand(max_len, D, generator=g, dtype=torch.float64) * 2 - 1) * np.pi).to(device)
        sc = 1 / np.sqrt(D)
        self.V = torch.nn.Parameter((torch.randn(D, r, generator=g, dtype=torch.float64) + 1j * torch.randn(D, r, generator=g, dtype=torch.float64)).to(device) * sc)
        self.U = torch.nn.Parameter((torch.randn(r, D, generator=g, dtype=torch.float64) + 1j * torch.randn(r, D, generator=g, dtype=torch.float64)).to(device) * sc)
        self.F = None
    def set_F(self, F_np): self.F_np = F_np; self.F = torch.tensor(F_np, dtype=torch.complex128, device=self.device)
    def encode(self, sentences):
        B = len(sentences); T = max(len(s) for s in sentences); ids = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, toks in enumerate(sentences): ids[b, :len(toks)] = torch.tensor([self.w.idx[t] for t in toks]); mask[b, :len(toks)] = 1
        return None, ids.to(self.device), mask.to(self.device)
    def states(self, _, ids, mask):
        T = ids.shape[1]; x = (self.pos[:T].unsqueeze(0) * self.F[ids] * mask.unsqueeze(-1)).sum(1)
        return (x @ self.V) @ self.U
    def train(self, world, rng, n_sent, templates, persons, objects, depth_train, lr=0.01, batch=32):
        sents, targets = [], []
        for n in range(n_sent):
            d = int(rng.integers(depth_train + 1)); ev = self.w.event(persons, objects, depth=d)
            sents.append(self.w.render(ev, rng.choice(list(templates)))); targets.append(world.state(ev))
        _, ids, mask = self.encode(sents); tgt = torch.tensor(np.array(targets), dtype=torch.complex128, device=self.device)
        opt = torch.optim.Adam([self.V, self.U], lr=lr)
        for ep in range(3):
            for i in range(0, n_sent, batch):
                sl = slice(i, i + batch); S = self.states(None, ids[sl], mask[sl]); t = tgt[sl]
                cos = torch.real((torch.conj(t) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * t.abs().pow(2).sum(-1).sqrt() + 1e-9)
                opt.zero_grad(); (-cos.mean()).backward(); opt.step()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--n_id", type=int, default=18); ap.add_argument("--depth_train", type=int, default=2)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lang", default="en"); ap.add_argument("--ranks", default="8,32,128,1024"); ap.add_argument("--only", default="AB")
    a = ap.parse_args(); D, N = a.D, a.N; v5.set_lang(a.lang); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    print(f"v10_controls: device={a.device}, D={D}, N={N}, seeds={a.seeds}; reference world ({v5.LANG}, prefixed reporting), training {v5.TRAIN_T}, depths 0-{a.depth_train}")
    w0 = v5.BigWorld(a.n_id, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); cues_ok, _ = v5.split_templates_rolecues(w0)
    classes = content_words(w0); words = sorted(classes); labels = [classes[x] for x in words]
    print(f"Control A: {len(words)} content words in 5 classes {dict(zip(*np.unique(labels, return_counts=True)))}; chance purity = {max(np.unique(labels, return_counts=True)[1])/len(words):.3f}; NN chance ~ {np.mean([(np.sum(np.array(labels)==l)-1)/(len(words)-1) for l in labels]):.3f}")
    resA = {"model (untied displacements)": [], "word2vec (skip-gram, same sentences)": []}
    cols = ["seen", "new ids", "seen trans.", "UNSEEN trans.", "ROLE CUES seen", "depth 1", "depth 2 (seen)", "depth 3 (UNSEEN)"]
    ranks = [int(r) for r in a.ranks.split(",")]; resB = {r: [] for r in ranks}; rolesB = {r: [] for r in ranks}
    for seed in range(a.seeds):
        w = v5.BigWorld(a.n_id, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v7.NestedRoleWorld(F, w, np.random.default_rng(100 + seed))
        allP, allO = w.persons, w.objects                      # control A uses all identities (categories of ALL words)
        if "A" in a.only:
            rng = np.random.default_rng(200 + seed); sents = []
            for n in range(N):
                d = int(rng.integers(a.depth_train + 1)); ev = w.event(allP, allO, depth=d); sents.append(w.render(ev, rng.choice(list(v5.TRAIN_T))))
            t0 = time.time(); m = UntiedPath(w, D, np.random.default_rng(200 + seed), a.device); m.set_F(F)
            # train the untied path model on exactly these sentences
            m.rng = np.random.default_rng(200 + seed); m.train(world, N, templates=v5.TRAIN_T, persons=allP, objects=allO, batch=a.batch, depth_train=a.depth_train)
            Xm = m.word_vectors(words); t1 = time.time() - t0
            t0 = time.time(); Xw = skipgram(sents, w.idx, device=a.device, seed=seed); Xw = Xw[[w.idx[x] for x in words]]; t2 = time.time() - t0
            for name, X in (("model (untied displacements)", Xm), ("word2vec (skip-gram, same sentences)", Xw)):
                pu, ari = purity_ari(labels, list(agglomerative(X, 5))); nn = nn_accuracy(X, labels); resA[name].append((pu, ari, nn))
                print(f"  seed {seed} A {name:38s} purity {pu:.3f}  ARI {ari:.3f}  NN-category {nn:.3f}   ({t1 if name.startswith('model') else t2:4.0f}s)", flush=True)
        if "B" in a.only:
            for r in ranks:
                t0 = time.time(); mb = Bottleneck(w, D, r, a.device, seed); mb.set_F(F); assert mb.F_np is world.F
                mb.train(world, np.random.default_rng(200 + seed), N, v5.TRAIN_T, trP, trO, a.depth_train)
                d2 = v7.decode_nested(mb, world, w, v5.TRAIN_T, trP, trO, 2, n=15); d3, pr3 = v7.decode_nested(mb, world, w, v5.TRAIN_T, trP, trO, 3, n=15, per_role=True)
                rr = [v7.decode_nested(mb, world, w, v5.TRAIN_T, trP, trO, 0, n=20), v7.decode_nested(mb, world, w, v5.TRAIN_T, teP, teO, 0, n=20),
                      v7.decode_nested(mb, world, w, held_seen, trP, trO, 0, n=30), v7.decode_nested(mb, world, w, held_new, trP, trO, 0, n=30),
                      v7.decode_nested(mb, world, w, cues_ok, trP, trO, 0, n=30), v7.decode_nested(mb, world, w, v5.TRAIN_T, trP, trO, 1, n=15), d2, d3]
                resB[r].append(rr); rolesB[r].append([pr3[k] for k in ("A", "B", "C", "X1", "X2", "X3")])
                print(f"  seed {seed} B bottleneck r={r:5d} ({time.time()-t0:4.0f}s) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in rr) + "  || depth 3: " + " ".join(f"{k} {pr3[k]:.2f}" for k in ("A", "B", "C", "X1", "X2", "X3")), flush=True)
    if "A" in a.only:
        print(f"\n== Control A, MEANS over {a.seeds} seeds: purity | ARI | nearest-neighbour category accuracy (k = 5 clusters, 54 words)")
        for name, v in resA.items():
            v = np.array(v); print(f"  {name:38s} " + " | ".join(f"{v[:,i].mean():.3f}±{v[:,i].std():.3f}" for i in range(3)))
    if "B" in a.only:
        print(f"\n== Control B, MEANS over {a.seeds} seeds: accuracy (margin) == columns: " + " | ".join(cols))
        for r in ranks:
            v = np.array(resB[r]); print(f"  bottleneck r={r:5d} " + " | ".join(f"{v[:,i,0].mean():.3f}±{v[:,i,0].std():.3f} (m {v[:,i,1].mean():.2f})" for i in range(len(cols))))
        print("== Control B, depth 3 (UNSEEN) per role: A | B | C | X1 | X2 | X3")
        for r in ranks:
            v = np.array(rolesB[r]); print(f"  bottleneck r={r:5d} " + " | ".join(f"{v[:,i].mean():.3f}" for i in range(6)))
        print("reference (RESULTS 7c, VSA bundle, learned gates): new ids 1.000, depth 3 0.996-1.000, all roles.")
    print(f"\nchecklist: [x] world addresses [x] shared F (assert) [x] chance printed [x] separate axes [x] margin [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] same data (13)")
