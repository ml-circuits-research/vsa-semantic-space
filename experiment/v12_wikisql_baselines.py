"""
v12_wikisql_baselines: the empirical-paper comparison on WikiSQL (judgment 2026-10-15), everything on the SAME input and objective.

Models (identical input: the sequence of FIXED identity vectors F[w] of the question tokens + the same 90 category ids; no table at the input;
identical objective -cos(S, S_sql); identical decoder with the world's addresses; the table's columns only as decoding candidates):
  - ours: v11 RealModel (path + bigram + soft chainL, gates learned by gradient).
  - transformer: linear(2D -> d_model) on F[w] + learned category embedding + sinusoidal positions -> 2-layer encoder -> masked mean pool ->
    linear(d_model -> 2D) -> complex S. Minimal tuning: lr in {1e-3, 3e-4} chosen ONCE on a validation split held out from TRAIN (2,000
    questions) at the 56k size, then fixed. No further tuning for ours (author decision: the curve is reported as it comes).
  - BoW + logistic regression (hashed bag of words; aggregator 6-way; selected column by per-(question, column) features) — no sequence.
Data-efficiency curve: 1k / 5k / 56k training questions; epochs scaled so that the number of gradient steps is comparable (60 / 20 / 3).
Metrics on dev (8,421) and test (15,878): selected column, aggregator, condition-1 column, FULL query (sel + agg + all condition columns),
cos(S, S_sql); and the HARD PAIRS built from SQL on the same table, each reported for the model AND for the bag-of-identities baseline
(cosine of sum_k F[w_k], i.e. pure lexical overlap) on the same row:
  H1 "different words, same SQL": same skeleton (sel, agg, condition columns), Jaccard(question tokens) < 0.3
  H2 "same words, different SQL": different skeleton, Jaccard >= 0.5
  plus the AUC of "same skeleton vs different" by cosine, model vs bag — if the bag separates as well, the separation is lexical, not role-based.
No leakage: skip-gram clusters from TRAIN only (asserted in v11); validation split carved from TRAIN; dev/test never used for any choice.
NOTE on the curve: the skip-gram clusters (categories) are trained on the WHOLE train pool (~54k questions + headers) at every size, for ours
AND for the transformer; at 1k/5k the model sees 1k/5k labelled (question, SQL) pairs but categories from the full unlabelled corpus. Any
"data efficiency" claim must say: "with distributional categories from the whole training corpus, without labels".
Run: python3 v12_wikisql_baselines.py [--sizes 1000,5000,56355] [--seeds 3] [--D 512]
"""
import guard  # process-level memory/time protection (see guard.py; the machine hung on 2026-10-15)
import argparse, json, re, time, math, collections
import numpy as np, torch, torch.nn as nn
import v11_wikisql as v11, v10_controls as v10

class TransformerSame(nn.Module):
    def __init__(self, D, K, d_model=128, n_heads=4, n_layers=2, d_ff=256, dropout=0.0, max_len=64):
        super().__init__()
        self.inp = nn.Linear(2 * D, d_model); self.cat = nn.Embedding(K, d_model)
        pe = torch.zeros(max_len, d_model); pos = torch.arange(max_len).unsqueeze(1); div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div); self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers); self.out = nn.Linear(d_model, 2 * D)
    def forward(self, Freal, ids, cats, mask):
        h = self.inp(Freal[ids]) + self.cat(cats) + self.pe[: ids.shape[1]].unsqueeze(0)
        h = self.enc(h, src_key_padding_mask=(mask == 0)); pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        o = self.out(pooled); D = o.shape[1] // 2; return torch.complex(o[:, :D], o[:, D:])

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg);
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos])) if len(pos) and len(neg) else float("nan")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1000,5000,56355"); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--k_clusters", type=int, default=48); ap.add_argument("--n_freq", type=int, default=40)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--t_lrs", default="1e-3,3e-4"); ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--epochs", default="60,20,3"); ap.add_argument("--only", default="all", help="all | transformer | ours (robustness checks, e.g. a longer transformer budget)")
    ap.add_argument("--cluster_on_subset", action="store_true", help="ablation: skip-gram categories trained ONLY on the training subset of each curve size (both models)")
    a = ap.parse_args(); dev_ = a.device; tok, load = v11.tok, v11.load
    train_all, ttabs = load("train"); dev, dtabs = load("dev"); test, tetabs = load("test")
    rng0 = np.random.default_rng(0); perm = rng0.permutation(len(train_all)); val = [train_all[i] for i in perm[:a.n_val]]; train_pool = [train_all[i] for i in perm[a.n_val:]]
    vocab = {}; vid = lambda w: vocab.setdefault(w, len(vocab))
    for rows, tabs in ((train_all, ttabs), (dev, dtabs), (test, tetabs)):
        for r in rows:
            for w in tok(r["question"]): vid(w)
            for h in tabs[r["table_id"]]:
                for w in tok(h): vid(w)
            for c in r["sql"]["conds"]:
                for w in tok(str(c[2])): vid(w)
    V = len(vocab); sizes = [int(x) for x in a.sizes.split(",")]; epochs = dict(zip(sizes, [int(x) for x in a.epochs.split(",")]))
    print(f"v12_wikisql_baselines: device={dev_}, D={a.D}; train pool {len(train_pool)} (+{len(val)} validation held out from train), dev {len(dev)}, test {len(test)}, vocabulary {V}; sizes {sizes}, epochs {epochs}")
    hid = lambda tabs, tid: [[vid(w) for w in tok(h)] for h in tabs[tid]]
    def build_categories(pool, label):
        """frequent tokens + skip-gram clusters from `pool` ONLY (never dev/test/val); returns (cat_of, K, singles)"""
        freq = collections.Counter(w for r in pool for w in tok(r["question"])); freq.update(w for r in pool for h in ttabs[r["table_id"]] for w in tok(h))
        singles = [w for w, _ in freq.most_common(a.n_freq)]
        sents = [tok(r["question"]) for r in pool] + [tok(h) for r in pool[:20000] for h in ttabs[r["table_id"]]]
        train_words = {w for s_ in sents for w in s_}
        print(f"clustering data ({label}): {len(sents)} sequences from {len(pool)} training questions only; dev/test/val not used ({len(train_words)} word types).")
        emb = v10.skipgram([s for s in sents if len(s) > 1], vocab, dim=64, window=2, epochs=3, device=dev_, seed=0)
        X = torch.tensor(emb, device=dev_); X = X / (X.norm(dim=1, keepdim=True) + 1e-8); gk = torch.Generator().manual_seed(0)
        C = X[torch.randperm(V, generator=gk)[:a.k_clusters].to(dev_)].clone()
        for it in range(25):
            asg = (X @ C.T).argmax(1)
            for c in range(a.k_clusters):
                mem = X[asg == c]
                if len(mem): C[c] = mem.mean(0) / (mem.mean(0).norm() + 1e-8)
        asg = (X @ C.T).argmax(1).cpu().numpy()
        cat_of = torch.tensor(np.array([a.k_clusters + singles.index(w) if w in singles else asg[i] for w, i in sorted(vocab.items(), key=lambda x: x[1])]))
        return cat_of, a.k_clusters + len(singles), singles
    cat_of, K, singles = build_categories(train_pool, "WHOLE train pool")
    world = v11.World(vocab, a.D, dev_, 0); Freal = torch.cat([world.F.real, world.F.imag], 1)
    def targets(rows, tabs):
        return torch.stack([world.sql_state({"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [[c, o, [vid(w) for w in tok(str(v))]] for c, o, v in r["sql"]["conds"]]}, hid(tabs, r["table_id"])) for r in rows])
    qids = lambda r: [vid(w) for w in tok(r["question"])]
    def batch_tensors(rows):
        ids, mask = v11.pad([qids(r) for r in rows], dev_); return ids, cat_of[ids.cpu()].to(dev_), mask
    splits = {"dev": (dev, dtabs, targets(dev, dtabs)), "test": (test, tetabs, targets(test, tetabs))}; tgt_val = targets(val, ttabs)
    # ---- hard pairs (dev), fixed once
    def skeleton(r): return (r["sql"]["sel"], r["sql"]["agg"], tuple(c[0] for c in r["sql"]["conds"]))
    def jacc(r1, r2): a_, b_ = set(tok(r1["question"])), set(tok(r2["question"])); return len(a_ & b_) / max(1, len(a_ | b_))
    bytab = collections.defaultdict(list)
    for i, r in enumerate(dev): bytab[r["table_id"]].append(i)
    pairs_same, pairs_diff, H1, H2 = [], [], [], []
    for idx in bytab.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]; same = skeleton(dev[i]) == skeleton(dev[j]); jc = jacc(dev[i], dev[j])
                (pairs_same if same else pairs_diff).append((i, j))
                if same and jc < 0.3: H1.append((i, j))
                if (not same) and jc >= 0.5: H2.append((i, j))
    print(f"dev same-table pairs: same skeleton {len(pairs_same)}, different {len(pairs_diff)}; HARD H1 'different words, same SQL' (Jaccard<0.3) {len(H1)}; H2 'same words, different SQL' (Jaccard>=0.5) {len(H2)}")
    def pair_report(S_dev, label):
        Sn = S_dev / S_dev.abs().pow(2).sum(-1, keepdim=True).sqrt()
        cs = lambda pairs: [float(torch.real((torch.conj(Sn[i]) * Sn[j]).sum())) for i, j in pairs]
        s, d, h1, h2 = cs(pairs_same), cs(pairs_diff), cs(H1), cs(H2)
        return f"{label}: same {np.mean(s):.3f} diff {np.mean(d):.3f} sep {np.mean(s)-np.mean(d):.3f} AUC {auc(s, d):.3f} | H1 {np.mean(h1):.3f} | H2 {np.mean(h2):.3f}"
    # bag-of-identities baseline for the pairs (pure lexical overlap in the same space)
    ids_dev, _, mask_dev = batch_tensors(dev); S_bag = (world.F[ids_dev] * mask_dev.unsqueeze(-1)).sum(1)
    print("  pairs, bag of identities (lexical only): " + pair_report(S_bag, "bag"))
    def full_eval(states_fn, name, with_test=True):
        out = {}
        for sp, (rows, tabs, tgt) in splits.items():
            if sp == "test" and not with_test: continue        # test split only at the largest training size (cost)
            S_all = torch.cat([states_fn(*batch_tensors(rows[i:i+256])) for i in range(0, len(rows), 256)])
            cs = v11.cosb(S_all, tgt).cpu().numpy(); sel_ok = agg_ok = c1_ok = full_ok = 0; n_c1 = 0
            for S, r in zip(S_all, rows):
                cols = hid(tabs, r["table_id"]); cb = torch.stack([world.bundle(c) for c in cols])
                sel = int(v11.cosb((S * torch.conj(world.R["SEL"])).unsqueeze(0).expand(len(cols), -1), cb).argmax())
                agg = int(v11.cosb((S * torch.conj(world.R["AGG"])).unsqueeze(0).expand(6, -1), world.F[torch.tensor(world.AGG, device=dev_)]).argmax())
                ok_c = True
                if r["sql"]["conds"]:
                    cks = [int(v11.cosb((S * torch.conj(world.R["COND"][k] * world.R["COL"])).unsqueeze(0).expand(len(cols), -1), cb).argmax()) for k in range(min(4, len(r["sql"]["conds"])))]
                    c1_ok += cks[0] == r["sql"]["conds"][0][0]; n_c1 += 1; ok_c = all(ck == c for ck, (c, o, v) in zip(cks, r["sql"]["conds"][:4]))
                sel_ok += sel == r["sql"]["sel"]; agg_ok += agg == r["sql"]["agg"]; full_ok += (sel == r["sql"]["sel"]) and (agg == r["sql"]["agg"]) and ok_c
            n = len(rows); out[sp] = (sel_ok / n, agg_ok / n, c1_ok / max(1, n_c1), full_ok / n, float(cs.mean()))
            line = f"  {name:40s} [{sp:4s}] sel {sel_ok/n:.3f} agg {agg_ok/n:.3f} cond1 {c1_ok/max(1,n_c1):.3f} FULL {full_ok/n:.3f} cos {cs.mean():.3f}"
            if sp == "dev": line += " | pairs " + pair_report(S_all, "model")
            print(line, flush=True)
        return out
    # ---- BoW + LR baseline per size (agg + sel; FULL = sel+agg only, flagged)
    H = 4096; hb = lambda r: torch.bincount(torch.tensor([hash(w) % H for w in tok(r["question"])]), minlength=H).float().clamp(max=1)
    def col_feats(r, tabs):
        q = set(tok(r["question"])); fs = []
        for j, h in enumerate(tabs[r["table_id"]]):
            ht = tok(h); ov = len(q & set(ht)); fs.append([ov, ov / (len(ht) + 1e-9), len(ht), float(any(w in q for w in ht))] + [1.0 if j == p else 0.0 for p in range(20)])
        return torch.tensor(fs)
    def bow_lr(train_rows):
        Xa = torch.stack([hb(r) for r in train_rows]).to(dev_); ya = torch.tensor([r["sql"]["agg"] for r in train_rows], device=dev_)
        Wa = nn.Linear(H, 6).to(dev_); opt = torch.optim.Adam(Wa.parameters(), lr=1e-2, weight_decay=1e-5)
        for ep in range(max(8, 400000 // max(1, len(train_rows)) // 10)):
            p_ = torch.randperm(len(train_rows), device=dev_)
            for i in range(0, len(train_rows), 256): sl = p_[i:i+256]; loss = nn.functional.cross_entropy(Wa(Xa[sl]), ya[sl]); opt.zero_grad(); loss.backward(); opt.step()
        Xs = torch.cat([col_feats(r, ttabs) for r in train_rows]).to(dev_); ys = torch.cat([torch.tensor([1.0 if j == r["sql"]["sel"] else 0.0 for j in range(len(ttabs[r["table_id"]]))]) for r in train_rows]).to(dev_)
        Ws = nn.Linear(Xs.shape[1], 1).to(dev_); opt = torch.optim.Adam(Ws.parameters(), lr=1e-2)
        for ep in range(6):
            p_ = torch.randperm(len(Xs), device=dev_)
            for i in range(0, len(Xs), 1024): sl = p_[i:i+1024]; loss = nn.functional.binary_cross_entropy_with_logits(Ws(Xs[sl]).squeeze(1), ys[sl]); opt.zero_grad(); loss.backward(); opt.step()
        for sp, (rows, tabs, _) in splits.items():
            if sp == "test" and len(train_rows) < max(sizes): continue
            sel = np.array([int(Ws(col_feats(r, tabs).to(dev_)).squeeze(1).argmax()) == r["sql"]["sel"] for r in rows]); agg = np.array([int(Wa(hb(r).to(dev_).unsqueeze(0)).argmax()) == r["sql"]["agg"] for r in rows])
            print(f"  {'BoW + LR':40s} [{sp:4s}] sel {sel.mean():.3f} agg {agg.mean():.3f} cond1  —    FULL(sel+agg only) {(sel & agg).mean():.3f}", flush=True)
    # ---- training loops
    def train_ours(rows, seed, n_ep):
        m = v11.RealModel(K, a.D, dev_, seed); opt = torch.optim.Adam(m.params, lr=a.lr); tgt = targets(rows, ttabs)
        seqs = [qids(r) for r in rows]; order = np.argsort([len(s) for s in seqs]); g = np.random.default_rng(seed)
        for ep in range(n_ep):
            buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]; g.shuffle(buckets)
            for b in buckets:
                ids, mask = v11.pad([seqs[i] for i in b], dev_); cats = cat_of[ids.cpu()].to(dev_)
                loss = -v11.cosb(m.states(world.F, cats, ids, mask), tgt[torch.tensor(b, device=dev_)]).mean(); opt.zero_grad(); loss.backward(); opt.step()
        return lambda ids, cats, mask: (torch.no_grad()(lambda: m.states(world.F, cats, ids, mask))())
    def train_tr(rows, seed, n_ep, lr):
        torch.manual_seed(seed); net = TransformerSame(a.D, K).to(dev_); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01); tgt = targets(rows, ttabs)
        seqs = [qids(r) for r in rows]; order = np.argsort([len(s) for s in seqs]); g = np.random.default_rng(seed)
        steps = n_ep * ((len(rows) + a.batch - 1) // a.batch); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for ep in range(n_ep):
            net.train(); buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]; g.shuffle(buckets)
            for b in buckets:
                ids, mask = v11.pad([seqs[i] for i in b], dev_); cats = cat_of[ids.cpu()].to(dev_)
                loss = -v11.cosb(net(Freal, ids, cats, mask), tgt[torch.tensor(b, device=dev_)]).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval(); return lambda ids, cats, mask: (torch.no_grad()(lambda: net(Freal, ids, cats, mask))()), net
    # ---- transformer lr chosen ONCE on the train-held-out validation at the largest size, seed 0
    t_lrs = [float(x) for x in a.t_lrs.split(",")]; best = None; big = train_pool[:max(sizes)]
    for lr in t_lrs:
        t0 = time.time(); fn, net = train_tr(big, 0, epochs[max(sizes)], lr); ids, cats, mask = batch_tensors(val)
        vc = float(v11.cosb(fn(ids, cats, mask), tgt_val).mean()); print(f"  transformer lr={lr:g}: validation cos {vc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if best is None or vc > best[0]: best = (vc, lr)
    t_lr = best[1]; print(f"  => transformer lr fixed at {t_lr:g} (validation held out from train; never dev/test)\n")
    # ---- the curve
    for n in sizes:
        rows = train_pool[:n]; print(f"== training size {n} ({epochs[n]} epochs) ==")
        if a.cluster_on_subset: cat_of, K, singles = build_categories(rows, f"ABLATION: the {n} training questions only")
        bow_lr(rows)
        for seed in range(a.seeds):
            if a.only in ("all", "ours"):
                t0 = time.time(); fn = train_ours(rows, seed, epochs[n]); full_eval(fn, f"ours (path+bigram+chainL) n={n} seed {seed} [{time.time()-t0:.0f}s]", with_test=(n == max(sizes)))
            if a.only in ("all", "transformer"):
                t0 = time.time(); fn, _ = train_tr(rows, seed, epochs[n], t_lr); full_eval(fn, f"transformer (same input) n={n} ep={epochs[n]} seed {seed} [{time.time()-t0:.0f}s]", with_test=(n == max(sizes)))
    print("\nchecklist: [x] same input (fixed identities + same categories, no table) [x] same objective and decoder [x] BoW+LR on every size [x] FULL-query accuracy [x] hard pairs with the lexical bag on the same row [x] validation from train only [x] no extra tuning for ours [x] test split reported")
