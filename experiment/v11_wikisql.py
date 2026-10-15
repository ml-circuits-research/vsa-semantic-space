"""
v11_wikisql: REAL TEXT with an executable consequence (00_READ_FIRST §5.7). WikiSQL: 56,355 English questions -> SQL over Wikipedia tables.
Decides whether there is an empirical paper (judgment 2026-10-15). Full model, learned gates, discovered categories, table visible or not.

World (the SQL query as a role-filler state; fixed random role vectors; compositional like the reference world):
    S_sql = SEL (x) bundle(tokens of selected column) + AGG (x) f(aggregator)
          + sum_k COND_k (x) [ COL (x) bundle(column) + OP (x) f(operator) + VAL (x) bundle(value tokens) ]
Model: the question (optionally preceded by the table header: 'h1 tokens | h2 tokens | ... SEP question') -> state through the v4/v8
codes: path + bigram + soft chainL with gates LEARNED by gradient (start closed). Identities f_w fixed random per word type (never
learned). Categories for real words — the model needs one displacement per CATEGORY: (i) the 40 most frequent tokens are their own
categories (function words: what, is, the, of, how, many, ...; plus '|' and SEP), (ii) all other words are assigned to k = 48 clusters of
skip-gram vectors trained on the same questions + headers (control A of round 7 showed distributional clustering recovers categories at
least as well as the model's displacements, so this is the honest choice, not a hidden parser). Objective: maximize cos(S, S_sql).
Evaluation on dev (8,421): decode from S with the world's addresses — selected column among the TABLE's columns (chance ~0.15),
aggregator among 6 (majority 0.72), first-condition column; cos(S, S_sql); equivalence: pairs of dev questions on the same table with the
same SQL skeleton (sel, agg, condition columns) vs a different skeleton — mean cosine and the separation (same - different); plus the
HARD pairs: same table, same selected column, different aggregator or different condition column.
Mandatory baseline: bag of words + logistic regression — aggregator: 6-way LR on the question BoW (hashed 4096); selected column: per
(question, column) LR on [token overlap count, overlap fraction of the header, header length, column position one-hot] -> argmax over the
table's columns. Trivial controls printed: chance, majority aggregator, 'overlap only' column pick.
Run: python3 v11_wikisql.py [--epochs 3] [--D 512] [--batch 32] [--seeds 1] [--inputs q,qh] [--n_train 56355] [--n_dev 8421]
"""
import argparse, json, re, time, math, collections
import numpy as np, torch
import v10_controls as v10

P = "corp/wikisql/data/"
def tok(s): return re.findall(r"[a-z0-9]+", s.lower())
def load(split):
    rows = [json.loads(l) for l in open(P + f"{split}.jsonl")]; tabs = {}
    for l in open(P + f"{split}.tables.jsonl"): t = json.loads(l); tabs[t["id"]] = t["header"]
    return rows, tabs

class World:
    """fixed random identities F (per word type) and role vectors; S_sql; bundles of header tokens"""
    def __init__(self, vocab, D, device, seed):
        g = np.random.default_rng(seed); self.V = len(vocab); self.D = D; self.device = device
        self.F = torch.tensor(np.exp(1j * g.uniform(-np.pi, np.pi, (self.V + 16, D))).astype(np.complex64), device=device)   # +16: SQL symbols
        self.AGG = [self.V + i for i in range(6)]; self.OPS = [self.V + 8 + i for i in range(3)]
        role = lambda: torch.tensor(np.exp(1j * g.uniform(-np.pi, np.pi, D)).astype(np.complex64), device=device)
        self.R = {"SEL": role(), "AGG": role(), "COL": role(), "OP": role(), "VAL": role(), "COND": [role() for _ in range(4)]}
    def bundle(self, ids): return self.F[torch.tensor(ids, device=self.device)].sum(0) if ids else torch.zeros(self.D, dtype=torch.complex64, device=self.device)
    def sql_state(self, sql, header_ids):
        S = self.R["SEL"] * self.bundle(header_ids[sql["sel"]]) + self.R["AGG"] * self.F[self.AGG[sql["agg"]]]
        for k, (c, o, v) in enumerate(sql["conds"][:4]):
            S = S + self.R["COND"][k] * (self.R["COL"] * self.bundle(header_ids[c]) + self.R["OP"] * self.F[self.OPS[o]] + self.R["VAL"] * self.bundle(v))
        return S

class RealModel:
    """path + bigram + soft chainL (gates as logits), categories given per token; the states() of v8 with explicit index tables"""
    def __init__(self, K, D, device, seed, codes=("path", "bigram", "chainL"), gate_init=-2.0):
        g = torch.Generator().manual_seed(seed); self.K, self.D, self.device, self.codes = K, D, device, codes
        u = lambda *s: torch.nn.Parameter(((torch.rand(*s, generator=g) * 2 - 1) * math.pi).to(device))
        self.th_path = u(K, D); self.th_bg0 = u(K, D); self.th_bg = u(K, K, D); self.th_A = u(K, K, D); self.th_L = u(K, K, D)
        self.logits = torch.nn.Parameter(torch.full((K,), gate_init, device=device))
        self.params = [self.th_path, self.th_bg0, self.th_bg, self.th_A, self.th_L, self.logits]
    def states(self, F, cats, ids, mask):
        E = lambda ph: torch.polar(torch.ones_like(ph), ph); B, T = cats.shape; terms = 0
        if "path" in self.codes: terms = terms + E(torch.cumsum(self.th_path[cats] * mask.unsqueeze(-1), 1))
        if "bigram" in self.codes: terms = terms + E(torch.cat([self.th_bg0[cats[:, :1]], self.th_bg[cats[:, :-1], cats[:, 1:]]], 1))
        if "chainL" in self.codes:
            g = torch.sigmoid(self.logits)[cats] * mask; Sx = torch.cat([torch.zeros(B, 1, device=self.device), torch.cumsum(torch.log1p(-g + 1e-7), 1)], 1)
            EA = E(self.th_A[cats.unsqueeze(2), cats.unsqueeze(1)]); EL = E(self.th_L[cats.unsqueeze(1), cats.unsqueeze(2)])
            addrs = []
            for k in range(T):
                p_root = torch.exp(Sx[:, k]).unsqueeze(-1).to(EA.dtype)
                if k == 0: addrs.append(p_root.expand(B, self.D)); continue
                prev = torch.stack(addrs, 1); p = (g[:, :k] * torch.exp(Sx[:, k].unsqueeze(1) - Sx[:, 1:k + 1])).to(EA.dtype)
                addrs.append(p_root + torch.einsum("bq,bqd->bd", p, prev * EA[:, :k, k]))
            addr = torch.stack(addrs, 1); addrLs = []
            for k in range(T):
                acc = torch.exp(Sx[:, T] - Sx[:, k + 1]).unsqueeze(-1).to(EA.dtype) * addr[:, k]
                if k < T - 1:
                    pL = (g[:, k + 1:] * torch.exp(Sx[:, k + 1:T] - Sx[:, k + 1].unsqueeze(1))).to(EA.dtype)
                    acc = acc + torch.einsum("bq,bqd->bd", pL, addr[:, k + 1:] * EL[:, k, k + 1:])
                addrLs.append(acc)
            gk = g.unsqueeze(-1).to(EA.dtype); terms = terms + gk * addr + (1 - gk) * torch.stack(addrLs, 1)
        return (terms * F[ids] * mask.unsqueeze(-1)).sum(1)

def cosb(S, T): return torch.real((torch.conj(T) * S).sum(-1)) / (S.abs().pow(2).sum(-1).sqrt() * T.abs().pow(2).sum(-1).sqrt() + 1e-8)

def pad(seqs, device, fill=0):
    T = max(len(s) for s in seqs); out = torch.full((len(seqs), T), fill, dtype=torch.long); mask = torch.zeros(len(seqs), T)
    for i, s in enumerate(seqs): out[i, :len(s)] = torch.tensor(s); mask[i, :len(s)] = 1
    return out.to(device), mask.to(device)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--D", type=int, default=512); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--inputs", default="q,qh"); ap.add_argument("--n_train", type=int, default=56355); ap.add_argument("--n_dev", type=int, default=8421)
    ap.add_argument("--k_clusters", type=int, default=48); ap.add_argument("--n_freq", type=int, default=40); ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_header", type=int, default=24); ap.add_argument("--split", default="dev", help="dev | test | dev,test")
    a = ap.parse_args(); dev_ = a.device
    train, ttabs = load("train"); splits = {sp: load(sp) for sp in a.split.split(",")}; train = train[:a.n_train]
    dev, dtabs = splits[a.split.split(",")[0]]; dev = dev[:a.n_dev]   # "dev" below = the first evaluation split; others evaluated at the end of training
    vocab = {}; vid = lambda w: vocab.setdefault(w, len(vocab))
    for rows, tabs in [(train, ttabs)] + [(r, t) for r, t in splits.values()]:
        for r in rows:
            for w in tok(r["question"]): vid(w)
            for h in tabs[r["table_id"]]:
                for w in tok(h): vid(w)
            for c in r["sql"]["conds"]:
                for w in tok(str(c[2])): vid(w)
    SEP, BAR = vid("<sep>"), vid("<bar>"); V = len(vocab)
    print(f"v11_wikisql: device={dev_}, D={a.D}, epochs={a.epochs}, batch={a.batch}; train {len(train)}, dev {len(dev)}, vocabulary {V}; inputs {a.inputs}")
    hid = lambda tabs, tid: [[vid(w) for w in tok(h)] for h in tabs[tid]]
    q_ids = lambda r: [vid(w) for w in tok(r["question"])]
    def seq(r, tabs, mode):
        q = q_ids(r)
        if mode == "q": return q
        h = []
        for col in hid(tabs, r["table_id"]): h += col + [BAR]
        return h[:a.max_header] + [SEP] + q
    # ---- categories: frequent tokens as singletons + skip-gram clusters for the rest
    freq = collections.Counter(w for r in train for w in tok(r["question"])); freq.update(w for r in train for h in ttabs[r["table_id"]] for w in tok(h))
    singles = [w for w, _ in freq.most_common(a.n_freq)] + ["<sep>", "<bar>"]
    t0 = time.time(); sents = [tok(r["question"]) for r in train] + [tok(h) for r in train[:20000] for h in ttabs[r["table_id"]]]
    emb = v10.skipgram([s for s in sents if len(s) > 1], vocab, dim=64, window=2, epochs=3, device=dev_, seed=0)
    X = torch.tensor(emb, device=dev_); X = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    gk = torch.Generator(device="cpu").manual_seed(0); C = X[torch.randperm(V, generator=gk)[:a.k_clusters].to(dev_)].clone()
    for it in range(25):                                          # k-means (cosine) on the normalized skip-gram vectors
        asg = (X @ C.T).argmax(1)
        for c in range(a.k_clusters):
            mem = X[asg == c]
            if len(mem): C[c] = mem.mean(0) / (mem.mean(0).norm() + 1e-8)
    asg = (X @ C.T).argmax(1).cpu().numpy()
    cat_of = np.array([a.k_clusters + singles.index(w) if w in singles else asg[i] for w, i in sorted(vocab.items(), key=lambda x: x[1])])
    K = a.k_clusters + len(singles); print(f"categories: {a.k_clusters} skip-gram clusters + {len(singles)} frequent tokens = {K} ({time.time()-t0:.0f}s). Singletons: {singles[:20]} ...")
    for c in range(3): print(f"   cluster {c} sample: {[w for w, i in vocab.items() if asg[i] == c][:12]}")
    world = World(vocab, a.D, dev_, 0)
    # ---- targets
    def targets(rows, tabs):
        out = []
        for r in rows:
            sql = {"sel": r["sql"]["sel"], "agg": r["sql"]["agg"], "conds": [[c, o, [vid(w) for w in tok(str(v))]] for c, o, v in r["sql"]["conds"]]}
            out.append(world.sql_state(sql, hid(tabs, r["table_id"])))
        return torch.stack(out)
    t0 = time.time(); tgt_train = targets(train, ttabs); tgt_dev = targets(dev, dtabs); print(f"targets built ({time.time()-t0:.0f}s)")
    # ---- trivial controls and BoW + LR baseline
    hdr_cols = [hid(dtabs, r["table_id"]) for r in dev]
    print(f"\nchance: selected column ~{np.mean([1/len(h) for h in hdr_cols]):.3f}; majority aggregator {np.mean([r['sql']['agg']==0 for r in dev]):.3f}")
    def overlap_pick(r, cols):
        q = set(tok(r["question"])); sc = [len(q & set(tok(h))) / (len(tok(h)) + 1e-9) for h in dtabs[r["table_id"]]]; return int(np.argmax(sc))
    print(f"trivial control 'overlap only' selected column: {np.mean([overlap_pick(r, c) == r['sql']['sel'] for r, c in zip(dev, hdr_cols)]):.3f}")
    # agg LR on hashed BoW
    H = 4096; hb = lambda r: torch.bincount(torch.tensor([hash(w) % H for w in tok(r["question"])]), minlength=H).float().clamp(max=1)
    Xa = torch.stack([hb(r) for r in train]).to(dev_); ya = torch.tensor([r["sql"]["agg"] for r in train], device=dev_)
    Wa = torch.nn.Linear(H, 6).to(dev_); opt = torch.optim.Adam(Wa.parameters(), lr=1e-2, weight_decay=1e-5)
    for ep in range(8):
        perm = torch.randperm(len(train), device=dev_)
        for i in range(0, len(train), 256): sl = perm[i:i+256]; loss = torch.nn.functional.cross_entropy(Wa(Xa[sl]), ya[sl]); opt.zero_grad(); loss.backward(); opt.step()
    Xd = torch.stack([hb(r) for r in dev]).to(dev_); acc_agg_lr = float((Wa(Xd).argmax(1).cpu() == torch.tensor([r["sql"]["agg"] for r in dev])).float().mean())
    # sel LR on (question, column) features
    def col_feats(r, tabs):
        q = set(tok(r["question"])); fs = []
        for j, h in enumerate(tabs[r["table_id"]]):
            ht = tok(h); ov = len(q & set(ht)); f = [ov, ov / (len(ht) + 1e-9), len(ht), float(any(w in q for w in ht))] + [1.0 if j == p else 0.0 for p in range(20)]
            fs.append(f)
        return torch.tensor(fs)
    Xs, ys, grp = [], [], []
    for r in train:
        f = col_feats(r, ttabs); Xs.append(f); ys.append(torch.tensor([1.0 if j == r["sql"]["sel"] else 0.0 for j in range(len(f))]))
    Xs = torch.cat(Xs).to(dev_); ys = torch.cat(ys).to(dev_); Ws = torch.nn.Linear(Xs.shape[1], 1).to(dev_); opt = torch.optim.Adam(Ws.parameters(), lr=1e-2)
    for ep in range(6):
        perm = torch.randperm(len(Xs), device=dev_)
        for i in range(0, len(Xs), 1024): sl = perm[i:i+1024]; loss = torch.nn.functional.binary_cross_entropy_with_logits(Ws(Xs[sl]).squeeze(1), ys[sl]); opt.zero_grad(); loss.backward(); opt.step()
    acc_sel_lr = np.mean([int(Ws(col_feats(r, dtabs).to(dev_)).squeeze(1).argmax()) == r["sql"]["sel"] for r in dev])
    both = np.mean([(int(Ws(col_feats(r, dtabs).to(dev_)).squeeze(1).argmax()) == r["sql"]["sel"]) and (int(Wa(hb(r).to(dev_).unsqueeze(0)).argmax()) == r["sql"]["agg"]) for r in dev])
    print(f"BASELINE BoW + logistic regression: selected column {acc_sel_lr:.3f}  aggregator {acc_agg_lr:.3f}  sel+agg jointly {both:.3f} (no condition prediction in this baseline)\n")
    # ---- the model
    def evaluate(m, cats_fn, mode, name, dev=dev, dtabs=dtabs, tgt_dev=tgt_dev, hdr_cols=hdr_cols):
        m_eval = []; sel_ok = agg_ok = c1_ok = full_ok = 0; n_c1 = 0; preds = []
        for i in range(0, len(dev), 256):
            rows = dev[i:i+256]; seqs = [seq(r, dtabs, mode) for r in rows]; ids, mask = pad(seqs, dev_); cats = torch.tensor(cat_of)[ids.cpu()].to(dev_)
            with torch.no_grad(): S = m.states(world.F, cats, ids, mask)
            preds.append(S)
        S_all = torch.cat(preds); cs = cosb(S_all, tgt_dev).cpu().numpy()
        for S, r, cols in zip(S_all, dev, hdr_cols):
            cb = torch.stack([world.bundle(c) for c in cols]); Sn = S
            sel = int(cosb((Sn * torch.conj(world.R["SEL"])).unsqueeze(0).expand(len(cols), -1), cb).argmax())
            agg = int(cosb((Sn * torch.conj(world.R["AGG"])).unsqueeze(0).expand(6, -1), world.F[torch.tensor(world.AGG, device=dev_)]).argmax())
            sel_ok += sel == r["sql"]["sel"]; agg_ok += agg == r["sql"]["agg"]; ok_c = True
            if r["sql"]["conds"]:
                c1 = int(cosb((Sn * torch.conj(world.R["COND"][0] * world.R["COL"])).unsqueeze(0).expand(len(cols), -1), cb).argmax()); c1_ok += c1 == r["sql"]["conds"][0][0]; n_c1 += 1
                ok_c = all(int(cosb((Sn * torch.conj(world.R["COND"][k] * world.R["COL"])).unsqueeze(0).expand(len(cols), -1), cb).argmax()) == c for k, (c, o, v) in enumerate(r["sql"]["conds"][:4]))
            full_ok += (sel == r["sql"]["sel"]) and (agg == r["sql"]["agg"]) and ok_c
        # equivalence pairs on the same table
        bytab = collections.defaultdict(list)
        for i, r in enumerate(dev): bytab[r["table_id"]].append(i)
        same, diff, hard = [], [], []
        Sn_all = S_all / S_all.abs().pow(2).sum(-1, keepdim=True).sqrt()
        for idx in bytab.values():
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    ra, rb = dev[idx[x]], dev[idx[y]]; ka = (ra["sql"]["sel"], ra["sql"]["agg"], tuple(c[0] for c in ra["sql"]["conds"])); kb = (rb["sql"]["sel"], rb["sql"]["agg"], tuple(c[0] for c in rb["sql"]["conds"]))
                    c = float(torch.real((torch.conj(Sn_all[idx[x]]) * Sn_all[idx[y]]).sum()))
                    if ka == kb: same.append(c)
                    else:
                        diff.append(c)
                        if ra["sql"]["sel"] == rb["sql"]["sel"]: hard.append(c)
        n = len(dev); print(f"  {name:44s} sel {sel_ok/n:.3f}  agg {agg_ok/n:.3f}  cond1-col {c1_ok/max(n_c1,1):.3f}  FULL (sel+agg+all cond cols) {full_ok/n:.3f} | cos(S, S_sql) {cs.mean():.3f} | same-table pairs: same skeleton {np.mean(same):.3f} (n={len(same)}), different {np.mean(diff):.3f} (n={len(diff)}), HARD (same sel) {np.mean(hard):.3f} (n={len(hard)}) | separation {np.mean(same)-np.mean(diff):.3f}", flush=True)
    for mode in a.inputs.split(","):
        for seed in range(a.seeds):
            m = RealModel(K, a.D, dev_, seed); opt = torch.optim.Adam(m.params, lr=a.lr)
            seqs = [seq(r, ttabs, mode) for r in train]; order = np.argsort([len(s) for s in seqs])        # length buckets
            t0 = time.time()
            for ep in range(a.epochs):
                gperm = np.random.default_rng(seed * 100 + ep); buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]; gperm.shuffle(buckets)
                for bi, b in enumerate(buckets):
                    ids, mask = pad([seqs[i] for i in b], dev_); cats = torch.tensor(cat_of)[ids.cpu()].to(dev_)
                    S = m.states(world.F, cats, ids, mask); loss = -cosb(S, tgt_train[torch.tensor(b, device=dev_)]).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                print(f"  [{mode} seed {seed}] epoch {ep} done ({time.time()-t0:.0f}s), last loss {-loss.item():.3f}", flush=True)
                evaluate(m, None, mode, f"model {mode} seed {seed} epoch {ep}")
            for sp, (rows, tabs) in splits.items():
                if sp == a.split.split(",")[0]: continue
                hc = [hid(tabs, r["table_id"]) for r in rows]; evaluate(m, None, mode, f"model {mode} seed {seed} FINAL on {sp} ({len(rows)})", dev=rows, dtabs=tabs, tgt_dev=targets(rows, tabs), hdr_cols=hc)
            gts = torch.sigmoid(m.logits).detach().cpu().numpy(); openc = [(singles[i - a.k_clusters] if i >= a.k_clusters else f"cluster{i}") for i in np.argsort(-gts)[:16]]
            print(f"  learned gates ({mode}): most open {[(c, round(float(gts[np.argsort(-gts)[j]]),2)) for j, c in enumerate(openc)]}")
    print("\nchecklist: [x] world addresses [x] fixed identities [x] chance + majority printed [x] BoW+LR baseline [x] hard pairs reported separately [ ] >=3 seeds (see --seeds) [x] no test calibration")
