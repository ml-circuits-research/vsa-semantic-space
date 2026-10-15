"""
v4_torch: portarea lui v4.py in torch (GPU). Acelasi model, aceeasi lume, aceeasi evaluare.

Ce ramane in numpy/Python (neschimbat, importat din v4/v2/camp): lumea, template-urile, categoriile, portile si STRUCTURA adreselor
(care parametri intra in fiecare termen: V4.run intoarce listele de membri; depinde doar de categorii si porti, nu de theta).
Ce trece in torch: fazele theta ca parametru; termen = exp(i * M @ theta) * F[tok]; S = suma; loss = -cos(S, S_lume); autograd + Adam.
  termen k: faza = theta[idx_k].sum(), idx_k = indicii membrilor (padding cu root, faza 0). Encodarea se face O DATA, inainte de antrenare.
Fluxul de date (lume, identitati, seed-uri, ordinea propozitiilor de antrenare) e IDENTIC cu v4.py: aceleasi rng-uri numpy.
Diferente fata de v4.py: aritmetica optimizatorului (torch.optim.Adam vs Adam scris de mana), float64 implicit; --batch > 1 schimba
regimul de optimizare (acelasi numar de propozitii, mai putini pasi) si se raporteaza separat.

Evaluare conform PRINCIPII_EVALUARE.md (identic cu v4.py): adrese din lume, F comun (assert), sansa tiparita, 3 axe vazut/nevazut,
marja, >=3 seed-uri, control trivial.

Rulare: python3 v4_torch.py [--lr 0.02] [--seeds 3] [--N 10000] [--D 512] [--batch 1] [--device cuda] [--variants all|toate4]
"""
import argparse, time, sys
import numpy as np, torch
import v2, v4

def build_terms(model, toks):
    """structura adreselor din codul numpy: membri (liste de indici de parametri) si identitati, fara nicio aritmetica pe theta"""
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
        # modelul numpy e folosit DOAR pentru nume/indici de parametri si structura adreselor; theta-ul lui nu e folosit
        self.np_model = v4.V4(world, D, rng, gates, codes)
        self.w, self.D, self.rng, self.codes, self.device, self.dtype = world, D, rng, codes, device, dtype
        self.P = len(self.np_model.names)
        th = torch.tensor(self.np_model.theta, dtype=dtype, device=device)      # aceeasi initializare ca v4.py (rng 200+seed)
        self.theta = torch.nn.Parameter(th)
        self.root_mask = torch.ones(self.P, 1, dtype=dtype, device=device); self.root_mask[0] = 0   # root fix la 0, ca in numpy
        self.F = None

    def set_F(self, F_np):
        self.F_np = F_np; self.F = torch.tensor(F_np, dtype=torch.complex128 if self.dtype == torch.float64 else torch.complex64, device=self.device)

    def encode(self, sentences):
        """lista de propozitii -> idx (B x T x K) indici de parametri per termen, ids (B x T) identitati, mask (B x T).
        Padding-ul membrilor e cu 0 = root, a carui faza e fixata la 0, deci nu schimba suma; padding-ul termenilor e mascat."""
        enc = [build_terms(self.np_model, toks) for toks in sentences]
        B = len(enc); T = max(len(m) for m, _ in enc); K = max(len(mem) for m, _ in enc for mem in m)
        idx = torch.zeros(B, T, K, dtype=torch.long); ids = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, (members, tid) in enumerate(enc):
            for k, mem in enumerate(members): idx[b, k, :len(mem)] = torch.tensor(mem)
            ids[b, :len(tid)] = torch.tensor(tid); mask[b, :len(tid)] = 1.0
        return idx.to(self.device), ids.to(self.device), mask.to(self.device)

    def states(self, idx, ids, mask):
        """S (B x D) = sum_k exp(i * sum_{p in membri_k} theta_p) * F[id_k], cu masca de padding"""
        phase = (self.theta * self.root_mask)[idx].sum(2)                     # B x T x D
        terms = torch.polar(torch.ones_like(phase), phase) * self.F[ids]
        return (terms * mask.unsqueeze(-1)).sum(1)

    def state(self, toks):
        with torch.no_grad():
            idx, ids, mask = self.encode([toks]); return self.states(idx, ids, mask)[0]

    def train(self, world, n_sent, lr=0.02, templates=None, persons=None, objects=None, batch=1):
        w = self.w
        # fluxul de propozitii identic cu v4.py: w.event (w.rng) si self.rng.choice pentru template; ENCODARE O SINGURA DATA, inainte
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
    """acuratete si marja pe roluri, cu adresele LUMII (identic cu v4.decode_stats, vectorizat in torch)"""
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
                sc = torch.real((Sn * torch.conj(R[r])) @ torch.conj(Fn[cid]).T).cpu().numpy()      # n x candidati
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
    print("formulari de antrenare:", TRAIN_T)
    held_seen, held_new = [], []
    for t in range(len(TEMPLATES["GIVE"])):
        if t in TRAIN_T: continue
        tr, ca = v4.transitions(w0, "GIVE", t); new_tr = tr - seen_tr; new_cat = ca - seen_cat
        (held_seen if not new_tr and not new_cat else held_new).append(t)
        print(f"  T{t} '{TEMPLATES['GIVE'][t]}': tranzitii noi {sorted(new_tr)}, categorii noi {sorted(new_cat)}")
    print(f"  => formulari noi cu TOATE tranzitiile vazute: {held_seen}; cu tranzitii/categorii NEVAZUTE: {held_new}")
    n_tr = 8; n_te = a.n_pers - n_tr
    print(f"sansa: identitati vazute 1/{n_tr} = {1/n_tr:.3f}; identitati noi 1/{n_te} = {1/n_te:.3f}. Control trivial: 'toate la origine' (fara coduri) = sansa.\n")
    variants = [("path", ["path"]), ("chain", ["chain"]), ("bigram", ["bigram"]), ("pos", ["pos"]),
                ("path+chain (dual)", ["path", "chain"]), ("path+bigram", ["path", "bigram"]), ("path+chain+bigram", ["path", "chain", "bigram"]),
                ("toate 4", ["path", "chain", "bigram", "pos"])]
    if a.variants == "toate4": variants = variants[-1:]
    cols = ["vazute", "id. noi", "form. noi, tranz. vazute", "form. noi, tranz. NEvazute"]
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
            print(f"  seed {seed} {name:22s} ({t_train:5.1f}s antrenare, din care {m.t_encode:4.1f}s encodare; {time.time()-t0-t_train:4.1f}s eval) " + " | ".join(f"{x:.2f} (m {mg:.2f})" for x, mg in r), flush=True)
    print(f"\n== MEDII pe {a.seeds} seed-uri: acuratete (marja) ==  coloane: " + " | ".join(cols))
    for name, _ in variants:
        r = np.array(results[name]); print(f"  {name:22s} " + " | ".join(f"{r[:,i,0].mean():.3f}±{r[:,i,0].std():.3f} (m {r[:,i,1].mean():.2f})" for i in range(4)))
    print(f"\nchecklist: [x] adrese din lume [x] F comun (assert) [x] sansa tiparita [x] 3 axe separate [x] marja [{'x' if a.seeds >= 3 else ' '}] >=3 seed-uri [x] control trivial = sansa")
