"""B6 — SPARSE BLOCK CODES and phase quantization (round 10 B6; judgment consolidated Part IV; order from judgment 45ccd1d F4), on the SAVED models
of the reference table 10b-quinquies (v16 --rel_gates open; logs/.ckpt_v16_relg_D1024_omit0.2_learned_learned_open_<seed>.pt; 5 seeds). No training.
Question: how many bits does a vector need? Four storage formats, each applied alone (one change per variant), evaluated with v16's readout on
v16's axes and rngs (the dense variant must reproduce 10b-quinquies' digits):
   (1) ROUTE BOOK quantized: every learned phase (theta) rounded to L levels (the tool's model/routebook.npz at log2(L) bits per phase);
   (2) IDENTITIES AND ROLES quantized: the word stars F and the legend's named stars R rounded to L levels (the catalog and the legend);
   (3) STATE quantized: the reader's chart S projected to unit phasors at L levels (the chart stored at D x log2(L) bits, magnitudes dropped);
   (4) STATE block-sparse: per block of 32 coordinates only the k largest |S| kept (k = 16 / 8 / 4: 50% / 25% / 12.5%), the rest zeroed.
   L in {16, 64, 256} (4 / 6 / 8 bits); bits per vector = D x log2(L) against the dense 64 bits per coordinate (complex64).
Plus the standard capacity curve (synthetic, no reader): n role-filler terms bundled at D = 1024 with 1,000 candidate fillers; cleanup accuracy per
filler vs n, for the dense state, the phase-quantized state (L = 16 / 64) and the block-sparse state (50% / 25%).
Criterion, written before the run: a format is RECOMMENDED when every reference axis (seen FULL, new ids, unseen transitions, depth 2, depth 3
UNSEEN, occupancy known / unknown / absent) stays within 0.02 of the dense value on the 5-seed mean; the recommendation is the smallest L and the
sparsest k meeting it; if none does, dense stays. Reported, not tuned. Run: python3 rescore_v16_b6.py"""
import argparse, numpy as np, torch, guard, v5_scale as v5, v8_gates as v8, v16_reference as v16

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--device", default="cuda"); ap.add_argument("--n_scale", type=float, default=1.0); a = ap.parse_args()
D, dev = a.D, a.device; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w0 = v16.RefWorld(18, np.random.default_rng(0)); held_seen, held_new = v5.split_templates(w0); verbs = sorted(set(v16.LEMMA.values()))
print(f"\nrescore_v16_b6 (B6): saved 10b-quinquies models, {a.seeds} seeds, D = {D}; formats: dense | routebook L | identities+roles L | state phasor L | state block top-k (block 32); bits per vector = D log2 L (dense: 64 per coordinate)")
qphase = lambda z, L: torch.polar(torch.ones_like(z.real), torch.round(torch.angle(z) / (2 * np.pi / L)) * (2 * np.pi / L))
def qphase_np(z, L): return np.exp(1j * np.round(np.angle(z) / (2 * np.pi / L)) * (2 * np.pi / L))
def block_topk(S, k, b=32):
    B = S.shape[0]; mag = S.abs().reshape(B, -1, b); idx = torch.topk(mag, k, dim=-1).indices; mask = torch.zeros_like(mag); mask.scatter_(-1, idx, 1.0)
    return S * mask.reshape(B, -1).to(S.dtype)
axes = ["seen FULL", "new ids FULL", "unseen trans. FULL", "depth 2 FULL", "depth 3 UNSEEN FULL", "occ known", "occ unknown", "occ absent"]
variants = [("dense", None, None)] + [(f"routebook L={L}", "theta", L) for L in (16, 64, 256)] + [(f"identities+roles L={L}", "FR", L) for L in (16, 64, 256)] + [(f"state phasor L={L}", "S", L) for L in (16, 64, 256)] + [(f"state block top-{k}/32", "topk", k) for k in (16, 8, 4)]
res = {v[0]: [] for v in variants}
for seed in range(a.seeds):
    w = v16.RefWorld(18, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
    F0 = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v16.RefRoleWorld(F0, w, np.random.default_rng(100 + seed)); R0 = {k: v.copy() for k, v in world.R.items()}
    st = torch.load(f"logs/.ckpt_v16_relg_D1024_omit0.2_learned_learned_open_{seed}.pt", map_location=dev)
    # calibration rows for the occupancy threshold, as in v16: depth-0 TRAINING sentences (120 with C, 120 without), regenerated with v16's rng
    trng = np.random.default_rng(200 + seed); cal_known, cal_abs = [], []
    for n in range(30000):
        d = int(trng.integers(3)); ev = w.event(trP, trO, depth=d)
        if trng.random() < 0.2: toks = w.render_omit(ev, int(trng.integers(len(v16.OMIT_C)))); (cal_abs.append(toks) if d == 0 and len(cal_abs) < 120 else None)
        else: toks = w.render(ev, int(trng.choice(v5.TRAIN_T))); (cal_known.append(toks) if d == 0 and len(cal_known) < 120 else None)
        if len(cal_known) >= 120 and len(cal_abs) >= 120: break
    for name, what, L in variants:
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], dev, gate_init="closed"); m.fix_gates(["to", "by", "from"], True)
        with torch.no_grad():
            for p_, v in zip([m.theta, m.logits], st["params"]): p_.copy_(v)
            if what == "theta": m.theta.copy_(torch.round(m.theta / (2 * np.pi / L)) * (2 * np.pi / L))
        F = F0; world.R = R0
        if what == "FR": F = qphase_np(F0, L); world.R = {k: qphase_np(v, L) for k, v in R0.items()}
        m.set_F(F); world.F = F; world.m = F[w.idx["<m>"]]
        states0 = m.states
        if what == "S": m.states = lambda c, i, k, L=L: qphase(states0(c, i, k), L)
        elif what == "topk": m.states = lambda c, i, k, K=L: block_topk(states0(c, i, k), K)
        def run(templs, P, O, depth, n):
            erng = np.random.default_rng(900 + depth + 7 * len(templs)); full = []
            for t in templs:
                evs = [w.event(P, O, depth=depth) for _ in range(max(1, int(n * a.n_scale)))]; toks = [w.render(ev, t) for ev in evs]; c, i_, k = m.encode(toks)
                with torch.no_grad(): S = m.states(c, i_, k)
                for j, ev in enumerate(evs):
                    o = v16.readout(S[j], m, world, w, depth, P, O, verbs); full.append(o["A"]["id"] == ev[1] and o["B"]["id"] == ev[2] and o["C"]["id"] == ev[3])
            return float(np.mean(full))
        seen = run(v5.TRAIN_T, trP, trO, 0, 30); newid = run(v5.TRAIN_T, teP, teO, 0, 20); unseen = run(held_new, trP, trO, 0, 20); d2 = run(v5.TRAIN_T, trP, trO, 2, 15); d3 = run(v5.TRAIN_T, trP, trO, 3, 15)
        def pres(toks_list, P):
            out = []
            for toks in toks_list:
                c, i_, k = m.encode([toks])
                with torch.no_grad(): out.append(v16.readout(m.states(c, i_, k)[0], m, world, w, 0, P, trO, verbs)["C"]["present"])
            return out
        thr = 0.5 * (np.mean(pres(cal_known, trP + teP)) + np.mean(pres(cal_abs, trP + teP)))
        erng = np.random.default_rng(555); kn, un, ab = [], [], []
        for k in range(int(120 * a.n_scale) or 1):
            ev = w.event(trP, trO, depth=0); evu = w.event(trP, trO, depth=0); evu = (evu[0], evu[1], evu[2], teP[k % len(teP)]) + evu[4:]
            kn.append(w.render(ev, 0)); un.append(w.render(evu, 0)); ab.append(w.render_omit(ev, k % len(v16.OMIT_C)))
        occ = (float(np.mean(np.array(pres(kn, trP + teP)) > thr)), float(np.mean(np.array(pres(un, trP + teP)) > thr)), float(np.mean(np.array(pres(ab, trP + teP)) < thr)))
        r = [seen, newid, unseen, d2, d3, *occ]; res[name].append(r); print(f"  seed {seed} | {name:26s} | " + " | ".join(f"{x:.3f}" for x in r), flush=True)
print("\n== MEANS over seeds: format | bits per coordinate | " + " | ".join(axes) + " | within 0.02 of dense on every axis?")
dense = np.array(res["dense"]).mean(0)
for name, what, L in variants:
    r = np.array(res[name]); mean = r.mean(0); bits = {"dense": 64, "theta": f"routebook {int(np.log2(L))}" if what == "theta" else None}
    bpc = "64" if what is None else (f"{int(np.log2(L))} (routebook)" if what == "theta" else f"{int(np.log2(L))} (catalog)" if what == "FR" else f"{int(np.log2(L))} (chart, phasor)" if what == "S" else f"{L}/32 of 64 (chart, sparse)")
    ok = bool(np.all(np.abs(mean - dense) <= 0.02 + 1e-9)); print(f"  {name:26s} | {bpc:22s} | " + " | ".join(f"{x:.3f}±{s:.3f}" for x, s in zip(mean, r.std(0))) + f" | {'YES' if ok else 'no'}")
# capacity curve (synthetic)
print("\n== capacity curve (synthetic, D = 1024, 1,000 candidate fillers, 200 trials): n terms bundled | dense | state phasor L=16 | L=64 | block top-16/32 | top-8/32  (cleanup accuracy per filler)")
g = torch.Generator().manual_seed(0); Fc = torch.polar(torch.ones(1000, D), (torch.rand(1000, D, generator=g) * 2 - 1) * np.pi)
for n in (4, 8, 16, 32, 64):
    accs = {k: [] for k in ("dense", "L16", "L64", "k16", "k8")}
    for t in range(200):
        Rr = torch.polar(torch.ones(n, D), (torch.rand(n, D, generator=g) * 2 - 1) * np.pi); ids = torch.randperm(1000, generator=g)[:n]; S = (Rr * Fc[ids]).sum(0, keepdim=True)
        for k, Sv in (("dense", S), ("L16", qphase(S, 16)), ("L64", qphase(S, 64)), ("k16", block_topk(S, 16)), ("k8", block_topk(S, 8))):
            ok = [int(torch.real((Sv[0] * torch.conj(Rr[j])) @ torch.conj(Fc).T).argmax()) == int(ids[j]) for j in range(n)]; accs[k].append(np.mean(ok))
    print(f"   {n:3d} | " + " | ".join(f"{np.mean(accs[k]):.3f}" for k in ("dense", "L16", "L64", "k16", "k8")))
print("\nchecklist: [x] saved params, no retraining [x] one change per format [x] v16's axes, rngs and readout (dense reproduces 10b-quinquies) [x] occupancy threshold from training rows (25) [x] bits per vector stated [x] criterion written before the run [x] capacity curve with the same formats [x] 5 seeds")
