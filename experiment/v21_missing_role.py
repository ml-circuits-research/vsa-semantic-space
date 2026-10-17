"""v21_missing_role — N6 (addendum N5-N7; released by judgment 752853c §1): a MISSING QUESTION in the legend — can the residual point to it?
World: the reference world of 10b-quinquies (v16: predicate role, occupancy marker, relation-marker gates to / by / from given open; path + chainL +
bigram), generated with the FULL world (every sentence of the TRAIN_T templates expresses the recipient C; recipient-less forms with share 0.2),
but the TRAINING TARGET LEGEND LACKS the recipient role: target = R_pred f_lemma + R_A (f_A + m) + R_B (f_B + m) [+ reporting], no C term.
Measured at depth 0 on fresh events (seen templates), 5 seeds:
  (i)   FULL on the remaining roles (A, B, predicate) — should hold;
  (ii)  the reconstruction residual (||S − alpha · recon|| / ||S||, recon = the legend's roles with their best-fit fillers, alpha the least-squares
        scale) on sentences WITH a recipient vs recipient-less sentences; gap = (with − without) / without; criterion: gap > 0.3;
  (iii) is the unexplained part addressable? cleanup of the hidden role's filler from the residual alone with
        (a) the ORACLE as the addendum writes it: the world's true R_C (which the reader never saw in a target);
        (b) random role vectors (5 draws) — must be at chance;
        (c) the reader's OWN address for the hidden slot, estimated on 120 CALIBRATION training sentences with their gold fillers
            (O = mean_i residual_i (x) conj(f_{C_i})), then applied to fresh sentences — stated as an addition to the addendum's design:
            a reader trained without the question has no reason to put the filler at the legend's vector, but it may put it somewhere consistent;
        criterion (addendum): > 0.8 with the oracle, 0 with random vectors;
  (iv)  the cosine structure of the residuals across sentences: mean cos between residuals of sentences with the SAME hidden filler vs DIFFERENT
        fillers; and the step from detectable to proposed without any vector: leave-one-out 1-nearest-neighbour filler accuracy over the residuals
        alone (cheap clustering), reported.
Decision rule, written before the run: (ii) and (iii-a) met -> "a missing question is detectable and addressable at the legend's vector"; (ii) met and
(iii-c) met but (iii-a) not -> "detectable and addressable at the READER's own address (the legend's vector is not where the reader put it)";
(ii) not met -> the residual does not expose the missing question in this world. Run: python3 v21_missing_role.py [--seeds 5]"""
import argparse, time, numpy as np, torch
import guard, trainer, v5_scale as v5, v8_gates as v8, v16_reference as v16

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--device", default="cuda"); ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--n", type=int, default=20); ap.add_argument("--tag", default="n6"); a = ap.parse_args()
D, N, dev = a.D, a.N, a.device; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; verbs = sorted(set(v16.LEMMA.values()))
print(f"v21_missing_role (N6): device={dev}, D={D}, N={N}, seeds={a.seeds}; reference world (10b-quinquies regime), training legend WITHOUT the recipient role; measured at depth 0; chance for the hidden filler 1/12")
tcos = lambda x, y: float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt() + 1e-12))
cols = ["FULL remaining roles", "rel. residual WITH C", "rel. residual WITHOUT C", "gap (with-without)/without", "hidden filler: oracle = true R_C", "random vectors (mean of 5)", "reader's own address (calibrated)", "cos residuals same filler", "different filler", "1-NN filler acc (no vector)"]
res = []
for seed in range(a.seeds):
    w = v16.RefWorld(18, np.random.default_rng(seed)); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v16.RefRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], dev, gate_init="closed"); m.set_F(F); m.fix_gates(["to", "by", "from"], True)
    trng = np.random.default_rng(200 + seed); sents, targets, cal = [], [], []
    for n in range(N):
        d = int(trng.integers(3)); ev = w.event(trP, trO, depth=d)
        if trng.random() < 0.2: toks = w.render_omit(ev, int(trng.integers(len(v16.OMIT_C))))
        else:
            toks = w.render(ev, int(trng.choice(v5.TRAIN_T)))
            if d == 0 and len(cal) < 120: cal.append((toks, ev[3]))
        sents.append(toks); targets.append(world.state(ev, toks, has_C=False))              # the legend LACKS the recipient: no C term in any target
    t0 = time.time(); cats, ids, mask = m.encode(sents); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=dev)
    trainer.train_loop(m, [m.theta, m.logits], cats, ids, mask, tgt, a.batch, a.lr, log_every=a.log_every, ckpt=f"logs/.ckpt_v21_{a.tag}_{seed}.pt", tag=f" seed {seed}")
    if dev == "cuda": torch.cuda.synchronize()
    R = {k: torch.tensor(v, dtype=m.F.dtype, device=dev) for k, v in world.R.items()}; mk = torch.tensor(world.m, dtype=m.F.dtype, device=dev); Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
    cidP = torch.tensor([w.idx[x] for x in trP], device=dev)
    def resid(S, toks):
        o = v16.readout(S, m, world, w, 0, trP, trO, verbs)
        recon = R["pred"] * m.F[w.idx[o["pred"]["id"]]] + R["A"] * (m.F[w.idx[o["A"]["id"]]] + mk) + R["B"] * (m.F[w.idx[o["B"]["id"]]] + mk)
        alpha = torch.real((torch.conj(recon) * S).sum()) / (recon.abs().pow(2).sum() + 1e-12); r = S - alpha * recon
        return o, r, float(r.abs().pow(2).sum().sqrt() / S.abs().pow(2).sum().sqrt())
    def states(toks_list):
        c, i_, k = m.encode(toks_list)
        with torch.no_grad(): return m.states(c, i_, k)
    # calibration: the reader's own address for the hidden slot, from 120 training sentences with gold recipients
    Sc = states([t for t, _ in cal]); O = torch.zeros(D, dtype=m.F.dtype, device=dev)
    for j, (toks, C) in enumerate(cal): _, r, _ = resid(Sc[j], toks); O = O + r * torch.conj(Fn[w.idx[C]])
    O = O / (O.abs() + 1e-12)
    rand = [torch.tensor(np.exp(1j * np.random.default_rng(7000 + 10 * seed + k).uniform(-np.pi, np.pi, D)), dtype=m.F.dtype, device=dev) for k in range(5)]
    erng = np.random.default_rng(900); full, rw, rwo, acc_o, acc_r, acc_e, resids, labels = [], [], [], [], [], [], [], []
    for t in v5.TRAIN_T:
        evs = [w.event(trP, trO, depth=0) for _ in range(a.n)]; toks = [w.render(ev, t) for ev in evs]; S = states(toks)
        for j, ev in enumerate(evs):
            o, r, rel = resid(S[j], toks[j]); full.append(o["A"]["id"] == ev[1] and o["B"]["id"] == ev[2] and o["pred"]["id"] == w.surface_verb(toks[j])); rw.append(rel); resids.append(r / (r.abs().pow(2).sum().sqrt() + 1e-12)); labels.append(ev[3])
            clean = lambda key: trP[int(torch.real((r * torch.conj(key)) @ torch.conj(Fn[cidP]).T).argmax())]
            acc_o.append(clean(R["C"]) == ev[3]); acc_e.append(clean(O) == ev[3]); acc_r.append(np.mean([clean(v) == ev[3] for v in rand]))
        evs2 = [w.event(trP, trO, depth=0) for _ in range(a.n)]; toks2 = [w.render_omit(ev, t % len(v16.OMIT_C)) for ev in evs2]; S2 = states(toks2)
        for j, ev in enumerate(evs2): rwo.append(resid(S2[j], toks2[j])[2])
    Rm = torch.stack(resids); G = torch.real(Rm @ torch.conj(Rm).T); same = np.array([[labels[i] == labels[j] for j in range(len(labels))] for i in range(len(labels))]); eye = np.eye(len(labels), dtype=bool)
    cs = float(G[torch.tensor(same & ~eye)].mean()); cd = float(G[torch.tensor(~same)].mean())
    G2 = G.clone(); G2.fill_diagonal_(-2.0); nn = G2.argmax(1).cpu().numpy(); knn = float(np.mean([labels[int(nn[i])] == labels[i] for i in range(len(labels))]))
    gap = (np.mean(rw) - np.mean(rwo)) / max(np.mean(rwo), 1e-9)
    r_ = [np.mean(full), np.mean(rw), np.mean(rwo), gap, np.mean(acc_o), np.mean(acc_r), np.mean(acc_e), cs, cd, knn]; res.append(r_)
    print(f"  seed {seed} ({time.time() - t0:.0f}s) " + " | ".join(f"{x:.3f}" for x in r_) + f"  || gates OPEN: {sorted(c for c, v in m.gates().items() if v > 0.5)}", flush=True)
r = np.array(res); print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:, i].mean():.3f}±{r[:, i].std():.3f}" for i in range(len(cols))))
g, ao, ar, ae = r[:, 3].mean(), r[:, 4].mean(), r[:, 5].mean(), r[:, 6].mean()
print(f"== criterion (addendum N6): residual gap > 0.3 relative -> {g:.3f} {'MET' if g > 0.3 else 'NOT met'}; hidden filler from the residual with the true R_C > 0.8 -> {ao:.3f} {'MET' if ao > 0.8 else 'NOT met'}; random vectors at chance (1/12 = 0.083) -> {ar:.3f}; with the reader's own calibrated address -> {ae:.3f} {'(> 0.8)' if ae > 0.8 else '(<= 0.8)'}; 1-NN over residuals alone -> {r[:, 9].mean():.3f}")
print("== decision rule (written before the run): (ii) and (iii-a) met -> detectable and addressable at the legend's vector; (ii) and (iii-c) met, (iii-a) not -> detectable and addressable at the reader's own address; (ii) not met -> not exposed in this world.")
print("\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] legend lacks the role in every target (16) [x] residual with the least-squares scale (18) [x] oracle as the addendum writes it AND the reader's own calibrated address, both stated [x] random-vector control [x] calibration rows = training sentences, scored rows fresh (25) [x] cosine structure and 1-NN without any vector [x] decision rule before the run [x] memory logged (21) [x] 5 seeds")
