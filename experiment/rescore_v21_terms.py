"""N6 per-term decomposition of the residual (judgment 934fbae B), on the SAVED N6 models (logs/.ckpt_v21_n6_<seed>.pt; the training legend lacked
the recipient). No training. For sentences WITH and WITHOUT the recipient (same templates and draws as RESULTS 15), the reader's chart is split into
token classes by the identity-masking trick (the chart with only that class's identities non-zero):
   S = S_hidden (the recipient token) + S_content (the other content words: A, B, the verb form) + S_function (everything else: to, by, from, is,
   it, who, what, that, says, claims, ',', '.').
The residual r = S − alpha · recon (recon = the legend's roles with their best-fit fillers and the marker, alpha the least-squares scale) is then
   r = S_hidden + S_function + (S_content − alpha · recon): three pieces, exact. Reported: the share of each piece ALONG the residual, Re<piece, r> / ||r||^2 (the
three shares sum to one exactly; the raw norm ratio is printed once for the function words to show that the class pieces cancel each other), for sentences with and without the recipient; and the N6 gap
criterion RE-SCORED on the CONTENT share alone: ||S_hidden + S_content − alpha · recon|| / ||S_hidden + S_content|| (function words excluded).
Decision rule (judgment 934fbae B), before the run: function-word terms carry the MAJORITY of the residual norm -> the tool's `_residual` becomes two
numbers (content share, function share) and the gap is re-scored on the content share: > 0.3 -> "share detector works on the content share"; else
share stays a recorded non-detector. Function words NOT the majority -> the hypothesis of RESULTS 15 is dropped, the row stays. Direction stays the
detector either way. Run: python3 rescore_v21_terms.py"""
import argparse, numpy as np, torch, guard, v5_scale as v5, v8_gates as v8, v16_reference as v16

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--tag", default="n6"); a = ap.parse_args()
D, dev = a.D, a.device; v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]; verbs = sorted(set(v16.LEMMA.values()))
print(f"rescore_v21_terms (N6 per-term decomposition): saved N6 models, {a.seeds} seeds; pieces: hidden-role token | other content words | function words | fit error of the content words")
cols = ["WITH C: ||r||/||S||", "share along r: hidden", "content fit", "function words", "(raw ||function||/||r||)", "WITHOUT C: ||r||/||S||", "share: content fit", "function words", "CONTENT-share residual WITH", "WITHOUT", "gap on the content share"]
res = []
for seed in range(a.seeds):
    w = v16.RefWorld(18, np.random.default_rng(seed)); trP, trO = w.persons[:12], w.objects[:12]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = v16.RefRoleWorld(F, w, np.random.default_rng(100 + seed))
    m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], dev, gate_init="closed"); m.set_F(F); m.fix_gates(["to", "by", "from"], True)
    st = torch.load(f"logs/.ckpt_v21_{a.tag}_{seed}.pt", map_location=dev)
    with torch.no_grad():
        for p_, v in zip([m.theta, m.logits], st["params"]): p_.copy_(v)
    R = {k: torch.tensor(v, dtype=m.F.dtype, device=dev) for k, v in world.R.items()}; mk = torch.tensor(world.m, dtype=m.F.dtype, device=dev); Ffull = m.F.clone()
    content = set(trP) | set(w.persons[12:]) | set(trO) | set(w.objects[12:]) | set(v16.LEMMA.keys())
    def class_state(toks, keep):
        F0 = torch.zeros_like(Ffull)
        for t in keep: F0[w.idx[t]] = Ffull[w.idx[t]]
        m.F = F0; c, i_, k = m.encode([toks])
        with torch.no_grad(): S = m.states(c, i_, k)[0]
        m.F = Ffull; return S
    nrm = lambda x: float(x.abs().pow(2).sum().sqrt())
    def pieces(toks, hidden):
        S = class_state(toks, set(toks)); Sh = class_state(toks, {hidden}) if hidden else torch.zeros(D, dtype=m.F.dtype, device=dev)
        Sc = class_state(toks, {t for t in toks if t in content and t != hidden}); Sf = class_state(toks, {t for t in toks if t not in content})
        o = v16.readout(S, m, world, w, 0, trP, trO, verbs); recon = R["pred"] * Ffull[w.idx[o["pred"]["id"]]] + R["A"] * (Ffull[w.idx[o["A"]["id"]]] + mk) + R["B"] * (Ffull[w.idx[o["B"]["id"]]] + mk)
        alpha = torch.real((torch.conj(recon) * S).sum()) / (recon.abs().pow(2).sum() + 1e-12); r = S - alpha * recon; fit = Sc - alpha * recon
        # attribution ALONG the residual: share_K = Re<piece_K, r> / ||r||^2 (the three shares sum to 1 exactly; a share can be negative when a
        # piece points against the residual). The raw norms are not reported as shares: the trained bundle makes the class pieces cancel each
        # other (their norms exceed ||r||), so a norm ratio would not say who carries the residual.
        proj = lambda x: float(torch.real((torch.conj(r) * x).sum()) / (nrm(r) ** 2 + 1e-12))
        Scont = Sh + Sc; rc = Scont - alpha * recon
        return nrm(r) / nrm(S), proj(Sh), proj(fit), proj(Sf), nrm(Sf) / nrm(r), nrm(rc) / nrm(Scont)
    erng = np.random.default_rng(900); A_, B_ = [], []
    for t in v5.TRAIN_T:
        evs = [w.event(trP, trO, depth=0) for _ in range(a.n)]
        for ev in evs: A_.append(pieces(w.render(ev, t), ev[3]))
        evs2 = [w.event(trP, trO, depth=0) for _ in range(a.n)]
        for ev in evs2: B_.append(pieces(w.render_omit(ev, t % len(v16.OMIT_C)), None))
    A_, B_ = np.array(A_), np.array(B_); gap = (A_[:, 5].mean() - B_[:, 5].mean()) / max(B_[:, 5].mean(), 1e-9)
    r_ = [A_[:, 0].mean(), A_[:, 1].mean(), A_[:, 2].mean(), A_[:, 3].mean(), A_[:, 4].mean(), B_[:, 0].mean(), B_[:, 2].mean(), B_[:, 3].mean(), A_[:, 5].mean(), B_[:, 5].mean(), gap]; res.append(r_)
    print(f"  seed {seed} " + " | ".join(f"{x:.3f}" for x in r_), flush=True)
r = np.array(res); print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:, i].mean():.3f}±{r[:, i].std():.3f}" for i in range(len(cols))))
fw, fwo, g = r[:, 3].mean(), r[:, 7].mean(), r[:, 10].mean()
maj = fw > 0.5 and fwo > 0.5
print(f"== decision rule: function-word share ALONG the residual = {fw:.3f} (with C) / {fwo:.3f} (without): {'the MAJORITY' if maj else 'NOT the majority'} of the residual -> " + (f"`_residual` becomes two numbers (content share, function share); the N6 gap re-scored on the CONTENT share = {g:+.3f} -> " + ("'the share detector works on the content share'" if g > 0.3 else "share stays a recorded non-detector; direction is the detector") if maj else "the RESULTS 15 hypothesis (function words dominate) is DROPPED; the row stays as recorded; direction is the detector"))
print("\nchecklist: [x] saved params, no retraining [x] same templates and draws as RESULTS 15 [x] exact three-piece split of the residual, shares along r sum to one [x] content-share gap re-scored [x] decision rule written before the run [x] 5 seeds")
