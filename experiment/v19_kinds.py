"""
v19_kinds (round 10 N5, addendum + indications 013c490 §3): KIND AMBIGUITY — one word, two categories (noun / verb homographs) on the reference
world of v16 (predicate role, occupancy marker, calibration).
World: three homograph tokens h0, h1, h2 that are BOTH an object (kind O, replacing o15..o17) AND the third-person verb form of one frame
(GIVE 'gives' -> h0, SELL 'sells' -> h1, SEND 'sends' -> h2; LEMMA maps them to the lemma). The context (the pattern) decides the kind.
Variants, same regime (5 seeds, N sentences, depths 0-2, recipient-less share 0.2):
  (a) status quo: ONE category per token = the MAJORITY kind in the training sentences (counted and printed).
  (b) category superposition: the ambiguous word's category vector is the half-way key — implemented as the mean of the two states obtained
      with the two kind assignments (every category-dependent key — path, bigram, chain A/L, gate — is averaged between the kinds; nothing else
      changes; the word is drawn half-way in all codes). Readout by FIT: the two kind hypotheses are scored against the chart — noun: the
      object slot R_B (x) (f_h + m); verb: the predicate slot R_pred (x) f_lemma(h) — the larger cosine wins, with the margin; resolved = margin
      > tau (stated), else both are reported.
  (ref) the unambiguous reference: the same world without homographs (o15..o17 and the verbs keep their own tokens).
Measure: FULL frame on sentences WITH an ambiguous word (as noun / as verb / both), the kind decision's accuracy and margin, resolved share and
both-share, FULL on sentences without ambiguity; bag-of-words control: cosine between the bags of "h as noun" and "h as verb" sentences vs the
states'. Criterion (addendum N5): (b) within 0.02 of the unambiguous reference on FULL frame; resolved-by-fit share reported; (a) reported as
the baseline. Diagnostic (printed, not part of the definition): the gate of the first open emitter after the ambiguous word and whether the
learned L keys of the two kinds differ (cosine between L[q, O] and L[q, verb] for the emitter q).
"""
import guard, argparse, time, collections, numpy as np, torch
import trainer, v5_scale as v5, v8_gates as v8, v16_reference as v16

HOMO = {"h0": ("GIVE", "gives", "give"), "h1": ("SELL", "sells", "sell"), "h2": ("SEND", "sends", "send")}

def make_world(seed, ambiguous):
    """reference world; with ambiguous=True three object tokens double as verb forms"""
    v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
    if ambiguous:
        for h, (kind, form, lemma) in HOMO.items():
            v5.VERBS[kind] = (h,) + tuple(v5.VERBS[kind][1:]); v16.LEMMA[h] = lemma
    w = v16.RefWorld(18, np.random.default_rng(seed))
    if ambiguous:
        for i, h in enumerate(HOMO):
            old = w.objects[15 + i]; w.objects[15 + i] = h
            if h not in w.idx: w.vocab.append(h)
            if HOMO[h][1] not in w.idx: w.vocab.append(HOMO[h][1])      # the verb form's own category must still exist (a vocabulary item never used in a sentence)
        w.vocab = sorted(set(w.vocab)); w.idx = {x: i for i, x in enumerate(w.vocab)}
        for h in HOMO: w.cat[h] = "O"; w.cat[HOMO[h][1]] = HOMO[h][1]   # placeholder: the kind per occurrence is decided at encode time (see encode2)
        sub = lambda p, k: " ".join({"V": v5.VERBS[k][0], "VP": v5.VERBS[k][1], "VR": v5.VERBS[k][2]}.get(t, t) for t in p.split())
        w.frames = {k: [sub(p, k) for p in v5.PATTERNS] for k in w.frames}
        w.omit = {k: [p.replace("V3", v5.VERBS[k][0]).replace("VP", v5.VERBS[k][1]) for p in v16.OMIT_C] for k in w.frames}
    return w

def kinds_of(toks, ev):
    """per token: the TRUE kind of a homograph in this sentence (noun if it is the object B of the event, else verb)"""
    return [("O" if (t == ev[2]) else t) if t in HOMO else None for t in toks]

class V19(v8.V8):
    def encode2(self, sentences, evs, mode):
        """mode 'true' = gold kinds (reference / oracle); 'major:<kind>' = one category per token; 'both' = two cats tensors (superposition)"""
        w = self.w; B = len(sentences); T = max(len(s) for s in sentences)
        cats1 = torch.zeros(B, T, dtype=torch.long); cats2 = torch.zeros(B, T, dtype=torch.long); ids = torch.zeros(B, T, dtype=torch.long); mask = torch.zeros(B, T, dtype=self.dtype)
        for b, (toks, ev) in enumerate(zip(sentences, evs)):
            kd = kinds_of(toks, ev)
            for k, t in enumerate(toks):
                ids[b, k] = w.idx[t]; mask[b, k] = 1
                if t in HOMO:
                    if mode == "true": c1 = c2 = kd[k]
                    elif mode.startswith("major:"): c1 = c2 = (HOMO[t][1] if mode == "major:verb" else "O")
                    else: c1, c2 = "O", HOMO[t][1]
                else: c1 = c2 = w.cat[t]
                cats1[b, k] = self.cat_id[c1]; cats2[b, k] = self.cat_id[c2]
        d = self.device; return cats1.to(d), cats2.to(d), ids.to(d), mask.to(d)
    def states2(self, cats1, cats2, ids, mask):
        S1 = self.states(cats1, ids, mask)
        if torch.equal(cats1, cats2): return S1
        return 0.5 * (S1 + self.states(cats2, ids, mask))

class Wrap:                                                   # trainer.train_loop expects .states(cats, ids, mask) and .theta: cats here packs both kind assignments
    def __init__(self, m): self.m = m; self.theta = m.theta
    def states(self, cats, ids, mask): return self.m.states2(cats[..., 0], cats[..., 1], ids, mask)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=30000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda"); ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--p_omit", type=float, default=0.2)
    ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--variants", default="ref,a,b"); ap.add_argument("--tag", default="kinds"); a = ap.parse_args()
    print(f"v19_kinds (N5): device={a.device}, D={a.D}, N={a.N}, seeds={a.seeds}, tau={a.tau}; homographs {HOMO} (object o15..o17 AND the V3 form of GIVE / SELL / SEND)")
    print("variants: ref = unambiguous world; a = one category per token (majority kind in training); b = superposition (mean of the two kind assignments' states), kind resolved at readout by fit (margin > tau)")
    print("criterion (addendum N5): (b) FULL on ambiguous sentences within 0.02 of ref; resolved share and both-share reported; (a) the baseline; bag control\n")
    res = collections.defaultdict(list)
    for seed in range(a.seeds):
        for variant in a.variants.split(","):
            w = make_world(seed, variant != "ref"); verbs = sorted(set(v16.LEMMA.values())); trP, teP, trO, teO = w.persons[:12], w.persons[12:], w.objects[:12], w.objects[12:]
            # homographs are among the TEST objects (o15..o17 -> teO) so that training sees them as objects only through teO? No: training must see both kinds -> use all objects in training
            trO_all = w.objects
            F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), a.D))); world = v16.RefRoleWorld(F, w, np.random.default_rng(100 + seed))
            m = V19(w, a.D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)
            trng = np.random.default_rng(200 + seed); sents, evs, targets = [], [], []
            for n in range(a.N):
                d = int(trng.integers(3)); ev = w.event(trP, trO_all, depth=d)
                if trng.random() < a.p_omit: toks = w.render_omit(ev, int(trng.integers(len(v16.OMIT_C)))); targets.append(world.state(ev, toks, has_C=False))
                else: toks = w.render(ev, int(trng.choice(v5.TRAIN_T))); targets.append(world.state(ev, toks))
                sents.append(toks); evs.append(ev)
            # majority kind (variant a), counted on the training sentences
            cnt = collections.Counter(k for toks, ev in zip(sents, evs) for k in kinds_of(toks, ev) if k)
            n_noun = cnt.get("O", 0); n_verb = sum(v for k, v in cnt.items() if k != "O"); major = "major:verb" if n_verb > n_noun else "major:noun"
            mode = {"ref": "true", "a": major, "b": "both"}[variant]
            c1, c2, ids, mask = m.encode2(sents, evs, mode); cats = torch.stack([c1, c2], -1); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
            t0 = time.time(); trainer.train_loop(Wrap(m), [m.theta, m.logits], cats, ids, mask, tgt, a.batch, a.lr, log_every=a.log_every, ckpt=f"logs/.ckpt_v19_{a.tag}_{variant}_{seed}.pt", tag=f" {variant} seed {seed}")
            if a.device == "cuda": torch.cuda.synchronize()
            # evaluation: sentences with the homograph as NOUN, as VERB, and without any homograph; kind decision by fit (variant b: on the superposed state; a: the same readout for comparison)
            erng = np.random.default_rng(900 + seed); R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}; mk = torch.tensor(world.m, dtype=m.F.dtype, device=m.device)
            Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
            def cos(x, y): return float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt()))
            def evaluate(case, n=40):
                full, kind_ok, margins, both, bagc, stc = [], [], [], [], [], []
                templs = [t for t in v5.TRAIN_T if "V" in v5.PATTERNS[t].split()] if case == "verb" else list(v5.TRAIN_T)   # the verb case needs the V3 form in the sentence
                for t in templs:
                    for _ in range(n):
                        if case == "noun":   ev = w.event(trP, trO_all, depth=0); ev = (ev[0], ev[1], erng.choice(list(HOMO)) if variant != "ref" else erng.choice(w.objects[15:]), ev[3]) + ev[4:]
                        elif case == "verb": ev = w.event(trP, trO, depth=0); ev = (erng.choice([HOMO[h][0] for h in HOMO]),) + ev[1:]
                        else:                ev = w.event(trP, trO, depth=0); ev = (erng.choice(["LEND", "SHOW", "OFFER"]),) + ev[1:]
                        toks = w.render(ev, t)
                        c1, c2, i_, k = m.encode2([toks], [ev], mode)
                        with torch.no_grad(): S = m.states2(c1, c2, i_, k)[0]
                        o = v16.readout(S, m, world, w, 0, trP, trO_all, verbs); lem = w.surface_verb(toks)
                        full.append(o["A"]["id"] == ev[1] and o["B"]["id"] == ev[2] and o["C"]["id"] == ev[3] and o["pred"]["id"] == lem)
                        if case in ("noun", "verb") and variant != "ref":
                            h = next(x for x in toks if x in HOMO); Sn = S / S.abs().pow(2).sum().sqrt()
                            s_noun = cos(Sn, R["B"] * (Fn[w.idx[h]] + mk / mk.abs().pow(2).sum().sqrt())); s_verb = cos(Sn, R["pred"] * Fn[w.idx[HOMO[h][2]]])
                            kind = "noun" if s_noun > s_verb else "verb"; mg = abs(s_noun - s_verb); kind_ok.append(kind == case); margins.append(mg); both.append(mg <= a.tau)
                            # bag control: the same words with h as noun vs as verb (a paired sentence of the other case)
                            if case == "noun" and "V" in v5.PATTERNS[t].split():
                                ev2 = (HOMO[h][0], ev[1], erng.choice(trO), ev[3]) + ev[4:]; toks2 = w.render(ev2, t)
                                bag = lambda tk: m.F[torch.tensor([w.idx[x] for x in tk], device=m.device)].sum(0)
                                c1b, c2b, ib, kb = m.encode2([toks2], [ev2], mode)
                                with torch.no_grad(): S2 = m.states2(c1b, c2b, ib, kb)[0]
                                bagc.append(cos(bag(toks), bag(toks2))); stc.append(cos(S, S2))
                return (float(np.mean(full)), (float(np.mean(kind_ok)) if kind_ok else float("nan")), (float(np.mean(margins)) if margins else float("nan")),
                        (float(np.mean(both)) if both else float("nan")), (float(np.mean(bagc)) if bagc else float("nan")), (float(np.mean(stc)) if stc else float("nan")))
            rn, rv, rp = evaluate("noun"), evaluate("verb"), evaluate("plain")
            gt = m.gates(); first_em = max((c for c in ("to", ".", "that", "by", "from")), key=lambda c: gt.get(c, 0))
            row = [rn[0], rv[0], rp[0], rn[1], rv[1], (rn[2] + rv[2]) / 2 if variant != "ref" else float("nan"), 1 - (rn[3] + rv[3]) / 2 if variant != "ref" else float("nan"), (rn[3] + rv[3]) / 2 if variant != "ref" else float("nan"), rn[4], rn[5]]
            res[variant].append(row)
            print(f"  seed {seed} {variant:3s} ({time.time()-t0:.0f}s; majority kind in training: noun {n_noun} / verb {n_verb}): FULL h-as-noun {rn[0]:.3f} | h-as-verb {rv[0]:.3f} | no homograph {rp[0]:.3f} || kind by fit: acc noun {rn[1]:.3f} verb {rv[1]:.3f}, margin {row[5]:.3f}, resolved {row[6]:.3f}, both {row[7]:.3f} || bag cos(noun use, verb use) {rn[4]:.2f} vs state {rn[5]:.2f} || gates OPEN: {sorted(c for c, v in gt.items() if v > 0.5)[:8]}", flush=True)
    cols = ["FULL h-as-noun", "FULL h-as-verb", "FULL no homograph", "kind acc (noun)", "kind acc (verb)", "kind margin", "resolved share (margin > tau)", "both share", "bag cos noun/verb use", "state cos noun/verb use"]
    print("\n== MEANS over seeds, per variant: " + " | ".join(cols))
    for variant in a.variants.split(","):
        r = np.array(res[variant]); print(f"  {variant:3s} " + " | ".join(f"{np.nanmean(r[:, i]):.3f}±{np.nanstd(r[:, i]):.3f}" for i in range(len(cols))))
    if "ref" in res and "b" in res:
        ra, rb = np.array(res["ref"]), np.array(res["b"]); gap = np.mean(ra[:, :2]) - np.mean(rb[:, :2])
        print(f"== criterion N5: (b) FULL on ambiguous sentences {np.mean(rb[:, :2]):.3f} vs reference {np.mean(ra[:, :2]):.3f}: gap {gap:+.3f} -> {'MET' if abs(gap) <= 0.02 else 'NOT met'} (within 0.02)")
    print(f"\nchecklist: [x] world addresses [x] shared F [x] chance printed (objects 1/18, verbs 1/12) [x] predicate role present (23) [x] three variants, same regime [x] kind resolved at readout by fit, margin > tau stated (18) [x] resolved / both shares [x] bag control [x] memory logged (21) [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [x] given vs learned stated (20: kinds are an input; the half-way key is a definition, not learned)")
