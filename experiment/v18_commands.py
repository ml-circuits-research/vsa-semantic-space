"""
v18_commands — round 10 B3: SCOPE OF NEGATION ON COMMANDS (chapter 8C of the positioning document). Model unchanged (v8 bundle, learned gates).

Command world (roles given; moves/addresses/gates learned — rule 20):
    S = R_act (x) f_action + R_pat (x) (f_noun + R_attr (x) f_attr) [+ R_dest (x) f_place] [+ R_princ (x) f_user] [+ R_cond (x) S_condition]
    negation (uniform with E1: R_neg wraps its scope):
      prohibition (scope = the whole command):            "do not delete the unsaved files"      S = R_neg (x) S_cmd
      negated attribute inside the patient (scope = attr): "delete the unsaved files" / "delete the files that are not saved"
                                                           patient slot = f_files + R_attr (x) (R_neg (x) f_saved)
      two commands, first prohibited:                     "do not delete the files , save them"   S = R_neg (x) S_1 + R_next (x) S_2
      negation under a condition:                         "if the files are unsaved , do not delete them"   S = R_cond (x) S_c + R_then (x) (R_neg (x) S_cmd)
    actions: delete, save, send, move, copy; patients: files, messages, reports, drafts, logs, photos (+ 3 new at test); attributes: saved,
    sent, old, shared, signed (each with an affixal negative form: unsaved, unsent, ...); destinations: the archive, the cloud, the trash,
    the folder; principals: the user, the admin.
Decoding (rule 17): hypotheses {none, prohibition, attr-negated, both} x the frame readouts; SCOPE reported separately from PRESENCE.
Measures: scope accuracy per form; pair separation between the affirmed and the negated command (gap > 0.5), bag-of-words control on the same
rows; confident-wrong rate on the pairs where ONLY the scope differs ("do not delete the saved files" vs "delete the unsaved files");
calibration (rule 18) on every readout; unseen depth: the command inside a report ("the admin says that ...") seen at depth 0-1 (share p),
tested at depth 2. LLM paraphrases (one LLM paraphrases, a second checks the frame): PLANNED — no LLM API on this machine; templated sentences
only in this run, said plainly.
Criteria: scope > 0.95 on templated sentences; confident-wrong on scope < 1% at coverage > 90% (reported, not tuned); gap > 0.5.
Run: python3 v18_commands.py [--seeds 5] [--N 40000] [--D 1024]
"""
import argparse, time
import numpy as np, torch
import guard, trainer, v2, v5_scale as v5, v7_nested as v7, v8_gates as v8

ACTIONS = ["delete", "save", "send", "move", "copy"]; PATIENTS = ["files", "messages", "reports", "drafts", "logs", "photos", "notes", "tickets", "backups"]
ATTRS = {"saved": "unsaved", "sent": "unsent", "shared": "unshared", "signed": "unsigned", "old": "new"}
DESTS = ["archive", "cloud", "trash", "folder"]; PRINCIPALS = ["user", "admin"]
FORMS = {   # (label, template, negation scope: None | "S" | "attr")
    "aff":        ("{act} the {attr} {pat} .", None),
    "aff_dest":   ("{act} the {attr} {pat} to the {dest} .", None),
    "prohib":     ("do not {act} the {attr} {pat} .", "S"),
    "prohib_never": ("never {act} the {attr} {pat} .", "S"),
    "attr_affix": ("{act} the {nattr} {pat} .", "attr"),
    "attr_rel":   ("{act} the {pat} that are not {attr} .", "attr"),
    "prohib_attr": ("do not {act} the {nattr} {pat} .", "S+attr"),
}
OUTER = ["the {X} says that <P>", "the {X} asks that <P>"]

class CmdWorld:
    def __init__(self, rng):
        self.rng = rng; words = set()
        for t, _ in FORMS.values(): words |= {x for x in t.split() if not x.startswith("{")}
        for o in OUTER: words |= {x for x in o.split() if not x.startswith("{") and x != "<P>"}
        self.vocab = sorted(words | set(ACTIONS) | set(PATIENTS) | set(ATTRS) | set(ATTRS.values()) | set(DESTS) | set(PRINCIPALS)); self.idx = {x: i for i, x in enumerate(self.vocab)}
        self.cat = {x: x for x in words}
        for x in ACTIONS: self.cat[x] = "ACT"
        for x in PATIENTS: self.cat[x] = "PAT"
        for x in ATTRS: self.cat[x] = "ATTR"
        for x in ATTRS.values(): self.cat[x] = "NATTR"
        for x in DESTS: self.cat[x] = "DEST"
        for x in PRINCIPALS: self.cat[x] = "PRINC"
    def event(self, rng, patients, form, depth=0):
        act = rng.choice(ACTIONS); pat = rng.choice(patients); attr = rng.choice(list(ATTRS)); dest = rng.choice(DESTS); sp = list(rng.choice(PRINCIPALS, depth)) if depth else []
        return {"form": form, "act": act, "pat": pat, "attr": attr, "dest": dest, "speakers": sp, "scope": FORMS[form][1]}
    def render(self, ev, rng):
        t = FORMS[ev["form"]][0].format(act=ev["act"], attr=ev["attr"], nattr=ATTRS[ev["attr"]], pat=ev["pat"], dest=ev["dest"]).rsplit(" .", 1)[0]
        for x in reversed(ev["speakers"]): t = OUTER[int(rng.integers(len(OUTER)))].replace("<P>", t).format(X=x)
        return (t + " .").split()

class CmdRoleWorld:
    def __init__(self, F, world, rng):
        self.F, self.w = F, world; D = F.shape[1]; u = lambda: np.exp(1j * rng.uniform(-np.pi, np.pi, D))
        self.R = {r: u() for r in ("act", "pat", "attr", "dest", "neg", "X", "rep")}
    def state(self, ev):
        w = self.w; f = lambda x: self.F[w.idx[x]]; sc = ev["scope"]
        attr = f(ev["attr"]); attr = self.R["neg"] * attr if sc in ("attr", "S+attr") else attr
        S = self.R["act"] * f(ev["act"]) + self.R["pat"] * (f(ev["pat"]) + self.R["attr"] * attr)
        if ev["form"] == "aff_dest": S = S + self.R["dest"] * f(ev["dest"])
        if sc in ("S", "S+attr"): S = self.R["neg"] * S
        for x in reversed(ev["speakers"]): S = self.R["X"] * f(x) + self.R["rep"] * S
        return S

def decode(S, m, world, w, depth, patients):
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}; Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum(-1).sqrt(); rep = torch.conj(R["rep"]) ** depth
    def best(key, cands):
        cid = torch.tensor([w.idx[x] for x in cands], device=m.device); sc = torch.real((Sn * key) @ torch.conj(Fn[cid]).T); top = torch.topk(sc, min(2, len(cands))); return cands[int(top.indices[0])], float(top.values[0]), float(top.values[0] - (top.values[1] if len(cands) > 1 else 0))
    H = {}
    for h in (None, "S", "attr", "S+attr"):
        outer = rep * (torch.conj(R["neg"]) if h in ("S", "S+attr") else 1)
        a = best(outer * torch.conj(R["act"]), ACTIONS); p = best(outer * torch.conj(R["pat"]), patients)
        at = best(outer * torch.conj(R["pat"]) * torch.conj(R["attr"]) * (torch.conj(R["neg"]) if h in ("attr", "S+attr") else 1), list(ATTRS))
        H[h] = ((a[1] + p[1] + at[1]) / 3, {"act": a, "pat": p, "attr": at})
    hs = sorted(H, key=lambda k: -H[k][0]); h = hs[0]; margin = H[hs[0]][0] - H[hs[1]][0]
    return h, H[h][1], margin

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--N", type=int, default=40000); ap.add_argument("--D", type=int, default=1024); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.02); ap.add_argument("--device", default="cuda"); ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--p_depth1", type=float, default=0.3)
    ap.add_argument("--log_every", type=int, default=500); ap.add_argument("--ckpt", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--tag", default="cmd")
    a = ap.parse_args(); D, N = a.D, a.N
    print(f"v18_commands (B3): device={a.device}, D={D}, N={N}, seeds={a.seeds}, tau={a.tau}; command world, negation as a role around its scope; forms {list(FORMS)}; templated sentences only (LLM paraphrases PLANNED, no API)")
    print(f"training: all forms at depth 0, and inside one report ('the admin says that ...') with share {a.p_depth1} at depth 1; test: depth 0 (seen forms, fresh events), new patients (3 unseen), depth 2 (UNSEEN)")
    print(f"chance: action 1/5, patient 1/6 (new 1/3), attribute 1/5, scope hypothesis 1/4\n")
    trPat, tePat = PATIENTS[:6], PATIENTS[6:]
    cols = ["d0: action", "patient", "attribute", "PRESENCE", "SCOPE", "FULL", "new patients FULL", "d1 FULL", "d2 UNSEEN: SCOPE", "d2 FULL", "scope-only pairs: confident-wrong", "coverage"]
    res = []; seps = []; forms_acc = {}
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed); w = CmdWorld(rng); cats = sorted(set(w.cat.values()))
        F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D))); world = CmdRoleWorld(F, w, np.random.default_rng(100 + seed))
        m = v8.V8(w, D, np.random.default_rng(200 + seed), ["path", "chainL", "bigram"], a.device, gate_init="closed"); m.set_F(F)   # V8 needs only world.vocab / idx / cat
        trng = np.random.default_rng(200 + seed); sents, targets = [], []
        for n in range(N):
            form = list(FORMS)[int(trng.integers(len(FORMS)))]; d = 1 if trng.random() < a.p_depth1 else 0
            ev = w.event(trng, trPat, form, depth=d); sents.append(w.render(ev, trng)); targets.append(world.state(ev))
        t0 = time.time(); cts, ids, mask = m.encode(sents); tgt = torch.tensor(np.array(targets), dtype=m.F.dtype, device=m.device)
        trainer.train_loop(m, [m.theta, m.logits], cts, ids, mask, tgt, a.batch, a.lr, log_every=a.log_every, ckpt=(f"logs/.ckpt_{a.tag}_{seed}.pt" if a.ckpt else None), resume=a.resume, tag=f" seed {seed}")
        if a.device == "cuda": torch.cuda.synchronize()
        t_train = time.time() - t0
        def run(depth, patients, n=40, forms=None):
            erng = np.random.default_rng(900 + depth); acc = {"act": [], "pat": [], "attr": [], "pres": [], "scope": [], "full": []}; per = {}
            for form in (forms or FORMS):
                for k in range(n):
                    ev = w.event(erng, patients, form, depth=depth); toks = w.render(ev, erng); c, i_, kk = m.encode([toks])
                    with torch.no_grad(): h, fill, mg = decode(m.states(c, i_, kk)[0], m, world, w, depth, patients)
                    truth = ev["scope"]; ok = [fill["act"][0] == ev["act"], fill["pat"][0] == ev["pat"], fill["attr"][0] == ev["attr"]]
                    acc["act"].append(ok[0]); acc["pat"].append(ok[1]); acc["attr"].append(ok[2]); acc["pres"].append((h is not None) == (truth is not None)); acc["scope"].append(h == truth); acc["full"].append(all(ok) and h == truth)
                    per.setdefault(form, []).append(h == truth)
            return {k: float(np.mean(v)) for k, v in acc.items()}, {f: float(np.mean(v)) for f, v in per.items()}
        d0, per0 = run(0, trPat); nw, _ = run(0, tePat, n=20); d1, _ = run(1, trPat, n=20); d2, per2 = run(2, trPat, n=20)
        # scope-only pairs: "do not delete the saved files" (S) vs "delete the unsaved files" (attr) — same words except the negation site
        erng = np.random.default_rng(77); cw, cov, cm, cb = [], [], [], []
        for k in range(120):
            base = w.event(erng, trPat, "prohib"); e1 = dict(base, form="prohib", scope="S"); e2 = dict(base, form="attr_affix", scope="attr"); e0 = dict(base, form="aff", scope=None)
            t1, t2, t0_ = w.render(e1, erng), w.render(e2, erng), w.render(e0, erng)
            with torch.no_grad():
                S1 = m.states(*m.encode([t1]))[0]; S2 = m.states(*m.encode([t2]))[0]; S0 = m.states(*m.encode([t0_]))[0]
            for S, truth in ((S1, "S"), (S2, "attr")):
                h, fill, mg = decode(S, m, world, w, 0, trPat); conf = mg >= a.tau; cov.append(conf); cw.append(conf and h != truth)
            cosf = lambda x, y: float(torch.real((torch.conj(x) * y).sum()) / (x.abs().pow(2).sum().sqrt() * y.abs().pow(2).sum().sqrt()))
            bag = lambda toks: m.F[m.encode([toks])[1][0]].sum(0)
            cm.append((cosf(S0, S1), cosf(S1, S2))); cb.append((cosf(bag(t0_), bag(t1)), cosf(bag(t1), bag(t2))))
        r = [d0["act"], d0["pat"], d0["attr"], d0["pres"], d0["scope"], d0["full"], nw["full"], d1["full"], d2["scope"], d2["full"], float(np.mean(cw)), float(np.mean(cov))]
        res.append(r); seps.append((np.mean(cm, 0), np.mean(cb, 0)))
        for f, v in per0.items(): forms_acc.setdefault(f, []).append(v)
        print(f"  seed {seed} ({t_train:.0f}s) " + " | ".join(f"{x:.3f}" for x in r) + f"  || cos(aff, prohib) {np.mean(cm,0)[0]:.2f} cos(prohib, attr-neg) {np.mean(cm,0)[1]:.2f}; bag {np.mean(cb,0)[0]:.2f}/{np.mean(cb,0)[1]:.2f} || gates OPEN: {sorted(c for c, v in m.gates().items() if v > 0.5)}", flush=True)
    r = np.array(res); print(f"\n== MEANS over {a.seeds} seeds: " + " | ".join(cols)); print("  " + " | ".join(f"{r[:,i].mean():.3f}±{r[:,i].std():.3f}" for i in range(len(cols))))
    print("== scope accuracy per form (depth 0): " + " ".join(f"{f}={np.mean(v):.2f}" for f, v in forms_acc.items()))
    s = np.array([x[0] for x in seps]); b = np.array([x[1] for x in seps]); print(f"== separation model | bag: cos(aff, prohibition) {s[:,0].mean():.3f} | {b[:,0].mean():.3f}; cos(prohibition, attr-negated) {s[:,1].mean():.3f} | {b[:,1].mean():.3f}")
    print(f"\nchecklist: [x] world addresses [x] shared F [x] chance printed [x] scope separately from presence (17) [x] scope-only pairs with confident-wrong and coverage (18) [x] bag control [x] unseen depth [x] memory logged (21) [{'x' if a.seeds >= 3 else ' '}] >=3 seeds [ ] LLM paraphrases (PLANNED) [x] given vs learned stated (20)")
