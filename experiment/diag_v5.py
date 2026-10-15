"""Diagnostic v5 (PRINCIPII_EVALUARE §11: traseaza inainte de a interpreta): acuratete per FORMULARE x ROL, seed 0, lume mare.
Intrebari: (1) de ce 'toate 4' da 0.84 pe T9/T10 (tranzitii vazute) cand path+chain+bigram da 0.998? care rol, care formulare?
           (2) la adancimea 3, ce se pierde: rolul X3 (niciodata tinta la antrenare => sansa prin constructie) sau si A/B/C/X1/X2?"""
import sys, numpy as np, torch, v2, v5_scale as v5

def per_role(m, world, w, t, persons, objects, depth, n=40):
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt()
    roles = [("A", 1, persons), ("B", 2, objects), ("C", 3, persons)] + [(v5.XROLES[i], 4 + i, persons) for i in range(depth)]
    out = {}
    for kind in w.frames:
        evs = [(kind,) + w.event(persons, objects, depth=depth)[1:] for _ in range(n)]
        idx, ids, mask = m.encode([w.render(ev, t) for ev in evs])
        with torch.no_grad(): S = m.states(idx, ids, mask)
        Sn = S / S.abs().pow(2).sum(-1, keepdim=True).sqrt()
        for r, ri, cands in roles:
            cid = torch.tensor([w.idx[x] for x in cands], device=m.device)
            sc = torch.real((Sn * torch.conj(R[r])) @ torch.conj(Fn[cid]).T).cpu().numpy()
            truth = np.array([cands.index(ev[ri]) for ev in evs]); pred = sc.argmax(1)
            out.setdefault(r, []).extend(pred == truth)
            # ce se prezice gresit: rolul cu care se confunda (acelasi tip de candidat)
            if r in ("A", "C") + tuple(v5.XROLES):
                for p_, ev in zip(pred, evs):
                    if cands[p_] != ev[ri]:
                        who = [rr for rr, rj, _ in roles if rj != ri and ev[rj] == cands[p_]]
                        out.setdefault(f"{r}->", []).append(who[0] if who else "?")
    return out

if __name__ == "__main__":
    seed = 0; D, N, batch = 1024, 30000, 8
    w = v5.BigWorld(18, np.random.default_rng(seed)); cats = sorted(set(w.cat.values()))
    trP, trO = w.persons[:12], w.objects[:12]
    F = np.exp(1j * np.random.default_rng(300 + seed).uniform(-np.pi, np.pi, (len(w.vocab), D)))
    world = v5.BigRoleWorld(F, w, np.random.default_rng(100 + seed))
    for name, codes in [("path+chain+bigram", ["path", "chain", "bigram"]), ("toate 4", ["path", "chain", "bigram", "pos"]), ("pos", ["pos"])]:
        m = v5.V5(w, D, np.random.default_rng(200 + seed), v2.HAND_GATES(cats), codes, "cuda"); m.set_F(F)
        m.train(world, N, templates=v5.TRAIN_T, persons=trP, objects=trO, batch=batch)
        print(f"\n== {name} ==")
        for t in [9, 10, 0, 3]:
            o = per_role(m, world, w, t, trP, trO, 0)
            conf = {k: dict(zip(*np.unique(v, return_counts=True))) for k, v in o.items() if k.endswith("->") and v}
            print(f"  T{t:2d} '{v5.PATTERNS[t]:34s}' " + " ".join(f"{r} {np.mean(o[r]):.2f}" for r in ("A", "B", "C")) + f"   confuzii: {conf}")
        for d in (1, 2, 3):
            o = per_role(m, world, w, 0, trP, trO, d, n=60)
            conf = {k: dict(zip(*np.unique(v, return_counts=True))) for k, v in o.items() if k.endswith("->") and v}
            print(f"  adancime {d} (T0): " + " ".join(f"{r} {np.mean(o[r]):.2f}" for r in ("A", "B", "C") + tuple(v5.XROLES[:d])) + f"   confuzii: {conf}")
