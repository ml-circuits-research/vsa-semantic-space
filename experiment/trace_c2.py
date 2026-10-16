"""Rule-11 trace for C2 (judgment a8a8fab E2), BEFORE training: one two-quantifier sentence inside TWO reporting frames (depth 2, the unseen
depth of the E2c regime). For the wide noun ('child' under 'each') and the narrow noun ('book' under 'a'): which chain address does each take
under (i) the learned gates of E2c (determiners CLOSED; open: that, to, from) and (ii) the same gates with the determiner categories GIVEN open
(the C2 flag)? The slot target needs the scope term at rho_rep + rho_rep + rho_sc + rho_A + f_child, i.e. the chain address of 'child' must
differ between 'each child' and 'every child' AND be re-addressed by both 'that'. No training; chain code only (v6.run_left)."""
import numpy as np, v5_scale as v5, v6_left as v6, v14_quantifiers as q
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w = q.QWorld(np.random.default_rng(0)); cats = sorted(set(w.cat.values()))
base = {c: c in ("that", "to", "from") for c in cats}; flag = dict(base); flag.update({c: True for c in ("every", "each", "some", "a", "certain")})
P = [x for x in w.vocab if w.cat[x] == "P"]; NP, NO = q.NOUNS_P[:3], q.NOUNS_O[:1]          # speakers and nouns from the world's own vocabulary
S1 = f"{P[0]} says that {P[1]} says that each {NP[0]} gives a {NO[0]} to the {NP[1]} ."
S2 = f"{P[0]} says that {P[1]} says that every {NP[0]} gives a {NO[0]} to the {NP[1]} ."
WIDE, NARROW, THIRD = NP[0], NO[0], NP[1]
for label, gates in (("(i) E2c learned gates: that, to, from open; determiners closed", base), ("(ii) C2 flag: determiners GIVEN open", flag)):
    m = v6.V6(w, 8, np.random.default_rng(0), gates, ["chainL"], "cpu"); m.np_model.gates = dict(gates); print(f"\n== {label}")
    addr = {}
    for s in (S1, S2):
        toks = s.split(); _, placed = v6.run_left(m.np_model, toks); print("  " + s)
        for t, mem in zip(toks, placed):
            names = [m.np_model.names[i] for i in mem if i]
            line = " + ".join(n.replace("A:", "").replace("L:", "L[") + ("]" if n.startswith("L:") else "") for n in names) if names else "root"
            if t in (WIDE, NARROW, THIRD): print(f"     {t:8s} <- {line}"); addr[(s, t)] = line
    same = addr[(S1, WIDE)] == addr[(S2, WIDE)]
    print(f"  chain address of the wide noun under 'each' vs 'every': {'IDENTICAL' if same else 'DIFFERENT'}; contains 'that' re-addressing: {'yes' if 'that' in addr[(S1, WIDE)] else 'no'}")
print("""
Target (slot world, depth 2): rho_rep + rho_rep + rho_sc + rho_A + f_wide for the wide noun; the narrow noun gets only rho_rep + rho_rep + rho_B + f_narrow.
Verdict: under (i) the chain address of the wide noun is IDENTICAL for 'each' and 'every' (that:that + that:to + L[to:NP] in both), so the only
terms that can distinguish them are path / bigram (bg:each:NP vs bg:every:NP), which carry no 'that' displacement => not re-addressed by the
frame => cannot land at rho_rep + rho_rep + rho_sc + rho_A. Under (ii) the wide noun's chain address contains the determiner's OWN category
(that:that + that:each + each:a + L[a:NP] vs that:that + that:every + every:a + L[a:NP]): a product of per-category vectors that includes the
two 'that' displacements and differs between the two determiners => the scope term CAN move with the frame. The flag is therefore a valid test
of H-a (addressing). What it cannot settle: whether the gradient makes (that:each + each:a) - (that:every + every:a) equal to rho_sc + (the
slot's own correction) — that is what the run measures. Cost to watch: with the determiners open EVERY noun's base address now passes through
its determiner ('the' stays closed, so definite and quantified nouns take different chains) => report nouns and determiners at depth 2, not
only the scope. One- and two-quantifier scope are reported separately (H-a vs H-b).""")
