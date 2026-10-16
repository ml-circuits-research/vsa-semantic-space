"""Rule-11 trace for E1 (judgment 3b4f837): the failing passive form T2 "B is given to C by A ." against T0 "A gives B to C ." inside THREE
reports (depth 3), with the learned gate set of 10b (that, '.', recipient verbs open; participles CLOSED), and the same with the participle
gates GIVEN open (E1 ii). Per form: the chain address of C, of A and of the verb / participle (the predicate term's carrier), and the per-
category keys they share. No training; chain code only (v6.run_left on the reference world of v16)."""
import numpy as np, v5_scale as v5, v6_left as v6, v16_reference as v16
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w = v16.RefWorld(18, np.random.default_rng(0)); cats = sorted(set(w.cat.values())); P, O = w.persons, w.objects
base = {c: c in ("that", ".", "accepts", "gets", "sees", "buys", "receives", "borrows") for c in cats}
part = dict(base); part.update({c: True for c in v16.PARTICIPLES})
X = f"{P[0]} says that {P[1]} says that {P[2]} says that"
S = {"T0 active": f"{X} {P[3]} gives {O[0]} to {P[4]} .", "T2 passive": f"{X} {O[0]} is given to {P[4]} by {P[3]} .", "T3 passive": f"{X} {O[0]} is given by {P[3]} to {P[4]} ."}
roles = {P[3]: "A", O[0]: "B", P[4]: "C", "gives": "pred", "given": "pred"}
relm = dict(base); relm.update({c: True for c in ("to", "by", "from")})
for label, gates in (("(i) 10b learned gates: that, '.', recipient verbs open; participles closed", base), ("(ii) E1: participle gates GIVEN open", part), ("(iii) E1b: relation markers to / by / from GIVEN open (as 7c learned them)", relm)):
    m = v6.V6(w, 8, np.random.default_rng(0), gates, ["chainL"], "cpu"); m.np_model.gates = dict(gates); print(f"\n== {label}")
    for name, s in S.items():
        toks = s.split(); _, placed = v6.run_left(m.np_model, toks); print(f"  {name}: {s}"); keys = {}
        for t, mem in zip(toks, placed):
            if t in roles:
                names = [m.np_model.names[i].replace("A:", "").replace("L:", "L[") + ("]" if m.np_model.names[i].startswith("L:") else "") for i in mem if i]
                keys[roles[t]] = set(names); print(f"     {roles[t]:4s} {t:8s} <- " + (" + ".join(names) if names else "root"))
        for a, b in (("C", "pred"), ("A", "pred"), ("A", "C")):
            print(f"     shared keys {a}/{b}: {len(keys[a] & keys[b])} of {len(keys[a])} / {len(keys[b])}  distinct in {a}: {sorted(keys[a] - keys[b])}")
print("""
Reading of the trace is written in RESULTS 10b-quater after the run (the trace is the evidence; the verdict is read from the shared-key counts:
if C's address in the passive is reached through 'to' alone and shares its 'that' prefix with the predicate's carrier while in T0 it carries the
verb's own key, the emitter test is the right one).""")
