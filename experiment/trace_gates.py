"""Trace for the principle note: the same sentences addressed with the HAND gates (all but P, O emit) and with the LEARNED gates
(round 7c, start closed: that, to, by, '.'), word by word (parameter names), using the hard chainL structure of v6_left (right expectations +
left re-addressing). No training involved: this shows WHICH parameters each word's address is made of under each gate set."""
import numpy as np, v2, v5_scale as v5, v6_left as v6
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w = v5.BigWorld(18, np.random.default_rng(0)); cats = sorted(set(w.cat.values()))
HAND = v2.HAND_GATES(cats); LEARNED = {c: c in ("that", "to", "by", ".") for c in cats}
sents = ["p0 gives o0 to p1 .", "p0 says that p1 gives o0 to p2 .", "p0 says that p1 claims that p2 gives o0 to p3 .", "o0 is given to p1 by p0 ."]
def show(label, gates):
    m = v6.V6(w, 8, np.random.default_rng(0), gates, ["chainL"], "cpu"); m.np_model.gates = dict(gates)
    print(f"\n== {label}: emitters = {sorted(c for c, g in gates.items() if g)}")
    for s in sents:
        toks = s.split(); _, placed = v6.run_left(m.np_model, toks)
        print("  " + s)
        for t, mem in zip(toks, placed):
            names = [m.np_model.names[i] for i in mem if i]
            print(f"     {t:7s} <- " + (" + ".join(n.replace("A:", "").replace("L:", "L[") + ("]" if n.startswith("L:") else "") for n in names) if names else "root"))
show("HAND gates (round 2 prior)", HAND); show("LEARNED gates (round 7c, start closed)", LEARNED)
