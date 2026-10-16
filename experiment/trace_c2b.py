"""Rule-11 trace for C2b (judgment 3b4f837 E2), BEFORE training: the determiner as a BEACON-RELATIVE emitter. Gates as learned in E2c (that, to,
from open; determiners closed). The quantified determiner draws the NEXT noun a second time at (the noun's own address) + REL[det] — the probe
is untouched. Checks (a) the wide noun's second address contains the determiner's key (REL[each]) AND the 'that' re-addressing (inherited from
the noun's own address), (b) the other words have IDENTICAL addresses under each / every, (c) in the two-quantifier sentence each determiner's
term is relative to its own slot. No training; chain code only (v6.run_left) + the REL rule applied symbolically."""
import numpy as np, v5_scale as v5, v6_left as v6, v14_quantifiers as q
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w = q.QWorld(np.random.default_rng(0)); cats = sorted(set(w.cat.values())); gates = {c: c in ("that", "to", "from") for c in cats}
P = [x for x in w.vocab if w.cat[x] == "P"]; NP, NO = q.NOUNS_P[:3], q.NOUNS_O[:1]; EM = {"every", "each", "some", "a", "certain"}
m = v6.V6(w, 8, np.random.default_rng(0), gates, ["chainL"], "cpu"); m.np_model.gates = dict(gates)
def trace(s):
    toks = s.split(); _, placed = v6.run_left(m.np_model, toks); out = {}
    for k, (t, mem) in enumerate(zip(toks, placed)):
        names = [m.np_model.names[i].replace("A:", "").replace("L:", "L[") + ("]" if m.np_model.names[i].startswith("L:") else "") for i in mem if i]
        base = " + ".join(names) if names else "root"; out[t] = base
        if w.cat[t] in ("NP", "NO") and k > 0 and toks[k - 1] in EM: out[t + " (2nd drawing)"] = f"[{base}] + REL[{toks[k-1]}]"
    return out
for s1, s2 in ((f"{P[0]} says that {P[1]} says that each {NP[0]} gives a {NO[0]} to the {NP[1]} .", f"{P[0]} says that {P[1]} says that every {NP[0]} gives a {NO[0]} to the {NP[1]} ."),
               (f"each {NP[0]} gives every {NO[0]} to the {NP[1]} .", f"a {NP[0]} gives a certain {NO[0]} to the {NP[1]} .")):
    t1, t2 = trace(s1), trace(s2); print(f"\n  {s1}\n  {s2}")
    for key in sorted(set(t1) | set(t2), key=lambda k: (k.endswith(")"), k)):
        a, b = t1.get(key, "-"), t2.get(key, "-"); print(f"     {key:22s} | {a:55s} | {b:55s} | {'same' if a == b else 'DIFFERENT'}")
print("""
Checks: (a) the wide noun's 2nd drawing = [its own address, which contains that:that + that:to] + REL[each] -> contains the determiner's key
and the 'that' re-addressing; (b) every other word (the narrow noun, the third noun, the verb) is 'same' under each / every — the probe is untouched;
(c) two quantifiers: each determiner's REL is added to ITS OWN noun's address (A's slot for 'each', B's slot for 'every'), not to a shared
cumulative position. What the trace cannot settle: whether the gradient makes REL[each] = rho_sc and what it does with REL[every] (wide in a
one-quantifier sentence, narrow next to 'each'): that conflict is H-b and the run measures it.""")
