"""Rule-11 trace for E2c, BEFORE training: for the wide-scope noun, which address must the model produce under each world target, and is it
a product (sum of phases) of per-category vectors the model has? Uses the v8 chain structure with the learned-gate set of 9c-bis as the
reference (that, to, '.', and the recipient verbs open) plus the determiner categories; no training."""
import numpy as np, v2, v5_scale as v5, v6_left as v6, v14_quantifiers as q
v5.set_lang("en"); v5.OUTER = [o for o in v5.OUTER if o.index("<P>") > 0]
w = q.QWorld(np.random.default_rng(0)); cats = sorted(set(w.cat.values()))
for label, gates in (("9c-bis learned gates (that, to, '.', recipient verbs)", {c: c in ("that", "to", ".", "sees", "buys", "accepts", "borrows", "receives", "gets") for c in cats}),
                     ("same + wide-scope determiners 'each', 'certain' as emitters", {c: c in ("that", "to", ".", "sees", "buys", "accepts", "borrows", "receives", "gets", "each", "certain") for c in cats})):
    m = v6.V6(w, 8, np.random.default_rng(0), gates, ["chainL"], "cpu"); m.np_model.gates = dict(gates)
    print(f"\n== {label}")
    for s in ("each child gives a book to the teacher .", "the child gives each book to a teacher .", "every child gives a certain book to the teacher ."):
        toks = s.split(); _, placed = v6.run_left(m.np_model, toks)
        print("  " + s)
        for t, mem in zip(toks, placed):
            names = [m.np_model.names[i] for i in mem if i]
            print(f"     {t:8s} <- " + (" + ".join(n.replace("A:", "").replace("L:", "L[") + ("]" if n.startswith("L:") else "") for n in names) if names else "root"))
print("""
Targets per world (phases add; rho_x = phase vector of x):
  role (9c-bis):  rho_sc + rho_fA              — fA is a ROLE identity, not any word: no word term address (x) f_word can produce it.
  slot (E2c):     rho_sc + rho_A + f_child     — the noun 'child' drawn a second time at address rho_sc + rho_A: the chain term of 'child'
                  must reach rho_sc + rho_A while its path/bigram terms stay at rho_A (the plain slot). With 'each' as an emitter, 'child'
                  takes L[each:NP] (a per-category displacement) => its chain address = address(each) + L[each:NP], a product of
                  per-category vectors; the determiner category differs from 'every' (L[every:NP]) => the two addresses can differ.
  noun (secondary): rho_sc + f_child           — reachable the same way but collides when the same noun fills two roles.
Verdict of the trace: the E2c target IS a product of per-category vectors the model has (the wide determiner's emitted slot), PROVIDED the
gate of 'each' / 'certain' opens; with the 9c-bis gates (determiners closed) the noun's only chain address comes from 'to' / '.', the same
for 'each' and 'every' => the scope difference must then be carried by path/bigram (bg:each:NP vs bg:every:NP — also per-category) => reachable.""")
