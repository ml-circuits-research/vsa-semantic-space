"""
frame.py — the canonical output of every model (requirement R1) and its mechanical translations (R2).

frame(S, ...)   -> nested {role: filler}: unbinding with the WORLD's role vectors + cleanup over known candidates. For the reference world
                   (v7_nested): {"A": .., "B": .., "C": .., "says": [X1, X2, ...]} with the nesting depth ESTIMATED from S itself (the depth at
                   which the inner roles decode best), not given.
diff(fa, fb)    -> {"missing": {role: filler_a}, "extra": {role: filler_b}, "different": {role: (a, b)}}.
to_prolog(f)    -> Prolog facts (one term per clause, nesting as a term):  says(p0, says(p1, gives(p2, o0, p3))).  plus role facts.
to_smt(f)       -> SMT-LIB 2 (Z3): sorts, constants, uninterpreted role functions and assertions; checked with z3 when available.
round_trip(...) -> text -> model state -> frame -> Prolog/SMT -> query -> answer, compared with the WORLD's answer (no learning anywhere).
No SWI-Prolog on this machine: the generated .pl text is valid Prolog and is executed by a minimal built-in evaluator for ground facts and
conjunctive queries with variables (unification only, no rules); Z3 is used for the SMT side (z3-solver 5.3.0).
Also: full_frame_accuracy(...) — all roles correct at once — to be reported in every evaluation from now on (R1).
"""
import numpy as np, torch

ROLES_INNER = ("A", "B", "C")

def frame(S, m, world, w, persons, objects, max_depth=3, tau=0.15):
    """nested frame from a state S (torch complex, D), with the depth estimated from S"""
    R = {r: torch.tensor(world.R[r], dtype=m.F.dtype, device=m.device) for r in world.R}
    Fn = m.F / m.F.abs().pow(2).sum(-1, keepdim=True).sqrt(); Sn = S / S.abs().pow(2).sum(-1).sqrt()
    cid = {"P": torch.tensor([w.idx[x] for x in persons], device=m.device), "O": torch.tensor([w.idx[x] for x in objects], device=m.device)}
    def best(role_vec, level, kind):
        unbind = torch.conj(role_vec) * torch.conj(R["rep"]) ** level
        sc = torch.real((Sn * unbind) @ torch.conj(Fn[cid[kind]]).T); j = int(sc.argmax()); return (persons if kind == "P" else objects)[j], float(sc[j])
    # depth = the level at which the inner roles decode best (mean of the three best scores)
    scores = [np.mean([best(R[r], d, "O" if r == "B" else "P")[1] for r in ROLES_INNER]) for d in range(max_depth + 1)]
    d = int(np.argmax(scores)); f = {"depth": d, "depth_scores": [round(x, 3) for x in scores]}
    for r in ROLES_INNER: f[r] = best(R[r], d, "O" if r == "B" else "P")[0]
    f["says"] = [best(R["X"], k, "P")[0] for k in range(d)]
    return f

def frame_from_event(ev, w):
    """the WORLD's frame for an event tuple (kind, A, B, C, X1, X2, X3|None...)"""
    speakers = [x for x in ev[4:] if x is not None]
    return {"depth": len(speakers), "A": ev[1], "B": ev[2], "C": ev[3], "says": speakers}

def diff(fa, fb):
    keys = [k for k in set(fa) | set(fb) if k not in ("depth_scores",)]
    out = {"missing": {}, "extra": {}, "different": {}}
    for k in sorted(keys):
        a, b = fa.get(k), fb.get(k)
        if list(a) == list(b) if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) else a == b: continue
        if a is None or a == []: out["extra"][k] = b
        elif b is None or b == []: out["missing"][k] = a
        else: out["different"][k] = (a, b)
    return out

def equal(fa, fb): d = diff(fa, fb); return not (d["missing"] or d["extra"] or d["different"])

def full_frame_accuracy(frames_pred, frames_true):
    return float(np.mean([equal(p, t) for p, t in zip(frames_pred, frames_true)]))

# ---------------- Prolog ----------------
def to_prolog(f, verb="gives", event_id="e"):
    """facts: role facts + one nested term. Valid SWI-Prolog syntax."""
    term = f"{verb}({f['A']}, {f['B']}, {f['C']})"
    for x in reversed(f["says"]): term = f"says({x}, {term})"
    facts = [f"event({event_id}, {term}).", f"agent({event_id}, {f['A']}).", f"theme({event_id}, {f['B']}).", f"recipient({event_id}, {f['C']}).",
             f"depth({event_id}, {f['depth']})."] + [f"speaker({event_id}, {k+1}, {x})." for k, x in enumerate(f["says"])]
    return "\n".join(facts)

def prolog_eval(program, query):
    """minimal evaluator: ground facts, conjunctive query of atoms with variables (capitalized); returns list of bindings"""
    import re
    def parse_atom(s):
        s = s.strip().rstrip("."); name, args = re.match(r"(\w+)\((.*)\)$", s).groups()
        out, depth, cur = [], 0, ""
        for ch in args:
            if ch == "," and depth == 0: out.append(cur.strip()); cur = ""
            else:
                cur += ch; depth += ch == "("; depth -= ch == ")"
        out.append(cur.strip()); return name, out
    facts = [parse_atom(l) for l in program.splitlines() if l.strip()]
    goals = [parse_atom(g) for g in re.split(r",\s*(?=\w+\()", query.strip().rstrip("."))]
    def unify(pat, val, b):
        if pat[:1].isupper():
            if pat in b: return b[pat] == val
            b[pat] = val; return True
        return pat == val
    def solve(i, b):
        if i == len(goals): yield dict(b); return
        name, args = goals[i]
        for fn, fargs in facts:
            if fn != name or len(fargs) != len(args): continue
            b2 = dict(b)
            if all(unify(p, v, b2) for p, v in zip(args, fargs)): yield from solve(i + 1, b2)
    return list(solve(0, {}))

# ---------------- SMT-LIB / Z3 ----------------
def to_smt(f, event_id="e"):
    """SMT-LIB 2 text: a sort for entities, constants, uninterpreted role functions, and assertions"""
    ents = sorted({f["A"], f["B"], f["C"], *f["says"]})
    lines = ["(declare-sort Entity 0)", "(declare-sort Event 0)", f"(declare-const {event_id} Event)"] + [f"(declare-const {x} Entity)" for x in ents]
    lines += ["(declare-fun agent (Event) Entity)", "(declare-fun theme (Event) Entity)", "(declare-fun recipient (Event) Entity)",
              "(declare-fun depth (Event) Int)", "(declare-fun speaker (Event Int) Entity)"]
    lines += [f"(assert (distinct {' '.join(ents)}))"] if len(ents) > 1 else []
    lines += [f"(assert (= (agent {event_id}) {f['A']}))", f"(assert (= (theme {event_id}) {f['B']}))", f"(assert (= (recipient {event_id}) {f['C']}))", f"(assert (= (depth {event_id}) {f['depth']}))"]
    lines += [f"(assert (= (speaker {event_id} {k+1}) {x}))" for k, x in enumerate(f["says"])]
    return "\n".join(lines)

def smt_check(smt_text, extra_assert):
    """z3: is smt_text + extra_assert satisfiable? (unsat => the extra assertion contradicts the frame)"""
    import z3
    s = z3.Solver(); s.from_string(smt_text + "\n" + extra_assert); return str(s.check())

# ---------------- round trip ----------------
def round_trip(m, world, w, sentences, events, persons, objects, verbose=3):
    """text -> S -> frame -> Prolog query 'who is the recipient / who said it' and SMT consistency, vs the world; returns accuracies"""
    ok_frame, ok_prolog, ok_smt = [], [], []
    for i, (toks, ev) in enumerate(zip(sentences, events)):
        with torch.no_grad():
            enc = m.encode([toks]); S = m.states(*enc)[0]
        f = frame(S, m, world, w, persons, objects); ft = frame_from_event(ev, w); ok_frame.append(equal(f, ft))
        prog = to_prolog(f); ans = prolog_eval(prog, "recipient(e, R), agent(e, A)"); ok_prolog.append(bool(ans) and ans[0]["R"] == ev[3] and ans[0]["A"] == ev[1])
        smt = to_smt(f); ok_smt.append(smt_check(smt, f"(assert (= (recipient e) {ev[3]}))") == "sat" and smt_check(smt, f"(assert (not (= (recipient e) {ev[3]})))") == "unsat")
        if i < verbose:
            print("  text:   " + " ".join(toks)); print(f"  frame:  {f}"); print(f"  world:  {ft}"); print(f"  diff:   {diff(f, ft)}")
            print("  prolog: " + prog.replace("\n", " ")); print(f"  query recipient(e,R), agent(e,A) -> {ans}")
            print("  smt:    " + smt.replace("\n", " ")[:160] + " ...  check(recipient = world) = sat, check(recipient != world) = " + smt_check(smt, f"(assert (not (= (recipient e) {ev[3]})))"))
    return float(np.mean(ok_frame)), float(np.mean(ok_prolog)), float(np.mean(ok_smt))

if __name__ == "__main__":
    # self-test of the symbolic side (no model): frame -> prolog -> query; frame -> smt -> z3; diff
    f = {"depth": 2, "A": "p2", "B": "o0", "C": "p3", "says": ["p0", "p1"]}
    prog = to_prolog(f); print(prog); print("query:", prolog_eval(prog, "speaker(e, 2, S), recipient(e, R)"))
    smt = to_smt(f); print(smt); print("z3 sat:", smt_check(smt, "(assert (= (agent e) p2))"), "| contradiction:", smt_check(smt, "(assert (= (agent e) p3))"))
    print("diff:", diff(f, {"depth": 2, "A": "p2", "B": "o1", "C": "p3", "says": ["p0"]}))
