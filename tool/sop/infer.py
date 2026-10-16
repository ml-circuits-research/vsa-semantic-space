"""
infer.py — the reader at inference, numpy only (spec 4-5): a `model/` folder -> SOP wires for sentences.

  1. tokens -> kinds (the lexicon in model/catalog.json; a token with no kind cannot be placed: it gets no displacement and is listed in
     `_unplaced`, never silently dropped);
  2. the chart S = sum_k address_k (x) f_k, address_k = the bundle of the learned codes (path + soft chainL + bigram [+ beacon-relative]),
     exactly the computation of experiment/v8_gates.V8.states, re-implemented in numpy from model/routebook.npz;
  3. readout with the WORLD's role vectors (rule 1), regenerated from the seed: structure hypotheses (reporting depth d, event negation)
     scored by the mean best cleanup cosine of the always-present roles; per role: presence (occupancy marker or best cosine, as a ratio
     to the structure score, against a threshold set on CALIBRATION rows = training sentences, model/config.json), identity by cleanup
     over the catalog words of the role's kinds, margin = top1 - top2 cosine;
  4. `_residual` = ||S - a * rec|| / ||S|| with rec = the state of the decoded frame (world.Compiler) and a the best real scale;
     `_status`: ok | abstain (some margin < tau) | unknown-event | no-sense (more than `no_sense_share` of the tokens unplaced).
What the v1 decoder reads: says* -> [not] -> event, with fillers word | negated word | entity(head, attr [negated], det). Other wire
types (if, must, may, and, `|` alternatives, `~`) are written by annotators and compiled into training targets, but not decoded (stated).
"""
import json, os
import numpy as np
from . import soplang, world
from .legend import Legend, tokenize

UNKNOWN_KIND = "unknown"


def sigmoid(x): return 1 / (1 + np.exp(-x))
E = lambda ph: np.exp(1j * ph)


class ModelError(Exception): pass


class Routebook:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.cats = [str(c) for c in z["cats"]]; self.cat_id = {c: i for i, c in enumerate(self.cats)}
        self.codes = [str(c) for c in z["codes"]]
        self.path, self.bg0, self.bg = z["path"].astype(float), z["bg0"].astype(float), z["bg"].astype(float)
        self.A, self.L, self.logits = z["A"].astype(float), z["L"].astype(float), z["logits"].astype(float)
        self.rel = z["rel"].astype(float) if "rel" in z.files else None
        self.rel_mask = z["rel_mask"].astype(float) if "rel_mask" in z.files else None
        self.taker_mask = z["taker_mask"].astype(float) if "taker_mask" in z.files else None
        self.D = self.path.shape[1]

    def addresses(self, c):
        """c: category indices (T,) -> addresses (T, D): the sum of the codes' terms before the identity, as V8.states"""
        c = np.asarray(c, dtype=int); T, D = len(c), self.D; terms = np.zeros((T, D), dtype=complex)
        if "path" in self.codes: terms += E(np.cumsum(self.path[c], 0))
        if "bigram" in self.codes: terms += E(np.concatenate([self.bg0[c[:1]], self.bg[c[:-1], c[1:]]], 0))
        if "chainL" in self.codes or "chain" in self.codes:
            g = sigmoid(self.logits)[c]; u = np.log1p(-g + 1e-9); Sx = np.concatenate([[0.0], np.cumsum(u)])
            EA = E(self.A[c[:, None], c[None, :]])                                # EA[q, k] = A[c_q, c_k]
            addr = np.zeros((T, D), dtype=complex)
            for k in range(T):
                p_root = np.exp(Sx[k])
                if k == 0: addr[0] = p_root; continue
                p = g[:k] * np.exp(Sx[k] - Sx[1:k + 1])
                addr[k] = p_root + (p[:, None] * addr[:k] * EA[:k, k]).sum(0)
            if "chainL" in self.codes:
                EL = E(self.L[c[None, :], c[:, None]])                            # EL[k, q] = L[c_q, c_k]
                addrL = np.zeros_like(addr)
                for k in range(T):
                    acc = np.exp(Sx[T] - Sx[k + 1]) * addr[k]
                    if k < T - 1:
                        pL = g[k + 1:] * np.exp(Sx[k + 1:T] - Sx[k + 1])
                        acc = acc + (pL[:, None] * addr[k + 1:] * EL[k, k + 1:]).sum(0)
                    addrL[k] = acc
                addr = g[:, None] * addr + (1 - g[:, None]) * addrL
            terms += addr
        if self.rel is not None and T > 1:
            sel = np.zeros(T); sel[1:] = self.rel_mask[c[:-1]] * self.taker_mask[c[1:]]
            relph = np.zeros((T, D)); relph[1:] = self.rel[c[:-1]]
            terms = terms + sel[:, None] * terms * E(relph)
        return terms


class Reader:
    def __init__(self, model_dir, legend_path=None, force=False):
        self.dir = model_dir
        need = ["config.json", "legend.json", "catalog.json", "routebook.npz"]
        for n in need:
            if not os.path.exists(os.path.join(model_dir, n)): raise ModelError(f"model incomplete: {n} missing in {model_dir}")
        with open(os.path.join(model_dir, "config.json")) as f: self.cfg = json.load(f)
        with open(os.path.join(model_dir, "catalog.json")) as f: self.cat = json.load(f)
        self.legend = Legend(os.path.join(model_dir, "legend.json"))
        if self.legend.sha != self.cfg["legend_sha256"] and not force:
            raise ModelError(f"model/legend.json hash {self.legend.sha} differs from config.json {self.cfg['legend_sha256']}")
        if legend_path is not None:
            other = Legend(legend_path)
            if other.sha != self.legend.sha and not force:
                raise ModelError(f"legend mismatch: {legend_path} is {other.sha}, the model was trained with {self.legend.sha} (--force to inspect)")
        self.rb = Routebook(os.path.join(model_dir, "routebook.npz")); self.D = self.rb.D
        self.st = world.Stars(self.cfg["seed"], self.D, self.cat.get("star_seed", world.GLOBAL_STAR_SEED)); self.comp = world.Compiler(self.legend, self.st)
        self.words = self.cat["words"]                      # word -> {"kind", "lemma"?}
        self.schema = self.cat["schema"]; self.th = self.cfg.get("thresholds", {})
        self.tau = self.cfg["thresholds"].get("tau", 0.05) if "thresholds" in self.cfg else 0.05
        self._cands = {}
        bg = os.path.join(model_dir, "background.npz")
        self.background = dict(np.load(bg)) if os.path.exists(bg) else None
        self.content_kinds = set(self.schema.get("content_kinds", []))

    # ---------------- words
    def kind(self, tok):
        e = self.words.get(tok) or self.words.get(tok.lower())
        if e is not None: return e["kind"]
        return self.legend.kind(tok)
    def lemma(self, tok):
        e = self.words.get(tok) or self.words.get(tok.lower()); return (e or {}).get("lemma", tok)

    def cands(self, kinds, filt=None):
        key = (tuple(sorted(kinds)), None if filt is None else frozenset(filt))
        if key not in self._cands:
            ws = sorted(w for w, e in self.words.items() if e["kind"] in kinds and (filt is None or w in filt))
            M = np.array([self.st.id(w) for w in ws]) / np.sqrt(self.D) if ws else np.zeros((0, self.D), complex)
            self._cands[key] = (ws, M)
        return self._cands[key]
    def verb_cands(self):
        if "__verbs" not in self._cands:
            ws = self.legend.all_verbs(); self._cands["__verbs"] = (ws, np.array([self.st.id(w) for w in ws]) / np.sqrt(self.D))
        return self._cands["__verbs"]

    # ---------------- chart
    def encode(self, tokens):
        kept, cidx, unknown = [], [], []
        for i, t in enumerate(tokens):
            k = self.kind(t)
            if k is None or k == UNKNOWN_KIND or k not in self.rb.cat_id:
                unknown.append(i)
                if UNKNOWN_KIND not in self.rb.cat_id: continue          # spec 6.3: a word with no kind gets the kind `unknown` (placed, never a filler)
                k = UNKNOWN_KIND
            kept.append(i); cidx.append(self.rb.cat_id[k])
        return kept, cidx, unknown
    def chart(self, tokens):
        kept, cidx, unknown = self.encode(tokens)
        if not kept: return np.zeros(self.D, complex), kept, unknown, np.zeros((0, self.D), complex)
        addr = self.rb.addresses(cidx); F = np.array([self.st.id(tokens[i]) for i in kept])
        return (addr * F).sum(0), kept, unknown, addr

    # ---------------- readout
    @staticmethod
    def best(x, cand):
        ws, M = cand
        if not ws: return None, 0.0, 0.0
        sc = np.real(M.conj() @ x)
        if len(ws) == 1: return ws[0], float(sc[0]), float(sc[0])
        j = np.argsort(-sc)[:2]; return ws[j[0]], float(sc[j[0]] - sc[j[1]]), float(sc[j[0]])

    def _filler_top(self, Sn, key, r, filt):
        rs = self.schema["roles"][r]
        if rs["main"] == "entity": return self.best(Sn * key, self.cands(rs["head_kinds"], filt))
        return self.best(Sn * key, self.cands(rs["kinds"], filt))

    def structure_score(self, Sn, K, filt=None):
        tops = [self._filler_top(Sn, K * np.conj(self.st.role(r)), r, filt)[2] for r in self.schema["core_roles"]]
        if self.legend.reader["predicate_role"]: tops.append(self.best(Sn * K * np.conj(self.st.op("pred")), self.verb_cands())[2])
        return float(np.mean(tops)) if tops else 0.0

    def read_role(self, Sn, K, r, core, filt=None):
        rs = self.schema["roles"][r]; key = K * np.conj(self.st.role(r)); out = {"margins": {}}
        if self.legend.reader["occupancy_marker"]:
            mk = self.st.marker() / np.sqrt(self.D); pres = float(np.real(np.sum(Sn * key * np.conj(mk))))
        else: pres = self._filler_top(Sn, key, r, filt)[2]
        ratio = pres / core if core > 1e-9 else 0.0; out["presence"] = ratio
        thr = self.th.get("presence", {}).get(r)
        present = (ratio >= thr) if thr is not None else rs["rate"] >= 0.5
        out["present"] = present
        if not present: return out
        if rs["main"] == "entity":
            h, mg, top = self.best(Sn * key, self.cands(rs["head_kinds"], filt)); out["margins"][r] = mg
            attr = None
            if rs.get("attr_kinds"):
                ka = key * np.conj(self.st.op("attr")); ca = self.cands(rs["attr_kinds"], filt)
                a1 = self.best(Sn * ka, ca); neg = False
                if rs.get("attr_neg"):
                    a2 = self.best(Sn * ka * np.conj(self.st.op("not")), ca)
                    if a2[2] > a1[2]: a1, neg = a2, True
                aratio = a1[2] / top if top > 1e-9 else 0.0; athr = self.th.get("attr_presence", {}).get(r); out["attr_ratio"] = aratio
                if (aratio >= athr) if athr is not None else rs.get("attr_rate", 0) >= 0.5:
                    attr = ["negword" if neg else "word", a1[0]]; out["margins"][r + ".attr"] = a1[1]
            det = None
            if rs.get("det_kinds"):
                d1 = self.best(Sn * key * np.conj(self.st.op("det")), self.cands(rs["det_kinds"], filt))
                dratio = d1[2] / top if top > 1e-9 else 0.0; dthr = self.th.get("det_presence", {}).get(r); out["det_ratio"] = dratio; out["det_best"] = (d1[0], d1[1])
                if (dratio >= dthr) if dthr is not None else rs.get("det_rate", 0) >= 0.5:
                    det = d1[0]; out["margins"][r + ".det"] = d1[1]
            out["filler"] = ["entity", h, attr, det]
        else:
            c = self.cands(rs["kinds"], filt); w1 = self.best(Sn * key, c); neg = False
            if rs.get("negword"):
                w2 = self.best(Sn * key * np.conj(self.st.op("not")), c)
                if w2[2] > w1[2]: w1, neg = w2, True
            out["filler"] = ["negword" if neg else "word", w1[0]]; out["margins"][r] = w1[1]
        return out

    def read(self, S, tokens, filt=None, force=None, force_depth=None):
        """S -> decoded frame (same layout as world.frame_of) with margins and presence scores.
        force = (depth, negation) GIVEN; force_depth = depth GIVEN, negation estimated; neither = both ESTIMATED from S (the tool's default;
        the repository scripts are given the depth — metrics report the three side by side)."""
        nS = np.linalg.norm(S)
        f = {"speakers": [], "neg": False, "event": None, "roles": {}, "shape": "ok", "lemma": None, "margins": {}, "presence": {}}
        if nS < 1e-9: f["shape"] = "unknown"; return f
        Sn = S / nS; sch = self.schema; rep = np.conj(self.st.op("says.scope")); nott = np.conj(self.st.op("not"))
        hyps = [(d, n) for d in range(sch["max_depth"] + 2) for n in ([False, True] if sch["neg_event"] else [False])]
        scores = {h: self.structure_score(Sn, rep ** h[0] * (nott if h[1] else 1), filt) for h in hyps}
        order = sorted(hyps, key=lambda h: -scores[h])
        h = force if force is not None else (max([x for x in hyps if x[0] == force_depth] or [(force_depth, False)], key=lambda x: scores.get(x, -9))
                                             if force_depth is not None else order[0])
        if h not in scores: scores[h] = self.structure_score(Sn, rep ** h[0] * (nott if h[1] else 1), filt)
        f["margins"]["structure"] = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
        d, neg = h; K = rep ** d * (nott if neg else 1); core = scores[h]; f["neg"] = neg; f["depth"] = d; f["core"] = core
        for k in range(d):
            sp, mg, _ = self.best(Sn * rep ** k * np.conj(self.st.op("says.speaker")), self.cands(sch["speaker_kinds"], filt))
            f["speakers"].append(sp); f["margins"][f"speaker.{k}"] = mg
        if self.legend.reader["predicate_role"]:
            lem, mg, _ = self.best(Sn * K * np.conj(self.st.op("pred")), self.verb_cands())
            f["lemma"] = lem; f["event"] = self.legend.event_of_lemma(lem); f["margins"]["event"] = mg
        else:
            f["event"] = sch.get("single_event") or next((self.legend.event_of_lemma(self.lemma(t)) for t in tokens if self.legend.event_of_lemma(self.lemma(t))), None)
        for r in sch["role_order"]:
            o = self.read_role(Sn, K, r, core, filt); f["presence"][r] = o["presence"]
            for part in ("attr", "det"):
                if part + "_ratio" in o: f["presence"][r + "." + part] = o[part + "_ratio"]
            if o.get("present") and o.get("filler") is not None:
                f["roles"][r] = o["filler"]; f["margins"].update(o["margins"])
                if "det_best" in o: f.setdefault("det_best", {})[r] = o["det_best"]
        st = self.legend.reader.get("scope_terms")
        if st: self.read_scope(Sn, K, core, f)
        if f["event"] is None: f["shape"] = "unknown"
        return f

    def read_scope(self, Sn, K, core, f):
        """the quantifier world v2 (C2c, RESULTS 9c-sexies): category-local terms R_wide (x) R_role (x) f_head (wide-scope marker) and
        R_q (x) R_role (x) f_head (plain quantifier) in the event. Readout rule (the legend's `scope_rule`): an R_wide term present -> that slot
        is wide; one R_q term alone -> the only quantifier; two R_q and no R_wide -> both. `_scope` is written only with two or more quantified
        slots (spec 3.3: absent with zero or one quantifier). Term presence = ratio to the structure score >= the threshold set on calibration
        rows (model/config.json thresholds.scope_term)."""
        thr = self.th.get("scope_term"); sw, sq = {}, {}
        for r, fl in f["roles"].items():
            if fl[0] != "entity" or fl[1] is None: continue
            fh = np.conj(self.st.id(fl[1])) / np.sqrt(self.D); base = Sn * K * np.conj(self.st.role(r)) * fh
            sw[r] = float(np.real(np.sum(base * np.conj(self.st.op("wide"))))) / max(core, 1e-9)
            sq[r] = float(np.real(np.sum(base * np.conj(self.st.op("q"))))) / max(core, 1e-9)
            f["presence"][r + ".wide"], f["presence"][r + ".q"] = sw[r], sq[r]
        if thr is None: return
        W = [r for r in sw if sw[r] >= thr]; Q = [r for r in sq if sq[r] >= thr and r not in W]
        # the PRESENCE of a quantifier on a slot is read from its category-local term (as v14's C2c decoder); the determiner WORD from the
        # determiner channel (argmax), reported as its own axis
        for r in sw:
            fl = f["roles"][r]; best = f.get("det_best", {}).get(r)
            if r in W or r in Q:
                if fl[3] is None and best: fl[3] = best[0]; f["margins"][r + ".det"] = best[1]
            elif fl[3] is not None: fl[3] = None; f["margins"].pop(r + ".det", None)
        if len(set(W) | set(Q)) < 2: f["scope"] = None; dec = [abs(v - thr) for v in list(sw.values()) + list(sq.values())]
        elif W: f["scope"] = max(W, key=lambda r: sw[r]); dec = [sw[f["scope"]] - thr] + [abs(sw[r] - thr) for r in sw if r != f["scope"]]
        else: f["scope"] = "both"; dec = [sq[r] - thr for r in Q]
        f["margins"]["scope"] = float(min(dec) * core) if dec else 1.0

    # ---------------- frame -> wires
    def frame_wires(self, f, ids, text=None, sentence=None, extra=None):
        """the decoded frame as SOP wires (top-level first). ids: an iterator of fresh wire ids"""
        wires = []; mg = f.get("margins", {})
        def new(t): w = soplang.Wire(next(ids), t); wires.append(w); return w
        if f["shape"] != "ok":
            w = new("unknown")
            if text is not None: w.set("_text", text)
            if sentence is not None: w.set("_sentence", str(sentence))
            return wires
        top = None; chain = []
        for k, sp in enumerate(f["speakers"]):
            w = new("says"); chain.append(w); top = top or w; w.set("speaker", sp)
        notw = None
        if f["neg"]:
            notw = new("not"); chain.append(notw); top = top or notw
            if "structure" in mg: notw.set("_margin.structure", fmt(mg["structure"]))
        ev = new(f["event"]); top = top or ev
        for a, b in zip(chain, chain[1:] + [ev]): a.set("scope", ("ref", b.id))
        for k, w in enumerate(chain):
            if w.type == "says": w.set("_margin.speaker", fmt(mg.get(f"speaker.{k}", 0)))
            w.set("_depth", str(k))
        later = []
        for r in self.legend.roles:
            if r not in f["roles"]: continue
            fl = f["roles"][r]
            if fl[0] == "word": ev.set(r, fl[1])
            elif fl[0] == "negword":
                nw = soplang.Wire(next(ids), "not"); nw.set("scope", fl[1]); later.append(nw); ev.set(r, ("ref", nw.id))
                if r in mg: nw.set("_margin.scope", fmt(mg[r]))
            elif fl[0] == "entity":
                en = soplang.Wire(next(ids), "entity"); later.append(en); ev.set(r, ("ref", en.id)); en.set("head", fl[1])
                if fl[2] is not None:
                    if fl[2][0] == "negword":
                        nw = soplang.Wire(next(ids), "not"); nw.set("scope", fl[2][1]); later.append(nw); en.set("attr", ("ref", nw.id))
                        if f"{r}.attr" in mg: nw.set("_margin.scope", fmt(mg[f"{r}.attr"]))
                    else: en.set("attr", fl[2][1])
                if fl[3] is not None: en.set("det", fl[3])
                if r in mg: en.set("_margin.head", fmt(mg[r]))
                for part in ("attr", "det"):
                    if f"{r}.{part}" in mg: en.set(f"_margin.{part}", fmt(mg[f"{r}.{part}"]))
        if f.get("scope"): ev.set("_scope", f["scope"])
        if f.get("lemma"): ev.set("_verb", f["lemma"])
        ev.set("_mood", self.mood(text))
        for r in self.legend.roles:
            if r in mg: ev.set(f"_margin.{r}", fmt(mg[r]))
        if "event" in mg: ev.set("_margin.event", fmt(mg["event"]))
        if "scope" in mg: ev.set("_margin.scope", fmt(mg["scope"]))
        evm = [mg[k] for k in mg if k in self.legend.roles or k == "event"]
        if evm: ev.set("_margin", fmt(min(evm)))
        wires += later
        # top-level lines first: _text, _sentence
        if text is not None or sentence is not None:
            head = []
            if text is not None: head.append(soplang.Line("_text", text.replace("\n", " ")))
            if sentence is not None: head.append(soplang.Line("_sentence", str(sentence)))
            top.lines = head + top.lines
        return wires

    def no_fit_residual(self):
        """dispatcher only (section 9): a residual at or above this value means the reader explains almost nothing of the chart. A STATED
        constant (config thresholds.no_fit_residual, default 0.9), not fitted: fitting it needs out-of-world sentences, which a world folder
        does not have; unseen depths of a world reach 0.6-0.8 (P0), random text 0.99."""
        return float(self.cfg.get("thresholds", {}).get("no_fit_residual", 0.9))

    def wire_lines(self, wires, status):
        """spec 5: every wire written by formalize has _margin, _status, _empty, _depth. _depth = distance from the top-level wire;
        _empty = the keys of the wire's type not filled ('-' when none); _margin = the minimum of the wire's own _margin.* lines (the
        top-level wire: over the whole sentence)."""
        by = {w.id: w for w in wires}; depth = {wires[0].id: 0}; q = [wires[0].id]
        while q:
            x = q.pop(0)
            for r in by[x].refs():
                if r in by and r not in depth: depth[r] = depth[x] + 1; q.append(r)
        for w in wires:
            keys = self.legend.keys_for(w.type) if w.type != "unknown" else set()
            order = [k for k in (self.legend.roles if w.type in self.legend.events else sorted(keys)) if k in keys]
            empty = [k for k in order if w.get(k) is None]
            if w.type in self.legend.events: empty = [k for k in self.schema["role_order"] if w.get(k) is None] + [k for k in empty if k not in self.schema["role_order"]]
            w.set("_empty", ",".join(empty) if empty else "-")
            if w.get("_margin") is None:
                ms = [float(l.text) for l in w.lines if l.key.startswith("_margin.")]
                w.set("_margin", fmt(min(ms)) if ms else "none")
            if w.get("_status") is None: w.set("_status", status)
            w.set("_depth", str(depth.get(w.id, 0)))

    def mood(self, text):
        """_mood comes from the normalizer, not the reader (stated): the legend's only mood, or '?' -> question, else the first mood"""
        ms = self.legend.moods
        if len(ms) == 1: return ms[0]
        if text and text.strip().endswith("?") and "question" in ms: return "question"
        return ms[0]

    def residual(self, S, wires, tokens):
        by = {w.id: w for w in wires}; rec = self.comp.wire(wires[0], by, tokens)
        nS, nr = np.linalg.norm(S), np.linalg.norm(rec)
        if nS < 1e-9 or nr < 1e-9: return 1.0
        Sn, rn = S / nS, rec / nr; a = max(0.0, float(np.real(np.vdot(rn, Sn))))
        return float(np.linalg.norm(Sn - a * rn))

    def placed_words(self, f):
        ws = set(f["speakers"]) | ({f["lemma"]} if f.get("lemma") else set())
        for fl in f["roles"].values(): ws |= set(world.filler_words(fl))
        return ws

    def formalize(self, text, sentence, ids, filt=None, force=None, force_depth=None):
        """one sentence -> (wires, frame, info). force / force_depth: see read() (evaluation only; the CLI estimates both)"""
        tokens = tokenize(text); S, kept, unknown, addr = self.chart(tokens)
        f = self.read(S, tokens, filt, force, force_depth) if kept else {"shape": "unknown", "roles": {}, "speakers": [], "margins": {}, "neg": False, "lemma": None}
        placed = self.placed_words(f) if f["shape"] == "ok" else set()
        unplaced = [i for i in unknown]
        for i in kept:
            t = tokens[i]; k = self.kind(t)
            if k in self.content_kinds and t not in placed and self.lemma(t) not in placed: unplaced.append(i)
        unplaced = sorted(set(unplaced))
        status = "ok"
        has_verb = any(self.legend.event_of_lemma(self.lemma(t)) for t in tokens) or not self.legend.reader["predicate_role"]
        if tokens and len(unplaced) > self.cfg.get("thresholds", {}).get("no_sense_share", 0.5) * len(tokens): status = "no-sense"
        elif f["shape"] != "ok" or not has_verb: status = "unknown-event"
        if status != "ok": f = dict(f, shape="unknown")
        wires = self.frame_wires(f, ids, text, sentence)
        top = wires[0]
        mins = [v for v in f.get("margins", {}).values()]
        if f["shape"] == "ok":
            top.set("_margin.structure", fmt(f["margins"]["structure"]))
            top.set("_margin", fmt(min(mins)) if mins else "0")
            if min(mins) < self.tau: status = "abstain"
        else:
            top.set("_margin", "0")
        top.set("_status", status)
        top.lines.insert(next((i + 1 for i, l in enumerate(top.lines) if l.key == "_sentence"), 0), soplang.line("_world", self.cfg.get("world", "?")))
        res = self.residual(S, wires, tokens) if f["shape"] == "ok" else 1.0
        top.set("_residual", fmt(res))
        if unplaced: top.set("_unplaced", ",".join(tokens[i] for i in unplaced))
        self.wire_lines(wires, status)
        info = {"tokens": tokens, "S": S, "kept": kept, "unplaced": unplaced, "addr": addr, "residual": res, "status": status}
        return wires, f, info


def fmt(x): return f"{x:.3f}"


def idgen(prefix="w", start=1):
    i = start
    while True: yield f"{prefix}{i}"; i += 1


def formalize_text(reader, lines, filt=None):
    """sentences -> Doc. `reader` is one Reader, or a list of Readers: the DISPATCHER BY FIT (spec v2 section 0) — every reader reads the
    sentence, the formalization with the smallest _residual wins (a status other than no-sense / unknown-event first), and its _world names
    the winner. Directives: one `!legend <world> sha256:...` per world (check.py)."""
    readers = reader if isinstance(reader, list) else [reader]
    doc = soplang.Doc(); doc.directives = ([("world", readers[0].cfg.get("world", "?"), None)] if len(readers) == 1 else [("dispatch", "by-fit (smallest _residual)", None)])
    doc.directives += [("legend", f"{r.cfg.get('world', '?')} {r.legend.sha}", None) for r in readers]
    doc.directives.append(("generated", "sop 0.1, " + ", ".join(f"model {r.cfg.get('world', '?')}/model ({r.cfg.get('trained', '?')[:10]})" for r in readers), None))
    k = 1; frames = []
    for n, text in enumerate(lines):
        cands = []
        for r in readers:
            wires, f, info = r.formalize(text, n, idgen(start=k), filt)
            rel = info["residual"] / max(1e-9, r.no_fit_residual())          # relative residual: >= 1.0 = this reader explains almost nothing
            mm = [v for v in f.get("margins", {}).values()]; mean_margin = float(np.mean(mm)) if mm else 0.0
            cands.append(((info["status"] in ("no-sense", "unknown-event"), rel, -mean_margin), r, wires, f, info))
        cands.sort(key=lambda c: c[0]); (key, r, wires, f, info) = cands[0]
        if len(readers) > 1:
            top = wires[0]
            if key[0] or key[1] >= 1.0:     # the best reader cannot place it or explains almost nothing of it: no known world
                w = soplang.Wire(f"w{k}", "unknown"); w.set("_text", text); w.set("_sentence", str(n)); w.set("_world", "none")
                w.set("_unplaced", ",".join(info["tokens"])) if info["tokens"] else None
                w.set("_margin", "0"); w.set("_status", "no-sense"); w.set("_empty", "-"); w.set("_depth", "0"); wires = [w]
            else:
                gap = (cands[1][0][1] - key[1]) if len(cands) > 1 else 1.0
                i = next(j for j, l in enumerate(top.lines) if l.key == "_world"); top.lines.insert(i + 1, soplang.line("_world_margin", fmt(gap)))
        doc.wires += wires; frames.append((f, info))
        k = max([k] + [int(w.id[1:]) + 1 for w in wires if w.id[1:].isdigit()])
    return doc, frames


def readers_of_root(root, force=False):
    """every <root>/<world>/model that loads (section 9 / spec 5 `--worlds`)"""
    out = []
    for name in sorted(os.listdir(root)):
        md = os.path.join(root, name, "model")
        if os.path.exists(os.path.join(md, "config.json")): out.append(Reader(md, None, force))
    return out
