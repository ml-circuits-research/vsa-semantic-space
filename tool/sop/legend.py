"""
legend.py — the world's closed vocabulary (spec 1): roles, events with their synonym table, operators, moods, kinds (word categories)
from a lexicon, value patterns; plus the `reader` block (an extension of the informal schema, see README): how a frame is written as a
target state (predicate role on/off, occupancy marker on/off, determiners that add no term).

Lexicon (kinds.tsv, next to legend.json): one word per line, `word<TAB>kind[<TAB>lemma]`; '#' comments. The kind is the word's category
(rule 20: categories are GIVEN, never learned); the lemma maps a surface form to the legend's verb (gives -> give) or a negative affix form
to its positive (unsaved -> saved).
"""
import hashlib, json, os, re

OPERATOR_KEYS = {"not": {"scope"}, "must": {"scope"}, "may": {"scope"}, "if": {"condition", "consequence"}, "says": {"speaker", "scope"},
                 "and": {"items"}}
ENTITY_KEYS = {"head", "attr", "quantity", "unit", "det"}
VERDICT_KEYS = {"left", "right"}
KNOWN_OPERATORS = set(OPERATOR_KEYS)
READER_DEFAULTS = {"predicate_role": True, "occupancy_marker": True, "definite_dets": ["the"]}
PUNCT = re.compile(r"([.,;:!?](?=\s|$))")


class LegendError(Exception): pass


def sha256_file(path):
    with open(path, "rb") as f: return "sha256:" + hashlib.sha256(f.read()).hexdigest()


class Legend:
    def __init__(self, path):
        self.path = os.path.abspath(path); self.dir = os.path.dirname(self.path)
        with open(path, encoding="utf-8") as f: d = json.load(f)
        self.raw = d; self.sha = sha256_file(path); self.version = d.get("version")
        self.roles = [r["id"] for r in d.get("roles", [])]; self.questions = {r["id"]: r.get("question", "") for r in d.get("roles", [])}
        self.events = {e["id"]: list(e.get("verbs", [])) for e in d.get("events", [])}
        self.operators = list(d.get("operators", [])); self.moods = list(d.get("moods", ["statement"]))
        self.reader = {**READER_DEFAULTS, **d.get("reader", {})}
        self.values = [(v["kind"], re.compile("^(?:" + v["pattern"] + ")$")) for v in d.get("values", [])]
        self.errors = self._validate(); self.warnings = []
        self.verb_event = {v: e for e, vs in self.events.items() for v in vs}
        kinds = d.get("kinds", {}); self.kinds_source = kinds.get("source", "lexicon")
        self.lexicon, self.lemma = {}, {}
        if self.kinds_source == "lexicon":
            lp = os.path.join(self.dir, kinds.get("lexicon", "kinds.tsv"))
            if os.path.exists(lp): self._load_lexicon(lp)
            else: self.warnings.append(f"lexicon {lp} not found (kinds unknown; sop check does not need them, training does)")
        else:
            self.errors.append("kinds.source 'tagger' is not implemented in v1 (use a lexicon)")   # TODO(section 6 missing): tagger kinds

    def _validate(self):
        e = []
        for r in self.roles:
            if r.startswith("_"): e.append(f"role {r!r}: a legend may not define roles starting with '_'")
            if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", r): e.append(f"role {r!r}: bad id")
        if len(set(self.roles)) != len(self.roles): e.append("duplicate role ids")
        for ev in self.events:
            if ev in KNOWN_OPERATORS or ev in ("entity", "unknown", "verdict"): e.append(f"event id {ev!r} collides with a reserved wire type")
        for op in self.operators:
            if op not in KNOWN_OPERATORS: e.append(f"operator {op!r} is not one of {sorted(KNOWN_OPERATORS)}")
        return e

    def _load_lexicon(self, path):
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"): continue
                parts = line.split("\t")
                if len(parts) < 2: self.errors.append(f"{path}:{n}: expected word<TAB>kind[<TAB>lemma]"); continue
                self.lexicon[parts[0]] = parts[1]
                if len(parts) > 2 and parts[2]: self.lemma[parts[0]] = parts[2]

    # -------- words --------
    def kind(self, tok):
        if tok in self.lexicon: return self.lexicon[tok]
        if tok.lower() in self.lexicon: return self.lexicon[tok.lower()]
        for k, pat in self.values:
            if pat.match(tok): return k
        return None
    def lemma_of(self, tok): return self.lemma.get(tok, self.lemma.get(tok.lower(), tok))
    def event_of_lemma(self, lemma): return self.verb_event.get(lemma)
    def all_verbs(self): return sorted(self.verb_event)
    def wire_types(self):
        return set(self.events) | {"entity", "unknown"} | set(self.operators)
    def keys_for(self, wtype):
        if wtype in self.events: return set(self.roles)
        if wtype == "entity": return ENTITY_KEYS
        if wtype in OPERATOR_KEYS: return OPERATOR_KEYS[wtype]
        if wtype == "verdict": return VERDICT_KEYS
        return set()


def tokenize(text):
    """whitespace tokens with punctuation split off (the constructed worlds write ' .' and ' ,' already)"""
    return PUNCT.sub(r" \1 ", text).split()
