"""
llm.py — the ONE place the loop calls a language model: llm_call(role, prompt, world_dir).

The model is named by ROLE and read from <world>/world.json `models` (never hard-coded in the loop):
    pipeline, annotate, second_look -> small class by default (claude-haiku-5-5)
    evolve (legend proposal and evolution) -> large class by default (claude-opus-5-5)
Nothing in the loop requires a model above the Opus class; a step that cannot be done at that class is reported as a limit.
Escalation (indications section 1): when the second-look disagreement rate exceeds world.json `second_look_escalate` (default 0.05),
`annotate` and `second_look` are moved to the evolve (large) model for that world — escalate(world_dir) rewrites world.json and says so.
Calls go through the official Anthropic Python SDK (`anthropic`). When the SDK or credentials are missing, LLMUnavailable is raised with the
reason, and every caller reports "API unavailable" instead of failing (this machine has no API key, 2026-10-16).
"""
import json, os

DEFAULT_MODELS = {"pipeline": "claude-haiku-5-5", "annotate": "claude-haiku-5-5", "second_look": "claude-haiku-5-5", "evolve": "claude-opus-5-5"}


class LLMUnavailable(Exception): pass


def world_json(world_dir):
    p = os.path.join(world_dir, "world.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def model_for(role, world_dir):
    return {**DEFAULT_MODELS, **world_json(world_dir).get("models", {})}.get(role, DEFAULT_MODELS["pipeline"])


def llm_call(role, prompt, world_dir, max_tokens=16000, system=None):
    """one request -> the text of the answer. Raises LLMUnavailable when no API can be reached."""
    model = model_for(role, world_dir)
    try:
        import anthropic
    except ImportError:
        raise LLMUnavailable("API unavailable: the anthropic SDK is not installed (pip install anthropic)")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.path.isdir(os.path.expanduser("~/.config/anthropic"))):
        raise LLMUnavailable("API unavailable: no credentials (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile)")
    try:
        client = anthropic.Anthropic()
        kw = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if system: kw["system"] = system
        resp = client.messages.create(**kw)
    except anthropic.APIConnectionError as e:
        raise LLMUnavailable(f"API unavailable: connection error ({e})")
    except anthropic.AuthenticationError as e:
        raise LLMUnavailable(f"API unavailable: authentication failed ({e})")
    if resp.stop_reason == "refusal":
        raise LLMUnavailable(f"model {model} declined the request (stop_reason refusal)")
    return "".join(b.text for b in resp.content if b.type == "text")


def escalate(world_dir, rate):
    """move annotate / second_look to the large class for this world (the evolve model); returns the change made"""
    p = os.path.join(world_dir, "world.json"); wj = world_json(world_dir); models = {**DEFAULT_MODELS, **wj.get("models", {})}
    large = models["evolve"]; before = {k: models[k] for k in ("annotate", "second_look")}
    models["annotate"] = models["second_look"] = large; wj["models"] = models
    wj.setdefault("escalations", []).append({"second_look_rate": rate, "threshold": wj.get("second_look_escalate", 0.05), "from": before, "to": large})
    tmp = p + ".tmp"; json.dump(wj, open(tmp, "w"), indent=1); os.replace(tmp, p)
    return {"from": before, "to": large}
