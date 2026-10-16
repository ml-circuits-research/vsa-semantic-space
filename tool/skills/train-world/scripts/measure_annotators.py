"""
measure_annotators.py — the ONE measurement that sets the `annotate` default (indications section 1; spec 6b, not received):
on the constructed commands world (gold frames exist), annotate N generated sentences (default 500) with the SMALL model (annotate default,
claude-haiku-5-5) and with the LARGE model (evolve default, claude-opus-5-5); FULL-frame agreement of each with the gold
(world.frame_of canonical frames, the same comparison as `sop train`'s FULL column). Rule: if small >= large - 0.02, the small default stands;
otherwise the default becomes the large class and the number is written in the skill's README. Both numbers are reported whatever they are.
When no API is reachable it prints "API unavailable" with the reason and exits 3 (nothing is measured, nothing is written as a result).
Run: python3 measure_annotators.py [--n 500] [--out <dir>]
"""
import argparse, json, os, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); TOOL = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, TOOL); sys.path.insert(0, HERE)
from sop import soplang, world
from sop.legend import Legend
from llm import llm_call, LLMUnavailable, DEFAULT_MODELS
import loop


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--out", default=None); a = ap.parse_args()
    out = a.out or tempfile.mkdtemp(prefix="sop_measure_")
    import export_worlds as ex
    wd = ex.export_commands(out, a.n, 1, 1, name="commands_measure")
    legend = Legend(os.path.join(wd, "legend.json")); gold = soplang.load(os.path.join(wd, "examples", "train.sop")); by = gold.by_id()
    tops = gold.top_level()[:a.n]
    try: llm_call("annotate", "Reply with OK.", wd, max_tokens=16)
    except LLMUnavailable as e:
        print(f"measure_annotators: API unavailable — {e}. Nothing measured; the annotate default stays {DEFAULT_MODELS['annotate']} "
              "as a stated default, NOT as a measured one (label: PLANNED)."); return 3
    res = {}
    for label, model in (("small", DEFAULT_MODELS["annotate"]), ("large", DEFAULT_MODELS["evolve"])):
        wj = json.load(open(os.path.join(wd, "world.json"))); wj["models"]["annotate"] = model; json.dump(wj, open(os.path.join(wd, "world.json"), "w"))
        agree = 0; ids = loop.soplang_ids()
        for t in tops:
            s = t.value("_text"); g = world.frame_of(t, by, legend)
            text = llm_call("annotate", loop.annotation_prompt_header(legend) + "\nSentence: " + s + "\nReply with the SOP wires only.", wd)
            ws = loop.parse_llm_wires(text, s, legend, ids, model, wd); f = world.frame_of(ws[0], {w.id: w for w in ws}, legend)
            agree += world.canon(f, False) == world.canon(g, False)
        res[label] = {"model": model, "n": len(tops), "FULL_agreement_with_gold": agree / len(tops)}
    small, large = res["small"]["FULL_agreement_with_gold"], res["large"]["FULL_agreement_with_gold"]
    res["decision"] = "small default stands" if small >= large - 0.02 else "default becomes the large class (write it in the skill README)"
    print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(out, "annotator_measurement.json"), "w"), indent=1); return 0


if __name__ == "__main__":
    sys.exit(main())
