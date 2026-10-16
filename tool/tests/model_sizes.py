"""
model_sizes.py — the size breakdown of model/ folders (author C4): bytes per file and per route-book array, with the array shapes.
The route book holds, per kind c (K kinds): path[c] (D phases), bigram from the start bg0[c] (D); per ordered kind PAIR (c1, c2): bigram
bg[c1, c2], chain-right A[c1, c2] and chain-left L[c1, c2] relative keys (D phases each); gates (K); beacon-relative rel[c] (D) when the
world gives emitters. So the size is ~ 3 K^2 D x 4 bytes (float32 phases): the per-category-pair relative keys dominate.
Run: python3 tool/tests/model_sizes.py <model dir> [...] [--json out.json]
"""
import argparse, json, os, sys
import numpy as np


def sizes(md):
    out = {"model": md, "files": {f: os.path.getsize(os.path.join(md, f)) for f in sorted(os.listdir(md)) if os.path.isfile(os.path.join(md, f))}}
    z = np.load(os.path.join(md, "routebook.npz"))
    arr = {k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype), "bytes": int(z[k].nbytes)} for k in z.files}
    K, D = len(z["cats"]), z["path"].shape[1]; tot = sum(v["bytes"] for v in arr.values())
    pair = sum(arr[k]["bytes"] for k in ("bg", "A", "L"))
    out.update(K=K, D=D, arrays=arr, routebook_uncompressed_bytes=tot, pair_keys_bytes=pair, pair_keys_share=pair / tot,
               formula_3K2D4=3 * K * K * D * 4)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="+"); ap.add_argument("--json"); a = ap.parse_args()
    res = [sizes(m) for m in a.models]
    for r in res:
        print(f"{r['model']}: K={r['K']} kinds, D={r['D']}; routebook.npz on disk {r['files']['routebook.npz'] / 1e6:.2f} MB, arrays {r['routebook_uncompressed_bytes'] / 1e6:.2f} MB "
              f"uncompressed; pair keys (bg, A, L: 3 x K^2 x D float32) {r['pair_keys_bytes'] / 1e6:.2f} MB = {r['pair_keys_share']:.1%}")
        for k, v in sorted(r["arrays"].items(), key=lambda x: -x[1]["bytes"]): print(f"   {k:10s} {str(v['shape']):18s} {v['dtype']:8s} {v['bytes']:>10d} B")
        print("   other files: " + ", ".join(f"{f} {b} B" for f, b in r["files"].items() if f != "routebook.npz"))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
