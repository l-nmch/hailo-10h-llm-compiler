#!/usr/bin/env python3
"""Step 5 — conv input-order repair (quantized HAR → convfixed HAR).

``set_kv_cache_global_params`` duplicates the graph into ``__prefill`` and
``__tbt`` scopes. With the historical recipe (default ew_add_fusing) this
misaligned `input` / `input_shapes` on residual-fused convs. Since the final
recipe disables ew_add_fusing (step 4), **zero** misaligned convs are found —
this step is kept as a cheap structural safety net: whenever a multi-input
conv's `input` list does not match its declared `input_shapes` widths, it
greedily reorders candidates by feature width.

Operates directly on the HAR tarball (plain JSON surgery), no SDK involved.

Usage:
    python s5_fix_convolutions.py
"""
import argparse
import glob
import json
import os
import tarfile
import tempfile

import config


def fix_duplicated_conv_inputs(har_in: str, har_out: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(har_in) as t:
            t.extractall(d)
        hn_path = [
            p for p in glob.glob(os.path.join(d, "*.hn"))
            if not p.endswith((".fp.hn", ".native.hn"))
        ][0]
        with open(hn_path) as f:
            hn = json.load(f)
        layers = hn["layers"]

        def feat(name):
            layer = layers.get(name)
            if layer and layer.get("output_shapes"):
                return layer["output_shapes"][0][-1]
            return None

        fixed = 0
        for layer in layers.values():
            if layer.get("type") != "conv":
                continue
            inputs, shapes = layer.get("input"), layer.get("input_shapes")
            if not inputs or not shapes or len(inputs) < 2:
                continue
            want = [s[-1] for s in shapes]
            if [feat(n) for n in inputs] == want:
                continue
            pool, new_inputs = list(inputs), []
            ok = True
            for width in want:
                candidate = next((n for n in pool if feat(n) == width), None)
                if candidate is None:
                    ok = False
                    break
                new_inputs.append(candidate)
                pool.remove(candidate)
            if ok and new_inputs != inputs:
                print(f"  {layer.get('name', '?')}: {inputs} -> {new_inputs}")
                layer["input"] = new_inputs
                fixed += 1

        with open(hn_path, "w") as f:
            json.dump(hn, f)
        with tarfile.open(har_out, "w") as t:
            for f in os.listdir(d):
                t.add(os.path.join(d, f), arcname=f)
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    config.load()  # picks up run_config.json written by step 1, if any
    P = config.paths()

    print("scanning for misaligned conv inputs...")
    n_fixed = fix_duplicated_conv_inputs(str(P.har_quantized), str(P.har_convfixed))
    print(f"convs repaired: {n_fixed}")
    print(f"convfixed HAR saved -> {P.har_convfixed}")
    print("[OK] step 5 complete")


if __name__ == "__main__":
    main()
