#!/usr/bin/env python3
"""Step 6 — quantized HAR → HEF.

Compiles the final HAR into a HEF containing the two network groups the
genai runtime expects, named exactly ``<scope>__prefill`` and
``<scope>__tbt`` via explicit network_group declarations.

With ``--include-base-scope``, a third network group is declared over the
unduplicated base scope and the result is written to ``model_basescope.hef``
instead of ``model.hef``. That variant exists solely for out-of-runtime
diagnostics: ``runtime/diagnostics/generate_base_scope.py`` drives it for
cache-free greedy generation (full-prefix recomputation). WARNING: a
three-group HEF tends to break hailo-ollama / genai.LLM() — three networks
to manage on-chip instead of two degrade runtime reliability. The official
compile recipe (extracted from qwen2_1.5b_instruct.q.har) declares only the
two groups; never deploy a base-scope HEF through the genai stack.

Includes the SDKPaths patch: outside an official Hailo release layout the
SDK's singleton paths object reports a non-release build directory that may
not exist; forcing release mode and redirecting its temp build dir avoids
spurious compile-time failures. This is a host-environment workaround only —
it changes nothing about the compiled artifact.

KV-cache memory layout: every cache input and output of both groups is
routed through an explicit tf_rgb <-> hailo_rgb format conversion, as the
official LLM compile scripts do. HailoRT manages each cache as a ring of
one-token entries shared by ``__prefill`` and ``__tbt``, which assumes a
token-major layout in memory; left to itself the compiler keeps the core's
native layout (and a different one per group), so every ring rotation
mixes tokens and multi-token generation degrades. See
docs/findings/tbt-cache-read.md.

compiler_optimization_level=0 keeps compile time bounded (~5-8 min with the
monolithic lm_head); raise it if you want the compiler to spend longer
searching for better placements.

Automatic spatial reshapes are disabled by default
(``allocator_param(enable_auto_spatial_reshapes=False)``). On deep models the
compiler otherwise runs its "Spatial Reshapes Flow" — a single-threaded,
iterative search for spatial-reshape insertion points — for roughly an hour
per network group before the first allocation bucket, and on every model
compiled here it concluded with no reshape inserted (the decision recorded in
the compiled HAR's ``.auto.alls``). ``--auto-spatial-reshapes`` restores the
compiler default. See docs/findings/compile-time-spatial-reshape-search.md.

Usage:
    python s6_compile_hef.py [--include-base-scope] [--auto-spatial-reshapes]
"""
import argparse
import tempfile

import config  # must precede numpy imports — sets NPY_PROMOTION_STATE et al.


def patch_sdk_paths() -> None:
    from hailo_sdk_common.paths_manager.paths import SDKPaths

    p = SDKPaths()
    if not p.is_release:
        p._is_release = True
        p._build_dir = tempfile.mkdtemp(prefix=type(p).HAILO_TEMP_DIR_PREFIX)
    print(f"SDKPaths patched: is_release={p.is_release}")


def cache_layout_conversions(runner, groups) -> list[str]:
    """Model-script lines forcing a token-major KV-cache layout in `groups`.

    One ``tf_rgb_to_hailo_rgb`` conversion after every cache input and one
    ``hailo_rgb_to_tf_rgb`` before every cache output (layers with
    ``io_type == "cache"``), so the prefill writes and the tbt reads the
    shared ring buffer one token per entry.
    """
    layers = runner.get_hn_dict()["layers"]
    lines = []
    for name, layer in sorted(layers.items()):
        group, short = name.split("/", 1)
        if group not in groups or layer.get("io_type") != "cache":
            continue
        if layer["type"] == "input_layer":
            succ = layer["output"][0]
            lines.append(
                f"{group}/tf_rgb_to_hailo_rgb_from_{short} = "
                f"format_conversion({name}, {succ}, tf_rgb_to_hailo_rgb)"
            )
        elif layer["type"] == "output_layer":
            pred = layer["input"][0]
            lines.append(
                f"{group}/hailo_rgb_to_tf_rgb_to_{short} = "
                f"format_conversion({pred}, {name}, hailo_rgb_to_tf_rgb)"
            )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    parser.add_argument("--include-base-scope", action="store_true",
                        help="also expose the unduplicated base scope as a "
                             "third network group (writes model_basescope.hef; "
                             "out-of-runtime diagnostics only — tends to break "
                             "hailo-ollama / genai.LLM())")
    parser.add_argument("--auto-spatial-reshapes", action="store_true",
                        help="keep the compiler's automatic spatial-reshape "
                             "search (DFC default; adds ~1 h per network group "
                             "on deep models, see module docstring)")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    config.load()  # picks up run_config.json written by step 1, if any
    P = config.paths()

    scope = config.NET_SCOPE
    patch_sdk_paths()

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(P.har_convfixed))

    # lm_head sharding (large VOCAB, e.g. Qwen3's ~152K tokens) is handled
    # upstream in s3_surgery_and_resources.py, pre-quantization, as N
    # genuinely independent output convs -- not here. An earlier attempt
    # used the native `defuse()` model-script command at this stage
    # instead; abandoned after it caused two severe compile-time failures
    # of its own at scale (subcluster starvation, then a deterministic
    # multi-context topology error) -- both traced to defuse's mandatory
    # auto-generated on-chip concat. See
    # docs/findings/large-vocab-lm-head-sharding.md and
    # docs/findings/large-body-multicontext-topology.md.

    base_group_line = ""
    if args.include_base_scope:
        base_group_line = f"{scope} = network_group([{scope}])"
    cache_lines = cache_layout_conversions(runner, (f"{scope}__prefill", f"{scope}__tbt"))
    assert cache_lines, "no KV-cache layers found — was step 4 run with set_kv_cache_global_params?"
    compile_script = "\n".join([
        "performance_param(compiler_optimization_level=0)",
        "" if args.auto_spatial_reshapes else "allocator_param(enable_auto_spatial_reshapes=False)",
        f"{scope}__prefill = network_group([{scope}__prefill])",
        f"{scope}__tbt = network_group([{scope}__tbt])",
        base_group_line,
        *cache_lines,
    ])
    print("=== compile script ===")
    print("\n".join(compile_script.strip().splitlines()[:4]))
    print(f"    ... + {len(cache_lines)} KV-cache layout conversions")

    runner.load_model_script(compile_script)
    hef_bytes = runner.compile()

    # The compiled HAR embeds the .auto.alls (the exact partition/placement
    # decisions the compiler made) alongside the quantized weights — extract
    # it with `hailo har extract <this> --auto-model-script-path x.alls` and
    # reuse via `hailo compiler <quantized.har> --model-script x.alls` for a
    # much faster recompile than re-solving placement from scratch. See the
    # Dataflow Compiler User Guide's "Automatic Model Script" section.
    runner.save_har(str(P.har_compiled))
    print(f"compiled HAR (embeds .auto.alls) -> {P.har_compiled}")

    hef_path = P.hef
    if args.include_base_scope:
        hef_path = P.hef.parent / "model_basescope.hef"
    with open(hef_path, "wb") as f:
        f.write(hef_bytes)
    print(f"HEF written -> {hef_path} ({len(hef_bytes) / 1024 / 1024:.2f} MiB)")
    if args.include_base_scope:
        print("base-scope HEF: diagnostics only (generate_base_scope.py) -- "
              "do NOT serve through hailo-ollama / genai.LLM()")
    else:
        print("[OK] step 6 complete")
        print("next: deploy to the device and register with hailo-ollama "
              "(see runtime/register_hailo_ollama.py)")


if __name__ == "__main__":
    main()
