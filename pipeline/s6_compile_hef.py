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

compiler_optimization_level=0 keeps compile time bounded (~5-8 min with the
monolithic lm_head); raise it if you want the compiler to spend longer
searching for better placements.

Usage:
    python s6_compile_hef.py [--include-base-scope]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    parser.add_argument("--include-base-scope", action="store_true",
                        help="also expose the unduplicated base scope as a "
                             "third network group (writes model_basescope.hef; "
                             "out-of-runtime diagnostics only — tends to break "
                             "hailo-ollama / genai.LLM())")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    config.load()  # picks up run_config.json written by step 1, if any
    P = config.paths()

    scope = config.NET_SCOPE
    patch_sdk_paths()

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(P.har_convfixed))

    defuse_lines = []
    if config.LM_HEAD_SHARDS > 1:
        # DFC's placer rejects a single lm_head matmul this wide once VOCAB
        # is large (e.g. Qwen3's ~152K tokens) -- see
        # docs/findings/large-vocab-lm-head-sharding.md. Fix: split it into
        # N physical layers via the native `defuse` model-script command,
        # one per network-group scope present in this HAR (base/__prefill/
        # __tbt each carry their own copy of the lm_head layer). The
        # compiler auto-reconcatenates the N pieces into one logical
        # output, so nothing downstream (runtime scripts, HN output count)
        # needs to know sharding happened.
        hn = runner.get_hn_model()
        active_scopes = [f"{scope}__prefill", f"{scope}__tbt"]
        if args.include_base_scope:
            active_scopes.append(scope)
        for sc in active_scopes:
            # Each scope has several output layers (the logits head plus one
            # pair of cache-write outputs per decoder layer) -- the lm_head
            # is the one whose original name is "logits" (s1_export_onnx.py's
            # single ONNX output name).
            out_layers = [
                n for n in hn.get_output_layers()
                if n.name.startswith(f"{sc}/") and any(
                    orig.startswith("logits") for orig in n.original_names
                )
            ]
            assert len(out_layers) == 1, f"expected exactly one logits output layer in {sc}, got {out_layers}"
            lm_head_layer = out_layers[0].inputs[0]
            n = config.LM_HEAD_SHARDS
            names = ", ".join([f"d{i}" for i in range(n)] + ["dc"])
            defuse_lines.append(f"{names} = defuse({lm_head_layer}, {n})")
        print(f"lm_head sharding: VOCAB={config.VOCAB} -> {config.LM_HEAD_SHARDS} shards per scope")

    base_group_line = ""
    if args.include_base_scope:
        base_group_line = f"{scope} = network_group([{scope}])"
    compile_script = "\n".join([
        "performance_param(compiler_optimization_level=0)",
        *defuse_lines,
        f"{scope}__prefill = network_group([{scope}__prefill])",
        f"{scope}__tbt = network_group([{scope}__tbt])",
        base_group_line,
    ])
    print("=== compile script ===")
    print(compile_script.strip())

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
