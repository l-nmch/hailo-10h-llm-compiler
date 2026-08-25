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

    base_group_line = ""
    if args.include_base_scope:
        base_group_line = f"{scope} = network_group([{scope}])"
    compile_script = f"""
performance_param(compiler_optimization_level=0)
{scope}__prefill = network_group([{scope}__prefill])
{scope}__tbt = network_group([{scope}__tbt])
{base_group_line}
"""
    print("=== compile script ===")
    print(compile_script.strip())

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(P.har_convfixed))
    runner.load_model_script(compile_script)
    hef_bytes = runner.compile()

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
