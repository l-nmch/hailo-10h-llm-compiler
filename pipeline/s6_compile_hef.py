#!/usr/bin/env python3
"""Step 6 — quantized HAR → HEF.

Compiles the final HAR into a HEF containing the two network groups the
genai runtime expects, named exactly ``<scope>__prefill`` and
``<scope>__tbt`` via explicit network_group declarations.

Includes the SDKPaths patch: outside an official Hailo release layout the
SDK's singleton paths object reports a non-release build directory that may
not exist; forcing release mode and redirecting its temp build dir avoids
spurious compile-time failures. This is a host-environment workaround only —
it changes nothing about the compiled artifact.

compiler_optimization_level=0 keeps compile time bounded (~5-8 min with the
monolithic lm_head); raise it if you want the compiler to spend longer
searching for better placements.

Usage:
    python s6_compile_hef.py
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
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    P = config.paths()

    scope = config.NET_SCOPE
    patch_sdk_paths()

    compile_script = f"""
performance_param(compiler_optimization_level=0)
{scope}__prefill = network_group([{scope}__prefill])
{scope}__tbt = network_group([{scope}__tbt])
"""
    print("=== compile script ===")
    print(compile_script.strip())

    from hailo_sdk_client import ClientRunner

    runner = ClientRunner(har=str(P.har_convfixed))
    runner.load_model_script(compile_script)
    hef_bytes = runner.compile()

    with open(P.hef, "wb") as f:
        f.write(hef_bytes)
    print(f"HEF written -> {P.hef} ({len(hef_bytes) / 1024 / 1024:.2f} MiB)")
    print("[OK] step 6 complete")
    print("next: deploy to the device and register with hailo-ollama "
          "(see runtime/register_hailo_ollama.py)")


if __name__ == "__main__":
    main()
