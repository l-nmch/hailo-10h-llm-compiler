#!/usr/bin/env python3
"""Register a compiled HEF into hailo-ollama's model store.

hailo-ollama discovers models through two locations:

1. a **content-addressed blob store**:
   ``<blob-store>/sha256_<sha256-of-hef>`` — the HEF bytes themselves;
2. a **manifest directory**:
   ``<models-root>/manifests/<family>/<tag>/manifest.json`` — pointing at
   the blob hash plus display metadata.

This script copies the HEF into the blob store and writes the manifest,
after which the model behaves exactly like an official one:

    hailo-ollama run <family>:<tag>

Note: the server only scans manifests at startup, so restart
(`hailo-ollama serve`) after registering.

Run on the device host (the machine with the Hailo-10H and hailo-ollama
installed). See ../docs/device-setup.md for installing that stack.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_blob_store() -> Path:
    return Path.home() / ".local" / "share" / "hailo-ollama" / "models" / "blob"


def default_models_root() -> Path:
    return Path("/usr/share/hailo-ollama/models")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a HEF into hailo-ollama's content-addressed model store."
    )
    parser.add_argument("--hef", required=True, type=Path, help="path to the compiled .hef")
    parser.add_argument("--family", required=True, help="model family name (e.g. tinystories25m)")
    parser.add_argument("--tag", required=True, help="model tag within the family")
    parser.add_argument("--parameter-size", default="25M", help="display size (e.g. 25M)")
    parser.add_argument("--quantization-level", default="INT4", help="display quantization level")
    parser.add_argument("--license", default=None, help="license string stored in the manifest")
    parser.add_argument("--parent-model", default="", help="display parent model")
    parser.add_argument(
        "--blob-store", type=Path, default=default_blob_store(),
        help=f"blob store location (default {default_blob_store()})",
    )
    parser.add_argument(
        "--models-root", type=Path, default=default_models_root(),
        help=f"manifests root (default {default_models_root()})",
    )
    args = parser.parse_args()

    hef = args.hef.resolve()
    if not hef.is_file():
        raise SystemExit(f"HEF not found: {hef}")

    digest = sha256_of(hef)
    blob_dir = args.blob_store.resolve()
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"sha256_{digest}"
    shutil.copy2(hef, blob_path)
    print(f"blob  -> {blob_path}")

    manifest_dir = args.models_root.resolve() / "manifests" / args.family / args.tag
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "hef_h10h": digest,
        "license": args.license,
        "details": {
            "parent_model": args.parent_model,
            "format": "hef",
            "family": args.family,
            "families": [args.family],
            "parameter_size": args.parameter_size,
            "quantization_level": args.quantization_level,
        },
    }
    manifest_path = manifest_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)
    print(f"manifest -> {manifest_path}")
    print(json.dumps(manifest, indent=2))

    print(
        "\nnext steps:\n"
        "  1. restart the server (manifests are scanned at startup only):\n"
        "       pkill -f 'hailo-ollama serve' && OLLAMA_HOST=0.0.0.0:8000 hailo-ollama serve &\n"
        f"  2. generate:\n       curl -s -H 'Content-Type: application/json' "
        f"http://localhost:8000/api/generate \\\n"
        f"         -d '{{\"model\":\"{args.family}:{args.tag}\",\"prompt\":\"Once upon a time\",\"stream\":false}}'"
    )


if __name__ == "__main__":
    main()
