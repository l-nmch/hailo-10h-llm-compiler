#!/usr/bin/env python3
"""Hot-patch the hailo-config.json embedded in an already-compiled HEF,
without recompiling the graph (weights/CCWs are never touched).

Uses the SDK's official HefWrapper API (protobuf + offset-based sections) —
no reverse engineering of the binary format. New config bytes are appended
to the additional-info section and the resource descriptor is repointed at
them; the old bytes remain as harmless dead space that serialization
preserves verbatim.

Typical use: fixing the pre_process_params key name or sizes in a HEF that
is otherwise final (see docs/findings/kv-cache-config-key.md).

Usage:
    python hotpatch_hailo_config.py <hef_in> <hef_out> \
        [--prefill-size 16] [--cache-size 24]
"""
import argparse
import json

import xxhash
from hailo_sdk_client.allocator.hef_wrapper import HefWrapper


def find_config_resource(hef: HefWrapper):
    matches = [
        r for r in hef.external_resources
        if "hailo-config" in r.name or "hailo_config" in r.name
    ]
    assert len(matches) <= 1, f"multiple hailo-config resources found: {[r.name for r in matches]}"
    return matches[0] if matches else None


def read_config_bytes(hef: HefWrapper, resource) -> bytes:
    relative = resource.offset - len(hef.ccws)
    return hef._additional_info[relative:relative + resource.size]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hef_in")
    parser.add_argument("hef_out")
    parser.add_argument("--prefill-size", type=int, default=16)
    parser.add_argument("--cache-size", type=int, default=24)
    args = parser.parse_args()

    hef = HefWrapper.from_hef_path(args.hef_in)
    print(f"[i] loaded {args.hef_in} (version {hef.version})")

    target = None
    print(f"[i] {len(list(hef.external_resources))} external resources:")
    for r in hef.external_resources:
        print(f"    name={r.name!r} offset={r.offset} size={r.size} signature={r.signature}")
        if "hailo-config" in r.name or "hailo_config" in r.name:
            target = r
    if target is None:
        raise SystemExit("[X] no hailo-config.json resource found")

    old_config = json.loads(read_config_bytes(hef, target))
    old_pp = old_config.get("pre_process_params", {})
    print(f"[i] current pre_process_params: {old_pp}")

    new_config = dict(old_config)
    new_config["pre_process_params"] = dict(old_pp)
    # The key the LLM server actually reads is ..._count (not ..._size).
    new_config["pre_process_params"].pop("prefill_input_tokens_size", None)
    new_config["pre_process_params"]["prefill_input_tokens_count"] = args.prefill_size
    new_config["pre_process_params"]["kv_cache_size"] = args.cache_size
    new_bytes = json.dumps(new_config, indent=2).encode("utf-8")
    print(f"[i] patched pre_process_params: {new_config['pre_process_params']} "
          f"({len(new_bytes)} bytes vs {target.size} before)")

    # Append the new bytes; repoint the descriptor; refresh the signature.
    new_additional_info = hef._additional_info + new_bytes
    target.offset = len(hef.ccws) + len(hef._additional_info)
    target.size = len(new_bytes)
    target.signature = xxhash.xxh3_64(new_bytes).intdigest()
    hef._additional_info = new_additional_info

    hef.save(args.hef_out)
    print(f"[OK] saved -> {args.hef_out}")

    # Round-trip verification.
    check = HefWrapper.from_hef_path(args.hef_out)
    r2 = find_config_resource(check)
    verified = json.loads(read_config_bytes(check, r2))
    assert verified["pre_process_params"]["prefill_input_tokens_count"] == args.prefill_size
    assert verified["pre_process_params"]["kv_cache_size"] == args.cache_size
    expected_sig = xxhash.xxh3_64(read_config_bytes(check, r2)).intdigest()
    assert r2.signature == expected_sig, "signature mismatch after round-trip"
    print(f"[OK] round-trip verified (signature {r2.signature})")


if __name__ == "__main__":
    main()
