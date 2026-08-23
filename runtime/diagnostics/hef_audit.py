#!/usr/bin/env python3
"""Static structural audit of a HEF against the genai.LLM() contract.

Emits one JSON report per HEF (stdout or --out-dir). Checks, without any
device attached:

- network group names/shapes/formats (the ``__prefill``/``__tbt`` contract);
- the embedded external resources (embeddings.bin, tokenizer.json,
  rope_theta_data.bin, hailo-config.json) with their decoded summaries;
- ``hailortcli parse-hef`` compatibility/compiler metadata.

Requires an environment where ``hailo_platform`` is importable and
``hailortcli`` is on PATH (i.e. a HailoRT installation).

Memory safety: ``hailo_platform.HEF._get_external_resources()`` only works
on in-memory buffers — reading multi-GB HEFs entirely into RAM can OOM
small hosts. This tool therefore parses the HEF's own binary header +
protobuf metadata directly (format from the public MIT-licensed HailoRT
source: hef_internal.hpp / hef.proto) to locate each external resource's
absolute offset+size, then seek+reads just that range.

Usage:
    python hef_audit.py <hef1> [hef2 ...] [--out-dir DIR]
"""
import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import hailo_platform as ph

# --- Minimal HEF-header + protobuf parsing ---------------------------------
# Reference: hailort/libhailort/src/hef/hef_internal.hpp and
# hailort/libhailort/hef.proto in the public hailort source (v5.3.0).

_DISTINCT_SIZE_BY_VERSION = {0: 20, 1: 16, 2: 32, 3: 44}  # per hef__header_distinct_t union

# A resource's declared "offset" is NOT absolute: it is relative to an
# "offset zero point" placed right after header + proto (+ padding for v3).
# See Hef::Impl::get_offset_zero_point() in hef.cpp:
#   v1/v2: header_size + proto_size
#   v3:    header_size + proto_size + padding_size
# Verified empirically against a known-good buffer-mode read.


def _read_varint(buf: bytes, pos: int):
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _parse_protobuf_fields(buf: bytes):
    """Minimal protobuf wire-format decoder: yields (field_no, wire_type, value)."""
    pos = 0
    out = []
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field_no, wire_type = tag >> 3, tag & 7
        if wire_type == 0:  # varint
            val, pos = _read_varint(buf, pos)
        elif wire_type == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            val = buf[pos:pos + length]
            pos += length
        elif wire_type == 5:  # 32-bit
            val = buf[pos:pos + 4]
            pos += 4
        elif wire_type == 1:  # 64-bit
            val = buf[pos:pos + 8]
            pos += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} at byte {pos}")
        out.append((field_no, wire_type, val))
    return out


def read_external_resources_index(path: Path) -> list:
    """[{name, offset, size, xxhash}, ...] parsed from the HEF header +
    ProtoHEFHef.external_resources (field 7), without loading the file body.
    ``offset`` is absolute within the file."""
    with open(path, "rb") as f:
        magic, version, proto_size = struct.unpack(">III", f.read(12))
        if magic != 0x01484546:
            raise ValueError(f"unexpected HEF magic {hex(magic)} (expected 0x01484546)")
        header_size = 12 + _DISTINCT_SIZE_BY_VERSION[version]
        f.seek(header_size)
        proto = f.read(proto_size)

        zero_point = 0
        if version == 3:
            f.seek(12)  # distinct union starts right after the common header
            distinct = f.read(_DISTINCT_SIZE_BY_VERSION[3])
            hef_padding_size = struct.unpack(">I", distinct[16:20])[0]
            zero_point = header_size + proto_size + hef_padding_size
        elif version in (1, 2):
            zero_point = header_size + proto_size

    resources = []
    for field_no, wire_type, val in _parse_protobuf_fields(proto):
        if field_no != 7 or wire_type != 2:
            continue
        entry = {}
        for sub_no, _sub_wt, sub_val in _parse_protobuf_fields(val):
            if sub_no == 1:
                entry["offset"] = sub_val + zero_point
            elif sub_no == 2:
                entry["size"] = sub_val
            elif sub_no == 3:
                entry["name"] = sub_val.decode("utf-8")
            elif sub_no == 4:
                entry["xxhash"] = sub_val
        resources.append(entry)
    return resources


def read_resource_bytes(path: Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vstream_summary(vs) -> dict:
    return {
        "name": vs.name,
        "network_name": vs.network_name,
        "shape": list(vs.shape),
        "format_type": str(vs.format.type),
        "format_order": str(vs.format.order),
        "direction": str(vs.direction),
    }


def decode_hailo_config(raw: bytes) -> dict:
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"parse_error": str(exc)}
    keys_of_interest = [
        "model_name", "stop_token_id", "eos_token_id",
        "default_generation_params", "chat_template",
        "pre_process_params", "input_layers_names_suffixes",
    ]
    out = {k: cfg[k] for k in keys_of_interest if k in cfg}
    out["_all_top_level_keys"] = sorted(cfg.keys())
    return out


def decode_rope_theta(raw: bytes) -> dict:
    n = len(raw) // 4
    vals = struct.unpack(f"<{n}f", raw[: n * 4]) if n else ()
    return {
        "byte_length": len(raw),
        "n_float32": n,
        "first_8_values": list(vals[:8]),
        "last_8_values": list(vals[-8:]) if n >= 8 else list(vals),
    }


def run_parse_hef(path: Path) -> dict:
    try:
        out = subprocess.run(  # noqa: S603
            ["hailortcli", "parse-hef", str(path)],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    result = {"raw": out}
    compat = re.search(r"HEF Compatible for:\s*(.+)", out)
    compiler = re.search(r"HEF Compiler Version:\s*(.+)", out)
    if compat:
        result["compatible_for"] = [c.strip() for c in compat.group(1).split(",")]
    if compiler:
        result["compiler_version"] = compiler.group(1).strip()

    groups = []
    for block in re.split(r"(?=Network group name:)", out)[1:]:
        name_m = re.search(r"Network group name:\s*([^\s,]+)", block)
        mode_m = re.search(r"(Single|Multi) Context - Number of contexts:\s*(\d+)", block)
        groups.append({
            "name": name_m.group(1) if name_m else None,
            "context_mode": mode_m.group(1) if mode_m else None,
            "num_contexts": int(mode_m.group(2)) if mode_m else None,
        })
    result["network_groups_from_cli"] = groups
    return result


def audit_hef(path: Path) -> dict:
    report = {
        "file": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_of(path),
    }

    # Structural queries use path-mode HEF — HailoRT handles it without
    # loading the whole file.
    hef = ph.HEF(str(path))
    group_names = hef.get_network_group_names()
    report["network_group_names"] = group_names

    report["network_groups"] = {}
    for gname in group_names:
        inputs = [vstream_summary(v) for v in hef.get_input_vstream_infos(gname)]
        outputs = [vstream_summary(v) for v in hef.get_output_vstream_infos(gname)]
        report["network_groups"][gname] = {
            "networks_names": hef.get_networks_names(gname),
            "input_vstreams": inputs,
            "output_vstreams": outputs,
            "num_inputs": len(inputs),
            "num_outputs": len(outputs),
        }

    # External resources: bounded seek+read per resource (at most the size of
    # one resource — e.g. embeddings.bin — never the whole HEF).
    resources_index = read_external_resources_index(path)
    report["external_resources"] = {"names": [r["name"] for r in resources_index], "detail": {}}
    for r in resources_index:
        raw = read_resource_bytes(path, r["offset"], r["size"])
        entry = {"size_bytes": r["size"], "file_offset": r["offset"], "xxhash": r["xxhash"]}
        name = r["name"]
        if name == "hailo-config.json":
            entry["decoded"] = decode_hailo_config(raw)
        elif name == "rope_theta_data.bin":
            entry["decoded"] = decode_rope_theta(raw)
        elif name == "tokenizer.json":
            try:
                tok = json.loads(raw.decode("utf-8"))
                entry["decoded"] = {
                    "top_level_keys": sorted(tok.keys()),
                    "vocab_size_guess": len(tok.get("model", {}).get("vocab", {})) or None,
                }
            except Exception as exc:  # noqa: BLE001
                entry["decode_error"] = str(exc)
        elif name == "embeddings.bin":
            entry["decoded"] = {"note": "size only (uint16 table), content not decoded"}
        report["external_resources"]["detail"][name] = entry
        del raw

    report["parse_hef_cli"] = run_parse_hef(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hefs", nargs="+")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for hef_path in args.hefs:
        path = Path(hef_path)
        print(f"[hef_audit] auditing {path}", file=sys.stderr)
        report = audit_hef(path)
        text = json.dumps(report, indent=2, default=str)
        if out_dir:
            out_path = out_dir / (path.stem + ".json")
            out_path.write_text(text)
            print(f"[hef_audit] wrote {out_path}", file=sys.stderr)
        else:
            print(text)


if __name__ == "__main__":
    main()
