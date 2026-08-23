#!/usr/bin/env python3
"""Single genai.LLM generation attempt — designed to run as a short-lived
subprocess of genai_generate.py, not interactively.

Prints exactly one JSON status line on stdout as its final line:

    RESULT_JSON:{"ok": true, "text": "...", ...}
    RESULT_JSON:{"ok": false, "error": "...", "error_type": "..."}

Why a subprocess at all? Two reasons documented in docs/findings/sdk-behavior-notes.md:

1. A known HailoRT host-side issue can surface as an EINTR/ioctl failure on
   HAILO_PCI_EP_ACCEPT inside a long-lived process; a fresh process per
   attempt sidesteps it entirely without touching firmware or drivers.
2. Any crash or hang in the native stack is contained by the parent's hard
   timeout instead of wedging a session.
"""
import json
import sys


def main() -> int:
    hef_path, prompt, max_generated_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])

    import hailo_platform as hpf
    from hailo_platform.genai import LLM

    llm = None
    vd = None
    try:
        vd_params = hpf.VDevice.create_params().group_id("SHARED")
        vd = hpf.VDevice(vd_params)
        llm = LLM(vd, hef_path)
        llm.clear_context()

        messages = [{"role": "user", "content": prompt}]
        out = llm.generate_all(messages, max_generated_tokens=max_generated_tokens)

        # `out` carries the full generated text plus stop reason/usage.
        text = out["text"] if isinstance(out, dict) else str(out)
        print("RESULT_JSON:" + json.dumps({"ok": True, "text": text}))
        return 0
    except Exception as exc:  # noqa: BLE001 — every failure must reach the parent as JSON
        print("RESULT_JSON:" + json.dumps({
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }))
        return 1
    finally:
        # Best-effort teardown; a device left with stale context breaks the
        # NEXT attempt otherwise.
        for release in (
            lambda: llm.clear_context(),
            lambda: llm.release(),
        ):
            try:
                release()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
