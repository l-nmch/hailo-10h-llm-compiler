#!/usr/bin/env python3
"""Robust genai.LLM generation with subprocess isolation and retries.

Wraps genai_worker.py: each generation attempt runs in a fresh, isolated
subprocess with a hard parent-side timeout, and failed attempts are retried
with exponential backoff. This absorbs a known host-side HailoRT failure
mode (EINTR/ioctl on HAILO_PCI_EP_ACCEPT in long-lived processes) without
any driver or firmware modification.

Usage (on the device host):
    python genai_generate.py --hef model.hef --prompt "Once upon a time" \
        [--max-generated-tokens 64] [--max-retries 8] [--attempt-timeout 60]
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

WORKER_PATH = Path(__file__).resolve().parent / "genai_worker.py"


def run_attempt(hef: Path, prompt: str, max_generated_tokens: int, attempt_timeout_s: float) -> dict:
    """Run one worker subprocess; returns the worker's RESULT_JSON dict."""
    cmd = [sys.executable, str(WORKER_PATH), str(hef), prompt, str(max_generated_tokens)]
    proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
        cmd,
        capture_output=True,
        text=True,
        timeout=attempt_timeout_s,
    )
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
    if result is None:
        # Worker died before reporting — surface its stderr as the error.
        tail = "\n".join(proc.stderr.splitlines()[-10:])
        result = {"ok": False, "error": f"no RESULT_JSON from worker; stderr tail:\n{tail}",
                  "error_type": "WorkerCrashed"}
    return result


def robust_generate(
    hef: Path,
    prompt: str,
    max_generated_tokens: int = 64,
    max_retries: int = 8,
    base_delay_s: float = 3.0,
    max_delay_s: float = 20.0,
    attempt_timeout_s: float = 60.0,
) -> dict:
    """Attempt generation up to max_retries times with exponential backoff.

    Returns the first successful result, or the last failure after retries
    are exhausted.
    """
    last_result = {"ok": False, "error": "not attempted", "error_type": "Internal"}
    for attempt in range(1, max_retries + 1):
        print(f"[attempt {attempt}/{max_retries}] ...", flush=True)
        try:
            last_result = run_attempt(hef, prompt, max_generated_tokens, attempt_timeout_s)
        except subprocess.TimeoutExpired:
            last_result = {
                "ok": False,
                "error": f"attempt exceeded {attempt_timeout_s:.0f}s (killed)",
                "error_type": "AttemptTimeout",
            }
        if last_result.get("ok"):
            return last_result

        delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
        delay *= 0.5 + random.random()  # jitter to avoid thundering re-open
        print(f"  failed ({last_result.get('error_type')}): "
              f"{str(last_result.get('error'))[:160]}")
        if attempt < max_retries:
            print(f"  retrying in {delay:.1f}s", flush=True)
            time.sleep(delay)
    return last_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hef", required=True, type=Path, help="HEF to serve from")
    parser.add_argument("--prompt", required=True, help="user prompt text")
    parser.add_argument("--max-generated-tokens", type=int, default=64)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--base-delay", type=float, default=3.0, help="first retry delay (s)")
    parser.add_argument("--max-delay", type=float, default=20.0, help="retry delay cap (s)")
    parser.add_argument("--attempt-timeout", type=float, default=60.0, help="hard timeout per attempt (s)")
    args = parser.parse_args()

    result = robust_generate(
        args.hef.resolve(),
        args.prompt,
        max_generated_tokens=args.max_generated_tokens,
        max_retries=args.max_retries,
        base_delay_s=args.base_delay,
        max_delay_s=args.max_delay,
        attempt_timeout_s=args.attempt_timeout,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
