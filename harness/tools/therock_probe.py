from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def _run_command(
    cmd: list[str],
    timeout_sec: int,
    env: dict[str, str] | None = None,
) -> tuple[int, float, str, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
        env=env,
    )
    elapsed = time.perf_counter() - started
    return proc.returncode, elapsed, proc.stdout, proc.stderr


def _resolve_rocm_sdk(therock_home: Path) -> str:
    candidates = [
        therock_home / "bin" / "rocm-sdk.exe",
        therock_home / "bin" / "rocm-sdk",
        therock_home / "Scripts" / "rocm-sdk.exe",
        therock_home / "scripts" / "rocm-sdk.exe",
        therock_home / "Lib" / "site-packages" / "_rocm_sdk_core" / "bin" / "rocm-sdk.exe",
        therock_home / "Lib" / "site-packages" / "_rocm_sdk_core" / "bin" / "rocm-sdk",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "rocm-sdk"


def _classify(stderr_tail: str) -> str:
    text = stderr_tail.lower()
    if "not found" in text or "is not recognized" in text:
        return "toolchain_missing"
    if "permission" in text or "access is denied" in text:
        return "path_env_misconfigured"
    if "record" in text and "permissionerror" in text:
        return "path_env_misconfigured"
    return "therock_source_build_blocker"


def run_probe(
    *,
    therock_home: Path,
    profile_id: str,
    iterations: int,
    timeout_sec: int,
    include_test: bool,
) -> dict[str, Any]:
    rocm_sdk = _resolve_rocm_sdk(therock_home)
    cmd_sets: list[list[str]] = [[rocm_sdk, "version"], [rocm_sdk, "targets"]]
    if include_test:
        cmd_sets.append([rocm_sdk, "test"])

    durations: list[float] = []
    success_runs = 0
    test_pass_count = 0
    stderr_tail = ""
    # Ensure conda-style ROCm command shims are visible when a conda env root is provided.
    env = os.environ.copy()
    path_parts = [
        str(therock_home / "Scripts"),
        str(therock_home / "scripts"),
        str(therock_home / "bin"),
        str(therock_home / "Lib" / "site-packages" / "_rocm_sdk_core" / "bin"),
    ]
    env["PATH"] = ";".join([*path_parts, env.get("PATH", "")])

    for _ in range(iterations):
        all_ok = True
        run_duration = 0.0
        for cmd in cmd_sets:
            code, elapsed, _out, err = _run_command(cmd, timeout_sec=timeout_sec, env=env)
            run_duration += elapsed
            if code != 0:
                all_ok = False
            if err:
                stderr_tail = err[-400:]
            if include_test and len(cmd) > 1 and cmd[1] == "test" and code == 0:
                test_pass_count += 1
        durations.append(run_duration)
        if all_ok:
            success_runs += 1

    pass_rate = success_runs / float(iterations)
    mean_duration = float(statistics.fmean(durations)) if durations else 0.0
    score = max(0.0, min(100.0, 100.0 * pass_rate - (mean_duration * 5.0)))
    blocker = None if success_runs > 0 else _classify(stderr_tail)

    return {
        "backend": "therock",
        "profile_id": profile_id,
        "iterations": iterations,
        "build_or_probe_sec": mean_duration,
        "test_pass_count": test_pass_count,
        "pass_rate": pass_rate,
        "stability_score": score,
        "score": score,
        "failure_signature": blocker,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TheRock SDK probe")
    parser.add_argument("--therock-home", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--include-test", default="1", help="0|1")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    include_test = args.include_test.strip() in {"1", "true", "yes"}
    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = run_probe(
            therock_home=Path(args.therock_home).resolve(),
            profile_id=args.profile_id,
            iterations=max(1, args.iterations),
            timeout_sec=max(1, args.timeout_sec),
            include_test=include_test,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime path
        payload = {
            "backend": "therock",
            "profile_id": args.profile_id,
            "build_or_probe_sec": 0.0,
            "test_pass_count": 0,
            "pass_rate": 0.0,
            "stability_score": 0.0,
            "score": 0.0,
            "failure_signature": "therock_source_build_blocker",
            "stderr_tail": str(exc),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[therock-probe] failed: {exc}")
        return 1

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[therock-probe] wrote {out_path}")
    return 0 if payload.get("pass_rate", 0.0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
