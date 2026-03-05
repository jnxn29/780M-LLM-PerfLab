from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _to_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {raw}")


def _run_once(
    zluda_with: Path,
    use_nvml: bool,
    timeout_sec: int,
) -> tuple[bool, float, str, str]:
    python_code = (
        "import ctypes; "
        "ctypes.WinDLL('nvcuda.dll'); "
        "print('zluda-probe-ok')"
    )
    cmd: list[str] = [str(zluda_with), sys.executable]
    if use_nvml:
        cmd.extend(["--nvml", str(zluda_with.parent / "nvml.dll")])
    # Separate wrapper flags from target executable flags.
    cmd.extend(["--", "-c", python_code])

    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ok = proc.returncode == 0 and "zluda-probe-ok" in proc.stdout
    return ok, elapsed_ms, proc.stdout.strip(), proc.stderr.strip()


def _classify_blocker(stderr_tail: str) -> str | None:
    text = stderr_tail.lower()
    if "timed out" in text:
        return "zluda_loader_blocker"
    if "not found" in text or "no such file" in text:
        return "toolchain_missing"
    if "nvcuda" in text or "dll" in text:
        return "zluda_loader_blocker"
    if not text:
        return None
    return "zluda_loader_blocker"


def _resolve_zluda_with(zluda_home: Path) -> tuple[Path, Path]:
    direct = zluda_home / "zluda_with.exe"
    nested = zluda_home / "zluda" / "zluda_with.exe"
    if direct.exists():
        return direct, zluda_home
    if nested.exists():
        return nested, nested.parent
    raise FileNotFoundError(
        f"zluda_with.exe not found: {direct} or {nested}"
    )


def run_probe(
    *,
    zluda_home: Path,
    profile_id: str,
    iterations: int,
    timeout_sec: int,
    use_nvml: bool,
) -> dict[str, Any]:
    zluda_with, resolved_home = _resolve_zluda_with(zluda_home)

    latencies: list[float] = []
    success_count = 0
    stderr_tail = ""
    stdout_tail = ""
    for _ in range(iterations):
        ok, elapsed_ms, out, err = _run_once(
            zluda_with=zluda_with,
            use_nvml=use_nvml,
            timeout_sec=timeout_sec,
        )
        latencies.append(elapsed_ms)
        if ok:
            success_count += 1
        if out:
            stdout_tail = out[-400:]
        if err:
            stderr_tail = err[-400:]

    pass_rate = success_count / float(iterations)
    error_rate = 1.0 - pass_rate
    latency_mean = float(statistics.fmean(latencies)) if latencies else 0.0
    # Stability-focused compatibility score in [0, 100].
    score = max(0.0, min(100.0, 100.0 * pass_rate - (latency_mean / 50.0)))
    blocker = None if success_count > 0 else _classify_blocker(stderr_tail)

    return {
        "backend": "zluda",
        "profile_id": profile_id,
        "iterations": iterations,
        "resolved_zluda_with": str(zluda_with),
        "resolved_zluda_home": str(resolved_home),
        "load_success": success_count > 0,
        "pass_rate": pass_rate,
        "error_rate": error_rate,
        "probe_latency_ms": latency_mean,
        "stability_score": score,
        "score": score,
        "blocker_signature": blocker,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ZLUDA compatibility probe")
    parser.add_argument("--zluda-home", required=True, help="ZLUDA install directory")
    parser.add_argument("--profile-id", required=True, help="Profile id for bookkeeping")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=20)
    parser.add_argument("--use-nvml", default="0", help="0|1")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = run_probe(
            zluda_home=Path(args.zluda_home).resolve(),
            profile_id=args.profile_id,
            iterations=max(1, args.iterations),
            timeout_sec=max(1, args.timeout_sec),
            use_nvml=_to_bool(args.use_nvml),
        )
    except Exception as exc:  # pragma: no cover - defensive runtime path
        payload = {
            "backend": "zluda",
            "profile_id": args.profile_id,
            "load_success": False,
            "pass_rate": 0.0,
            "error_rate": 1.0,
            "probe_latency_ms": 0.0,
            "stability_score": 0.0,
            "score": 0.0,
            "blocker_signature": "zluda_loader_blocker",
            "stderr_tail": str(exc),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[zluda-probe] failed: {exc}", file=sys.stderr)
        return 1

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[zluda-probe] wrote {out_path}")
    return 0 if payload.get("load_success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
