from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


HIP_SAMPLE = r"""
#include <iostream>
int main() {
  std::cout << "hip-probe-ok" << std::endl;
  return 0;
}
"""


def _find_hipcc(hip_path: Path) -> str:
    candidates = [
        hip_path / "bin" / "hipcc.bat",
        hip_path / "bin" / "hipcc.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return "hipcc"


def _find_rocm_device_lib_path(hip_path: Path) -> Path | None:
    candidates = [
        hip_path / "lib" / "llvm" / "amdgcn" / "bitcode",
        hip_path / "amdgcn" / "bitcode",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _run(cmd: list[str], timeout_sec: int, env: dict[str, str] | None = None) -> tuple[int, float, str, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    elapsed = time.perf_counter() - started
    return proc.returncode, elapsed, proc.stdout, proc.stderr


def _classify(stderr_tail: str) -> str:
    text = stderr_tail.lower()
    if "hipcc" in text and "not found" in text:
        return "toolchain_missing"
    if "cannot find" in text and "clang" in text:
        return "toolchain_missing"
    if "cannot find rocm device library" in text:
        return "toolchain_missing"
    if "fatal error" in text or "error:" in text:
        return "hip_runtime_conflict"
    return "hip_runtime_conflict"


def run_probe(
    *,
    hip_path: Path,
    profile_id: str,
    iterations: int,
    timeout_sec: int,
    opt_level: str,
    arch: str,
) -> dict[str, Any]:
    hipcc = _find_hipcc(hip_path)
    rocm_device_lib_path = _find_rocm_device_lib_path(hip_path)
    compile_times: list[float] = []
    run_times: list[float] = []
    success_count = 0
    stderr_tail = ""

    env = os.environ.copy()
    env["HIP_PATH"] = str(hip_path)
    env["ROCM_PATH"] = str(hip_path)
    env["PATH"] = f"{hip_path / 'bin'};{env.get('PATH', '')}"

    with tempfile.TemporaryDirectory(prefix="hip_probe_") as tmp:
        tmp_path = Path(tmp)
        src_path = tmp_path / "probe.cpp"
        src_path.write_text(HIP_SAMPLE, encoding="utf-8")
        exe_path = tmp_path / "probe.exe"

        for _ in range(iterations):
            compile_cmd = [
                hipcc,
                str(src_path),
                f"-{opt_level}",
                f"--offload-arch={arch}",
                f"--rocm-path={hip_path}",
                "-o",
                str(exe_path),
            ]
            if rocm_device_lib_path is not None:
                compile_cmd.insert(
                    -2,
                    f"--rocm-device-lib-path={rocm_device_lib_path}",
                )
            code, c_sec, _c_out, c_err = _run(compile_cmd, timeout_sec=timeout_sec, env=env)
            compile_times.append(c_sec)
            if c_err:
                stderr_tail = c_err[-400:]
            if code != 0 or (not exe_path.exists()):
                run_times.append(0.0)
                continue

            run_cmd = [str(exe_path)]
            r_code, r_sec, r_out, r_err = _run(run_cmd, timeout_sec=timeout_sec, env=env)
            run_times.append(r_sec * 1000.0)
            if r_err:
                stderr_tail = r_err[-400:]
            if r_code == 0 and "hip-probe-ok" in r_out:
                success_count += 1

    pass_rate = success_count / float(iterations)
    compile_sec = float(statistics.fmean(compile_times)) if compile_times else 0.0
    run_ms = float(statistics.fmean(run_times)) if run_times else 0.0
    score = max(0.0, min(100.0, 100.0 * pass_rate - compile_sec * 3.0 - run_ms / 50.0))
    blocker = None if success_count > 0 else _classify(stderr_tail)

    return {
        "backend": "hip",
        "profile_id": profile_id,
        "iterations": iterations,
        "compile_sec": compile_sec,
        "run_ms": run_ms,
        "pass_rate": pass_rate,
        "stability_score": score,
        "score": score,
        "failure_signature": blocker,
        "stderr_tail": stderr_tail,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="HIP runtime probe")
    parser.add_argument("--hip-path", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--opt-level", default="O2", choices=["O0", "O1", "O2", "O3"])
    parser.add_argument("--arch", default="gfx1103")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        payload = run_probe(
            hip_path=Path(args.hip_path).resolve(),
            profile_id=args.profile_id,
            iterations=max(1, args.iterations),
            timeout_sec=max(1, args.timeout_sec),
            opt_level=args.opt_level,
            arch=args.arch,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime path
        payload = {
            "backend": "hip",
            "profile_id": args.profile_id,
            "compile_sec": 0.0,
            "run_ms": 0.0,
            "pass_rate": 0.0,
            "stability_score": 0.0,
            "score": 0.0,
            "failure_signature": "hip_runtime_conflict",
            "stderr_tail": str(exc),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[hip-probe] failed: {exc}")
        return 1

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[hip-probe] wrote {out_path}")
    return 0 if payload.get("pass_rate", 0.0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
