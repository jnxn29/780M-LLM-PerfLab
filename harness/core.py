from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def split_args(raw: str | None) -> list[str]:
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    return shlex.split(text, posix=(os.name != "nt"))


def commandline_from_argv() -> str:
    tokens = [str(Path("harness") / "benchctl.py"), *sys.argv[1:]]
    return "python " + " ".join(shlex.quote(token) for token in tokens)


def now_utc_iso(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def current_git_commit() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode == 0:
        return proc.stdout.strip() or "unknown"
    return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def powershell_json(command: str) -> Any | None:
    ps_command = (
        "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new(); "
        + command
    )
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_command],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def collect_host_fingerprint() -> dict[str, Any]:
    host: dict[str, Any] = {
        "os_build": platform.version(),
        "cpu": platform.processor() or "unknown",
        "gpu": "unknown",
        "driver": "unknown",
        "ram_gb": 0.0,
    }

    if platform.system() != "Windows":
        return host

    os_info = powershell_json(
        "Get-CimInstance Win32_OperatingSystem "
        "| Select-Object Caption,Version,BuildNumber "
        "| ConvertTo-Json -Compress"
    )
    cpu_info = powershell_json(
        "Get-CimInstance Win32_Processor "
        "| Select-Object Name,NumberOfCores,NumberOfLogicalProcessors "
        "| ConvertTo-Json -Compress"
    )
    gpu_info = powershell_json(
        "Get-CimInstance Win32_VideoController "
        "| Select-Object Name,DriverVersion "
        "| ConvertTo-Json -Compress"
    )
    mem_info = powershell_json(
        "Get-CimInstance Win32_ComputerSystem "
        "| Select-Object TotalPhysicalMemory "
        "| ConvertTo-Json -Compress"
    )

    if isinstance(os_info, dict):
        host["os_build"] = str(os_info.get("Version") or os_info.get("BuildNumber") or host["os_build"])

    if isinstance(cpu_info, list):
        cpu_info = cpu_info[0] if cpu_info else {}
    if isinstance(cpu_info, dict):
        host["cpu"] = str(cpu_info.get("Name") or host["cpu"]).strip()

    selected_gpu: dict[str, Any] = {}
    if isinstance(gpu_info, list):
        for item in gpu_info:
            if "AMD" in str(item.get("Name", "")):
                selected_gpu = item
                break
        if not selected_gpu and gpu_info:
            selected_gpu = gpu_info[0]
    elif isinstance(gpu_info, dict):
        selected_gpu = gpu_info
    if selected_gpu:
        host["gpu"] = str(selected_gpu.get("Name", host["gpu"]))
        host["driver"] = str(selected_gpu.get("DriverVersion", host["driver"]))

    if isinstance(mem_info, dict):
        total_mem = mem_info.get("TotalPhysicalMemory")
        if isinstance(total_mem, (int, float)) and total_mem > 0:
            host["ram_gb"] = round(float(total_mem) / (1024**3), 2)

    return host


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_jsonl_with_lineno(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{lineno}: each line must be a JSON object")
            rows.append((lineno, payload))
    return rows


def nearest_rank_percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = max(1, math.ceil((p / 100.0) * len(sorted_values)))
    return float(sorted_values[rank - 1])


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
