from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

SENSITIVE_KEYS = ("token", "key", "password", "secret")


def build_cli_args(base_args: dict[str, Any], params: dict[str, Any], out_dir: Path) -> list[str]:
    merged = {**base_args, **params, "out": str(out_dir)}
    args: list[str] = []
    for key, value in merged.items():
        flag = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def redact_cli_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    pending_mask = False
    for token in args:
        lowered = token.lower()
        if pending_mask:
            redacted.append("***")
            pending_mask = False
            continue
        if any(k in lowered for k in SENSITIVE_KEYS):
            if "=" in token:
                key = token.split("=", 1)[0]
                redacted.append(f"{key}=***")
            else:
                redacted.append(token)
                pending_mask = True
            continue
        redacted.append(token)
    return redacted


def read_summary_metrics(summary_csv: Path) -> dict[str, float]:
    if not summary_csv.exists():
        return {}
    with summary_csv.open("r", encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh), None)
    if not row:
        return {}
    out: dict[str, float] = {}
    for key in ("tps_mean", "rps_mean", "ttft_p50_ms", "itl_p50_ms"):
        try:
            out[key] = float(row.get(key, "") or 0.0)
        except ValueError:
            out[key] = 0.0
    return out


def collect_artifacts(command: str, run_dir: Path) -> dict[str, dict[str, Any]]:
    rules = {
        "servebench": ["run.json", "metrics.jsonl", "summary.csv"],
        "enginebench": ["run.json", "metrics.jsonl", "summary.csv"],
        "compare-runs": ["compare_meta.json", "comparison_summary.csv", "comparison_report.md"],
    }
    files = rules.get(command, [])
    data: dict[str, dict[str, Any]] = {}
    for rel in files:
        path = run_dir / rel
        data[rel] = {"path": str(path), "exists": path.exists(), "size_bytes": (path.stat().st_size if path.exists() else 0)}
    return data

