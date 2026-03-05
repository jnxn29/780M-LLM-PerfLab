#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RGP CSV capture manifest from run roots.")
    parser.add_argument(
        "--run-roots",
        nargs="+",
        required=True,
        help="Run roots to extract stage_b profiles from.",
    )
    parser.add_argument(
        "--required-backends",
        default="llama,mlc,torch_rocm",
        help="Comma-separated backend list (default: llama,mlc,torch_rocm).",
    )
    parser.add_argument(
        "--rgp-raw-root",
        default="reports/rgp_raw",
        help="Target root for RGP CSV placement.",
    )
    parser.add_argument(
        "--out-csv",
        default="reports/perf_timeline/rgp_capture_manifest.csv",
        help="Output manifest csv path.",
    )
    return parser.parse_args()


def normalize_backend(name: str) -> str:
    text = (name or "").strip().lower()
    mapping = {
        "llama_cpp": "llama",
        "mlc_llm": "mlc",
        "ort_dml": "ort",
        "torch_rocm": "torch_rocm",
    }
    return mapping.get(text, text)


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def select_stage_b_profile(leaderboard_path: Path, backend: str) -> str:
    if not leaderboard_path.exists():
        return ""
    with leaderboard_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [
        row
        for row in rows
        if str(row.get("stage") or "").strip().lower() == "stage_b_recheck"
        and normalize_backend(str(row.get("backend") or "")) == backend
        and str(row.get("status") or "").strip().lower() == "success"
    ]
    if not matches:
        return ""
    matches.sort(key=lambda r: (parse_float(r.get("ttft_p50_ms")), -parse_float(r.get("tps_mean"))))
    return str(matches[0].get("profile_id") or "").strip()


def main() -> int:
    args = parse_args()
    run_roots = [Path(item).resolve() for item in args.run_roots]
    backends = [normalize_backend(item.strip()) for item in str(args.required_backends).split(",") if item.strip()]
    if not backends:
        backends = ["llama", "mlc", "torch_rocm"]
    rgp_raw_root = Path(args.rgp_raw_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for run_root in run_roots:
        snapshot = run_root.name
        leaderboard_path = run_root / "leaderboard.csv"
        for backend in backends:
            profile_id = select_stage_b_profile(leaderboard_path, backend)
            status = "ok" if profile_id else "missing_stage_b_profile"
            target_csv_name = f"{snapshot}_{backend}_{profile_id}.csv" if profile_id else ""
            target_path = str((rgp_raw_root / target_csv_name).resolve()) if profile_id else ""
            rows.append(
                {
                    "snapshot_id": snapshot,
                    "backend": backend,
                    "profile_id": profile_id,
                    "target_csv_name": target_csv_name,
                    "target_path": target_path,
                    "run_root": str(run_root),
                    "status": status,
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "snapshot_id",
                "backend",
                "profile_id",
                "target_csv_name",
                "target_path",
                "run_root",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"rgp_capture_manifest_csv: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
