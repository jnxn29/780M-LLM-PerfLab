#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_BACKENDS = ["llama", "mlc", "ort", "torch_rocm"]


@dataclass
class BackendBest:
    backend: str
    profile_id: str
    ttft_ms: float
    tps: float


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build four-track before/after metrics from run roots.")
    parser.add_argument("--before-run-root", required=True, help="Path to before run root (contains leaderboard.csv)")
    parser.add_argument("--after-run-root", required=True, help="Path to after run root (contains leaderboard.csv)")
    parser.add_argument("--out-csv", required=True, help="Output CSV path")
    parser.add_argument("--out-md", required=True, help="Output markdown path")
    parser.add_argument(
        "--required-backends",
        default=",".join(DEFAULT_BACKENDS),
        help=f"Comma separated backends required for output (default: {','.join(DEFAULT_BACKENDS)})",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_backend(name: str) -> str:
    n = (name or "").strip().lower()
    mapping = {
        "llama_cpp": "llama",
        "mlc_llm": "mlc",
        "ort_dml": "ort",
        "torch_rocm": "torch_rocm",
        "llama": "llama",
        "mlc": "mlc",
        "ort": "ort",
    }
    return mapping.get(n, n)


def _load_best_by_backend(run_root: Path) -> dict[str, BackendBest]:
    leaderboard = run_root / "leaderboard.csv"
    if not leaderboard.exists():
        raise ValueError(f"missing leaderboard: {leaderboard}")
    best: dict[str, BackendBest] = {}
    with leaderboard.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = str(row.get("stage") or "").strip().lower()
            status = str(row.get("status") or "").strip().lower()
            if stage != "stage_a" or status != "success":
                continue
            backend = _normalize_backend(str(row.get("backend") or ""))
            if not backend:
                continue
            tps = _parse_float(row.get("tps_mean"))
            ttft = _parse_float(row.get("ttft_p50_ms"))
            if tps is None or ttft is None:
                continue
            profile_id = str(row.get("profile_id") or "")
            candidate = BackendBest(
                backend=backend,
                profile_id=profile_id,
                ttft_ms=ttft,
                tps=tps,
            )
            previous = best.get(backend)
            if previous is None:
                best[backend] = candidate
                continue
            # Prefer higher TPS, then lower TTFT.
            if candidate.tps > previous.tps or (
                candidate.tps == previous.tps and candidate.ttft_ms < previous.ttft_ms
            ):
                best[backend] = candidate
    return best


def _ttft_improve(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError(f"invalid before ttft: {before}")
    return ((before - after) / before) * 100.0


def _tps_ratio(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError(f"invalid before tps: {before}")
    return after / before


def _write_csv(
    path: Path,
    *,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "backend",
        "before_snapshot",
        "after_snapshot",
        "before_profile",
        "after_profile",
        "before_ttft_ms",
        "after_ttft_ms",
        "ttft_improve_pct",
        "before_tps",
        "after_tps",
        "tps_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "backend": row["backend"],
                    "before_snapshot": row["before_snapshot"],
                    "after_snapshot": row["after_snapshot"],
                    "before_profile": row["before_profile"],
                    "after_profile": row["after_profile"],
                    "before_ttft_ms": f"{row['before_ttft_ms']:.6f}",
                    "after_ttft_ms": f"{row['after_ttft_ms']:.6f}",
                    "ttft_improve_pct": f"{row['ttft_improve_pct']:.6f}",
                    "before_tps": f"{row['before_tps']:.6f}",
                    "after_tps": f"{row['after_tps']:.6f}",
                    "tps_ratio": f"{row['tps_ratio']:.6f}",
                }
            )


def _build_md(
    *,
    rows: list[dict[str, Any]],
    csv_path_display: str,
) -> str:
    lines: list[str] = []
    lines.append("# Four-track Before/After (Stage-A best TPS profile)")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc_iso()}`")
    lines.append(f"csv: `{csv_path_display}`")
    lines.append("")
    lines.append(
        "| backend | before_snapshot | after_snapshot | before_profile | after_profile | before_ttft_ms | after_ttft_ms | ttft_improve_pct | before_tps | after_tps | tps_ratio |"
    )
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append(
            "| "
            + f"{row['backend']} | {row['before_snapshot']} | {row['after_snapshot']} | "
            + f"{row['before_profile']} | {row['after_profile']} | "
            + f"{row['before_ttft_ms']:.3f} | {row['after_ttft_ms']:.3f} | {row['ttft_improve_pct']:.3f} | "
            + f"{row['before_tps']:.3f} | {row['after_tps']:.3f} | {row['tps_ratio']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    before_root = Path(args.before_run_root).resolve()
    after_root = Path(args.after_run_root).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_md = Path(args.out_md).resolve()
    required_backends = [item.strip().lower() for item in str(args.required_backends).split(",") if item.strip()]
    if not required_backends:
        required_backends = DEFAULT_BACKENDS[:]

    try:
        before_best = _load_best_by_backend(before_root)
        after_best = _load_best_by_backend(after_root)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    missing_before = [b for b in required_backends if b not in before_best]
    missing_after = [b for b in required_backends if b not in after_best]
    if missing_before or missing_after:
        if missing_before:
            print(f"error: missing before success backends: {','.join(missing_before)}", file=sys.stderr)
        if missing_after:
            print(f"error: missing after success backends: {','.join(missing_after)}", file=sys.stderr)
        return 3

    before_meta = _read_json(before_root / "pipeline_meta.json")
    after_meta = _read_json(after_root / "pipeline_meta.json")
    before_snapshot = before_root.name
    after_snapshot = after_root.name
    if isinstance(before_meta.get("paths"), dict):
        before_snapshot = Path(str(before_meta.get("paths", {}).get("out_root") or before_snapshot)).name
    if isinstance(after_meta.get("paths"), dict):
        after_snapshot = Path(str(after_meta.get("paths", {}).get("out_root") or after_snapshot)).name

    rows: list[dict[str, Any]] = []
    for backend in required_backends:
        before = before_best[backend]
        after = after_best[backend]
        rows.append(
            {
                "backend": backend,
                "before_snapshot": before_snapshot,
                "after_snapshot": after_snapshot,
                "before_profile": before.profile_id,
                "after_profile": after.profile_id,
                "before_ttft_ms": before.ttft_ms,
                "after_ttft_ms": after.ttft_ms,
                "ttft_improve_pct": _ttft_improve(before.ttft_ms, after.ttft_ms),
                "before_tps": before.tps,
                "after_tps": after.tps,
                "tps_ratio": _tps_ratio(before.tps, after.tps),
            }
        )

    _write_csv(out_csv, rows=rows)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    try:
        csv_path_display = str(out_csv.relative_to(out_md.parent)).replace("\\", "/")
    except ValueError:
        csv_path_display = out_csv.name
    out_md.write_text(_build_md(rows=rows, csv_path_display=csv_path_display), encoding="utf-8")

    print(f"four_track_before_after_csv: {out_csv}")
    print(f"four_track_before_after_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
