#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect non-ORT tuning runs and build strict release gate.")
    parser.add_argument("--baseline-run-root", required=True, help="Baseline run root (expects leaderboard.csv)")
    parser.add_argument(
        "--candidate-run-roots",
        nargs="+",
        required=True,
        help="Candidate run roots (expects leaderboard.csv for each)",
    )
    parser.add_argument(
        "--required-backends",
        default="llama,mlc,torch_rocm",
        help="Comma-separated required backends (default: llama,mlc,torch_rocm).",
    )
    parser.add_argument("--ttft-improve-min", type=float, default=15.0, help="Minimum TTFT improvement percent.")
    parser.add_argument("--tps-ratio-min", type=float, default=0.95, help="Minimum TPS ratio.")
    parser.add_argument("--rgp-summary-csv", default="", help="Optional rgp_summary.csv for evidence linkage.")
    parser.add_argument(
        "--out-root",
        default="reports/perf_timeline",
        help="Output directory for non-ORT tuning artifacts.",
    )
    return parser.parse_args()


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_backend(name: str) -> str:
    text = (name or "").strip().lower()
    mapping = {
        "llama_cpp": "llama",
        "mlc_llm": "mlc",
        "ort_dml": "ort",
        "torch_rocm": "torch_rocm",
    }
    return mapping.get(text, text)


def _load_rgp_summary(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("snapshot_id") or "").strip(),
            normalize_backend(str(row.get("backend") or "")),
            str(row.get("profile_id") or "").strip(),
        )
        data[key] = row
    return data


def _load_stage_b_rows(leaderboard_path: Path) -> list[dict[str, Any]]:
    if not leaderboard_path.exists():
        raise ValueError(f"missing leaderboard: {leaderboard_path}")
    with leaderboard_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("stage") or "").strip().lower() != "stage_b_recheck":
            continue
        selected.append(row)
    return selected


def _select_backend_row(stage_b_rows: list[dict[str, Any]], backend: str) -> dict[str, Any] | None:
    matches = [row for row in stage_b_rows if normalize_backend(str(row.get("backend") or "")) == backend]
    if not matches:
        return None
    matches.sort(
        key=lambda r: (
            str(r.get("status") or "").strip().lower() != "success",
            parse_float(r.get("ttft_p50_ms")),
            -parse_float(r.get("tps_mean")),
        )
    )
    return matches[0]


def _row_hard_pass(row: dict[str, Any] | None) -> bool:
    if row is None:
        return False
    status = str(row.get("status") or "").strip().lower()
    samples = parse_float(row.get("samples"))
    ttft = parse_float(row.get("ttft_p50_ms"))
    tps = parse_float(row.get("tps_mean"))
    blocker = str(row.get("blocker_signature") or "").strip()
    return (
        status == "success"
        and samples > 0
        and ttft > 0
        and tps > 0
        and (blocker == "" or blocker.lower() == "nan")
    )


def _safe_pct(before: float, after: float) -> float:
    if before <= 0:
        return float("nan")
    return ((before - after) / before) * 100.0


def _safe_ratio(before: float, after: float) -> float:
    if before <= 0:
        return float("nan")
    return after / before


def _aggregate_candidate(
    baseline_root: Path,
    candidate_root: Path,
    required_backends: list[str],
    ttft_improve_min: float,
    tps_ratio_min: float,
    rgp_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_stage_b = _load_stage_b_rows(baseline_root / "leaderboard.csv")
    candidate_stage_b = _load_stage_b_rows(candidate_root / "leaderboard.csv")

    detail_rows: list[dict[str, Any]] = []
    strict_pass_all = True
    hard_pass_all = True
    avg_ttft_values: list[float] = []
    avg_tps_values: list[float] = []

    for backend in required_backends:
        baseline_row = _select_backend_row(baseline_stage_b, backend)
        candidate_row = _select_backend_row(candidate_stage_b, backend)
        baseline_ttft = parse_float((baseline_row or {}).get("ttft_p50_ms"))
        baseline_tps = parse_float((baseline_row or {}).get("tps_mean"))
        candidate_ttft = parse_float((candidate_row or {}).get("ttft_p50_ms"))
        candidate_tps = parse_float((candidate_row or {}).get("tps_mean"))
        ttft_improve_pct = _safe_pct(baseline_ttft, candidate_ttft)
        tps_ratio = _safe_ratio(baseline_tps, candidate_tps)

        hard_gate = _row_hard_pass(candidate_row)
        strict_gate = hard_gate and ttft_improve_pct >= ttft_improve_min and tps_ratio >= tps_ratio_min
        hard_pass_all = hard_pass_all and hard_gate
        strict_pass_all = strict_pass_all and strict_gate

        if candidate_ttft > 0:
            avg_ttft_values.append(candidate_ttft)
        if candidate_tps > 0:
            avg_tps_values.append(candidate_tps)

        cand_profile = str((candidate_row or {}).get("profile_id") or "").strip()
        rgp_key = (candidate_root.name, backend, cand_profile)
        rgp = rgp_rows.get(rgp_key)
        rgp_status = str((rgp or {}).get("status") or "")
        rgp_bpt = parse_float((rgp or {}).get("dram_bytes_per_token"))

        detail_rows.append(
            {
                "baseline_run_root": str(baseline_root),
                "candidate_run_root": str(candidate_root),
                "candidate_run_name": candidate_root.name,
                "backend": backend,
                "baseline_profile_id": str((baseline_row or {}).get("profile_id") or ""),
                "candidate_profile_id": cand_profile,
                "baseline_ttft_ms": baseline_ttft,
                "candidate_ttft_ms": candidate_ttft,
                "ttft_improve_pct": ttft_improve_pct,
                "baseline_tps": baseline_tps,
                "candidate_tps": candidate_tps,
                "tps_ratio": tps_ratio,
                "candidate_samples": parse_float((candidate_row or {}).get("samples")),
                "candidate_blocker_signature": str((candidate_row or {}).get("blocker_signature") or ""),
                "hard_gate": hard_gate,
                "strict_gate_pass": strict_gate,
                "rgp_status": rgp_status,
                "rgp_dram_bytes_per_token": rgp_bpt,
            }
        )

    avg_ttft = sum(avg_ttft_values) / len(avg_ttft_values) if avg_ttft_values else float("nan")
    avg_tps = sum(avg_tps_values) / len(avg_tps_values) if avg_tps_values else float("nan")
    candidate_summary = {
        "candidate_run_root": str(candidate_root),
        "candidate_run_name": candidate_root.name,
        "hard_gate": hard_pass_all,
        "strict_gate_pass": strict_pass_all,
        "avg_ttft_ms": avg_ttft,
        "avg_tps": avg_tps,
    }
    return detail_rows, candidate_summary


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "baseline_run_root",
        "candidate_run_root",
        "candidate_run_name",
        "backend",
        "baseline_profile_id",
        "candidate_profile_id",
        "baseline_ttft_ms",
        "candidate_ttft_ms",
        "ttft_improve_pct",
        "baseline_tps",
        "candidate_tps",
        "tps_ratio",
        "candidate_samples",
        "candidate_blocker_signature",
        "hard_gate",
        "strict_gate_pass",
        "rgp_status",
        "rgp_dram_bytes_per_token",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            for key in [
                "baseline_ttft_ms",
                "candidate_ttft_ms",
                "ttft_improve_pct",
                "baseline_tps",
                "candidate_tps",
                "tps_ratio",
                "candidate_samples",
                "rgp_dram_bytes_per_token",
            ]:
                value = row_out.get(key)
                if isinstance(value, float):
                    row_out[key] = f"{value:.6f}"
            writer.writerow(row_out)


def _build_report_md(
    summary_rows: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    summary_csv: Path,
    gate_json: Path,
    required_backends: list[str],
    ttft_improve_min: float,
    tps_ratio_min: float,
) -> str:
    lines: list[str] = []
    lines.append("# Non-ORT Tuning Report")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc_iso()}`")
    lines.append(f"required_backends: `{','.join(required_backends)}`")
    lines.append(f"strict_thresholds: `ttft_improve_pct>={ttft_improve_min}, tps_ratio>={tps_ratio_min}`")
    lines.append("")
    lines.append(f"- summary_csv: `{summary_csv}`")
    lines.append(f"- gate_json: `{gate_json}`")
    lines.append("")
    lines.append("## Candidate Summary")
    lines.append("")
    lines.append("| candidate_run_name | hard_gate | strict_gate_pass | avg_ttft_ms | avg_tps |")
    lines.append("| --- | --- | --- | ---: | ---: |")
    for cand in candidate_summaries:
        lines.append(
            "| "
            + f"{cand['candidate_run_name']} | {cand['hard_gate']} | {cand['strict_gate_pass']} | "
            + f"{cand['avg_ttft_ms']:.3f} | {cand['avg_tps']:.3f} |"
        )
    lines.append("")
    lines.append("## Backend Detail")
    lines.append("")
    lines.append(
        "| candidate_run_name | backend | baseline_ttft_ms | candidate_ttft_ms | ttft_improve_pct | baseline_tps | candidate_tps | tps_ratio | hard_gate | strict_gate_pass | rgp_status | rgp_dram_bytes_per_token |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |")
    for row in summary_rows:
        lines.append(
            "| "
            + f"{row['candidate_run_name']} | {row['backend']} | "
            + f"{row['baseline_ttft_ms']:.3f} | {row['candidate_ttft_ms']:.3f} | {row['ttft_improve_pct']:.3f} | "
            + f"{row['baseline_tps']:.3f} | {row['candidate_tps']:.3f} | {row['tps_ratio']:.3f} | "
            + f"{row['hard_gate']} | {row['strict_gate_pass']} | {row['rgp_status']} | {row['rgp_dram_bytes_per_token']:.3f} |"
        )
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    if selected is None:
        lines.append("- selected_candidate: `none`")
        lines.append("- strict_gate_pass: `false`")
    else:
        lines.append(f"- selected_candidate: `{selected['candidate_run_name']}`")
        lines.append("- strict_gate_pass: `true`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    baseline_root = Path(args.baseline_run_root).resolve()
    candidate_roots = [Path(item).resolve() for item in args.candidate_run_roots]
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    required_backends = [normalize_backend(item.strip()) for item in str(args.required_backends).split(",") if item.strip()]
    if not required_backends:
        required_backends = ["llama", "mlc", "torch_rocm"]

    rgp_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if str(args.rgp_summary_csv).strip():
        rgp_rows = _load_rgp_summary(Path(args.rgp_summary_csv).resolve())

    summary_rows: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    for candidate_root in candidate_roots:
        detail_rows, candidate_summary = _aggregate_candidate(
            baseline_root=baseline_root,
            candidate_root=candidate_root,
            required_backends=required_backends,
            ttft_improve_min=float(args.ttft_improve_min),
            tps_ratio_min=float(args.tps_ratio_min),
            rgp_rows=rgp_rows,
        )
        summary_rows.extend(detail_rows)
        candidate_summaries.append(candidate_summary)

    candidate_summaries.sort(
        key=lambda row: (
            row["strict_gate_pass"] is not True,
            row["hard_gate"] is not True,
            row["avg_ttft_ms"],
            -row["avg_tps"],
            row["candidate_run_name"],
        )
    )
    selected = next((row for row in candidate_summaries if row["strict_gate_pass"] is True), None)
    gate_pass = selected is not None

    summary_csv = out_root / "non_ort_tuning_summary.csv"
    report_md = out_root / "non_ort_tuning_report.md"
    gate_json = out_root / "non_ort_release_gate.json"
    _write_summary_csv(summary_csv, summary_rows)
    report_md.write_text(
        _build_report_md(
            summary_rows=summary_rows,
            candidate_summaries=candidate_summaries,
            selected=selected,
            summary_csv=summary_csv,
            gate_json=gate_json,
            required_backends=required_backends,
            ttft_improve_min=float(args.ttft_improve_min),
            tps_ratio_min=float(args.tps_ratio_min),
        ),
        encoding="utf-8",
    )
    gate_payload = {
        "generated_at_utc": now_utc_iso(),
        "pass": gate_pass,
        "required_backends": required_backends,
        "ttft_improve_min": float(args.ttft_improve_min),
        "tps_ratio_min": float(args.tps_ratio_min),
        "baseline_run_root": str(baseline_root),
        "candidate_run_roots": [str(item) for item in candidate_roots],
        "selected_candidate_run_root": selected["candidate_run_root"] if selected else "",
        "selected_candidate_run_name": selected["candidate_run_name"] if selected else "",
        "candidates": candidate_summaries,
        "summary_csv": str(summary_csv),
        "report_md": str(report_md),
    }
    gate_json.write_text(json.dumps(gate_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"non_ort_tuning_summary_csv: {summary_csv}")
    print(f"non_ort_tuning_report_md: {report_md}")
    print(f"non_ort_release_gate_json: {gate_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
