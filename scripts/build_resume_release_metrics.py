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


RUN_ORDER = [
    "r1_baseline",
    "r2_flash_only",
    "r3_compile_only",
    "r4_flash_compile",
    "r5_phi3_confirm",
]


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build resume-oriented three-point metrics (B0/C06/C11) and sidecar summary."
    )
    parser.add_argument("--baseline-meta", required=True, help="Path to B0 torch_push_pipeline_meta.json")
    parser.add_argument("--mid-meta", required=True, help="Path to C06 torch_push_pipeline_meta.json")
    parser.add_argument("--optimized-meta", required=True, help="Path to C11 torch_push_pipeline_meta.json")
    parser.add_argument("--sidecar-summary", required=True, help="Path to operator_sidecar_summary.csv")
    parser.add_argument(
        "--sidecar-recommendation",
        required=True,
        help="Path to promotion_recommendation.json",
    )
    parser.add_argument(
        "--repeat-a-meta",
        default=None,
        help="Optional path to repeat-A torch_push_pipeline_meta.json (for C12A/C12B style stability report).",
    )
    parser.add_argument(
        "--repeat-b-meta",
        default=None,
        help="Optional path to repeat-B torch_push_pipeline_meta.json (for C12A/C12B style stability report).",
    )
    parser.add_argument("--repeat-a-label", default="C12A", help="Label for repeat-A point.")
    parser.add_argument("--repeat-b-label", default="C12B", help="Label for repeat-B point.")
    parser.add_argument(
        "--four-track-csv",
        default=None,
        help="Optional path to four_track_before_after.csv for source traceability.",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for generated output files")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json: {path}: {exc}") from exc


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc


def _to_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid float for {field}: {value}") from exc


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_push_meta(meta: dict[str, Any], *, label: str, path: Path) -> None:
    if not isinstance(meta, dict):
        raise ValueError(f"{label} meta must be an object: {path}")
    runs = meta.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"{label} meta missing runs object: {path}")


@dataclass
class RunPoint:
    ttft_ms: float
    tps: float
    eligible: bool
    torch_device: str
    fallback_triggered: bool
    runtime_device_fallback: bool


@dataclass
class ThreePointRow:
    run_name: str
    b0_ttft: float
    c06_ttft: float
    c11_ttft: float
    b0_tps: float
    c06_tps: float
    c11_tps: float
    b0_to_c11_ttft_improve_pct: float
    c06_to_c11_ttft_improve_pct: float
    b0_to_c11_tps_ratio: float
    c11_hard_gate: bool


@dataclass
class RepeatRow:
    run_name: str
    b0_ttft: float
    c06_ttft: float
    c11_ttft: float
    repeat_a_ttft: float
    repeat_b_ttft: float
    b0_tps: float
    c06_tps: float
    c11_tps: float
    repeat_a_tps: float
    repeat_b_tps: float
    b0_to_c11_ttft_improve_pct: float
    b0_to_repeat_a_ttft_improve_pct: float
    b0_to_repeat_b_ttft_improve_pct: float
    b0_to_c11_tps_ratio: float
    b0_to_repeat_a_tps_ratio: float
    b0_to_repeat_b_tps_ratio: float
    hard_gate_c11: bool
    hard_gate_repeat_a: bool
    hard_gate_repeat_b: bool


def _extract_run_point(meta: dict[str, Any], run_name: str) -> RunPoint | None:
    runs = meta.get("runs") or {}
    if run_name not in runs:
        return None
    run_info = runs.get(run_name) or {}
    stage_b = run_info.get("stage_b")
    if not isinstance(stage_b, dict):
        return None
    return RunPoint(
        ttft_ms=_to_float(stage_b.get("ttft_p50_ms"), field=f"{run_name}.stage_b.ttft_p50_ms"),
        tps=_to_float(stage_b.get("tps_mean"), field=f"{run_name}.stage_b.tps_mean"),
        eligible=bool(run_info.get("eligible", False)),
        torch_device=str(run_info.get("torch_device") or ""),
        fallback_triggered=bool(run_info.get("fallback_triggered", False)),
        runtime_device_fallback=bool(run_info.get("runtime_device_fallback", False)),
    )


def _ordered_run_names(*metas: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for run in RUN_ORDER:
        seen.add(run)
        names.append(run)
    extra: set[str] = set()
    for meta in metas:
        runs = meta.get("runs") or {}
        for key in runs.keys():
            if key not in seen:
                extra.add(key)
    names.extend(sorted(extra))
    return names


def _calc_ttft_improve_pct(before: float, after: float) -> float:
    if before <= 0.0:
        raise ValueError(f"ttft baseline must be >0, got {before}")
    return ((before - after) / before) * 100.0


def _calc_ratio(before: float, after: float) -> float:
    if before <= 0.0:
        raise ValueError(f"ratio baseline must be >0, got {before}")
    return after / before


def _compact_path_ref(path: Path) -> str:
    parts = list(path.parts)
    if len(parts) >= 3 and parts[-2] == "meta":
        return f"{parts[-3]}/meta/{parts[-1]}"
    return path.name


def _csv_path_display(csv_path: Path, md_path: Path) -> str:
    try:
        return str(csv_path.relative_to(md_path.parent)).replace("\\", "/")
    except ValueError:
        return csv_path.name


def _build_three_point_rows(
    baseline_meta: dict[str, Any],
    mid_meta: dict[str, Any],
    optimized_meta: dict[str, Any],
) -> tuple[list[ThreePointRow], list[dict[str, str]]]:
    rows: list[ThreePointRow] = []
    excluded: list[dict[str, str]] = []
    for run_name in _ordered_run_names(baseline_meta, mid_meta, optimized_meta):
        b0 = _extract_run_point(baseline_meta, run_name)
        c06 = _extract_run_point(mid_meta, run_name)
        c11 = _extract_run_point(optimized_meta, run_name)
        if b0 is None or c06 is None or c11 is None:
            missing_labels: list[str] = []
            if b0 is None:
                missing_labels.append("B0")
            if c06 is None:
                missing_labels.append("C06")
            if c11 is None:
                missing_labels.append("C11")
            excluded.append(
                {
                    "run_name": run_name,
                    "reason": "missing_stage_b_or_run",
                    "missing_points": ",".join(missing_labels),
                }
            )
            continue
        b0_to_c11_ttft_improve_pct = _calc_ttft_improve_pct(b0.ttft_ms, c11.ttft_ms)
        c06_to_c11_ttft_improve_pct = _calc_ttft_improve_pct(c06.ttft_ms, c11.ttft_ms)
        b0_to_c11_tps_ratio = _calc_ratio(b0.tps, c11.tps)
        c11_hard_gate = (
            c11.eligible
            and c11.torch_device == "cuda"
            and (not c11.fallback_triggered)
            and (not c11.runtime_device_fallback)
        )
        rows.append(
            ThreePointRow(
                run_name=run_name,
                b0_ttft=b0.ttft_ms,
                c06_ttft=c06.ttft_ms,
                c11_ttft=c11.ttft_ms,
                b0_tps=b0.tps,
                c06_tps=c06.tps,
                c11_tps=c11.tps,
                b0_to_c11_ttft_improve_pct=b0_to_c11_ttft_improve_pct,
                c06_to_c11_ttft_improve_pct=c06_to_c11_ttft_improve_pct,
                b0_to_c11_tps_ratio=b0_to_c11_tps_ratio,
                c11_hard_gate=c11_hard_gate,
            )
        )
    return rows, excluded


def _build_repeat_rows(
    baseline_meta: dict[str, Any],
    mid_meta: dict[str, Any],
    optimized_meta: dict[str, Any],
    repeat_a_meta: dict[str, Any],
    repeat_b_meta: dict[str, Any],
) -> tuple[list[RepeatRow], list[dict[str, str]]]:
    rows: list[RepeatRow] = []
    excluded: list[dict[str, str]] = []
    for run_name in _ordered_run_names(
        baseline_meta,
        mid_meta,
        optimized_meta,
        repeat_a_meta,
        repeat_b_meta,
    ):
        b0 = _extract_run_point(baseline_meta, run_name)
        c06 = _extract_run_point(mid_meta, run_name)
        c11 = _extract_run_point(optimized_meta, run_name)
        repeat_a = _extract_run_point(repeat_a_meta, run_name)
        repeat_b = _extract_run_point(repeat_b_meta, run_name)
        if b0 is None or c06 is None or c11 is None or repeat_a is None or repeat_b is None:
            missing_labels: list[str] = []
            if b0 is None:
                missing_labels.append("B0")
            if c06 is None:
                missing_labels.append("C06")
            if c11 is None:
                missing_labels.append("C11")
            if repeat_a is None:
                missing_labels.append("repeat_a")
            if repeat_b is None:
                missing_labels.append("repeat_b")
            excluded.append(
                {
                    "run_name": run_name,
                    "reason": "missing_stage_b_or_run",
                    "missing_points": ",".join(missing_labels),
                }
            )
            continue
        hard_gate_c11 = (
            c11.eligible
            and c11.torch_device == "cuda"
            and (not c11.fallback_triggered)
            and (not c11.runtime_device_fallback)
        )
        hard_gate_repeat_a = (
            repeat_a.eligible
            and repeat_a.torch_device == "cuda"
            and (not repeat_a.fallback_triggered)
            and (not repeat_a.runtime_device_fallback)
        )
        hard_gate_repeat_b = (
            repeat_b.eligible
            and repeat_b.torch_device == "cuda"
            and (not repeat_b.fallback_triggered)
            and (not repeat_b.runtime_device_fallback)
        )
        rows.append(
            RepeatRow(
                run_name=run_name,
                b0_ttft=b0.ttft_ms,
                c06_ttft=c06.ttft_ms,
                c11_ttft=c11.ttft_ms,
                repeat_a_ttft=repeat_a.ttft_ms,
                repeat_b_ttft=repeat_b.ttft_ms,
                b0_tps=b0.tps,
                c06_tps=c06.tps,
                c11_tps=c11.tps,
                repeat_a_tps=repeat_a.tps,
                repeat_b_tps=repeat_b.tps,
                b0_to_c11_ttft_improve_pct=_calc_ttft_improve_pct(b0.ttft_ms, c11.ttft_ms),
                b0_to_repeat_a_ttft_improve_pct=_calc_ttft_improve_pct(b0.ttft_ms, repeat_a.ttft_ms),
                b0_to_repeat_b_ttft_improve_pct=_calc_ttft_improve_pct(b0.ttft_ms, repeat_b.ttft_ms),
                b0_to_c11_tps_ratio=_calc_ratio(b0.tps, c11.tps),
                b0_to_repeat_a_tps_ratio=_calc_ratio(b0.tps, repeat_a.tps),
                b0_to_repeat_b_tps_ratio=_calc_ratio(b0.tps, repeat_b.tps),
                hard_gate_c11=hard_gate_c11,
                hard_gate_repeat_a=hard_gate_repeat_a,
                hard_gate_repeat_b=hard_gate_repeat_b,
            )
        )
    return rows, excluded


def _rows_to_csv(path: Path, rows: list[ThreePointRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "B0_ttft",
        "C06_ttft",
        "C11_ttft",
        "B0_tps",
        "C06_tps",
        "C11_tps",
        "B0_to_C11_ttft_improve_pct",
        "C06_to_C11_ttft_improve_pct",
        "B0_to_C11_tps_ratio",
        "C11_hard_gate",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_name": row.run_name,
                    "B0_ttft": f"{row.b0_ttft:.3f}",
                    "C06_ttft": f"{row.c06_ttft:.3f}",
                    "C11_ttft": f"{row.c11_ttft:.3f}",
                    "B0_tps": f"{row.b0_tps:.3f}",
                    "C06_tps": f"{row.c06_tps:.3f}",
                    "C11_tps": f"{row.c11_tps:.3f}",
                    "B0_to_C11_ttft_improve_pct": f"{row.b0_to_c11_ttft_improve_pct:.3f}",
                    "C06_to_C11_ttft_improve_pct": f"{row.c06_to_c11_ttft_improve_pct:.3f}",
                    "B0_to_C11_tps_ratio": f"{row.b0_to_c11_tps_ratio:.3f}",
                    "C11_hard_gate": str(bool(row.c11_hard_gate)),
                }
            )


def _repeat_rows_to_csv(path: Path, rows: list[RepeatRow], *, repeat_a_label: str, repeat_b_label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "B0_ttft",
        "C06_ttft",
        "C11_ttft",
        f"{repeat_a_label}_ttft",
        f"{repeat_b_label}_ttft",
        "B0_tps",
        "C06_tps",
        "C11_tps",
        f"{repeat_a_label}_tps",
        f"{repeat_b_label}_tps",
        "B0_to_C11_ttft_improve_pct",
        f"B0_to_{repeat_a_label}_ttft_improve_pct",
        f"B0_to_{repeat_b_label}_ttft_improve_pct",
        "B0_to_C11_tps_ratio",
        f"B0_to_{repeat_a_label}_tps_ratio",
        f"B0_to_{repeat_b_label}_tps_ratio",
        "hard_gate_C11",
        f"hard_gate_{repeat_a_label}",
        f"hard_gate_{repeat_b_label}",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_name": row.run_name,
                    "B0_ttft": f"{row.b0_ttft:.3f}",
                    "C06_ttft": f"{row.c06_ttft:.3f}",
                    "C11_ttft": f"{row.c11_ttft:.3f}",
                    f"{repeat_a_label}_ttft": f"{row.repeat_a_ttft:.3f}",
                    f"{repeat_b_label}_ttft": f"{row.repeat_b_ttft:.3f}",
                    "B0_tps": f"{row.b0_tps:.3f}",
                    "C06_tps": f"{row.c06_tps:.3f}",
                    "C11_tps": f"{row.c11_tps:.3f}",
                    f"{repeat_a_label}_tps": f"{row.repeat_a_tps:.3f}",
                    f"{repeat_b_label}_tps": f"{row.repeat_b_tps:.3f}",
                    "B0_to_C11_ttft_improve_pct": f"{row.b0_to_c11_ttft_improve_pct:.3f}",
                    f"B0_to_{repeat_a_label}_ttft_improve_pct": f"{row.b0_to_repeat_a_ttft_improve_pct:.3f}",
                    f"B0_to_{repeat_b_label}_ttft_improve_pct": f"{row.b0_to_repeat_b_ttft_improve_pct:.3f}",
                    "B0_to_C11_tps_ratio": f"{row.b0_to_c11_tps_ratio:.3f}",
                    f"B0_to_{repeat_a_label}_tps_ratio": f"{row.b0_to_repeat_a_tps_ratio:.3f}",
                    f"B0_to_{repeat_b_label}_tps_ratio": f"{row.b0_to_repeat_b_tps_ratio:.3f}",
                    "hard_gate_C11": str(bool(row.hard_gate_c11)),
                    f"hard_gate_{repeat_a_label}": str(bool(row.hard_gate_repeat_a)),
                    f"hard_gate_{repeat_b_label}": str(bool(row.hard_gate_repeat_b)),
                }
            )


def _relative_delta(a: float, b: float) -> float:
    denom = min(a, b)
    if denom <= 0.0:
        return float("inf")
    return abs(a - b) / denom


def _build_repeat_stability(
    rows: list[RepeatRow],
    *,
    repeat_a_label: str,
    repeat_b_label: str,
    excluded: list[dict[str, str]],
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    all_hard = True
    all_tps_95 = True
    all_stable = True
    for row in rows:
        ttft_delta = _relative_delta(row.repeat_a_ttft, row.repeat_b_ttft)
        tps_delta = _relative_delta(row.repeat_a_tps, row.repeat_b_tps)
        stable_ttft = ttft_delta <= 0.15
        stable_tps = tps_delta <= 0.10
        stable = stable_ttft and stable_tps
        hard = row.hard_gate_c11 and row.hard_gate_repeat_a and row.hard_gate_repeat_b
        tps_95 = (
            row.b0_to_c11_tps_ratio >= 0.95
            and row.b0_to_repeat_a_tps_ratio >= 0.95
            and row.b0_to_repeat_b_tps_ratio >= 0.95
        )
        all_hard = all_hard and hard
        all_tps_95 = all_tps_95 and tps_95
        all_stable = all_stable and stable
        per_run.append(
            {
                "run_name": row.run_name,
                "hard_gate": hard,
                "tps_95_guard": tps_95,
                f"{repeat_a_label}_vs_{repeat_b_label}_ttft_delta_ratio": round(ttft_delta, 6),
                f"{repeat_a_label}_vs_{repeat_b_label}_tps_delta_ratio": round(tps_delta, 6),
                "repeat_stable_ttft": stable_ttft,
                "repeat_stable_tps": stable_tps,
                "repeat_stable": stable,
            }
        )
    return {
        "generated_at_utc": now_utc_iso(),
        "repeat_labels": {"a": repeat_a_label, "b": repeat_b_label},
        "all_hard": all_hard,
        "all_tps_95": all_tps_95,
        "repeat_stability_overall": all_stable,
        "excluded_runs": excluded,
        "by_run": per_run,
    }


def _build_kpi_summary(
    rows: list[ThreePointRow],
    excluded: list[dict[str, str]],
    optimized_meta: dict[str, Any],
    baseline_meta_path: Path,
    mid_meta_path: Path,
    optimized_meta_path: Path,
) -> dict[str, Any]:
    if len(rows) < 1:
        raise ValueError("no complete runs available after exclusion")
    all_hard = all(row.c11_hard_gate for row in rows)
    all_tps_95 = all(row.b0_to_c11_tps_ratio >= 0.95 for row in rows)
    by_run = {row.run_name: row for row in rows}
    r5 = by_run.get("r5_phi3_confirm")
    r1_r4 = [
        by_run[name]
        for name in ("r1_baseline", "r2_flash_only", "r3_compile_only", "r4_flash_compile")
        if name in by_run
    ]
    all_ttft_targets = False
    r1_r4_min_ttft_improve_pct = None
    r5_ttft_improve_pct = None
    r5_tps_ratio = None
    r5_tps_guardrail_pass = False
    if r5 is not None and r1_r4:
        r1_r4_min_ttft_improve_pct = min(item.b0_to_c11_ttft_improve_pct for item in r1_r4)
        r5_ttft_improve_pct = r5.b0_to_c11_ttft_improve_pct
        r5_tps_ratio = r5.b0_to_c11_tps_ratio
        r5_tps_guardrail_pass = r5_tps_ratio >= 0.95
        all_ttft_targets = r5_ttft_improve_pct >= 30.0 and r1_r4_min_ttft_improve_pct >= 10.0
    avg_ttft = sum(item.b0_to_c11_ttft_improve_pct for item in rows) / float(len(rows))
    config = optimized_meta.get("config") or {}
    return {
        "generated_at_utc": now_utc_iso(),
        "snapshot_inputs": {
            "baseline_meta": _compact_path_ref(baseline_meta_path),
            "mid_meta": _compact_path_ref(mid_meta_path),
            "optimized_meta": _compact_path_ref(optimized_meta_path),
        },
        "included_runs": [row.run_name for row in rows],
        "excluded_runs": excluded,
        "kpi": {
            "r1_r5_avg_ttft_improve_pct": round(avg_ttft, 3),
            "r1_r4_min_ttft_improve_pct": round(r1_r4_min_ttft_improve_pct, 3)
            if r1_r4_min_ttft_improve_pct is not None
            else None,
            "r5_ttft_improve_pct": round(r5_ttft_improve_pct, 3) if r5_ttft_improve_pct is not None else None,
            "r5_tps_ratio": round(r5_tps_ratio, 3) if r5_tps_ratio is not None else None,
            "r5_tps_guardrail_pass": r5_tps_guardrail_pass,
            "hard_gate": all_hard,
            "all_hard": all_hard,
            "all_tps_95": all_tps_95,
            "all_ttft_targets": all_ttft_targets,
        },
        "optimized_runtime": {
            "ttft_measurement_mode": "streaming_first_token"
            if bool(config.get("aiperf_streaming_enabled", False))
            else "non_stream_fallback_possible",
            "aiperf_endpoint_type": str(config.get("aiperf_endpoint_type") or ""),
            "tiny_override": {
                "concurrency": int(config.get("tiny_profile_override_concurrency", 0) or 0),
                "request_count": int(config.get("tiny_profile_override_request_count", 0) or 0),
                "prompt_tokens_mean": int(config.get("tiny_profile_override_prompt_tokens_mean", 0) or 0),
                "output_tokens_mean": int(config.get("tiny_profile_override_output_tokens_mean", 0) or 0),
            },
            "phi3_override": {
                "concurrency": int(config.get("phi3_confirm_concurrency", 0) or 0),
                "request_count": int(config.get("phi3_confirm_request_count", 0) or 0),
                "prompt_tokens_mean": int(config.get("phi3_confirm_prompt_tokens_mean", 0) or 0),
                "output_tokens_mean": int(config.get("phi3_confirm_output_tokens_mean", 0) or 0),
            },
        },
    }


def _build_three_point_md(
    rows: list[ThreePointRow],
    kpi_summary: dict[str, Any],
    csv_path_display: str,
) -> str:
    lines: list[str] = []
    lines.append("# Resume Three-point Metrics (B0/C06/C11)")
    lines.append("")
    lines.append(f"generated_at_utc: `{kpi_summary.get('generated_at_utc', '')}`")
    lines.append(f"csv: `{csv_path_display}`")
    lines.append("")
    lines.append(
        "| run_name | B0_ttft | C06_ttft | C11_ttft | B0_tps | C06_tps | C11_tps | B0_to_C11_ttft_improve_pct | C06_to_C11_ttft_improve_pct | B0_to_C11_tps_ratio | C11_hard_gate |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        lines.append(
            "| "
            + f"{row.run_name} | {row.b0_ttft:.3f} | {row.c06_ttft:.3f} | {row.c11_ttft:.3f} | "
            + f"{row.b0_tps:.3f} | {row.c06_tps:.3f} | {row.c11_tps:.3f} | "
            + f"{row.b0_to_c11_ttft_improve_pct:.3f} | {row.c06_to_c11_ttft_improve_pct:.3f} | "
            + f"{row.b0_to_c11_tps_ratio:.3f} | {row.c11_hard_gate} |"
        )
    lines.append("")
    kpi = kpi_summary.get("kpi") or {}
    lines.append("## KPI Summary")
    lines.append("")
    lines.append(f"- r1_r5_avg_ttft_improve_pct: `{kpi.get('r1_r5_avg_ttft_improve_pct')}`")
    lines.append(f"- r1_r4_min_ttft_improve_pct: `{kpi.get('r1_r4_min_ttft_improve_pct')}`")
    lines.append(f"- r5_ttft_improve_pct: `{kpi.get('r5_ttft_improve_pct')}`")
    lines.append(f"- r5_tps_ratio: `{kpi.get('r5_tps_ratio')}`")
    lines.append(f"- r5_tps_guardrail_pass: `{kpi.get('r5_tps_guardrail_pass')}`")
    lines.append(f"- all_hard: `{kpi.get('all_hard')}`")
    lines.append(f"- all_tps_95: `{kpi.get('all_tps_95')}`")
    lines.append(f"- all_ttft_targets: `{kpi.get('all_ttft_targets')}`")
    lines.append("")
    excluded = kpi_summary.get("excluded_runs") or []
    if excluded:
        lines.append("## Excluded Runs")
        lines.append("")
        lines.append("| run_name | reason | missing_points |")
        lines.append("| --- | --- | --- |")
        for item in excluded:
            lines.append(
                f"| {item.get('run_name','')} | {item.get('reason','')} | {item.get('missing_points','')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_repeat_md(
    rows: list[RepeatRow],
    repeat_stability: dict[str, Any],
    *,
    repeat_a_label: str,
    repeat_b_label: str,
    csv_path_display: str,
) -> str:
    lines: list[str] = []
    lines.append("# Resume Three-point + Repeat Metrics")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc_iso()}`")
    lines.append(f"csv: `{csv_path_display}`")
    lines.append("")
    lines.append(
        "| run_name | B0_ttft | C06_ttft | C11_ttft | "
        + f"{repeat_a_label}_ttft | {repeat_b_label}_ttft | "
        + "B0_tps | C06_tps | C11_tps | "
        + f"{repeat_a_label}_tps | {repeat_b_label}_tps | "
        + "B0_to_C11_ttft_improve_pct | "
        + f"B0_to_{repeat_a_label}_ttft_improve_pct | B0_to_{repeat_b_label}_ttft_improve_pct | "
        + "B0_to_C11_tps_ratio | "
        + f"B0_to_{repeat_a_label}_tps_ratio | B0_to_{repeat_b_label}_tps_ratio | "
        + f"hard_gate_C11 | hard_gate_{repeat_a_label} | hard_gate_{repeat_b_label} |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |"
    )
    for row in rows:
        lines.append(
            "| "
            + f"{row.run_name} | {row.b0_ttft:.3f} | {row.c06_ttft:.3f} | {row.c11_ttft:.3f} | "
            + f"{row.repeat_a_ttft:.3f} | {row.repeat_b_ttft:.3f} | "
            + f"{row.b0_tps:.3f} | {row.c06_tps:.3f} | {row.c11_tps:.3f} | "
            + f"{row.repeat_a_tps:.3f} | {row.repeat_b_tps:.3f} | "
            + f"{row.b0_to_c11_ttft_improve_pct:.3f} | {row.b0_to_repeat_a_ttft_improve_pct:.3f} | {row.b0_to_repeat_b_ttft_improve_pct:.3f} | "
            + f"{row.b0_to_c11_tps_ratio:.3f} | {row.b0_to_repeat_a_tps_ratio:.3f} | {row.b0_to_repeat_b_tps_ratio:.3f} | "
            + f"{row.hard_gate_c11} | {row.hard_gate_repeat_a} | {row.hard_gate_repeat_b} |"
        )
    lines.append("")
    lines.append("## Repeat Stability")
    lines.append("")
    lines.append(f"- all_hard: `{repeat_stability.get('all_hard')}`")
    lines.append(f"- all_tps_95: `{repeat_stability.get('all_tps_95')}`")
    lines.append(f"- repeat_stability_overall: `{repeat_stability.get('repeat_stability_overall')}`")
    lines.append("")
    lines.append("| run_name | ttft_delta_ratio | tps_delta_ratio | repeat_stable_ttft | repeat_stable_tps | repeat_stable |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for item in repeat_stability.get("by_run", []):
        lines.append(
            "| "
            + f"{item.get('run_name','')} | "
            + f"{item.get(f'{repeat_a_label}_vs_{repeat_b_label}_ttft_delta_ratio','')} | "
            + f"{item.get(f'{repeat_a_label}_vs_{repeat_b_label}_tps_delta_ratio','')} | "
            + f"{item.get('repeat_stable_ttft','')} | {item.get('repeat_stable_tps','')} | {item.get('repeat_stable','')} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _write_sidecar_md(
    out_path: Path,
    sidecar_rows: list[dict[str, str]],
    sidecar_reco: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# Sidecar Operator Summary")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc_iso()}`")
    lines.append("")
    selected = sidecar_reco.get("selected") or {}
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- selected_experiment_id: `{sidecar_reco.get('selected_experiment_id', '')}`")
    lines.append(f"- baseline_r5_tps: `{sidecar_reco.get('baseline_r5_tps', '')}`")
    lines.append(f"- tps_guardrail: `{sidecar_reco.get('tps_guardrail', '')}`")
    lines.append(f"- selected_ttft_p50_ms: `{selected.get('ttft_p50_ms', '')}`")
    lines.append(f"- selected_tps_mean: `{selected.get('tps_mean', '')}`")
    lines.append(f"- selected_sdpa_profile_effective: `{selected.get('sdpa_profile_effective', '')}`")
    lines.append(f"- selected_compile_enabled: `{selected.get('compile_enabled', '')}`")
    lines.append("")
    lines.append("## O1~O4 Snapshot")
    lines.append("")
    lines.append(
        "| experiment_id | sdpa_profile_effective | compile_enabled | compile_mode | ttft_p50_ms | tps_mean | eligible | ttft_signal_source | fallback_triggered | runtime_device_fallback |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for row in sidecar_rows:
        lines.append(
            "| "
            + f"{row.get('experiment_id','')} | {row.get('sdpa_profile_effective','')} | "
            + f"{row.get('compile_enabled','')} | {row.get('compile_mode','')} | "
            + f"{row.get('ttft_p50_ms','')} | {row.get('tps_mean','')} | {row.get('eligible','')} | "
            + f"{row.get('ttft_signal_source','')} | {row.get('fallback_triggered','')} | "
            + f"{row.get('runtime_device_fallback','')} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_meta_path = Path(args.baseline_meta).resolve()
    mid_meta_path = Path(args.mid_meta).resolve()
    optimized_meta_path = Path(args.optimized_meta).resolve()
    sidecar_summary_path = Path(args.sidecar_summary).resolve()
    sidecar_reco_path = Path(args.sidecar_recommendation).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repeat_a_meta_path = Path(args.repeat_a_meta).resolve() if args.repeat_a_meta else None
    repeat_b_meta_path = Path(args.repeat_b_meta).resolve() if args.repeat_b_meta else None
    repeat_a_label = str(args.repeat_a_label or "C12A").strip()
    repeat_b_label = str(args.repeat_b_label or "C12B").strip()
    four_track_csv_path = Path(args.four_track_csv).resolve() if args.four_track_csv else None
    if bool(repeat_a_meta_path) != bool(repeat_b_meta_path):
        print("error: repeat-a-meta and repeat-b-meta must be provided together", file=sys.stderr)
        return 2

    try:
        baseline_meta = _read_json(baseline_meta_path)
        mid_meta = _read_json(mid_meta_path)
        optimized_meta = _read_json(optimized_meta_path)
        _validate_push_meta(baseline_meta, label="baseline", path=baseline_meta_path)
        _validate_push_meta(mid_meta, label="mid", path=mid_meta_path)
        _validate_push_meta(optimized_meta, label="optimized", path=optimized_meta_path)

        sidecar_rows = _read_csv_rows(sidecar_summary_path)
        sidecar_reco = _read_json(sidecar_reco_path)
        repeat_a_meta = _read_json(repeat_a_meta_path) if repeat_a_meta_path else None
        repeat_b_meta = _read_json(repeat_b_meta_path) if repeat_b_meta_path else None
        if four_track_csv_path is not None and (not four_track_csv_path.exists()):
            raise ValueError(f"missing file: {four_track_csv_path}")
        if repeat_a_meta_path and repeat_b_meta_path:
            _validate_push_meta(repeat_a_meta or {}, label="repeat-a", path=repeat_a_meta_path)
            _validate_push_meta(repeat_b_meta or {}, label="repeat-b", path=repeat_b_meta_path)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        rows, excluded = _build_three_point_rows(
            baseline_meta=baseline_meta,
            mid_meta=mid_meta,
            optimized_meta=optimized_meta,
        )
        kpi_summary = _build_kpi_summary(
            rows=rows,
            excluded=excluded,
            optimized_meta=optimized_meta,
            baseline_meta_path=baseline_meta_path,
            mid_meta_path=mid_meta_path,
            optimized_meta_path=optimized_meta_path,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    out_csv = out_dir / "three_point_metrics.csv"
    out_md = out_dir / "three_point_metrics.md"
    out_json = out_dir / "resume_kpi_summary.json"
    out_sidecar_md = out_dir / "sidecar_operator_summary.md"
    out_repeat_csv = out_dir / "three_point_plus_repeats.csv"
    out_repeat_md = out_dir / "three_point_plus_repeats.md"
    out_repeat_stability = out_dir / "repeat_stability.json"

    _rows_to_csv(out_csv, rows)
    out_md.write_text(
        _build_three_point_md(rows=rows, kpi_summary=kpi_summary, csv_path_display=_csv_path_display(out_csv, out_md)),
        encoding="utf-8",
    )
    if four_track_csv_path:
        kpi_summary["snapshot_inputs"]["four_track_csv"] = four_track_csv_path.name
    out_json.write_text(json.dumps(kpi_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_sidecar_md(out_path=out_sidecar_md, sidecar_rows=sidecar_rows, sidecar_reco=sidecar_reco)
    if repeat_a_meta is not None and repeat_b_meta is not None:
        repeat_rows, repeat_excluded = _build_repeat_rows(
            baseline_meta=baseline_meta,
            mid_meta=mid_meta,
            optimized_meta=optimized_meta,
            repeat_a_meta=repeat_a_meta,
            repeat_b_meta=repeat_b_meta,
        )
        _repeat_rows_to_csv(out_repeat_csv, repeat_rows, repeat_a_label=repeat_a_label, repeat_b_label=repeat_b_label)
        repeat_stability = _build_repeat_stability(
            repeat_rows,
            repeat_a_label=repeat_a_label,
            repeat_b_label=repeat_b_label,
            excluded=repeat_excluded,
        )
        out_repeat_stability.write_text(json.dumps(repeat_stability, ensure_ascii=False, indent=2), encoding="utf-8")
        out_repeat_md.write_text(
            _build_repeat_md(
                repeat_rows,
                repeat_stability,
                repeat_a_label=repeat_a_label,
                repeat_b_label=repeat_b_label,
                csv_path_display=_csv_path_display(out_repeat_csv, out_repeat_md),
            ),
            encoding="utf-8",
        )

    print(f"three_point_metrics_csv: {out_csv}")
    print(f"three_point_metrics_md: {out_md}")
    print(f"resume_kpi_summary_json: {out_json}")
    print(f"sidecar_operator_summary_md: {out_sidecar_md}")
    if repeat_a_meta is not None and repeat_b_meta is not None:
        print(f"three_point_plus_repeats_csv: {out_repeat_csv}")
        print(f"three_point_plus_repeats_md: {out_repeat_md}")
        print(f"repeat_stability_json: {out_repeat_stability}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
