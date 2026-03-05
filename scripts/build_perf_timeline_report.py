#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

RUNNING_STATES = {"initialized", "running_preflight", "running_stage_a", "running_stage_b"}
INTERRUPTED_SIGNATURE = "pipeline_interrupted_no_active_runner"
MAINLINE_BACKENDS = {"llama", "mlc", "ort"}
EXPERIMENTAL_BACKENDS = {"torch_rocm"}
FIXED_MAINLINE_SNAPSHOTS = [
    "r14_15_mainline_r20",
    "r14_15_mainline_r21",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build performance timeline report from local benchmark snapshots."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing benchmark result snapshots.",
    )
    parser.add_argument(
        "--out-root",
        default="reports/perf_timeline",
        help="Output directory for timeline CSV/PNG/Markdown artifacts.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=int,
        default=20,
        help="If a pipeline is still running but stale for this duration and no runner exists, classify as interrupted.",
    )
    parser.add_argument(
        "--skip-process-probe",
        action="store_true",
        help="Skip active runner process detection (useful for deterministic offline reporting/tests).",
    )
    return parser.parse_args()


def now_utc() -> datetime:
    return datetime.now(UTC)


def parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    text = str(text).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def iso_utc(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_backend(name: str | None) -> str:
    if not name:
        return "unknown"
    text = str(name).strip().lower()
    mapping = {
        "llama_cpp": "llama",
        "llama": "llama",
        "mlc_llm": "mlc",
        "mlc": "mlc",
        "ort_dml": "ort",
        "ort": "ort",
        "torch_rocm": "torch_rocm",
    }
    return mapping.get(text, text)


def read_run_context(run_dir: Path) -> dict[str, str]:
    run_json = read_json(run_dir / "run.json") or {}
    backend = normalize_backend((run_json.get("backend") or {}).get("name"))
    workload = str((run_json.get("workload") or {}).get("path") or "")
    model_ref = str((run_json.get("model") or {}).get("path") or "")
    return {
        "backend": backend,
        "workload": workload,
        "model_ref": model_ref,
    }


def detect_any_grid_runner() -> bool:
    if os.name != "nt":
        return False
    script = r"""
$matches = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
  $_.CommandLine -and (
    $_.CommandLine -match 'run_gpu_util_uplift_windows\.ps1' -or
    $_.CommandLine -match 'harness[\\/]benchctl\.py\s+servebench' -or
    $_.CommandLine -match '-m\s+aiperf\s+profile'
  )
}
if (@($matches).Count -gt 0) { '1' } else { '0' }
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "1" in proc.stdout


def detect_snapshot_runner(snapshot_dir: Path) -> bool:
    if os.name != "nt":
        return False
    snap = str(snapshot_dir).replace("'", "''")
    script = f"""
$matches = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {{
  $_.CommandLine -and
  $_.CommandLine -like '*{snap}*' -and (
    $_.CommandLine -match 'run_gpu_util_uplift_windows\\.ps1' -or
    $_.CommandLine -match 'harness[\\\\/]benchctl\\.py\\s+servebench' -or
    $_.CommandLine -match '-m\\s+aiperf\\s+profile'
  )
}}
if (@($matches).Count -gt 0) {{ '1' }} else {{ '0' }}
"""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return False
    return proc.returncode == 0 and "1" in proc.stdout


def classify_health(
    meta: dict[str, Any],
    snapshot_dir: Path,
    stale_minutes: int,
    any_runner_active: bool,
) -> tuple[str, str]:
    progress = meta.get("progress") or {}
    state = str(progress.get("state") or "")
    artifacts_ready = bool(meta.get("artifacts_ready", False))
    if artifacts_ready or state == "completed":
        return "completed", ""
    if state == "failed":
        return "failed", ""
    if state in RUNNING_STATES or state.startswith("running_"):
        last_update = parse_utc(progress.get("last_update_utc"))
        generated = parse_utc(meta.get("generated_at_utc"))
        observed = last_update or generated
        stale = False
        if observed is not None:
            stale = now_utc() - observed >= timedelta(minutes=max(1, stale_minutes))
        active_here = detect_snapshot_runner(snapshot_dir)
        if stale and not active_here and not any_runner_active:
            return "interrupted", INTERRUPTED_SIGNATURE
        return "running", ""
    return "unknown", ""


def extract_grid_points(
    snapshot_dir: Path,
    snapshot_id: str,
    snapshot_time: datetime,
    meta: dict[str, Any],
    health: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    leaderboard = snapshot_dir / "leaderboard.csv"
    if not leaderboard.exists():
        return points

    try:
        rows = pd.read_csv(leaderboard)
    except Exception:
        return points

    default_workload = str(((meta.get("config") or {}).get("workload")) or "")
    for row in rows.to_dict(orient="records"):
        run_dir_text = str(row.get("run_dir") or "")
        run_ctx = read_run_context(Path(run_dir_text)) if run_dir_text else {}
        backend = normalize_backend(str(row.get("backend") or run_ctx.get("backend") or ""))
        workload = run_ctx.get("workload") or default_workload
        model_ref = run_ctx.get("model_ref") or ""
        if not model_ref and backend == "torch_rocm":
            model_ref = str(((meta.get("config") or {}).get("torch_model_id")) or "")
        group_key = f"{workload}|{model_ref}|{backend}"
        points.append(
            {
                "snapshot_id": snapshot_id,
                "snapshot_time_utc": iso_utc(snapshot_time),
                "snapshot_health": health,
                "source_type": "grid_leaderboard",
                "backend": backend,
                "workload": workload,
                "model_ref": model_ref,
                "group_key": group_key,
                "stage": str(row.get("stage") or ""),
                "profile_id": str(row.get("profile_id") or ""),
                "status": str(row.get("status") or ""),
                "attempt": int(float(row.get("attempt") or 0)),
                "ttft_p50_ms": to_float(row.get("ttft_p50_ms")),
                "itl_p50_ms": to_float(row.get("itl_p50_ms")),
                "tps_mean": to_float(row.get("tps_mean")),
                "rps_mean": to_float(row.get("rps_mean")),
                "gpu_util_avg": to_float(row.get("gpu_util_avg")),
                "telemetry_effective": str(row.get("telemetry_effective") or ""),
                "telemetry_degraded": str(row.get("telemetry_degraded") or "").strip().lower() in {"true", "1", "yes"},
                "sampler_status": str(row.get("sampler_status") or ""),
                "blocker_signature": str(row.get("blocker_signature") or ""),
                "run_dir": run_dir_text,
            }
        )
    return points


def extract_summary_points(
    snapshot_dir: Path,
    snapshot_id: str,
    snapshot_time: datetime,
    health: str,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for summary_path in snapshot_dir.rglob("summary.csv"):
        if "compare" in {part.lower() for part in summary_path.parts}:
            continue
        run_dir = summary_path.parent
        context = read_run_context(run_dir)
        backend = context["backend"]
        workload = context["workload"]
        model_ref = context["model_ref"]
        if not backend:
            lower_parts = {p.lower() for p in run_dir.parts}
            if "llama" in lower_parts:
                backend = "llama"
            elif "mlc" in lower_parts:
                backend = "mlc"
            elif "ort_dml" in lower_parts:
                backend = "ort"
        if not backend:
            continue
        try:
            frame = pd.read_csv(summary_path)
        except Exception:
            continue
        if frame.empty:
            continue
        if "track" in frame.columns:
            serving = frame.loc[frame["track"] == "serving"]
            selected = serving.iloc[0] if not serving.empty else frame.iloc[0]
        else:
            selected = frame.iloc[0]
        group_key = f"{workload}|{model_ref}|{backend}"
        points.append(
            {
                "snapshot_id": snapshot_id,
                "snapshot_time_utc": iso_utc(snapshot_time),
                "snapshot_health": health,
                "source_type": "run_summary",
                "backend": backend,
                "workload": workload,
                "model_ref": model_ref,
                "group_key": group_key,
                "stage": "single",
                "profile_id": run_dir.name,
                "status": "success",
                "attempt": 1,
                "ttft_p50_ms": to_float(selected.get("ttft_p50_ms")),
                "itl_p50_ms": to_float(selected.get("itl_p50_ms")),
                "tps_mean": to_float(selected.get("tps_mean")),
                "rps_mean": to_float(selected.get("rps_mean")),
                "gpu_util_avg": float("nan"),
                "telemetry_effective": "",
                "telemetry_degraded": False,
                "sampler_status": "",
                "blocker_signature": "",
                "run_dir": str(run_dir),
            }
        )
    return points


def collect_compare_winners(snapshot_dir: Path) -> dict[str, str]:
    winners: dict[str, str] = {}
    compare_root = snapshot_dir / "compare"
    if not compare_root.exists():
        return winners
    for meta_path in compare_root.rglob("compare_meta.json"):
        pair = meta_path.parent.name
        meta = read_json(meta_path) or {}
        overall = meta.get("overall") or {}
        winner = str(overall.get("winner") or "")
        if pair and winner:
            winners[pair] = winner
    return winners


def load_snapshot_rows(
    meta_path: Path,
    stale_minutes: int,
    any_runner_active: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    meta = read_json(meta_path) or {}
    snapshot_dir = meta_path.parent
    snapshot_id = snapshot_dir.name
    generated = parse_utc(meta.get("generated_at_utc")) or datetime.fromtimestamp(
        meta_path.stat().st_mtime,
        tz=UTC,
    )
    schema_version = str(meta.get("schema_version") or "")

    health, resolution_signature = classify_health(
        meta=meta,
        snapshot_dir=snapshot_dir,
        stale_minutes=stale_minutes,
        any_runner_active=any_runner_active,
    )

    if schema_version.startswith("gpu_util_uplift"):
        points = extract_grid_points(
            snapshot_dir=snapshot_dir,
            snapshot_id=snapshot_id,
            snapshot_time=generated,
            meta=meta,
            health=health,
        )
    else:
        points = extract_summary_points(
            snapshot_dir=snapshot_dir,
            snapshot_id=snapshot_id,
            snapshot_time=generated,
            health=health,
        )

    success_points = [p for p in points if p.get("status") == "success"]
    failed_points = [p for p in points if p.get("status") == "failed"]

    blocker_counter: Counter[str] = Counter()
    stage_results = meta.get("stage_results") or {}
    for stage_name in ("stage_a", "stage_b"):
        for item in stage_results.get(stage_name, []) or []:
            signature = str(item.get("blocker_signature") or "").strip()
            if signature:
                blocker_counter[signature] += 1
    if not blocker_counter and meta.get("blocker_signature"):
        blocker_counter[str(meta.get("blocker_signature"))] += 1
    if resolution_signature:
        blocker_counter[resolution_signature] += 1

    progress = meta.get("progress") or {}
    stage_a_total = int(progress.get("stage_a_total") or 0)
    stage_b_total = int(progress.get("stage_b_total") or 0)
    stage_a_completed = int(progress.get("stage_a_completed") or 0)
    stage_b_completed = int(progress.get("stage_b_completed") or 0)
    success_ratio_stage_a = (stage_a_completed / stage_a_total) if stage_a_total > 0 else float("nan")
    success_ratio_stage_b = (stage_b_completed / stage_b_total) if stage_b_total > 0 else float("nan")
    progress_velocity = int(progress.get("completed_profiles") or len(success_points))
    telemetry_points = [p for p in points if p.get("source_type") == "grid_leaderboard"]
    telemetry_total = len(telemetry_points)
    telemetry_enabled = len(
        [p for p in telemetry_points if str(p.get("telemetry_effective") or "").strip().lower() == "enabled"]
    )
    telemetry_disabled = len(
        [p for p in telemetry_points if str(p.get("telemetry_effective") or "").strip().lower() == "disabled"]
    )
    telemetry_degraded = len([p for p in telemetry_points if bool(p.get("telemetry_degraded", False))])
    telemetry_coverage_ratio = (telemetry_enabled / telemetry_total) if telemetry_total > 0 else float("nan")
    telemetry_degraded_ratio = (telemetry_degraded / telemetry_total) if telemetry_total > 0 else float("nan")

    best_tps = max(
        [to_float(p.get("tps_mean")) for p in success_points if to_float(p.get("tps_mean")) > 0.0],
        default=float("nan"),
    )
    best_ttft = min(
        [
            to_float(p.get("ttft_p50_ms"))
            for p in success_points
            if to_float(p.get("ttft_p50_ms")) > 0.0
        ],
        default=float("nan"),
    )
    blocker_top = blocker_counter.most_common(1)[0][0] if blocker_counter else ""

    compare_winners = collect_compare_winners(snapshot_dir)
    summary = {
        "snapshot_id": snapshot_id,
        "snapshot_time_utc": iso_utc(generated),
        "snapshot_dir": str(snapshot_dir),
        "schema_version": schema_version,
        "execution_health": health,
        "resolution_signature": resolution_signature,
        "pipeline_state": str(progress.get("state") or ""),
        "artifacts_ready": bool(meta.get("artifacts_ready", False)),
        "stage_a_completed": stage_a_completed,
        "stage_a_total": stage_a_total,
        "stage_b_completed": stage_b_completed,
        "stage_b_total": stage_b_total,
        "success_ratio_stage_a": success_ratio_stage_a,
        "success_ratio_stage_b": success_ratio_stage_b,
        "progress_velocity": progress_velocity,
        "telemetry_total_points": telemetry_total,
        "telemetry_enabled_points": telemetry_enabled,
        "telemetry_disabled_points": telemetry_disabled,
        "telemetry_degraded_points": telemetry_degraded,
        "telemetry_coverage_ratio": telemetry_coverage_ratio,
        "telemetry_degraded_ratio": telemetry_degraded_ratio,
        "success_points": len(success_points),
        "failed_points": len(failed_points),
        "best_tps_per_snapshot": best_tps,
        "best_ttft_per_snapshot": best_ttft,
        "blocker_top": blocker_top,
        "blocker_counts_json": json.dumps(blocker_counter, ensure_ascii=False, sort_keys=True),
        "compare_winners_json": json.dumps(compare_winners, ensure_ascii=False, sort_keys=True),
    }

    blocker_rows = [
        {
            "snapshot_id": snapshot_id,
            "snapshot_time_utc": iso_utc(generated),
            "blocker_signature": sig,
            "count": count,
        }
        for sig, count in blocker_counter.items()
    ]
    return points, summary, blocker_rows


def plot_empty(output: Path, title: str, text: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_metric_trend(points: pd.DataFrame, metric: str, output: Path, title: str, ylabel: str) -> None:
    successful = points.loc[points["status"] == "success"].copy()
    successful = successful.loc[successful[metric] > 0]
    successful = successful.loc[
        successful["workload"].astype(str).str.strip().ne("")
        & successful["model_ref"].astype(str).str.strip().ne("")
        & successful["backend"].astype(str).str.strip().ne("")
    ]
    if successful.empty:
        plot_empty(output, title, "No successful samples")
        return

    successful["snapshot_dt"] = pd.to_datetime(successful["snapshot_time_utc"], utc=True)
    agg_fn = "min" if metric == "ttft_p50_ms" else "max"
    grouped = (
        successful.groupby(["group_key", "snapshot_time_utc", "snapshot_dt"], as_index=False)[metric]
        .agg(agg_fn)
        .sort_values("snapshot_dt")
    )
    group_order = (
        grouped.groupby("group_key")[metric]
        .agg(["count", "last"])
        .sort_values(["count", "last"], ascending=[False, metric == "ttft_p50_ms"])
        .head(8)
        .index.tolist()
    )
    grouped = grouped.loc[grouped["group_key"].isin(group_order)]
    if grouped.empty:
        plot_empty(output, title, "No grouped data")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for group_key, frame in grouped.groupby("group_key"):
        frame = frame.sort_values("snapshot_dt")
        label = group_key.split("|")[-1]
        ax.plot(frame["snapshot_dt"], frame[metric], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Snapshot Time (UTC)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def filter_points_by_backends(points: pd.DataFrame, backends: set[str]) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    return points.loc[points["backend"].isin(list(backends))].copy()


def plot_mainline_success_ratio(points: pd.DataFrame, output: Path) -> None:
    if points.empty:
        plot_empty(output, "Mainline Success Ratio Trend", "No point data")
        return
    frame = points.copy()
    frame = frame.loc[frame["backend"].isin(list(MAINLINE_BACKENDS))]
    if frame.empty:
        plot_empty(output, "Mainline Success Ratio Trend", "No mainline samples")
        return

    frame["snapshot_dt"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    frame = frame.sort_values("snapshot_dt")
    frame["is_success"] = frame["status"].isin(["success", "dry-run"]).astype(int)

    stage_a = frame.loc[frame["stage"] == "stage_a"]
    stage_b = frame.loc[frame["stage"] == "stage_b_recheck"]

    stage_a_ratio = pd.DataFrame()
    stage_b_ratio = pd.DataFrame()
    if not stage_a.empty:
        stage_a_ratio = (
            stage_a.groupby(["snapshot_id", "snapshot_time_utc", "snapshot_dt"], as_index=False)
            .agg(total=("is_success", "count"), success=("is_success", "sum"))
            .assign(ratio=lambda x: x["success"] / x["total"])
        )
    if not stage_b.empty:
        stage_b_ratio = (
            stage_b.groupby(["snapshot_id", "snapshot_time_utc", "snapshot_dt"], as_index=False)
            .agg(total=("is_success", "count"), success=("is_success", "sum"))
            .assign(ratio=lambda x: x["success"] / x["total"])
        )

    if stage_a_ratio.empty and stage_b_ratio.empty:
        plot_empty(output, "Mainline Success Ratio Trend", "No stage_a/stage_b samples")
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))
    if not stage_a_ratio.empty:
        ax.plot(
            stage_a_ratio["snapshot_dt"],
            stage_a_ratio["ratio"],
            marker="o",
            label="mainline_stage_a_success_ratio",
        )
    if not stage_b_ratio.empty:
        ax.plot(
            stage_b_ratio["snapshot_dt"],
            stage_b_ratio["ratio"],
            marker="o",
            label="mainline_stage_b_success_ratio",
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Mainline Stage Success Ratio Trend")
    ax.set_xlabel("Snapshot Time (UTC)")
    ax.set_ylabel("Success Ratio")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_success_ratio(summary: pd.DataFrame, output: Path) -> None:
    frame = summary.copy()
    frame["snapshot_dt"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    frame = frame.sort_values("snapshot_dt")
    frame = frame.loc[
        frame["success_ratio_stage_a"].notna() | frame["success_ratio_stage_b"].notna()
    ]
    if frame.empty:
        plot_empty(output, "Success Ratio Trend", "No stage ratio data")
        return

    fig, ax = plt.subplots(figsize=(12, 4.8))
    if frame["success_ratio_stage_a"].notna().any():
        ax.plot(
            frame["snapshot_dt"],
            frame["success_ratio_stage_a"],
            marker="o",
            label="stage_a_success_ratio",
        )
    if frame["success_ratio_stage_b"].notna().any():
        ax.plot(
            frame["snapshot_dt"],
            frame["success_ratio_stage_b"],
            marker="o",
            label="stage_b_success_ratio",
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Stage Success Ratio Trend")
    ax.set_xlabel("Snapshot Time (UTC)")
    ax.set_ylabel("Success Ratio")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_blocker_mix(blockers: pd.DataFrame, output: Path) -> None:
    if blockers.empty:
        plot_empty(output, "Blocker Mix", "No blocker signatures")
        return
    frame = blockers.copy()
    frame["snapshot_dt"] = pd.to_datetime(frame["snapshot_time_utc"], utc=True)
    frame = frame.sort_values("snapshot_dt")
    top_signatures = (
        frame.groupby("blocker_signature")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )
    frame["blocker_group"] = frame["blocker_signature"].where(
        frame["blocker_signature"].isin(top_signatures),
        "other",
    )
    pivot = (
        frame.groupby(["snapshot_id", "snapshot_dt", "blocker_group"], as_index=False)["count"]
        .sum()
        .pivot_table(
            index=["snapshot_id", "snapshot_dt"],
            columns="blocker_group",
            values="count",
            fill_value=0,
        )
        .sort_index(level="snapshot_dt")
    )
    if pivot.empty:
        plot_empty(output, "Blocker Mix", "No blocker counts after pivot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x_labels = [idx[0] for idx in pivot.index]
    bottom = pd.Series([0] * len(pivot), index=pivot.index, dtype=float)
    for column in pivot.columns:
        values = pivot[column]
        ax.bar(x_labels, values, bottom=bottom, label=str(column))
        bottom += values
    ax.set_title("Blocker Signature Mix by Snapshot")
    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def format_pct_delta(before: float, after: float, lower_is_better: bool) -> str:
    if pd.isna(before) or before == 0 or pd.isna(after):
        return "n/a"
    ratio = (after - before) / before
    if lower_is_better:
        ratio = -ratio
    return f"{ratio * 100:.1f}%"


def build_weekly_gain_lines(
    points: pd.DataFrame,
    backends: set[str] | None = None,
) -> list[str]:
    success = points.loc[points["status"] == "success"].copy()
    if backends:
        success = success.loc[success["backend"].isin(list(backends))]
    success = success.loc[
        success["workload"].astype(str).str.strip().ne("")
        & success["model_ref"].astype(str).str.strip().ne("")
        & success["backend"].astype(str).str.strip().ne("")
    ]
    if success.empty:
        return ["- no same-key successful samples yet"]
    success["snapshot_dt"] = pd.to_datetime(success["snapshot_time_utc"], utc=True)
    lines: list[str] = []
    for backend in sorted(success["backend"].dropna().unique().tolist()):
        backend_rows = success.loc[success["backend"] == backend]
        if backend_rows.empty:
            continue
        by_group_snapshot = (
            backend_rows.groupby(
                ["group_key", "workload", "model_ref", "snapshot_time_utc", "snapshot_dt"],
                as_index=False,
            )
            .agg(
                tps_best=("tps_mean", "max"),
                ttft_best=("ttft_p50_ms", "min"),
            )
            .sort_values("snapshot_dt")
        )
        if by_group_snapshot.empty:
            continue

        group_stats = (
            by_group_snapshot.groupby(["group_key", "workload", "model_ref"], as_index=False)
            .agg(
                snapshots=("snapshot_time_utc", "nunique"),
                latest_dt=("snapshot_dt", "max"),
            )
            .sort_values(["snapshots", "latest_dt"], ascending=[False, False])
        )
        group_stats = group_stats.loc[group_stats["snapshots"] >= 2]
        if group_stats.empty:
            continue

        chosen = group_stats.iloc[0]
        chosen_key = str(chosen["group_key"])
        chosen_rows = by_group_snapshot.loc[by_group_snapshot["group_key"] == chosen_key].sort_values("snapshot_dt")
        if chosen_rows.shape[0] < 2:
            continue

        first = chosen_rows.iloc[0]
        last = chosen_rows.iloc[-1]
        tps_gain = format_pct_delta(first["tps_best"], last["tps_best"], lower_is_better=False)
        ttft_gain = format_pct_delta(first["ttft_best"], last["ttft_best"], lower_is_better=True)
        workload = str(chosen["workload"])
        model_ref = str(chosen["model_ref"])
        lines.append(
            f"- `{backend}`: best TPS `{first['tps_best']:.3f} -> {last['tps_best']:.3f}` "
            f"({tps_gain}), best TTFT `{first['ttft_best']:.3f} -> {last['ttft_best']:.3f}` "
            f"({ttft_gain}); key=`{workload} | {model_ref}`"
        )
    if not lines:
        return ["- no comparable same-key backend snapshots yet"]
    return lines


def build_fixed_mainline_throughput_lines(points: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append("## Mainline Throughput Net Gain (Fixed Snapshots)")
    lines.append("")
    lines.append(
        "Fixed snapshots (same-key mainline comparison): "
        + ", ".join([f"`{name}`" for name in FIXED_MAINLINE_SNAPSHOTS])
    )
    lines.append("")
    lines.append(
        "| backend | "
        + " | ".join(FIXED_MAINLINE_SNAPSHOTS)
        + " | net_gain_vs_first_available |"
    )
    lines.append("| --- | " + " | ".join([":---:" for _ in FIXED_MAINLINE_SNAPSHOTS]) + " | ---: |")

    success = points.loc[points["status"] == "success"].copy()
    success = success.loc[success["backend"].isin(list(MAINLINE_BACKENDS))]
    if success.empty:
        lines.append("| n/a | n/a | n/a | n/a | n/a |")
        lines.append("")
        return lines

    for backend in sorted(MAINLINE_BACKENDS):
        backend_rows = success.loc[success["backend"] == backend].copy()
        snapshot_values: list[float] = []
        for snapshot_id in FIXED_MAINLINE_SNAPSHOTS:
            snap_rows = backend_rows.loc[
                (backend_rows["snapshot_id"] == snapshot_id)
                & (backend_rows["tps_mean"] > 0)
            ]
            if snap_rows.empty:
                snapshot_values.append(float("nan"))
            else:
                snapshot_values.append(float(snap_rows["tps_mean"].max()))

        first_available = float("nan")
        last_available = float("nan")
        for value in snapshot_values:
            if not pd.isna(value):
                first_available = value
                break
        for value in reversed(snapshot_values):
            if not pd.isna(value):
                last_available = value
                break

        net_gain = "n/a"
        if not pd.isna(first_available) and first_available != 0 and not pd.isna(last_available):
            ratio = (last_available - first_available) / first_available
            net_gain = f"{ratio * 100:.1f}%"

        rendered_values = [
            "n/a" if pd.isna(v) else f"{v:.3f}" for v in snapshot_values
        ]
        lines.append(
            "| "
            + backend
            + " | "
            + " | ".join(rendered_values)
            + f" | {net_gain} |"
        )

    lines.append("")
    return lines


def build_markdown_report(
    out_root: Path,
    summary: pd.DataFrame,
    points: pd.DataFrame,
    blockers: pd.DataFrame,
) -> None:
    report_path = out_root / "timeline_report.md"
    lines: list[str] = []
    lines.append("# Performance Timeline Report")
    lines.append("")
    lines.append(f"Generated at: `{iso_utc(now_utc())}`")
    lines.append("")
    lines.append("## Current Health")
    lines.append("")

    if summary.empty:
        lines.append("- no snapshots found")
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    summary_sorted = summary.copy()
    summary_sorted["snapshot_dt"] = pd.to_datetime(summary_sorted["snapshot_time_utc"], utc=True)
    summary_sorted = summary_sorted.sort_values("snapshot_dt")
    latest = summary_sorted.iloc[-1]
    lines.append(f"- latest snapshot: `{latest['snapshot_id']}`")
    lines.append(f"- execution_health: `{latest['execution_health']}`")
    lines.append(f"- pipeline_state: `{latest['pipeline_state']}`")
    lines.append(f"- artifacts_ready: `{bool(latest['artifacts_ready'])}`")
    if str(latest.get("resolution_signature", "")).strip():
        lines.append(f"- resolution_signature: `{latest['resolution_signature']}`")
    if str(latest.get("blocker_top", "")).strip():
        lines.append(f"- blocker_top: `{latest['blocker_top']}`")
    lines.append("")

    lines.append("## Telemetry Health")
    lines.append("")
    telemetry_total = int(latest.get("telemetry_total_points", 0) or 0)
    telemetry_enabled = int(latest.get("telemetry_enabled_points", 0) or 0)
    telemetry_disabled = int(latest.get("telemetry_disabled_points", 0) or 0)
    telemetry_degraded = int(latest.get("telemetry_degraded_points", 0) or 0)
    lines.append(
        f"- telemetry points: `{telemetry_total}` (enabled={telemetry_enabled}, disabled={telemetry_disabled}, degraded={telemetry_degraded})"
    )
    coverage = latest.get("telemetry_coverage_ratio")
    degraded_ratio = latest.get("telemetry_degraded_ratio")
    if pd.notna(coverage):
        lines.append(f"- telemetry coverage ratio: `{float(coverage):.3f}`")
    else:
        lines.append("- telemetry coverage ratio: `n/a`")
    if pd.notna(degraded_ratio):
        lines.append(f"- telemetry degraded ratio: `{float(degraded_ratio):.3f}`")
    else:
        lines.append("- telemetry degraded ratio: `n/a`")
    lines.append("")

    lines.append("## Weekly Gains (Same-key Groups)")
    lines.append("")
    lines.append("### Mainline Backends (`llama+mlc+ort`)")
    lines.append("")
    lines.extend(build_weekly_gain_lines(points, backends=MAINLINE_BACKENDS))
    lines.append("")
    lines.append("### Experimental Backends")
    lines.append("")
    lines.extend(build_weekly_gain_lines(points, backends=EXPERIMENTAL_BACKENDS))
    lines.append("")
    lines.extend(build_fixed_mainline_throughput_lines(points))

    lines.append("## Trend Charts")
    lines.append("")
    lines.append("- `mainline_tps_trend.png`")
    lines.append("- `mainline_ttft_trend.png`")
    lines.append("- `mainline_success_rate.png`")
    lines.append("- `blocker_mix.png`")
    lines.append("")
    lines.append("![](mainline_tps_trend.png)")
    lines.append("")
    lines.append("![](mainline_ttft_trend.png)")
    lines.append("")
    lines.append("![](mainline_success_rate.png)")
    lines.append("")
    lines.append("![](blocker_mix.png)")
    lines.append("")
    lines.append("## All-backend Charts (Context)")
    lines.append("")
    lines.append("- `tps_trend.png`")
    lines.append("- `ttft_trend.png`")
    lines.append("- `success_rate_trend.png`")
    lines.append("")
    lines.append("![](tps_trend.png)")
    lines.append("")
    lines.append("![](ttft_trend.png)")
    lines.append("")
    lines.append("![](success_rate_trend.png)")
    lines.append("")

    lines.append("## RGP DRAM Evidence")
    lines.append("")
    rgp_summary_path = out_root / "rgp_summary.csv"
    rgp_report_path = out_root / "rgp_report.md"
    if rgp_summary_path.exists():
        try:
            rgp_frame = pd.read_csv(rgp_summary_path)
            rgp_total = int(len(rgp_frame))
            rgp_ok = int((rgp_frame["status"] == "ok").sum()) if "status" in rgp_frame.columns else 0
            rgp_missing = (
                int((rgp_frame["status"] == "rgp_columns_missing").sum())
                if "status" in rgp_frame.columns
                else 0
            )
            lines.append(f"- summary rows: `{rgp_total}` (`ok={rgp_ok}`, `rgp_columns_missing={rgp_missing}`)")
        except Exception:
            lines.append("- summary rows: `unknown` (failed to parse rgp_summary.csv)")
    else:
        lines.append("- no `rgp_summary.csv` found under `reports/perf_timeline/`")
    lines.append("- `rgp_report.md`")
    lines.append("- `rgp_bytes_per_token.png`")
    lines.append("- `rgp_dram_read_write.png`")
    lines.append("")
    lines.append("![](rgp_bytes_per_token.png)")
    lines.append("")
    lines.append("![](rgp_dram_read_write.png)")
    lines.append("")
    if rgp_report_path.exists():
        lines.append(f"- detailed report: `{rgp_report_path.name}`")
    lines.append("")

    lines.append("## Snapshot Table (Recent)")
    lines.append("")
    lines.append("| snapshot_id | health | state | artifacts_ready | stage_a | stage_b | best_tps | best_ttft | blocker_top |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | --- |")
    recent = summary_sorted.tail(10)
    for _, row in recent.iterrows():
        stage_a = f"{int(row['stage_a_completed'])}/{int(row['stage_a_total'])}"
        stage_b = f"{int(row['stage_b_completed'])}/{int(row['stage_b_total'])}"
        best_tps = "n/a" if pd.isna(row["best_tps_per_snapshot"]) else f"{row['best_tps_per_snapshot']:.3f}"
        best_ttft = "n/a" if pd.isna(row["best_ttft_per_snapshot"]) else f"{row['best_ttft_per_snapshot']:.3f}"
        lines.append(
            "| "
            + f"{row['snapshot_id']} | {row['execution_health']} | {row['pipeline_state']} | "
            + f"{bool(row['artifacts_ready'])} | {stage_a} | {stage_b} | "
            + f"{best_tps} | {best_ttft} | {row['blocker_top']} |"
        )
    lines.append("")

    if not blockers.empty:
        top_blockers = (
            blockers.groupby("blocker_signature")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
        )
        lines.append("## Blocker Top List")
        lines.append("")
        for signature, count in top_blockers.items():
            lines.append(f"- `{signature}`: {int(count)}")
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(results_root.rglob("pipeline_meta.json"))
    any_runner_active = False if args.skip_process_probe else detect_any_grid_runner()

    all_points: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    blocker_rows: list[dict[str, Any]] = []

    for meta_path in meta_paths:
        points, summary, blockers = load_snapshot_rows(
            meta_path=meta_path,
            stale_minutes=args.stale_minutes,
            any_runner_active=any_runner_active,
        )
        all_points.extend(points)
        summaries.append(summary)
        blocker_rows.extend(blockers)

    points_df = pd.DataFrame(all_points)
    summary_df = pd.DataFrame(summaries)
    blockers_df = pd.DataFrame(blocker_rows)

    if not points_df.empty:
        points_df = points_df.sort_values(["snapshot_time_utc", "backend", "stage", "profile_id"])
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["snapshot_time_utc", "snapshot_id"])
    if not blockers_df.empty:
        blockers_df = blockers_df.sort_values(["snapshot_time_utc", "blocker_signature"])

    points_path = out_root / "timeline_points.csv"
    summary_path = out_root / "timeline_summary.csv"
    latest_path = out_root / "latest_snapshot.json"

    points_df.to_csv(points_path, index=False, encoding="utf-8")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    plot_metric_trend(
        points=points_df if not points_df.empty else pd.DataFrame(),
        metric="tps_mean",
        output=out_root / "tps_trend.png",
        title="Best TPS Trend by Same-key Group",
        ylabel="TPS (higher is better)",
    )
    plot_metric_trend(
        points=points_df if not points_df.empty else pd.DataFrame(),
        metric="ttft_p50_ms",
        output=out_root / "ttft_trend.png",
        title="Best TTFT Trend by Same-key Group",
        ylabel="TTFT p50 ms (lower is better)",
    )
    plot_success_ratio(
        summary=summary_df if not summary_df.empty else pd.DataFrame(),
        output=out_root / "success_rate_trend.png",
    )
    plot_blocker_mix(
        blockers=blockers_df if not blockers_df.empty else pd.DataFrame(),
        output=out_root / "blocker_mix.png",
    )

    mainline_points = (
        filter_points_by_backends(points_df, MAINLINE_BACKENDS)
        if not points_df.empty
        else pd.DataFrame()
    )
    plot_metric_trend(
        points=mainline_points,
        metric="tps_mean",
        output=out_root / "mainline_tps_trend.png",
        title="Mainline Best TPS Trend by Same-key Group",
        ylabel="TPS (higher is better)",
    )
    plot_metric_trend(
        points=mainline_points,
        metric="ttft_p50_ms",
        output=out_root / "mainline_ttft_trend.png",
        title="Mainline Best TTFT Trend by Same-key Group",
        ylabel="TTFT p50 ms (lower is better)",
    )
    plot_mainline_success_ratio(
        points=points_df if not points_df.empty else pd.DataFrame(),
        output=out_root / "mainline_success_rate.png",
    )

    build_markdown_report(
        out_root=out_root,
        summary=summary_df if not summary_df.empty else pd.DataFrame(),
        points=points_df if not points_df.empty else pd.DataFrame(),
        blockers=blockers_df if not blockers_df.empty else pd.DataFrame(),
    )

    latest_payload: dict[str, Any] = {
        "generated_at_utc": iso_utc(now_utc()),
        "snapshot_count": int(len(summary_df)),
        "points_count": int(len(points_df)),
    }
    if not summary_df.empty:
        latest_row = summary_df.iloc[-1].to_dict()
        latest_payload["latest_snapshot"] = latest_row
    latest_path.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[timeline] wrote: {points_path}")
    print(f"[timeline] wrote: {summary_path}")
    print(f"[timeline] wrote: {out_root / 'timeline_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
