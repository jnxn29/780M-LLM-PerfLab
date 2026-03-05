#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and summarize torch_rocm push runs.")
    parser.add_argument(
        "--runs-root",
        default="<repo-root>/results/r14_16_torch_rocm_push_r1/",
        help="Root directory containing r1..r5 push sub-runs.",
    )
    parser.add_argument(
        "--out-root",
        default="reports/perf_timeline",
        help="Output directory for summary/report/plots.",
    )
    return parser.parse_args()


@dataclass
class StageBMetric:
    run_name: str
    profile_id: str
    telemetry_degraded: bool
    tps_mean: float
    rps_mean: float
    ttft_p50_ms: float
    status: str
    eligible: bool
    blocker_signature: str
    torch_device: str
    fallback_triggered: bool
    runtime_device_fallback: bool
    ttft_signal_source: str
    ttft_request_latency_fallback_count: int


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _read_leaderboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_ttft_signal(run_dir: Path) -> tuple[str, int]:
    adapt_meta = _read_json(run_dir / "raw_tool_output" / "aiperf_adapt_meta.json")
    fallback_counts = adapt_meta.get("fallback_counts") or {}
    if not isinstance(fallback_counts, dict):
        return "unknown", 0
    ttft_from_request_latency = _to_int(fallback_counts.get("ttft_from_request_latency"), 0)
    ttft_from_time_to_first_output_token = _to_int(
        fallback_counts.get("ttft_from_time_to_first_output_token"), 0
    )
    if ttft_from_request_latency > 0:
        return "request_latency_fallback", ttft_from_request_latency
    if ttft_from_time_to_first_output_token > 0:
        return "time_to_first_output_token", 0
    # When adapter metadata exists and no TTFT fallbacks were counted, treat TTFT as first-token sourced.
    if adapt_meta:
        return "time_to_first_token", 0
    return "unknown", 0


def extract_stage_b(run_name: str, run_root: Path) -> StageBMetric:
    meta = _read_json(run_root / "pipeline_meta.json")
    leaderboard = _read_leaderboard(run_root / "leaderboard.csv")
    row = None
    if not leaderboard.empty and "stage" in leaderboard.columns:
        stage_b = leaderboard[(leaderboard["stage"] == "stage_b_recheck") & (leaderboard["status"] == "success")]
        if not stage_b.empty:
            stage_b = stage_b.sort_values(
                by=["telemetry_degraded", "tps_mean", "rps_mean", "ttft_p50_ms"],
                ascending=[True, False, False, True],
            )
            row = stage_b.iloc[0]
    if row is None:
        return StageBMetric(
            run_name=run_name,
            profile_id="",
            telemetry_degraded=False,
            tps_mean=0.0,
            rps_mean=0.0,
            ttft_p50_ms=0.0,
            status="missing_stage_b",
            eligible=False,
            blocker_signature=str(meta.get("blocker_signature") or "stage_b_missing"),
            torch_device=str((meta.get("torch_server") or {}).get("device") or ""),
            fallback_triggered=bool((meta.get("torch_server") or {}).get("fallback_triggered", False)),
            runtime_device_fallback=bool((meta.get("torch_server") or {}).get("runtime_device_fallback", False)),
            ttft_signal_source="unknown",
            ttft_request_latency_fallback_count=0,
        )
    torch_server = meta.get("torch_server") or {}
    run_dir = Path(str(row.get("run_dir", "") or "")).resolve()
    ttft_signal_source, ttft_fallback_count = _resolve_ttft_signal(run_dir)
    eligible = (
        bool(meta.get("artifacts_ready", False))
        and str(torch_server.get("device") or "") == "cuda"
        and not bool(torch_server.get("runtime_device_fallback", False))
        and not bool(torch_server.get("fallback_triggered", False))
    )
    return StageBMetric(
        run_name=run_name,
        profile_id=str(row.get("profile_id", "")),
        telemetry_degraded=_to_bool_text(row.get("telemetry_degraded")),
        tps_mean=float(row.get("tps_mean", 0.0) or 0.0),
        rps_mean=float(row.get("rps_mean", 0.0) or 0.0),
        ttft_p50_ms=float(row.get("ttft_p50_ms", 0.0) or 0.0),
        status=str(row.get("status", "")),
        eligible=eligible,
        blocker_signature=str(meta.get("blocker_signature") or ""),
        torch_device=str(torch_server.get("device") or ""),
        fallback_triggered=bool(torch_server.get("fallback_triggered", False)),
        runtime_device_fallback=bool(torch_server.get("runtime_device_fallback", False)),
        ttft_signal_source=ttft_signal_source,
        ttft_request_latency_fallback_count=ttft_fallback_count,
    )


def _plot_tps(frame: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    x = frame["run_name"].tolist()
    y = frame["tps_mean"].tolist()
    plt.plot(x, y, marker="o")
    plt.title("Torch ROCm Push - TPS (stage_b_recheck)")
    plt.ylabel("tps_mean")
    plt.xticks(rotation=25, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _plot_ttft(frame: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    x = frame["run_name"].tolist()
    y = frame["ttft_p50_ms"].tolist()
    plt.plot(x, y, marker="o")
    plt.title("Torch ROCm Push - TTFT p50 (stage_b_recheck)")
    plt.ylabel("ttft_p50_ms")
    plt.xticks(rotation=25, ha="right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _render_report(
    summary_path: Path,
    tps_png: Path,
    ttft_png: Path,
    records: list[StageBMetric],
    selected_run: str | None,
    phi3_status: str,
    phi3_blocker: str | None,
) -> str:
    lines: list[str] = []
    lines.append("# Torch ROCm Push Report")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc_iso()}`")
    lines.append("")
    lines.append("## Selection Summary")
    lines.append("")
    lines.append(f"- selected_tiny_run: `{selected_run or 'n/a'}`")
    lines.append(f"- phi3_confirm_status: `{phi3_status}`")
    lines.append(f"- phi3_blocker_signature: `{phi3_blocker or ''}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- summary_csv: `{summary_path}`")
    lines.append(f"- tps_plot: `{tps_png}`")
    lines.append(f"- ttft_plot: `{ttft_png}`")
    lines.append("")
    lines.append("## Tiny/Phi3 Stage-B Snapshot")
    lines.append("")
    lines.append("| run_name | profile_id | eligible | telemetry_degraded | tps_mean | rps_mean | ttft_p50_ms | ttft_signal_source | ttft_request_latency_fallback_count | device | fallback_triggered | runtime_device_fallback | blocker_signature |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- | --- | --- |")
    for item in records:
        lines.append(
            "| "
            + f"{item.run_name} | {item.profile_id} | {item.eligible} | {item.telemetry_degraded} | "
            + f"{item.tps_mean:.3f} | {item.rps_mean:.3f} | {item.ttft_p50_ms:.3f} | {item.ttft_signal_source} | {item.ttft_request_latency_fallback_count} | {item.torch_device} | "
            + f"{item.fallback_triggered} | {item.runtime_device_fallback} | {item.blocker_signature} |"
        )
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append(f"![TPS trend]({tps_png.name})")
    lines.append("")
    lines.append(f"![TTFT trend]({ttft_png.name})")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    run_names = [
        "r1_baseline",
        "r2_flash_only",
        "r3_compile_only",
        "r4_flash_compile",
        "r5_phi3_confirm",
    ]
    records: list[StageBMetric] = [extract_stage_b(name, runs_root / name) for name in run_names]
    summary_frame = pd.DataFrame([item.__dict__ for item in records])

    summary_csv = out_root / "torch_rocm_push_summary.csv"
    report_md = out_root / "torch_rocm_push_report.md"
    tps_png = out_root / "torch_rocm_push_tps.png"
    ttft_png = out_root / "torch_rocm_push_ttft.png"

    summary_frame.to_csv(summary_csv, index=False)
    _plot_tps(summary_frame, tps_png)
    _plot_ttft(summary_frame, ttft_png)

    push_meta = _read_json(runs_root / "meta" / "torch_push_pipeline_meta.json")
    tiny_selection = push_meta.get("tiny_selection") or {}
    phi3 = push_meta.get("phi3_confirmation") or {}
    selected_run = tiny_selection.get("selected_run_name")
    phi3_passed = bool(phi3.get("passed", False))
    phi3_status = "passed" if phi3_passed else "failed"
    phi3_blocker = phi3.get("blocker_signature")
    if not phi3_passed and not phi3_blocker:
        phi3_blocker = "torch_phi3_confirmation_failed"

    report_text = _render_report(
        summary_path=summary_csv,
        tps_png=tps_png,
        ttft_png=ttft_png,
        records=records,
        selected_run=selected_run,
        phi3_status=phi3_status,
        phi3_blocker=phi3_blocker,
    )
    report_md.write_text(report_text, encoding="utf-8")
    print(f"torch_rocm_push_report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

