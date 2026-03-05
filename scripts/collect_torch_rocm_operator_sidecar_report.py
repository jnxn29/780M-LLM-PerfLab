#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and summarize torch_rocm operator sidecar runs.")
    parser.add_argument(
        "--runs-root",
        default="<repo-root>/results/r14_16_torch_rocm_operator_sidecar_r1/",
        help="Root directory containing operator sidecar sub-runs (e.g. O1..O4).",
    )
    parser.add_argument(
        "--out-root",
        default="reports/perf_timeline",
        help="Output directory for summary/report/plots.",
    )
    parser.add_argument(
        "--meta",
        default="",
        help="Optional operator sidecar meta file. Defaults to <runs-root>/operator_sidecar_meta.json.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    for encoding in ("utf-8-sig", "utf-8", "utf-16"):
        try:
            return json.loads(path.read_text(encoding=encoding))
        except Exception:
            continue
    return {}


def _read_leaderboard(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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
    if adapt_meta:
        return "time_to_first_token", 0
    return "unknown", 0


def _stage_b_row(leaderboard: pd.DataFrame) -> pd.Series | None:
    if leaderboard.empty or "stage" not in leaderboard.columns:
        return None
    stage_b = leaderboard[(leaderboard["stage"] == "stage_b_recheck") & (leaderboard["status"] == "success")]
    if stage_b.empty:
        return None
    stage_b = stage_b.sort_values(
        by=["telemetry_degraded", "tps_mean", "rps_mean", "ttft_p50_ms"],
        ascending=[True, False, False, True],
    )
    return stage_b.iloc[0]


def _extract_record(exp: dict[str, Any], run_root: Path) -> dict[str, Any]:
    pipeline_meta = _read_json(run_root / "pipeline_meta.json")
    leaderboard = _read_leaderboard(run_root / "leaderboard.csv")
    row = _stage_b_row(leaderboard)
    torch_server = pipeline_meta.get("torch_server") or {}
    op_opt = torch_server.get("operator_optimizations") or {}
    sdpa_state = op_opt.get("sdpa_kernel_state") or {}
    cfg = pipeline_meta.get("config") or {}

    configured_sdpa = str(exp.get("sdpa_profile") or cfg.get("torch_sdpa_kernel_profile") or "")
    effective_sdpa = str(sdpa_state.get("profile_effective") or cfg.get("torch_sdpa_kernel_profile") or configured_sdpa)
    compile_requested = bool(exp.get("compile_requested", cfg.get("torch_enable_compile", False)))
    compile_enabled = bool(op_opt.get("torch_compile_enabled", False))
    compile_mode = str(op_opt.get("torch_compile_mode") or exp.get("compile_mode") or cfg.get("torch_compile_mode") or "")
    compile_error = str(op_opt.get("torch_compile_error") or "")
    blocker_signature = str(pipeline_meta.get("blocker_signature") or "")
    fallback_triggered = bool(torch_server.get("fallback_triggered", False))
    runtime_device_fallback = bool(torch_server.get("runtime_device_fallback", False))
    torch_device = str(torch_server.get("device") or "")

    if row is None:
        return {
            "experiment_id": str(exp.get("experiment_id") or run_root.name),
            "sdpa_profile_configured": configured_sdpa,
            "sdpa_profile_effective": effective_sdpa,
            "compile_requested": compile_requested,
            "compile_enabled": compile_enabled,
            "compile_mode": compile_mode,
            "compile_error": compile_error,
            "ttft_p50_ms": 0.0,
            "tps_mean": 0.0,
            "eligible": False,
            "fallback_triggered": fallback_triggered,
            "runtime_device_fallback": runtime_device_fallback,
            "ttft_signal_source": "unknown",
            "ttft_request_latency_fallback_count": 0,
            "status": "missing_stage_b",
            "blocker_signature": blocker_signature if blocker_signature else "stage_b_missing",
            "torch_device": torch_device,
            "run_root": str(run_root),
        }

    run_dir = Path(str(row.get("run_dir", "") or ""))
    if not run_dir.is_absolute():
        run_dir = (run_root / run_dir).resolve()
    ttft_signal_source, ttft_fallback_count = _resolve_ttft_signal(run_dir)
    eligible = (
        bool(pipeline_meta.get("artifacts_ready", False))
        and torch_device == "cuda"
        and not runtime_device_fallback
        and not fallback_triggered
    )
    return {
        "experiment_id": str(exp.get("experiment_id") or run_root.name),
        "sdpa_profile_configured": configured_sdpa,
        "sdpa_profile_effective": effective_sdpa,
        "compile_requested": compile_requested,
        "compile_enabled": compile_enabled,
        "compile_mode": compile_mode,
        "compile_error": compile_error,
        "ttft_p50_ms": _to_float(row.get("ttft_p50_ms")),
        "tps_mean": _to_float(row.get("tps_mean")),
        "eligible": bool(eligible),
        "fallback_triggered": fallback_triggered,
        "runtime_device_fallback": runtime_device_fallback,
        "ttft_signal_source": ttft_signal_source,
        "ttft_request_latency_fallback_count": ttft_fallback_count,
        "status": str(row.get("status", "")),
        "blocker_signature": blocker_signature,
        "torch_device": torch_device,
        "run_root": str(run_root),
    }


def _plot(frame: pd.DataFrame, y_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(10, 4.5))
    if not frame.empty:
        x = frame["experiment_id"].tolist()
        y = frame[y_col].tolist()
        plt.plot(x, y, marker="o")
        plt.xticks(rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(y_col)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()


def _build_recommendation(frame: pd.DataFrame, baseline_r5_tps: float) -> dict[str, Any]:
    tps_guardrail = max(0.0, 0.95 * baseline_r5_tps)
    qualified = frame[
        (frame["eligible"] == True)  # noqa: E712
        & (frame["fallback_triggered"] == False)  # noqa: E712
        & (frame["runtime_device_fallback"] == False)  # noqa: E712
        & (frame["ttft_signal_source"] != "request_latency_fallback")
        & (frame["tps_mean"] >= tps_guardrail)
    ]

    recommendation: dict[str, Any] = {
        "generated_at_utc": now_utc_iso(),
        "baseline_r5_tps": baseline_r5_tps,
        "tps_guardrail": tps_guardrail,
        "rule": {
            "hard_constraints": [
                "eligible == true",
                "fallback_triggered == false",
                "runtime_device_fallback == false",
                "ttft_signal_source != request_latency_fallback",
                "tps_mean >= 0.95 * baseline_r5_tps",
            ],
            "selection": "min ttft_p50_ms among qualified",
        },
        "candidate_count": int(frame.shape[0]),
        "qualified_count": int(qualified.shape[0]),
        "selected_experiment_id": None,
        "selected": None,
        "qualified_experiments": qualified["experiment_id"].tolist(),
    }
    if not qualified.empty:
        selected = qualified.sort_values(by=["ttft_p50_ms", "tps_mean"], ascending=[True, False]).iloc[0]
        recommendation["selected_experiment_id"] = str(selected["experiment_id"])
        recommendation["selected"] = {
            "experiment_id": str(selected["experiment_id"]),
            "ttft_p50_ms": float(selected["ttft_p50_ms"]),
            "tps_mean": float(selected["tps_mean"]),
            "sdpa_profile_effective": str(selected["sdpa_profile_effective"]),
            "compile_requested": bool(selected["compile_requested"]),
            "compile_enabled": bool(selected["compile_enabled"]),
            "compile_mode": str(selected["compile_mode"]),
        }
    return recommendation


def _render_report(
    summary_csv: Path,
    tps_png: Path,
    ttft_png: Path,
    recommendation_json: Path,
    frame: pd.DataFrame,
    recommendation: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# Torch ROCm Operator Sidecar Report")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc_iso()}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- summary_csv: `{summary_csv}`")
    lines.append(f"- recommendation_json: `{recommendation_json}`")
    lines.append(f"- tps_plot: `{tps_png}`")
    lines.append(f"- ttft_plot: `{ttft_png}`")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- baseline_r5_tps: `{recommendation['baseline_r5_tps']:.6f}`")
    lines.append(f"- tps_guardrail: `{recommendation['tps_guardrail']:.6f}`")
    lines.append(f"- selected_experiment_id: `{recommendation.get('selected_experiment_id') or ''}`")
    lines.append("")
    lines.append("## Sidecar Snapshot")
    lines.append("")
    lines.append("| experiment_id | sdpa_profile_configured | sdpa_profile_effective | compile_requested | compile_enabled | compile_mode | ttft_p50_ms | tps_mean | eligible | ttft_signal_source | ttft_request_latency_fallback_count | fallback_triggered | runtime_device_fallback |")
    lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- | --- |")
    for _, row in frame.iterrows():
        lines.append(
            "| "
            + f"{row['experiment_id']} | {row['sdpa_profile_configured']} | {row['sdpa_profile_effective']} | "
            + f"{row['compile_requested']} | {row['compile_enabled']} | {row['compile_mode']} | "
            + f"{float(row['ttft_p50_ms']):.3f} | {float(row['tps_mean']):.3f} | {row['eligible']} | "
            + f"{row['ttft_signal_source']} | {int(row['ttft_request_latency_fallback_count'])} | "
            + f"{row['fallback_triggered']} | {row['runtime_device_fallback']} |"
        )
    lines.append("")
    lines.append("## Charts")
    lines.append("")
    lines.append(f"![Operator Sidecar TPS]({tps_png.name})")
    lines.append("")
    lines.append(f"![Operator Sidecar TTFT]({ttft_png.name})")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta).resolve() if str(args.meta).strip() else (runs_root / "operator_sidecar_meta.json")
    sidecar_meta = _read_json(meta_path)
    experiments = sidecar_meta.get("experiments") or []

    if not experiments:
        experiments = [
            {"experiment_id": child.name}
            for child in sorted(runs_root.iterdir())
            if child.is_dir() and child.name.upper().startswith("O")
        ]

    records: list[dict[str, Any]] = []
    for exp in experiments:
        exp_id = str(exp.get("experiment_id") or "")
        if not exp_id:
            continue
        run_root = runs_root / exp_id
        records.append(_extract_record(exp=exp, run_root=run_root))

    frame = pd.DataFrame(records)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "experiment_id",
                "sdpa_profile_configured",
                "sdpa_profile_effective",
                "compile_requested",
                "compile_enabled",
                "compile_mode",
                "compile_error",
                "ttft_p50_ms",
                "tps_mean",
                "eligible",
                "fallback_triggered",
                "runtime_device_fallback",
                "ttft_signal_source",
                "ttft_request_latency_fallback_count",
                "status",
                "blocker_signature",
                "torch_device",
                "run_root",
            ]
        )
    else:
        frame = frame.sort_values(by=["experiment_id"]).reset_index(drop=True)

    summary_csv = out_root / "operator_sidecar_summary.csv"
    report_md = out_root / "operator_sidecar_report.md"
    tps_png = out_root / "operator_sidecar_tps.png"
    ttft_png = out_root / "operator_sidecar_ttft.png"
    recommendation_json = out_root / "promotion_recommendation.json"

    frame.to_csv(summary_csv, index=False)
    _plot(frame, y_col="tps_mean", title="Torch ROCm Operator Sidecar - TPS", out_path=tps_png)
    _plot(frame, y_col="ttft_p50_ms", title="Torch ROCm Operator Sidecar - TTFT p50", out_path=ttft_png)

    baseline_r5_tps = _to_float(sidecar_meta.get("baseline_r5_tps"), 0.0)
    recommendation = _build_recommendation(frame=frame, baseline_r5_tps=baseline_r5_tps)
    recommendation_json.write_text(json.dumps(recommendation, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep a copy with run artifacts for local traceability.
    (runs_root / "promotion_recommendation.json").write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_text = _render_report(
        summary_csv=summary_csv,
        tps_png=tps_png,
        ttft_png=ttft_png,
        recommendation_json=recommendation_json,
        frame=frame,
        recommendation=recommendation,
    )
    report_md.write_text(report_text, encoding="utf-8")
    print(f"operator_sidecar_report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

