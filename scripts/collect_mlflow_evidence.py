#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from integrations.mlflow_logger import log_mlflow_run  # noqa: E402


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log non-ORT tuning evidence to MLflow and write local evidence files.")
    parser.add_argument(
        "--non-ort-summary-csv",
        default="reports/perf_timeline/non_ort_tuning_summary.csv",
        help="non_ort_tuning_summary.csv path",
    )
    parser.add_argument(
        "--non-ort-gate-json",
        default="",
        help="Optional non_ort_release_gate.json for explicit selected candidate",
    )
    parser.add_argument("--tracking-uri", default="file:reports/mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="780m_non_ort_release", help="MLflow experiment name")
    parser.add_argument("--run-name-prefix", default="release_evidence", help="MLflow run name prefix")
    parser.add_argument("--out-root", default="reports/perf_timeline", help="Output directory for evidence files")
    return parser.parse_args()


def parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _load_gate(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _aggregate_candidates(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        name = str(row.get("candidate_run_name") or "").strip()
        if not name:
            continue
        by_name.setdefault(name, []).append(row)
    aggregated: list[dict[str, Any]] = []
    for name, items in by_name.items():
        ttfts = [parse_float(item.get("candidate_ttft_ms")) for item in items]
        tpses = [parse_float(item.get("candidate_tps")) for item in items]
        improves = [parse_float(item.get("ttft_improve_pct")) for item in items]
        ratios = [parse_float(item.get("tps_ratio")) for item in items]
        hard = all(parse_bool(item.get("hard_gate")) for item in items)
        strict = all(parse_bool(item.get("strict_gate_pass")) for item in items)
        aggregated.append(
            {
                "candidate_run_name": name,
                "candidate_run_root": str(items[0].get("candidate_run_root") or ""),
                "hard_gate": hard,
                "strict_gate_pass": strict,
                "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else float("nan"),
                "avg_tps": sum(tpses) / len(tpses) if tpses else float("nan"),
                "avg_ttft_improve_pct": sum(improves) / len(improves) if improves else float("nan"),
                "avg_tps_ratio": sum(ratios) / len(ratios) if ratios else float("nan"),
            }
        )
    aggregated.sort(
        key=lambda row: (
            row["strict_gate_pass"] is not True,
            row["hard_gate"] is not True,
            row["avg_ttft_ms"],
            -row["avg_tps"],
            row["candidate_run_name"],
        )
    )
    return aggregated


def _select_candidate(aggregated: list[dict[str, Any]], gate_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    selected_name = ""
    if gate_payload:
        selected_name = str(gate_payload.get("selected_candidate_run_name") or "").strip()
    if selected_name:
        for row in aggregated:
            if row["candidate_run_name"] == selected_name:
                return row
    for row in aggregated:
        if row["strict_gate_pass"] is True:
            return row
    return aggregated[0] if aggregated else None


def _render_md(payload: dict[str, Any], out_json: Path) -> str:
    lines: list[str] = []
    lines.append("# MLflow Evidence")
    lines.append("")
    lines.append(f"generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append(f"json: `{out_json}`")
    lines.append("")
    lines.append(f"- mlflow_used: `{payload['mlflow_used']}`")
    lines.append(f"- mlflow_error: `{payload['mlflow_error']}`")
    lines.append(f"- selected_candidate_run_name: `{payload['selected_candidate_run_name']}`")
    lines.append(f"- selected_candidate_run_root: `{payload['selected_candidate_run_root']}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    lines.append(f"| avg_ttft_ms | {payload['metrics'].get('avg_ttft_ms', 0.0):.6f} |")
    lines.append(f"| avg_tps | {payload['metrics'].get('avg_tps', 0.0):.6f} |")
    lines.append(f"| avg_ttft_improve_pct | {payload['metrics'].get('avg_ttft_improve_pct', 0.0):.6f} |")
    lines.append(f"| avg_tps_ratio | {payload['metrics'].get('avg_tps_ratio', 0.0):.6f} |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary_csv = Path(args.non_ort_summary_csv).resolve()
    if not summary_csv.exists():
        print(f"error: missing file: {summary_csv}", file=sys.stderr)
        return 2

    gate_payload: dict[str, Any] | None = None
    if str(args.non_ort_gate_json).strip():
        gate_path = Path(args.non_ort_gate_json).resolve()
        if not gate_path.exists():
            print(f"error: missing file: {gate_path}", file=sys.stderr)
            return 2
        gate_payload = _load_gate(gate_path)

    rows = _load_summary(summary_csv)
    aggregated = _aggregate_candidates(rows)
    selected = _select_candidate(aggregated, gate_payload)
    if selected is None:
        print("error: no candidate rows found in non-ORT summary", file=sys.stderr)
        return 2

    run_name = f"{args.run_name_prefix}_{selected['candidate_run_name']}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    params = {
        "selected_candidate_run_name": selected["candidate_run_name"],
        "selected_candidate_run_root": selected["candidate_run_root"],
        "strict_gate_pass": selected["strict_gate_pass"],
        "hard_gate": selected["hard_gate"],
        "source_summary_csv": str(summary_csv),
    }
    metrics = {
        "avg_ttft_ms": float(selected["avg_ttft_ms"]),
        "avg_tps": float(selected["avg_tps"]),
        "avg_ttft_improve_pct": float(selected["avg_ttft_improve_pct"]),
        "avg_tps_ratio": float(selected["avg_tps_ratio"]),
    }
    used, error = log_mlflow_run(
        enabled=True,
        required=False,
        uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=run_name,
        params=params,
        metrics=metrics,
        artifacts_dir=summary_csv.parent,
    )

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "mlflow_evidence.json"
    out_md = out_root / "mlflow_evidence.md"
    payload = {
        "generated_at_utc": now_utc_iso(),
        "tracking_uri": args.tracking_uri,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "selected_candidate_run_name": selected["candidate_run_name"],
        "selected_candidate_run_root": selected["candidate_run_root"],
        "mlflow_used": bool(used),
        "mlflow_error": error,
        "metrics": metrics,
        "source_summary_csv": str(summary_csv),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_render_md(payload, out_json), encoding="utf-8")

    print(f"mlflow_evidence_json: {out_json}")
    print(f"mlflow_evidence_md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
