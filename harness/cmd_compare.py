from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

from cmd_servebench import cmd_servebench
from core import commandline_from_argv, now_utc_iso, read_json, sha256_file, write_json

METRIC_SPECS = [
    ("ttft_p50_ms", "TTFT p50 (ms)", "lower"),
    ("ttft_p95_ms", "TTFT p95 (ms)", "lower"),
    ("itl_p50_ms", "ITL p50 (ms)", "lower"),
    ("itl_p95_ms", "ITL p95 (ms)", "lower"),
    ("tps_mean", "TPS mean", "higher"),
    ("rps_mean", "RPS mean", "higher"),
]
LATENCY_TIE_THRESHOLD_PCT = 3.0
THROUGHPUT_TIE_THRESHOLD_PCT = 2.0
OVERALL_RULE = "equal_vote"


def _require_field(payload: dict[str, Any], key: str, prefix: str = "") -> Any:
    value = payload.get(key)
    if value in (None, ""):
        raise ValueError(f"scenario:{prefix}{key} missing")
    return value


def _load_compare_scenario(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"scenario file not found: {path}")
    if yaml is None:
        raise RuntimeError(
            "pyyaml is required for scenario parsing. Install: python -m pip install pyyaml"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenario:root must be an object")

    name = str(_require_field(data, "name"))
    workload = str(_require_field(data, "workload"))

    baseline_raw = _require_field(data, "baseline")
    candidate_raw = _require_field(data, "candidate")
    if not isinstance(baseline_raw, dict):
        raise ValueError("scenario:baseline must be an object")
    if not isinstance(candidate_raw, dict):
        raise ValueError("scenario:candidate must be an object")

    for side_name, side in (("baseline", baseline_raw), ("candidate", candidate_raw)):
        _require_field(side, "label", f"{side_name}.")
        _require_field(side, "backend", f"{side_name}.")
        _require_field(side, "backend_version", f"{side_name}.")
        _require_field(side, "model", f"{side_name}.")
        _require_field(side, "server_url", f"{side_name}.")
        _require_field(side, "tool_output_jsonl", f"{side_name}.")

    return {
        "name": name,
        "workload": workload,
        "baseline": baseline_raw,
        "candidate": candidate_raw,
    }


def _host_port_from_url(raw: str) -> tuple[str, int]:
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"invalid server_url: {raw}")
    if parsed.port is not None:
        return parsed.hostname, int(parsed.port)
    if parsed.scheme == "http":
        return parsed.hostname, 80
    if parsed.scheme == "https":
        return parsed.hostname, 443
    raise ValueError(f"invalid server_url (unsupported scheme): {raw}")


def _build_servebench_args(
    *,
    side: dict[str, Any],
    workload_path: str,
    out_dir: Path,
    run_id: str,
) -> argparse.Namespace:
    server_url = str(side["server_url"])
    server_host, server_port = _host_port_from_url(server_url)
    return argparse.Namespace(
        backend=str(side["backend"]),
        backend_version=str(side["backend_version"]),
        workload=workload_path,
        tool="aiperf",
        out=str(out_dir),
        run_id=run_id,
        model=str(side["model"]),
        mode="replay",
        tool_run_cmd=None,
        tool_output_jsonl=str(side["tool_output_jsonl"]),
        server_mode="attach",
        server_bin=None,
        server_bin_args="",
        server_host=server_host,
        server_port=server_port,
        server_health_timeout_sec=60,
        server_health_interval_sec=1.0,
        server_extra_args="",
        server_url=server_url,
        mlc_mode="server",
        mlc_max_num_sequence=None,
        mlc_max_total_seq_length=None,
        mlc_prefill_chunk_size=None,
    )


def read_serving_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"summary file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one summary row")
    row = rows[0]
    parsed: dict[str, float] = {}
    for key, _, _ in METRIC_SPECS:
        if key not in row or row[key] in (None, ""):
            raise ValueError(f"{path}: missing field `{key}`")
        try:
            parsed[key] = float(row[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: field `{key}` must be numeric") from exc
    return parsed


def _judge_better(direction: str, delta_abs: float) -> str:
    eps = 1e-12
    if abs(delta_abs) <= eps:
        return "tie"
    if direction == "lower":
        return "candidate better" if delta_abs < 0 else "baseline better"
    return "candidate better" if delta_abs > 0 else "baseline better"


def _threshold_pct_for_direction(direction: str) -> float:
    if direction == "lower":
        return LATENCY_TIE_THRESHOLD_PCT
    return THROUGHPUT_TIE_THRESHOLD_PCT


def _judge_winner_with_threshold(
    *,
    direction: str,
    baseline_value: float,
    delta_abs: float,
    delta_pct: float,
) -> tuple[str, float]:
    threshold_pct = _threshold_pct_for_direction(direction)
    eps = 1e-12
    if baseline_value != 0 and not math.isnan(delta_pct):
        if abs(delta_pct) < threshold_pct:
            return "tie", threshold_pct
    if abs(delta_abs) <= eps:
        return "tie", threshold_pct
    if direction == "lower":
        return ("candidate better" if delta_abs < 0 else "baseline better"), threshold_pct
    return ("candidate better" if delta_abs > 0 else "baseline better"), threshold_pct


def build_comparison_metric_rows(
    *,
    baseline_summary: dict[str, float],
    candidate_summary: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    votes = {"candidate": 0, "baseline": 0, "tie": 0}
    per_metric_winner: dict[str, str] = {}
    for key, display_name, direction in METRIC_SPECS:
        baseline_value = baseline_summary[key]
        candidate_value = candidate_summary[key]
        delta_abs = candidate_value - baseline_value
        delta_pct = float("nan")
        if baseline_value != 0:
            delta_pct = ((candidate_value / baseline_value) - 1.0) * 100.0
        winner, threshold_pct = _judge_winner_with_threshold(
            direction=direction,
            baseline_value=baseline_value,
            delta_abs=delta_abs,
            delta_pct=delta_pct,
        )
        if winner == "candidate better":
            votes["candidate"] += 1
        elif winner == "baseline better":
            votes["baseline"] += 1
        else:
            votes["tie"] += 1
        per_metric_winner[key] = winner
        rows.append(
            {
                "metric": key,
                "display": display_name,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "better": _judge_better(direction, delta_abs),
                "threshold_pct": threshold_pct,
                "winner": winner,
            }
        )

    overall_winner = "tie"
    if votes["candidate"] > votes["baseline"]:
        overall_winner = "candidate better"
    elif votes["baseline"] > votes["candidate"]:
        overall_winner = "baseline better"
    overall = {
        "winner": overall_winner,
        "vote_counts": {"candidate": votes["candidate"], "baseline": votes["baseline"], "tie": votes["tie"]},
        "rule": OVERALL_RULE,
        "thresholds_pct": {
            "latency": LATENCY_TIE_THRESHOLD_PCT,
            "throughput": THROUGHPUT_TIE_THRESHOLD_PCT,
        },
        "per_metric_winner": per_metric_winner,
    }
    return rows, overall


def write_comparison_outputs(
    *,
    out_dir: Path,
    run_id: str,
    baseline_label: str,
    candidate_label: str,
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    rows: Sequence[dict[str, Any]],
    overall: dict[str, Any],
    source_label: str,
    source_value: str,
    source_sha256: str | None,
    repro_commands: Sequence[str],
) -> tuple[Path, Path]:
    summary_csv = out_dir / "comparison_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "baseline",
                "candidate",
                "delta_abs",
                "delta_pct",
                "better",
                "threshold_pct",
                "winner",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "metric": row["metric"],
                    "baseline": round(float(row["baseline"]), 6),
                    "candidate": round(float(row["candidate"]), 6),
                    "delta_abs": round(float(row["delta_abs"]), 6),
                    "delta_pct": "nan"
                    if math.isnan(float(row["delta_pct"]))
                    else round(float(row["delta_pct"]), 6),
                    "better": row["better"],
                    "threshold_pct": round(float(row["threshold_pct"]), 3),
                    "winner": row["winner"],
                }
            )

    report_md = out_dir / "comparison_report.md"
    vote_counts = overall["vote_counts"]
    lines = [
        "# ServingBench Comparison Report",
        "",
        "## Inputs",
        f"- run_id: `{run_id}`",
        f"- {source_label}: `{source_value}`",
    ]
    if source_sha256 is not None:
        lines.append(f"- {source_label}_sha256: `{source_sha256}`")
    lines.extend(
        [
            f"- baseline: `{baseline_label}` -> `{baseline_run_dir}`",
            f"- candidate: `{candidate_label}` -> `{candidate_run_dir}`",
            f"- generated_at_utc: `{now_utc_iso()}`",
            "",
            "## Overall Winner",
            f"- winner: `{overall['winner']}`",
            f"- votes: candidate={vote_counts['candidate']}, baseline={vote_counts['baseline']}, tie={vote_counts['tie']}",
            f"- rule: `{overall['rule']}`",
            f"- thresholds: latency={LATENCY_TIE_THRESHOLD_PCT}%, throughput={THROUGHPUT_TIE_THRESHOLD_PCT}%",
            "",
            "## Metric Deltas",
            "",
            "| Metric | Baseline | Candidate | delta_abs | delta_pct | Better | Winner |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        delta_pct_text = "nan" if math.isnan(float(row["delta_pct"])) else f"{row['delta_pct']:.3f}%"
        lines.append(
            f"| {row['display']} | {row['baseline']:.3f} | {row['candidate']:.3f} | "
            f"{row['delta_abs']:.3f} | {delta_pct_text} | {row['better']} | {row['winner']} |"
        )
    lines.extend(["", "## Reproduction", "```bash"])
    lines.extend(str(cmd) for cmd in repro_commands)
    lines.extend(["```", ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return summary_csv, report_md


def cmd_compare(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    scenario = _load_compare_scenario(scenario_path)
    scenario_sha256 = sha256_file(scenario_path)
    run_id = args.run_id or (
        f"compare_{now_utc_iso().replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}"
    )

    baseline = scenario["baseline"]
    candidate = scenario["candidate"]
    baseline_label = str(baseline["label"])
    candidate_label = str(candidate["label"])
    if baseline_label == candidate_label:
        raise ValueError("scenario:baseline.label and scenario:candidate.label must be different")

    baseline_out = runs_dir / baseline_label
    candidate_out = runs_dir / candidate_label

    baseline_args = _build_servebench_args(
        side=baseline,
        workload_path=str(scenario["workload"]),
        out_dir=baseline_out,
        run_id=f"{run_id}_{baseline_label}",
    )
    candidate_args = _build_servebench_args(
        side=candidate,
        workload_path=str(scenario["workload"]),
        out_dir=candidate_out,
        run_id=f"{run_id}_{candidate_label}",
    )

    if cmd_servebench(baseline_args) != 0:
        raise RuntimeError("baseline servebench failed")
    if cmd_servebench(candidate_args) != 0:
        raise RuntimeError("candidate servebench failed")

    baseline_summary = read_serving_summary(baseline_out / "summary.csv")
    candidate_summary = read_serving_summary(candidate_out / "summary.csv")
    rows, overall = build_comparison_metric_rows(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
    )

    comparison_summary, comparison_report = write_comparison_outputs(
        out_dir=out_dir,
        run_id=run_id,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_run_dir=baseline_out,
        candidate_run_dir=candidate_out,
        rows=rows,
        overall=overall,
        source_label="scenario",
        source_value=str(scenario_path),
        source_sha256=scenario_sha256,
        repro_commands=[f"python harness/benchctl.py compare --scenario {scenario_path} --out {out_dir}"],
    )

    baseline_run_json = read_json(baseline_out / "run.json")
    candidate_run_json = read_json(candidate_out / "run.json")
    write_json(
        out_dir / "compare_meta.json",
        {
            "run_id": run_id,
            "timestamp_utc": now_utc_iso(),
            "scenario_path": str(scenario_path),
            "scenario_sha256": scenario_sha256,
            "commandline": commandline_from_argv(),
            "runs": {
                "baseline": {"label": baseline_label, "dir": str(baseline_out)},
                "candidate": {"label": candidate_label, "dir": str(candidate_out)},
            },
            "commands": [
                f"python harness/benchctl.py compare --scenario {scenario_path} --out {out_dir}",
                baseline_run_json.get("commandline"),
                candidate_run_json.get("commandline"),
            ],
            "outputs": {
                "comparison_summary_csv": str(comparison_summary),
                "comparison_report_md": str(comparison_report),
            },
            "overall": overall,
        },
    )

    print(f"[compare] wrote {comparison_summary}")
    print(f"[compare] wrote {comparison_report}")
    print(f"[compare] wrote {out_dir / 'compare_meta.json'}")
    return 0
