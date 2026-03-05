#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from bench_ops import DEFAULT_METRICS_SCHEMA, DEFAULT_RUN_SCHEMA
from cmd_doctor_validate_report import (
    cmd_aiperf_adapt,
    cmd_doctor,
    cmd_report,
    cmd_validate,
    cmd_validate_run,
)
from cmd_compare import cmd_compare
from cmd_compare_runs import cmd_compare_runs
from cmd_expctl import cmd_expctl
from cmd_enginebench import cmd_enginebench
from cmd_servebench import cmd_servebench


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="780M-LLM-PerfLab benchmark control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Capture host + runtime fingerprint")
    doctor.add_argument("--out", required=True, help="Output directory or doctor.json path")
    doctor.add_argument("--run-id", default=None, help="Optional run id")
    doctor.set_defaults(func=cmd_doctor)

    validate = sub.add_parser("validate", help="Validate metrics JSONL against schema")
    validate.add_argument("--input", required=True, help="Path to metrics.jsonl")
    validate.add_argument(
        "--schema",
        default=str(DEFAULT_METRICS_SCHEMA),
        help="Schema path (default: schemas/metrics_v0.schema.json)",
    )
    validate.set_defaults(func=cmd_validate)

    validate_run = sub.add_parser("validate-run", help="Validate run.json against schema")
    validate_run.add_argument("--input", required=True, help="Path to run.json")
    validate_run.add_argument(
        "--schema",
        default=str(DEFAULT_RUN_SCHEMA),
        help="Schema path (default: schemas/run_v0.schema.json)",
    )
    validate_run.set_defaults(func=cmd_validate_run)

    report = sub.add_parser("report", help="Generate summary.csv + report.md from a run directory")
    report.add_argument("--input", required=True, help="Run directory containing metrics.jsonl")
    report.add_argument("--out", required=True, help="Output directory for report files")
    report.set_defaults(func=cmd_report)

    aiperf_adapt = sub.add_parser(
        "aiperf-adapt",
        help="Convert AIPerf profile_export.jsonl to replay JSONL",
    )
    aiperf_adapt.add_argument("--input", required=True, help="AIPerf profile_export.jsonl")
    aiperf_adapt.add_argument("--output", required=True, help="Replay JSONL output path")
    aiperf_adapt.add_argument(
        "--default-concurrency",
        type=int,
        default=1,
        help="Default concurrency if record row has no concurrency field",
    )
    aiperf_adapt.set_defaults(func=cmd_aiperf_adapt)

    enginebench = sub.add_parser("enginebench", help="Engine benchmark with llama-bench")
    enginebench.add_argument("--backend", required=True, help="Backend name, e.g. llama_bench")
    enginebench.add_argument("--backend-version", default="unknown", help="Backend version or commit")
    enginebench.add_argument("--workload", required=True, help="Engine workload YAML path")
    enginebench.add_argument("--model", required=True, help="Model path")
    enginebench.add_argument("--bench-bin", required=True, help="llama-bench binary path or command")
    enginebench.add_argument(
        "--bench-bin-args",
        default="",
        help="Arguments injected after bench binary and before auto args",
    )
    enginebench.add_argument(
        "--bench-extra-args",
        default="",
        help="Extra args appended after auto args",
    )
    enginebench.add_argument("--out", required=True, help="Output run directory")
    enginebench.add_argument("--run-id", default=None, help="Optional run id override")
    enginebench.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of best configs shown in report, sorted by tg_tps_mean",
    )
    enginebench.set_defaults(func=cmd_enginebench)

    servebench = sub.add_parser("servebench", help="Serving benchmark scaffold")
    servebench.add_argument("--backend", required=True, help="Backend name, e.g. llama_cpp")
    servebench.add_argument("--backend-version", default="unknown", help="Backend version or commit")
    servebench.add_argument("--workload", required=True, help="Workload YAML path")
    servebench.add_argument("--tool", default="aiperf", help="Serving benchmark tool name")
    servebench.add_argument("--out", required=True, help="Output run directory")
    servebench.add_argument("--run-id", default=None, help="Optional run id override")
    servebench.add_argument("--model", default=None, help="Model path for metadata / managed mode")
    servebench.add_argument("--mode", choices=["mock", "replay"], default="mock")
    servebench.add_argument("--tool-run-cmd", default=None, help="External tool command")
    servebench.add_argument("--tool-output-jsonl", default=None, help="Replay input jsonl path")
    servebench.add_argument("--server-mode", choices=["managed", "attach"], default="managed")
    servebench.add_argument("--server-bin", default=None, help="Server binary path or command")
    servebench.add_argument(
        "--server-bin-args",
        default="",
        help="Arguments injected after server binary and before auto args",
    )
    servebench.add_argument("--server-host", default="127.0.0.1", help="Server host")
    servebench.add_argument("--server-port", type=int, default=8080, help="Server port")
    servebench.add_argument("--server-health-timeout-sec", type=int, default=60, help="Health timeout")
    servebench.add_argument(
        "--server-health-interval-sec",
        type=float,
        default=1.0,
        help="Health check interval seconds",
    )
    servebench.add_argument(
        "--server-extra-args",
        default="",
        help="Extra args appended after auto args",
    )
    servebench.add_argument("--server-url", default=None, help="Attach base URL override")
    servebench.add_argument(
        "--mlc-mode",
        default="server",
        help="MLC serve mode for managed backend=mlc_* (default: server)",
    )
    servebench.add_argument(
        "--mlc-max-num-sequence",
        type=int,
        default=None,
        help="MLC override: max_num_sequence",
    )
    servebench.add_argument(
        "--mlc-max-total-seq-length",
        type=int,
        default=None,
        help="MLC override: max_total_seq_length",
    )
    servebench.add_argument(
        "--mlc-prefill-chunk-size",
        type=int,
        default=None,
        help="MLC override: prefill_chunk_size",
    )
    servebench.add_argument(
        "--mlc-device",
        default=None,
        help="MLC runtime device (for example: vulkan:0, cpu:0)",
    )
    servebench.add_argument(
        "--mlc-opt",
        default=None,
        help="MLC runtime optimization preset/flags passed to `mlc_llm serve --opt`",
    )
    servebench.add_argument(
        "--mlc-model-lib",
        default=None,
        help="Optional prebuilt MLC model library path passed to `mlc_llm serve --model-lib`",
    )
    servebench.set_defaults(func=cmd_servebench)

    compare = sub.add_parser(
        "compare",
        help="Run replay/attach serving comparison from scenario YAML",
    )
    compare.add_argument("--scenario", required=True, help="Compare scenario YAML path")
    compare.add_argument("--out", required=True, help="Output directory for compare artifacts")
    compare.add_argument("--run-id", default=None, help="Optional compare run id")
    compare.set_defaults(func=cmd_compare)

    compare_runs = sub.add_parser(
        "compare-runs",
        help="Generate serving comparison from two existing run directories",
    )
    compare_runs.add_argument("--baseline-run", required=True, help="Baseline run directory")
    compare_runs.add_argument("--candidate-run", required=True, help="Candidate run directory")
    compare_runs.add_argument("--out", required=True, help="Output directory for compare artifacts")
    compare_runs.add_argument("--baseline-label", default=None, help="Optional baseline label override")
    compare_runs.add_argument("--candidate-label", default=None, help="Optional candidate label override")
    compare_runs.add_argument(
        "--strict-aiperf-observed",
        action="store_true",
        help="Require both runs to have aiperf_adapt_meta with zero fallback counts",
    )
    compare_runs.add_argument(
        "--strict-aiperf-allow-fallback",
        default="",
        help=(
            "Comma-separated fallback keys allowed in strict mode "
            "(only used with --strict-aiperf-observed)"
        ),
    )
    compare_runs.add_argument("--run-id", default=None, help="Optional compare run id")
    compare_runs.set_defaults(func=cmd_compare_runs)

    expctl = sub.add_parser(
        "expctl",
        help="Experiment runner with matrix expansion + resource sampling",
    )
    expctl.add_argument("--spec", required=True, help="Experiment spec YAML path")
    expctl.add_argument("--out", required=True, help="Experiment output root")
    expctl.add_argument("--run-db", default=None, help="Optional sqlite run db path")
    expctl.add_argument("--resume", action="store_true", help="Resume completed runs from run db")
    expctl.add_argument("--max-workers", type=int, default=1, help="Reserved worker count (v1 serial only)")
    expctl.add_argument("--fail-fast", action="store_true", help="Stop after first failed run")
    expctl.set_defaults(func=cmd_expctl)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - error handling path
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
