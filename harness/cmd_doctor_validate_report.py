from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import ops as ops
from adapters.aiperf_adapter import adapt_aiperf_profile_export_jsonl
from core import commandline_from_argv


def cmd_doctor(args: argparse.Namespace) -> int:
    out_path = ops.resolve_doctor_output(args.out)
    run_id = args.run_id or f"doctor_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    payload = ops.build_doctor_payload(
        run_id=run_id,
        commandline=commandline_from_argv(),
        backend_name="doctor",
        backend_version="v0",
        workload_path="unknown",
        workload_sha256="00000000",
    )
    payload["doctor"]["git_available"] = shutil.which("git") is not None
    ops.write_json(out_path, payload)
    print(f"[doctor] wrote {out_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    schema_path = Path(args.schema)
    count = ops.validate_metrics_file(input_path, schema_path)
    print(f"[validate] ok: {count} rows ({input_path})")
    return 0


def cmd_validate_run(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    schema_path = Path(args.schema)
    ops.validate_json_file(input_path, schema_path)
    print(f"[validate-run] ok: {input_path}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    summary_path, report_path = ops.generate_report(input_dir, out_dir)
    print(f"[report] wrote {summary_path}")
    print(f"[report] wrote {report_path}")
    return 0


def cmd_aiperf_adapt(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    stats = adapt_aiperf_profile_export_jsonl(
        input_path=input_path,
        output_path=output_path,
        default_concurrency=int(args.default_concurrency),
    )
    print(f"[aiperf-adapt] wrote {output_path}")
    print(
        "[aiperf-adapt] input_rows={input_rows} output_rows={output_rows} skipped_rows={skipped_rows}".format(
            **stats
        )
    )
    return 0
