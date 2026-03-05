from __future__ import annotations

import argparse
import shlex
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import ops as ops
from adapters.llama_bench_adapter import extract_llama_bench_metric
from core import commandline_from_argv, write_jsonl


def cmd_enginebench(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_tool_output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    bench_raw_dir = raw_dir / "llama_bench"
    bench_raw_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id or f"enginebench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    workload_path = Path(args.workload)
    workload = ops.load_workload(workload_path)

    if "engine" not in workload:
        raise ValueError("engine workload requires `engine` section with sweep fields")

    if not args.bench_bin:
        raise ValueError("enginebench requires --bench-bin")
    bench_tokens = ops.split_args(args.bench_bin)
    if not bench_tokens:
        raise ValueError("enginebench requires non-empty --bench-bin")
    bench_program = bench_tokens[0]
    if ("/" in bench_program or "\\" in bench_program) and not Path(bench_program).exists():
        raise FileNotFoundError(f"bench binary not found: {bench_program}")

    model_raw = str(args.model)
    if "\x00" in model_raw:
        raise ValueError("enginebench --model contains NUL byte")
    model_path = Path(model_raw).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"enginebench requires --model as existing file: {model_path}")
    model_sha256 = ops.maybe_sha256_model(model_path)

    backend_name = args.backend
    backend_version = args.backend_version
    repetitions = int(workload["repetitions"])
    if repetitions < 1:
        raise ValueError("workload.repetitions must be >= 1")

    engine_cfg = workload["engine"]
    combinations: list[dict[str, int]] = []
    for threads in engine_cfg["threads"]:
        for batch in engine_cfg["batch"]:
            for ubatch in engine_cfg["ubatch"]:
                for ngl in engine_cfg["ngl"]:
                    combinations.append(
                        {
                            "threads": int(threads),
                            "batch": int(batch),
                            "ubatch": int(ubatch),
                            "ngl": int(ngl),
                        }
                    )
    if not combinations:
        raise ValueError("engine sweep has no combinations")

    doctor_payload = ops.build_doctor_payload(
        run_id=run_id,
        commandline=commandline_from_argv(),
        backend_name=backend_name,
        backend_version=backend_version,
        workload_path=str(workload_path),
        workload_sha256=workload["sha256"],
    )
    ops.write_json(out_dir / "doctor.json", doctor_payload)

    metrics_rows: list[dict[str, Any]] = []
    command_rows: list[dict[str, Any]] = []
    start = datetime.now(timezone.utc).replace(microsecond=0)
    git_commit = ops.current_git_commit()
    host = doctor_payload["system"]

    for index, combo in enumerate(combinations):
        per_test: dict[str, dict[str, float]] = {}
        for test_kind in ("pp", "tg", "pg"):
            command = ops.build_llama_bench_command(
                args=args,
                model_path=model_path,
                repetitions=repetitions,
                threads=combo["threads"],
                batch=combo["batch"],
                ubatch=combo["ubatch"],
                ngl=combo["ngl"],
                prompt_tokens=int(workload["prompt_tokens"]),
                output_tokens=int(workload["output_tokens"]),
                test_kind=test_kind,
            )

            stdout_path = bench_raw_dir / f"combo_{index + 1}_{test_kind}.jsonl"
            stderr_path = bench_raw_dir / f"combo_{index + 1}_{test_kind}.stderr.log"
            ops.run_llama_bench_command(command, stdout_path, stderr_path)
            metric = extract_llama_bench_metric(stdout_path, test_kind)
            per_test[test_kind] = metric

            command_rows.append(
                {
                    "combo_index": index + 1,
                    "test_kind": test_kind,
                    "command": " ".join(shlex.quote(token) for token in command),
                    "stdout_path": str(stdout_path),
                    "stderr_path": str(stderr_path),
                }
            )

        metrics_rows.append(
            {
                "schema_version": "metrics_v0",
                "run_id": run_id,
                "timestamp_utc": ops.now_utc_iso(start + timedelta(seconds=index)),
                "git_commit": git_commit,
                "track": "engine",
                "backend": {"name": backend_name, "version": backend_version},
                "model": {"path": str(model_path), "sha256": model_sha256},
                "workload": {
                    "name": workload["name"],
                    "path": workload["path"],
                    "sha256": workload["sha256"],
                },
                "system": host,
                "request": {
                    "id": f"{run_id}-{index + 1}",
                    "prompt_tokens": int(workload["prompt_tokens"]),
                    "output_tokens": int(workload["output_tokens"]),
                },
                "config": {
                    **combo,
                    "repetitions": repetitions,
                },
                "metrics": {
                    "pp_tps_mean": per_test["pp"]["mean"],
                    "pp_tps_stddev": per_test["pp"]["stddev"],
                    "tg_tps_mean": per_test["tg"]["mean"],
                    "tg_tps_stddev": per_test["tg"]["stddev"],
                    "pg_tps_mean": per_test["pg"]["mean"],
                    "pg_tps_stddev": per_test["pg"]["stddev"],
                },
                "tool": {"name": "llama_bench", "mode": "jsonl"},
            }
        )

    if not metrics_rows:
        raise ValueError("enginebench produced no metric rows")

    write_jsonl(out_dir / "metrics.jsonl", metrics_rows)
    ops.write_json(raw_dir / "llama_bench_commands.json", {"commands": command_rows})

    sorted_rows = sorted(
        metrics_rows,
        key=lambda row: float(row["metrics"]["tg_tps_mean"]),
        reverse=True,
    )
    top_k = max(1, int(args.top_k))
    best_rows = sorted_rows[:top_k]
    best_configs: list[dict[str, Any]] = []
    for row in best_rows:
        best_configs.append(
            {
                "tg_tps_mean": float(row["metrics"]["tg_tps_mean"]),
                "pp_tps_mean": float(row["metrics"]["pp_tps_mean"]),
                "pg_tps_mean": float(row["metrics"]["pg_tps_mean"]),
                "config": row.get("config", {}),
            }
        )

    run_payload = {
        "schema_version": "run_v0",
        "run_id": run_id,
        "timestamp_utc": ops.now_utc_iso(),
        "git_commit": ops.current_git_commit(),
        "commandline": commandline_from_argv(),
        "backend": {"name": backend_name, "version": backend_version},
        "model": {"path": str(model_path), "sha256": model_sha256},
        "workload": {"path": str(workload_path), "sha256": workload["sha256"]},
        "system": doctor_payload["system"],
        "enginebench": {
            "bench_bin": args.bench_bin,
            "bench_bin_args": args.bench_bin_args,
            "bench_extra_args": args.bench_extra_args,
            "repetitions": repetitions,
            "sweep": engine_cfg,
            "top_k": top_k,
            "best_configs": best_configs,
        },
    }
    run_payload["reproducibility"] = {"commands": ops.make_repro_commands(out_dir, run_payload)}
    ops.write_json(out_dir / "run.json", run_payload)

    summary_path, report_path = ops.generate_report(out_dir, out_dir)
    print(f"[enginebench] wrote {out_dir / 'doctor.json'}")
    print(f"[enginebench] wrote {out_dir / 'run.json'}")
    print(f"[enginebench] wrote {out_dir / 'metrics.jsonl'}")
    print(f"[enginebench] wrote {summary_path}")
    print(f"[enginebench] wrote {report_path}")
    return 0
