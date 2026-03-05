from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import ops as ops
from adapters.aiperf_adapter import adapt_aiperf_profile_export_jsonl
from adapters.mlc_server import build_mlc_overrides
from core import commandline_from_argv, write_jsonl


def cmd_servebench(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_tool_output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    aiperf_artifact_dir = raw_dir / "aiperf"
    aiperf_records_jsonl = aiperf_artifact_dir / "profile_export.jsonl"

    run_id = args.run_id or f"servebench_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    workload_path = Path(args.workload)
    workload = ops.load_workload(workload_path)

    model_ref = str(args.model or "models/demo.gguf")
    if "\x00" in model_ref:
        raise ValueError("servebench --model contains NUL byte")
    model_path = Path(model_ref).expanduser().resolve()
    model_sha256 = ops.maybe_sha256_model(model_path)
    backend_name = args.backend
    backend_version = args.backend_version
    mlc_backend = ops.is_mlc_backend(backend_name)
    base_url = (args.server_url or f"http://{args.server_host}:{args.server_port}").rstrip("/")

    tool_output_jsonl: Path | None = None
    if args.tool_output_jsonl:
        tool_output_jsonl = Path(args.tool_output_jsonl)
    elif args.tool_run_cmd:
        tool_output_jsonl = raw_dir / "tool_output.jsonl"

    if args.server_mode == "managed":
        if not args.server_bin:
            raise ValueError("managed mode requires --server-bin")
        if args.model is None:
            raise ValueError("managed mode requires --model")
        if not mlc_backend:
            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"managed mode requires --model as existing file: {model_path}")
        else:
            ops.validate_mlc_managed_model_ref(model_ref)
            if model_ref.startswith("HF://"):
                ops.enforce_no_blocking_git_https_rewrite()

    doctor_payload = ops.build_doctor_payload(
        run_id=run_id,
        commandline=commandline_from_argv(),
        backend_name=backend_name,
        backend_version=backend_version,
        workload_path=str(workload_path),
        workload_sha256=workload["sha256"],
    )
    ops.write_json(out_dir / "doctor.json", doctor_payload)

    server_proc: subprocess.Popen[str] | None = None
    stdout_handle = None
    stderr_handle = None
    server_stderr_path = raw_dir / "server_stderr.log"
    server_stdout_path = raw_dir / "server_stdout.log"
    server_retry_stdout_path = raw_dir / "server_retry_stdout.log"
    server_retry_stderr_path = raw_dir / "server_retry_stderr.log"
    server_command: list[str] = []
    server_cmdline = ""
    mlc_overrides = build_mlc_overrides(args) if mlc_backend else {}
    mlc_capabilities: dict[str, object] = {}
    mlc_runtime_effective: dict[str, object] = {
        "effective_opt": None,
        "effective_device": None,
        "effective_model_lib": None,
    }
    mlc_downgraded_flags: list[str] = []
    mlc_compile_fallback: dict[str, object] = {
        "triggered": False,
        "reason_signature": None,
        "compile_command": None,
        "compile_returncode": None,
        "compile_stdout_log": None,
        "compile_stderr_log": None,
        "compiled_model_lib": None,
        "retry_started": False,
        "retry_succeeded": False,
        "compile_attempts": [],
        "selected_compile_strategy": None,
        "used_validation_bypass": False,
        "degraded_release": False,
        "compile_final_returncode": None,
    }

    def _close_current_stdio_handles() -> None:
        nonlocal stdout_handle, stderr_handle
        if stdout_handle is not None:
            stdout_handle.close()
            stdout_handle = None
        if stderr_handle is not None:
            stderr_handle.close()
            stderr_handle = None

    def _start_managed_server(
        command: list[str],
        *,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        nonlocal server_proc, server_command, server_cmdline, stdout_handle, stderr_handle
        server_command = command
        server_cmdline = " ".join(shlex.quote(part) for part in server_command)
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        server_proc = subprocess.Popen(
            server_command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

    try:
        if args.server_mode == "managed" and mlc_backend:
            mlc_capabilities = ops.probe_mlc_serve_capabilities(args.server_bin, args.server_bin_args)
            runtime_effective, downgraded_flags = ops.resolve_mlc_runtime_args(args, mlc_capabilities)
            mlc_runtime_effective = runtime_effective
            mlc_downgraded_flags = downgraded_flags

        if args.server_mode == "managed":
            server_command = ops.build_server_command(
                args,
                model_ref,
                mlc_runtime_effective=mlc_runtime_effective if mlc_backend else None,
            )
            _start_managed_server(
                server_command,
                stdout_path=server_stdout_path,
                stderr_path=server_stderr_path,
            )
            try:
                ops.wait_for_server_ready(
                    base_url=base_url,
                    timeout_sec=args.server_health_timeout_sec,
                    interval_sec=args.server_health_interval_sec,
                    managed_proc=server_proc,
                    managed_stderr_path=server_stderr_path,
                )
            except Exception as startup_exc:
                startup_text = str(startup_exc)
                first_stderr = ""
                if server_stderr_path.exists():
                    first_stderr = server_stderr_path.read_text(encoding="utf-8", errors="replace")
                combined_text = startup_text + ("\n" + first_stderr if first_stderr else "")
                should_try_compile_fallback = (
                    mlc_backend
                    and mlc_runtime_effective.get("effective_model_lib") in (None, "")
                    and ops.is_mlc_jit_compile_failure(combined_text)
                )
                if not should_try_compile_fallback:
                    raise

                mlc_compile_fallback["triggered"] = True
                mlc_compile_fallback["reason_signature"] = startup_text[:500]
                ops.stop_managed_server(server_proc)
                server_proc = None
                _close_current_stdio_handles()

                compile_dir = raw_dir / "mlc_compile"
                compiled_model_lib = (compile_dir / "model_o0.dll").resolve()
                compile_command = ops.build_mlc_compile_command(
                    args=args,
                    model_ref=model_ref,
                    output_path=compiled_model_lib,
                    mlc_runtime_effective=mlc_runtime_effective if mlc_backend else None,
                )
                compile_cmdline = " ".join(shlex.quote(part) for part in compile_command)
                mlc_compile_fallback["compile_command"] = compile_cmdline
                mlc_compile_fallback["compile_stdout_log"] = str(compile_dir / "compile_o0_default_stdout.log")
                mlc_compile_fallback["compile_stderr_log"] = str(compile_dir / "compile_o0_default_stderr.log")
                try:
                    compile_meta = ops.run_mlc_compile_with_retry(
                        command=compile_command,
                        compile_dir=compile_dir,
                    )
                    effective_model_lib_str = str(
                        compile_meta.get("effective_output_path") or compiled_model_lib
                    )
                    compiled_model_lib = Path(effective_model_lib_str).expanduser().resolve()
                    compile_attempts = list(compile_meta.get("attempts", []))
                    final_attempt = compile_attempts[-1] if compile_attempts else {}
                    mlc_compile_fallback["compile_attempts"] = compile_attempts
                    mlc_compile_fallback["selected_compile_strategy"] = compile_meta.get("selected_compile_strategy")
                    mlc_compile_fallback["used_validation_bypass"] = bool(
                        compile_meta.get("used_validation_bypass", False)
                    )
                    mlc_compile_fallback["degraded_release"] = bool(
                        compile_meta.get("used_validation_bypass", False)
                    )
                    mlc_compile_fallback["compile_final_returncode"] = int(
                        compile_meta.get("final_returncode", 0)
                    )
                    mlc_compile_fallback["compile_returncode"] = int(
                        compile_meta.get("final_returncode", 0)
                    )
                    mlc_compile_fallback["compile_output_relocated"] = bool(
                        compile_meta.get("output_relocated", False)
                    )
                    mlc_compile_fallback["compile_requested_output"] = compile_meta.get(
                        "requested_output_path"
                    )
                    mlc_compile_fallback["compile_staged_output"] = compile_meta.get(
                        "staged_output_path"
                    )
                    if final_attempt:
                        mlc_compile_fallback["compile_command"] = final_attempt.get(
                            "commandline", mlc_compile_fallback["compile_command"]
                        )
                        mlc_compile_fallback["compile_stdout_log"] = final_attempt.get(
                            "stdout_log", mlc_compile_fallback["compile_stdout_log"]
                        )
                        mlc_compile_fallback["compile_stderr_log"] = final_attempt.get(
                            "stderr_log", mlc_compile_fallback["compile_stderr_log"]
                        )
                    mlc_compile_fallback["compiled_model_lib"] = str(compiled_model_lib)
                except Exception as compile_exc:
                    raise RuntimeError(
                        f"mlc compile fallback failed: {compile_exc}. "
                        f"See {compile_dir}"
                    ) from compile_exc

                mlc_runtime_effective["effective_model_lib"] = str(compiled_model_lib)
                mlc_compile_fallback["retry_started"] = True
                retry_command = ops.build_server_command(
                    args,
                    model_ref,
                    mlc_runtime_effective=mlc_runtime_effective if mlc_backend else None,
                )
                _start_managed_server(
                    retry_command,
                    stdout_path=server_retry_stdout_path,
                    stderr_path=server_retry_stderr_path,
                )
                try:
                    ops.wait_for_server_ready(
                        base_url=base_url,
                        timeout_sec=args.server_health_timeout_sec,
                        interval_sec=args.server_health_interval_sec,
                        managed_proc=server_proc,
                        managed_stderr_path=server_retry_stderr_path,
                    )
                    mlc_compile_fallback["retry_succeeded"] = True
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"mlc managed retry with compiled model-lib failed: {retry_exc}. "
                        f"See {server_retry_stderr_path}"
                    ) from retry_exc
        else:
            ops.wait_for_server_ready(
                base_url=base_url,
                timeout_sec=args.server_health_timeout_sec,
                interval_sec=args.server_health_interval_sec,
                managed_proc=None,
                managed_stderr_path=None,
            )

        if args.tool_run_cmd:
            aiperf_artifact_dir.mkdir(parents=True, exist_ok=True)
            ops.run_tool_command(
                command=args.tool_run_cmd,
                raw_dir=raw_dir,
                base_url=base_url,
                model=model_ref,
                tool_output_jsonl=tool_output_jsonl,
                aiperf_artifact_dir=aiperf_artifact_dir,
                aiperf_records_jsonl=aiperf_records_jsonl,
            )
            if args.tool == "aiperf" and args.mode == "replay" and args.tool_output_jsonl is None:
                if not aiperf_records_jsonl.exists():
                    raise FileNotFoundError(
                        f"aiperf records file not found: {aiperf_records_jsonl}. "
                        "Expected tool command to write PERFLAB_AIPERF_RECORDS_JSONL"
                    )
                if tool_output_jsonl is None:
                    raise ValueError("internal error: tool_output_jsonl unresolved for aiperf replay adaptation")
                adapt_stats = adapt_aiperf_profile_export_jsonl(
                    input_path=aiperf_records_jsonl,
                    output_path=tool_output_jsonl,
                    default_concurrency=1,
                )
                ops.write_json(
                    raw_dir / "aiperf_adapt_meta.json",
                    {
                        "input_path": str(aiperf_records_jsonl),
                        "output_path": str(tool_output_jsonl),
                        **adapt_stats,
                    },
                )

        if args.mode == "mock":
            metrics_rows = ops.generate_mock_serving_rows(
                run_id=run_id,
                backend_name=backend_name,
                backend_version=backend_version,
                model_path=model_ref,
                model_sha256=model_sha256,
                workload=workload,
                tool_name=args.tool,
            )
        else:
            if tool_output_jsonl is None:
                raise ValueError("replay mode requires --tool-output-jsonl or --tool-run-cmd")
            if not tool_output_jsonl.exists():
                raise FileNotFoundError(f"replay input jsonl not found: {tool_output_jsonl}")

            replay_rows = ops.normalize_replay_rows(tool_output_jsonl, workload, run_id)
            metrics_rows = ops.build_metrics_rows_from_replay(
                run_id=run_id,
                backend_name=backend_name,
                backend_version=backend_version,
                model_path=model_ref,
                model_sha256=model_sha256,
                workload=workload,
                tool_name=args.tool,
                normalized_rows=replay_rows,
            )

        write_jsonl(out_dir / "metrics.jsonl", metrics_rows)

        run_payload = {
            "schema_version": "run_v0",
            "run_id": run_id,
            "timestamp_utc": ops.now_utc_iso(),
            "git_commit": ops.current_git_commit(),
            "commandline": commandline_from_argv(),
            "backend": {"name": backend_name, "version": backend_version},
            "model": {"path": model_ref, "sha256": model_sha256},
            "workload": {"path": str(workload_path), "sha256": workload["sha256"]},
            "system": doctor_payload["system"],
            "server": {
                "mode": args.server_mode,
                "url": base_url,
                "host": args.server_host,
                "port": args.server_port,
                "bin": args.server_bin,
                "bin_args": args.server_bin_args,
                "extra_args": args.server_extra_args,
                "health_timeout_sec": args.server_health_timeout_sec,
                "health_interval_sec": args.server_health_interval_sec,
                "resolved_command": server_cmdline,
                "mlc_mode": args.mlc_mode if mlc_backend else None,
                "mlc_overrides": mlc_overrides if mlc_backend else None,
                "mlc_capabilities": mlc_capabilities if mlc_backend else None,
                "mlc_downgraded_flags": mlc_downgraded_flags if mlc_backend else [],
                "mlc_runtime_effective": mlc_runtime_effective if mlc_backend else None,
                "mlc_compile_fallback": mlc_compile_fallback if mlc_backend else None,
            },
            "tool": {
                "name": args.tool,
                "mode": args.mode,
                "run_cmd": args.tool_run_cmd,
                "output_path": str(tool_output_jsonl) if tool_output_jsonl else None,
                "aiperf_records_path": str(aiperf_records_jsonl)
                if args.tool == "aiperf"
                else None,
            },
        }
        run_payload["reproducibility"] = {"commands": ops.make_repro_commands(out_dir, run_payload)}
        ops.write_json(out_dir / "run.json", run_payload)

        summary_path, report_path = ops.generate_report(out_dir, out_dir)
        print(f"[servebench] wrote {out_dir / 'doctor.json'}")
        print(f"[servebench] wrote {out_dir / 'run.json'}")
        print(f"[servebench] wrote {out_dir / 'metrics.jsonl'}")
        print(f"[servebench] wrote {summary_path}")
        print(f"[servebench] wrote {report_path}")
        return 0
    finally:
        ops.stop_managed_server(server_proc)
        _close_current_stdio_handles()
