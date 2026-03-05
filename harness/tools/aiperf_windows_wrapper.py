#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SAFETY_PROMPT_TOKENS = 200
DEFAULT_SAFETY_OUTPUT_TOKENS = 200
DEFAULT_ZMQ_PORT_COUNT = 8
DEFAULT_REQUEST_TIMEOUT_SECONDS = 900


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def _safe_print(text: str, *, stderr: bool = False) -> None:
    if not text:
        return
    stream = sys.stderr if stderr else sys.stdout
    try:
        stream.write(text)
    except UnicodeEncodeError:
        try:
            stream.buffer.write(text.encode("utf-8", errors="replace"))
        except Exception:
            stream.write(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _has_option(args: list[str], names: list[str]) -> bool:
    for token in args:
        for name in names:
            if token == name or token.startswith(f"{name}="):
                return True
    return False


def _extract_option(args: list[str], names: list[str]) -> str | None:
    for idx, token in enumerate(args):
        for name in names:
            if token == name:
                if idx + 1 < len(args):
                    return args[idx + 1]
                return None
            if token.startswith(f"{name}="):
                return token.split("=", 1)[1]
    return None


def _pop_option(args: list[str], names: list[str]) -> tuple[str | None, list[str]]:
    value: str | None = None
    filtered: list[str] = []
    idx = 0
    while idx < len(args):
        token = args[idx]
        matched = False
        for name in names:
            if token == name:
                matched = True
                if idx + 1 >= len(args):
                    raise ValueError(f"missing value for wrapper option: {name}")
                value = args[idx + 1]
                idx += 2
                break
            if token.startswith(f"{name}="):
                matched = True
                value = token.split("=", 1)[1]
                idx += 1
                break
        if matched:
            continue
        filtered.append(token)
        idx += 1
    return value, filtered


def _extract_float(args: list[str], names: list[str]) -> float | None:
    raw = _extract_option(args, names)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _resolve_concurrency(user_args: list[str]) -> int:
    raw = _extract_option(user_args, ["--concurrency"])
    if raw is None:
        return 1
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return value if value > 0 else 1


def _load_workload_tokens(path: Path) -> tuple[int, int]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"workload file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid workload yaml object: {path}")

    if "prompt_tokens" not in payload:
        raise ValueError(f"{path}: prompt_tokens missing")
    if "output_tokens" not in payload:
        raise ValueError(f"{path}: output_tokens missing")

    try:
        prompt_tokens = int(payload["prompt_tokens"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: prompt_tokens must be integer") from exc
    try:
        output_tokens = int(payload["output_tokens"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: output_tokens must be integer") from exc

    if prompt_tokens < 0 or output_tokens < 0:
        raise ValueError(f"{path}: prompt_tokens/output_tokens must be >= 0")
    return prompt_tokens, output_tokens


def _build_profile_command(
    model_name: str,
    user_args: list[str],
    env: dict[str, str],
    workload_tokens: tuple[int, int] | None,
) -> tuple[list[str], dict[str, Any]]:
    injected: list[str] = []
    safe_defaults: dict[str, Any] = {}
    artifact_dir_env = env.get("PERFLAB_AIPERF_ARTIFACT_DIR", "").strip()

    if not _has_option(user_args, ["--output-artifact-dir", "--artifact-dir"]):
        if not artifact_dir_env:
            raise ValueError(
                "missing output artifact directory: set PERFLAB_AIPERF_ARTIFACT_DIR or pass --output-artifact-dir"
            )
        injected.extend(["--output-artifact-dir", artifact_dir_env])

    if not _has_option(user_args, ["--profile-export-prefix", "--profile-export-file"]):
        injected.extend(["--profile-export-prefix", "profile_export"])

    if not _has_option(user_args, ["--export-level", "--profile-export-level"]):
        injected.extend(["--export-level", "records"])

    if not _has_option(user_args, ["--ui-type", "--ui"]):
        injected.extend(["--ui", "none"])

    if not _has_option(user_args, ["--zmq-host", "--zmq-ipc-path"]):
        injected.extend(["--zmq-host", "127.0.0.1"])

    safe_prompt_tokens = (
        workload_tokens[0] if workload_tokens is not None else DEFAULT_SAFETY_PROMPT_TOKENS
    )
    safe_output_tokens = (
        workload_tokens[1] if workload_tokens is not None else DEFAULT_SAFETY_OUTPUT_TOKENS
    )

    if not _has_option(user_args, ["--prompt-input-tokens-mean", "--synthetic-input-tokens-mean", "--isl"]):
        injected.extend(["--prompt-input-tokens-mean", str(safe_prompt_tokens)])
        safe_defaults["prompt_input_tokens_mean"] = safe_prompt_tokens

    if not _has_option(user_args, ["--prompt-output-tokens-mean", "--output-tokens-mean", "--osl"]):
        injected.extend(["--output-tokens-mean", str(safe_output_tokens)])
        safe_defaults["output_tokens_mean"] = safe_output_tokens

    if not _has_option(user_args, ["--conversation-turn-mean", "--session-turns-mean"]):
        injected.extend(["--conversation-turn-mean", "1"])
        safe_defaults["conversation_turn_mean"] = 1

    if not _has_option(user_args, ["--conversation-turn-stddev", "--session-turns-stddev"]):
        injected.extend(["--conversation-turn-stddev", "0"])
        safe_defaults["conversation_turn_stddev"] = 0

    if not _has_option(user_args, ["--num-prompts", "--num-dataset-entries"]):
        concurrency = _resolve_concurrency(user_args)
        num_prompts = max(4, concurrency * 2)
        injected.extend(["--num-prompts", str(num_prompts)])
        safe_defaults["num_prompts"] = num_prompts

    if not _has_option(user_args, ["--request-timeout-seconds"]):
        injected.extend(["--request-timeout-seconds", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)])
        safe_defaults["request_timeout_seconds"] = DEFAULT_REQUEST_TIMEOUT_SECONDS

    # Windows stability defaults: keep internal service fan-out conservative.
    if not _has_option(user_args, ["--record-processors"]):
        injected.extend(["--record-processors", "1"])
        safe_defaults["record_processors"] = 1

    if not _has_option(user_args, ["--max-workers", "--workers-max"]):
        workers_max = min(max(_resolve_concurrency(user_args), 1), 4)
        injected.extend(["--max-workers", str(workers_max)])
        safe_defaults["max_workers"] = workers_max

    full_args = [*injected, *user_args]
    command = [sys.executable, "-m", "aiperf", "profile", model_name, *full_args]
    return command, {
        "injected_args": injected,
        "safe_defaults_applied": safe_defaults,
        "resolved_safety_tokens": {
            "prompt_input_tokens_mean": _extract_float(
                full_args, ["--prompt-input-tokens-mean", "--synthetic-input-tokens-mean", "--isl"]
            ),
            "output_tokens_mean": _extract_float(
                full_args, ["--prompt-output-tokens-mean", "--output-tokens-mean", "--osl"]
            ),
        },
    }


def _write_sitecustomize(compat_dir: Path) -> Path:
    compat_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize = compat_dir / "sitecustomize.py"
    sitecustomize.write_text(
        "\n".join(
            [
                "import multiprocessing.context as _ctx",
                "if not hasattr(_ctx, 'ForkProcess') and hasattr(_ctx, 'SpawnProcess'):",
                "    _ctx.ForkProcess = _ctx.SpawnProcess",
                "try:",
                "    import asyncio as _asyncio",
                "    if hasattr(_asyncio, 'WindowsSelectorEventLoopPolicy'):",
                "        _asyncio.set_event_loop_policy(_asyncio.WindowsSelectorEventLoopPolicy())",
                "except Exception:",
                "    pass",
                "try:",
                "    import aiperf.controller.system_mixins as _sm",
                "    if hasattr(_sm, 'SignalHandlerMixin'):",
                "        _sm.SignalHandlerMixin.setup_signal_handlers = lambda self, callback: None",
                "except Exception:",
                "    pass",
                "try:",
                "    import os as _os",
                "    _ports_raw = _os.environ.get('PERFLAB_AIPERF_ZMQ_PORTS', '')",
                "    _parts = [int(_x.strip()) for _x in _ports_raw.split(',') if _x.strip()]",
                "    if len(_parts) == 8:",
                "        from aiperf.common.config.zmq_config import ZMQTCPConfig as _ZT",
                "        _mf = _ZT.model_fields",
                "        _mf['records_push_pull_port'].default = _parts[0]",
                "        _mf['credit_router_port'].default = _parts[1]",
                "        _dm = _mf['dataset_manager_proxy_config'].default",
                "        _dm.frontend_port = _parts[2]",
                "        _dm.backend_port = _parts[3]",
                "        _eb = _mf['event_bus_proxy_config'].default",
                "        _eb.frontend_port = _parts[4]",
                "        _eb.backend_port = _parts[5]",
                "        _ri = _mf['raw_inference_proxy_config'].default",
                "        _ri.frontend_port = _parts[6]",
                "        _ri.backend_port = _parts[7]",
                "except Exception:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    return sitecustomize


def _allocate_tcp_ports(count: int = DEFAULT_ZMQ_PORT_COUNT) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def _validate_profile_export(jsonl_path: Path) -> tuple[int, int]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"profile export jsonl not found: {jsonl_path}")

    total_rows = 0
    valid_rows = 0
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total_rows += 1
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        has_error = payload.get("error") not in (None, "", {}, [])
        metrics = payload.get("metrics")
        if not has_error and isinstance(metrics, dict):
            valid_rows += 1

    if valid_rows < 1:
        raise ValueError(f"profile export has no valid metric rows: {jsonl_path}")
    return total_rows, valid_rows


def _resolve_profile_export_path(args: list[str]) -> Path:
    output_dir = _extract_option(args, ["--output-artifact-dir", "--artifact-dir"])
    if not output_dir:
        raise ValueError("unable to resolve --output-artifact-dir")
    prefix = _extract_option(args, ["--profile-export-prefix", "--profile-export-file"]) or "profile_export"
    return Path(output_dir) / f"{prefix}.jsonl"


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_profile(model_name: str, user_args: list[str], *, workload_path: str | None = None) -> int:
    _ensure_utf8_stdio()
    env = os.environ.copy()
    workload_tokens: tuple[int, int] | None = None
    if workload_path is not None:
        workload_tokens = _load_workload_tokens(Path(workload_path))
    command, injected_meta = _build_profile_command(model_name, user_args, env, workload_tokens)

    allocated_zmq_ports: list[int] | None = None
    profile_export: Path | None = None
    proc: subprocess.CompletedProcess[str] | None = None
    total_rows = 0
    valid_rows = 0
    validation_error: str | None = None
    compat_patch_dir: str | None = None
    sitecustomize_meta_path: str | None = None

    with tempfile.TemporaryDirectory(prefix="aiperf_win_compat_") as compat_dir_raw:
        compat_dir = Path(compat_dir_raw)
        compat_patch_dir = str(compat_dir)
        sitecustomize_path = _write_sitecustomize(compat_dir)

        child_env = env.copy()
        py_path = child_env.get("PYTHONPATH", "").strip()
        child_env["PYTHONPATH"] = f"{compat_dir}{os.pathsep}{py_path}" if py_path else str(compat_dir)
        child_env["PYTHONIOENCODING"] = "utf-8"
        if not _has_option(user_args, ["--zmq-ipc-path"]):
            allocated_zmq_ports = _allocate_tcp_ports()
            child_env["PERFLAB_AIPERF_ZMQ_PORTS"] = ",".join(str(port) for port in allocated_zmq_ports)

        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=child_env,
            encoding="utf-8",
            errors="replace",
        )

        _safe_print(proc.stdout, stderr=False)
        _safe_print(proc.stderr, stderr=True)

        final_args = command[4:]
        profile_export = _resolve_profile_export_path(final_args)
        try:
            total_rows, valid_rows = _validate_profile_export(profile_export)
        except Exception as exc:
            validation_error = str(exc)

        # Persist a copy of runtime patch for audit after temp dir cleanup.
        sitecustomize_snapshot = profile_export.parent / "win_compat_sitecustomize.py"
        try:
            sitecustomize_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sitecustomize_path, sitecustomize_snapshot)
            sitecustomize_meta_path = str(sitecustomize_snapshot)
        except OSError:
            sitecustomize_meta_path = str(sitecustomize_path)

    if profile_export is None:
        raise RuntimeError("unable to resolve profile export output path")

    if proc is None:
        raise RuntimeError("aiperf wrapper failed to launch subprocess")

    meta_path = profile_export.parent / "win_compat_meta.json"
    meta = {
        "wrapper_command": " ".join(command),
        "aiperf_returncode": proc.returncode,
        "injected_args": injected_meta["injected_args"],
        "safe_defaults_applied": injected_meta["safe_defaults_applied"],
        "resolved_safety_tokens": injected_meta["resolved_safety_tokens"],
        "workload_path": workload_path,
        "compat_patch_dir": compat_patch_dir,
        "compat_patch_dir_ephemeral": True,
        "sitecustomize_path": sitecustomize_meta_path,
        "allocated_zmq_tcp_ports": allocated_zmq_ports,
        "profile_export_jsonl": str(profile_export),
        "profile_export_rows": total_rows,
        "profile_export_valid_rows": valid_rows,
        "validation_error": validation_error,
    }
    _write_meta(meta_path, meta)

    if proc.returncode != 0:
        print(f"error: wrapped aiperf failed with code {proc.returncode}", file=sys.stderr)
        return proc.returncode
    if validation_error is not None:
        print(f"error: {validation_error}", file=sys.stderr)
        return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows compatibility wrapper for `python -m aiperf profile`.")
    parser.add_argument("subcommand", choices=["profile"], help="Only `profile` is supported in this wrapper")
    parser.add_argument("model_name", help="AIPerf profile model name argument")
    parser.add_argument("aiperf_args", nargs=argparse.REMAINDER, help="Arguments forwarded to `aiperf profile`")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.subcommand != "profile":
        print(f"error: unsupported subcommand: {args.subcommand}", file=sys.stderr)
        return 2
    try:
        workload_path, forwarded = _pop_option(list(args.aiperf_args), ["--workload"])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        return run_profile(args.model_name, forwarded, workload_path=workload_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if os.environ.get("PERFLAB_DEBUG_EXCEPTIONS", "").strip() == "1":
            print(f"error: {exc}", file=sys.stderr)
        else:
            print(
                "error: wrapper execution failed (set PERFLAB_DEBUG_EXCEPTIONS=1 for details)",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
