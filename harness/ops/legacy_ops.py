from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapters.mlc_server import build_mlc_overrides, build_mlc_overrides_arg
from core import (
    collect_host_fingerprint,
    current_git_commit,
    mean,
    nearest_rank_percentile,
    now_utc_iso,
    read_json,
    read_jsonl_with_lineno,
    sha256_file,
    split_args,
    write_json,
)

try:
    import jsonschema
except ImportError:
    jsonschema = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_ROOT_CANDIDATE = Path(__file__).resolve().parents[2]
ROOT = _ROOT_CANDIDATE if (_ROOT_CANDIDATE / "schemas").exists() else Path(__file__).resolve().parents[1]
DEFAULT_METRICS_SCHEMA = ROOT / "schemas" / "metrics_v0.schema.json"
DEFAULT_RUN_SCHEMA = ROOT / "schemas" / "run_v0.schema.json"
DEFAULT_WORKLOAD = {
    "name": "w0_chat_200_200",
    "prompt_tokens": 200,
    "output_tokens": 200,
    "concurrency": [1, 4],
    "repetitions": 2,
}
DEFAULT_ENGINE_SWEEP = {
    "threads": [4],
    "batch": [512],
    "ubatch": [256],
    "ngl": [99],
}


def resolve_doctor_output(raw_out: str) -> Path:
    out_path = Path(raw_out)
    if out_path.suffix.lower() == ".json":
        return out_path
    return out_path / "doctor.json"

def build_doctor_payload(
    *,
    run_id: str,
    commandline: str,
    backend_name: str,
    backend_version: str,
    workload_path: str,
    workload_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "run_v0",
        "run_id": run_id,
        "timestamp_utc": now_utc_iso(),
        "git_commit": current_git_commit(),
        "commandline": commandline,
        "backend": {"name": backend_name, "version": backend_version},
        "workload": {"path": workload_path, "sha256": workload_sha256},
        "system": collect_host_fingerprint(),
        "doctor": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count_logical": os.cpu_count(),
        },
    }

def _read_positive_int_list(value: Any, *, field_name: str) -> list[int]:
    if isinstance(value, int):
        values = [int(value)]
    elif isinstance(value, list):
        values = [int(v) for v in value]
    else:
        raise ValueError(f"{field_name} must be int or list[int]")
    if not values or any(v < 1 for v in values):
        raise ValueError(f"{field_name} must contain positive integers")
    return values

def _read_non_negative_int_list(value: Any, *, field_name: str) -> list[int]:
    if isinstance(value, int):
        values = [int(value)]
    elif isinstance(value, list):
        values = [int(v) for v in value]
    else:
        raise ValueError(f"{field_name} must be int or list[int]")
    if not values or any(v < 0 for v in values):
        raise ValueError(f"{field_name} must contain non-negative integers")
    return values

def load_workload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"workload file not found: {path}")
    if yaml is None:
        raise RuntimeError("pyyaml is required for workload parsing. Install: python -m pip install pyyaml")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = payload if isinstance(payload, dict) else {}

    name = str(data.get("name") or path.stem)
    prompt_tokens = int(data.get("prompt_tokens", DEFAULT_WORKLOAD["prompt_tokens"]))
    output_tokens = int(data.get("output_tokens", DEFAULT_WORKLOAD["output_tokens"]))

    raw_concurrency = data.get("concurrency", DEFAULT_WORKLOAD["concurrency"])
    concurrency = _read_positive_int_list(raw_concurrency, field_name="workload.concurrency")

    repetitions = int(data.get("repetitions", DEFAULT_WORKLOAD["repetitions"]))
    if repetitions < 1:
        raise ValueError("workload.repetitions must be >= 1")

    workload: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "sha256": sha256_file(path),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "repetitions": repetitions,
    }

    engine_raw = data.get("engine")
    if engine_raw is not None:
        if not isinstance(engine_raw, dict):
            raise ValueError("workload.engine must be an object")
        workload["engine"] = {
            "threads": _read_positive_int_list(
                engine_raw.get("threads", DEFAULT_ENGINE_SWEEP["threads"]),
                field_name="workload.engine.threads",
            ),
            "batch": _read_positive_int_list(
                engine_raw.get("batch", DEFAULT_ENGINE_SWEEP["batch"]),
                field_name="workload.engine.batch",
            ),
            "ubatch": _read_positive_int_list(
                engine_raw.get("ubatch", DEFAULT_ENGINE_SWEEP["ubatch"]),
                field_name="workload.engine.ubatch",
            ),
            "ngl": _read_non_negative_int_list(
                engine_raw.get("ngl", DEFAULT_ENGINE_SWEEP["ngl"]),
                field_name="workload.engine.ngl",
            ),
        }

    return workload

def generate_mock_serving_rows(
    *,
    run_id: str,
    backend_name: str,
    backend_version: str,
    model_path: str,
    model_sha256: str,
    workload: dict[str, Any],
    tool_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    host = collect_host_fingerprint()
    git_commit = current_git_commit()
    start = datetime.now(timezone.utc).replace(microsecond=0)

    index = 0
    for concurrency in workload["concurrency"]:
        for rep in range(workload["repetitions"]):
            timestamp = now_utc_iso(start + timedelta(seconds=index))
            ttft_ms = round(320.0 + 45.0 * concurrency + 8.0 * rep, 3)
            itl_ms = round(22.0 + 3.6 * concurrency + 0.9 * rep, 3)
            tps = round(max(1.0, 33.0 - 2.1 * concurrency - 0.4 * rep), 3)
            rps = round((tps * concurrency) / max(1, workload["output_tokens"]), 3)

            rows.append(
                {
                    "schema_version": "metrics_v0",
                    "run_id": run_id,
                    "timestamp_utc": timestamp,
                    "git_commit": git_commit,
                    "track": "serving",
                    "backend": {"name": backend_name, "version": backend_version},
                    "model": {"path": model_path, "sha256": model_sha256},
                    "workload": {
                        "name": workload["name"],
                        "path": workload["path"],
                        "sha256": workload["sha256"],
                    },
                    "system": host,
                    "request": {
                        "id": f"{run_id}-{index + 1}",
                        "concurrency": concurrency,
                        "prompt_tokens": workload["prompt_tokens"],
                        "output_tokens": workload["output_tokens"],
                    },
                    "metrics": {
                        "ttft_ms": ttft_ms,
                        "itl_ms": itl_ms,
                        "tps": tps,
                        "rps": rps,
                    },
                    "tool": {"name": tool_name, "mode": "mock"},
                }
            )
            index += 1
    return rows

def _pick_number(payload: dict[str, Any], keys: list[str], *, path: Path, lineno: int) -> float:
    for key in keys:
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: `{key}` must be a number") from exc
    expected = ", ".join(keys)
    raise ValueError(f"{path}:{lineno}: missing numeric field, expected one of [{expected}]")

def _pick_int(payload: dict[str, Any], keys: list[str], default: int) -> int:
    for key in keys:
        if key in payload:
            try:
                return int(payload[key])
            except (TypeError, ValueError):
                return default
    return default

def _pick_str(payload: dict[str, Any], keys: list[str], default: str) -> str:
    for key in keys:
        if key in payload:
            value = str(payload[key]).strip()
            if value:
                return value
    return default

def normalize_replay_rows(path: Path, workload: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    source_rows = read_jsonl_with_lineno(path)
    if not source_rows:
        raise ValueError(f"tool output has no rows: {path}")

    start = datetime.now(timezone.utc).replace(microsecond=0)
    rows: list[dict[str, Any]] = []
    for index, (lineno, payload) in enumerate(source_rows):
        ttft_ms = _pick_number(payload, ["ttft_ms", "ttft", "latency_to_first_token_ms"], path=path, lineno=lineno)
        itl_ms = _pick_number(payload, ["itl_ms", "itl", "inter_token_latency_ms"], path=path, lineno=lineno)
        tps = _pick_number(payload, ["tps", "tokens_per_second", "output_tps"], path=path, lineno=lineno)
        rps = _pick_number(payload, ["rps", "requests_per_second"], path=path, lineno=lineno)

        timestamp_utc = _pick_str(payload, ["timestamp_utc", "timestamp"], "")
        if not timestamp_utc:
            timestamp_utc = now_utc_iso(start + timedelta(seconds=index))

        rows.append(
            {
                "request_id": _pick_str(payload, ["request_id", "id"], f"{run_id}-{index + 1}"),
                "timestamp_utc": timestamp_utc,
                "concurrency": _pick_int(payload, ["concurrency", "n_concurrency"], workload["concurrency"][0]),
                "prompt_tokens": _pick_int(payload, ["prompt_tokens", "input_tokens"], workload["prompt_tokens"]),
                "output_tokens": _pick_int(payload, ["output_tokens", "completion_tokens"], workload["output_tokens"]),
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "tps": tps,
                "rps": rps,
            }
        )

    return rows

def build_metrics_rows_from_replay(
    *,
    run_id: str,
    backend_name: str,
    backend_version: str,
    model_path: str,
    model_sha256: str,
    workload: dict[str, Any],
    tool_name: str,
    normalized_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    host = collect_host_fingerprint()
    git_commit = current_git_commit()
    metrics_rows: list[dict[str, Any]] = []

    for row in normalized_rows:
        metrics_rows.append(
            {
                "schema_version": "metrics_v0",
                "run_id": run_id,
                "timestamp_utc": row["timestamp_utc"],
                "git_commit": git_commit,
                "track": "serving",
                "backend": {"name": backend_name, "version": backend_version},
                "model": {"path": model_path, "sha256": model_sha256},
                "workload": {
                    "name": workload["name"],
                    "path": workload["path"],
                    "sha256": workload["sha256"],
                },
                "system": host,
                "request": {
                    "id": row["request_id"],
                    "concurrency": row["concurrency"],
                    "prompt_tokens": row["prompt_tokens"],
                    "output_tokens": row["output_tokens"],
                },
                "metrics": {
                    "ttft_ms": row["ttft_ms"],
                    "itl_ms": row["itl_ms"],
                    "tps": row["tps"],
                    "rps": row["rps"],
                },
                "tool": {"name": tool_name, "mode": "replay"},
            }
        )

    return metrics_rows

def _read_engine_metric(row: dict[str, Any], key: str) -> float:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("metrics field must be an object")
    if key not in metrics:
        raise ValueError(f"missing engine metric field: {key}")
    try:
        return float(metrics[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"engine metric must be numeric: {key}") from exc

def _read_engine_config(row: dict[str, Any]) -> dict[str, int]:
    config = row.get("config", {})
    if not isinstance(config, dict):
        return {"threads": 0, "batch": 0, "ubatch": 0, "ngl": 0}

    def parse_int(name: str) -> int:
        try:
            return int(config.get(name, 0))
        except (TypeError, ValueError):
            return 0

    return {
        "threads": parse_int("threads"),
        "batch": parse_int("batch"),
        "ubatch": parse_int("ubatch"),
        "ngl": parse_int("ngl"),
    }

def generate_report(input_dir: Path, out_dir: Path) -> tuple[Path, Path]:
    metrics_path = input_dir / "metrics.jsonl"
    run_path = input_dir / "run.json"

    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"missing metrics file: {metrics_path}")

    rows = [payload for _, payload in read_jsonl_with_lineno(metrics_path)]
    if not rows:
        raise ValueError(f"no metrics rows found: {metrics_path}")

    track = str(rows[0].get("track", "serving"))
    mixed_tracks = [str(r.get("track", "serving")) for r in rows if str(r.get("track", "serving")) != track]
    if mixed_tracks:
        raise ValueError("metrics.jsonl contains mixed track values")

    top_k = 3
    run_meta: dict[str, Any] = {}
    if run_path.exists():
        run_meta = read_json(run_path)
        engine_meta = run_meta.get("enginebench")
        if isinstance(engine_meta, dict):
            try:
                top_k = max(1, int(engine_meta.get("top_k", 3)))
            except (TypeError, ValueError):
                top_k = 3

    summary: dict[str, Any]
    lines = [
        "# 780M-LLM-PerfLab Report",
        "",
        "## Inputs",
        f"- run_dir: `{input_dir}`",
        f"- metrics_file: `{metrics_path}`",
        f"- generated_at_utc: `{now_utc_iso()}`",
        "",
    ]

    if track == "serving":
        ttft_values: list[float] = []
        itl_values: list[float] = []
        tps_values: list[float] = []
        rps_values: list[float] = []

        for row in rows:
            metrics = row.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValueError("metrics field must be an object")
            try:
                ttft_values.append(float(metrics["ttft_ms"]))
                itl_values.append(float(metrics["itl_ms"]))
                tps_values.append(float(metrics["tps"]))
                rps_values.append(float(metrics["rps"]))
            except KeyError as exc:
                raise ValueError(f"missing serving metric field: {exc.args[0]}") from exc

        summary = {
            "run_id": str(rows[0].get("run_id", "unknown")),
            "track": track,
            "samples": len(rows),
            "ttft_p50_ms": round(nearest_rank_percentile(ttft_values, 50), 3),
            "ttft_p95_ms": round(nearest_rank_percentile(ttft_values, 95), 3),
            "itl_p50_ms": round(nearest_rank_percentile(itl_values, 50), 3),
            "itl_p95_ms": round(nearest_rank_percentile(itl_values, 95), 3),
            "tps_mean": round(mean(tps_values), 3),
            "rps_mean": round(mean(rps_values), 3),
        }

        lines.extend(
            [
                "## Serving Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Samples | {summary['samples']} |",
                f"| TTFT p50 (ms) | {summary['ttft_p50_ms']} |",
                f"| TTFT p95 (ms) | {summary['ttft_p95_ms']} |",
                f"| ITL p50 (ms) | {summary['itl_p50_ms']} |",
                f"| ITL p95 (ms) | {summary['itl_p95_ms']} |",
                f"| TPS mean | {summary['tps_mean']} |",
                f"| RPS mean | {summary['rps_mean']} |",
                "",
            ]
        )
    elif track == "engine":
        pp_values = [_read_engine_metric(row, "pp_tps_mean") for row in rows]
        tg_values = [_read_engine_metric(row, "tg_tps_mean") for row in rows]
        pg_values = [_read_engine_metric(row, "pg_tps_mean") for row in rows]

        sorted_rows = sorted(rows, key=lambda r: _read_engine_metric(r, "tg_tps_mean"), reverse=True)
        best_row = sorted_rows[0]
        best_cfg = _read_engine_config(best_row)

        summary = {
            "run_id": str(rows[0].get("run_id", "unknown")),
            "track": track,
            "samples": len(rows),
            "pp_tps_mean": round(mean(pp_values), 3),
            "tg_tps_mean": round(mean(tg_values), 3),
            "pg_tps_mean": round(mean(pg_values), 3),
            "best_tg_tps_mean": round(_read_engine_metric(best_row, "tg_tps_mean"), 3),
            "best_threads": best_cfg["threads"],
            "best_batch": best_cfg["batch"],
            "best_ubatch": best_cfg["ubatch"],
            "best_ngl": best_cfg["ngl"],
        }

        lines.extend(
            [
                "## Engine Summary",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Samples | {summary['samples']} |",
                f"| PP mean t/s | {summary['pp_tps_mean']} |",
                f"| TG mean t/s | {summary['tg_tps_mean']} |",
                f"| PG mean t/s | {summary['pg_tps_mean']} |",
                f"| Best TG t/s | {summary['best_tg_tps_mean']} |",
                f"| Best Config (t,b,ub,ngl) | {summary['best_threads']},{summary['best_batch']},{summary['best_ubatch']},{summary['best_ngl']} |",
                "",
                f"## Top-{top_k} Configs (by TG t/s)",
                "",
                "| Rank | TG t/s | PP t/s | PG t/s | Threads | Batch | UBatch | NGL |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )

        for rank, row in enumerate(sorted_rows[:top_k], start=1):
            cfg = _read_engine_config(row)
            lines.append(
                f"| {rank} | {round(_read_engine_metric(row, 'tg_tps_mean'), 3)} | "
                f"{round(_read_engine_metric(row, 'pp_tps_mean'), 3)} | "
                f"{round(_read_engine_metric(row, 'pg_tps_mean'), 3)} | "
                f"{cfg['threads']} | {cfg['batch']} | {cfg['ubatch']} | {cfg['ngl']} |"
            )
        lines.append("")
    else:
        raise ValueError(f"unsupported track for report: {track}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    report_md = out_dir / "report.md"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    repro_commands: list[str] = []
    if run_meta:
        repro = run_meta.get("reproducibility", {})
        if isinstance(repro, dict):
            commands = repro.get("commands", [])
            if isinstance(commands, list):
                repro_commands = [str(cmd) for cmd in commands]

    if repro_commands:
        lines.extend([
            "## Reproduction Commands",
            "```bash",
            *repro_commands,
            "```",
            "",
        ])

    report_md.write_text("\n".join(lines), encoding="utf-8")
    return summary_csv, report_md

def validate_metrics_file(input_path: Path, schema_path: Path) -> int:
    if jsonschema is None:
        raise RuntimeError("jsonschema is required. Install: python -m pip install jsonschema")
    if not input_path.exists():
        raise FileNotFoundError(f"metrics input not found: {input_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema not found: {schema_path}")

    schema = read_json(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    rows = read_jsonl_with_lineno(input_path)
    if not rows:
        raise ValueError(f"no rows found: {input_path}")

    errors: list[str] = []
    for lineno, payload in rows:
        for err in validator.iter_errors(payload):
            path = ".".join(str(p) for p in err.absolute_path) or "$"
            errors.append(f"{input_path}:{lineno}:{path}: {err.message}")

    if errors:
        raise ValueError("\n".join(errors))
    return len(rows)

def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path

def validate_json_file(input_path: Path, schema_path: Path) -> None:
    if jsonschema is None:
        raise RuntimeError("jsonschema is required. Install: python -m pip install jsonschema")
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema not found: {schema_path}")

    schema = read_json(schema_path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{input_path}: invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{input_path}: root must be a JSON object")

    validator = jsonschema.Draft202012Validator(schema)
    errors: list[str] = []
    for err in sorted(validator.iter_errors(payload), key=lambda e: list(e.absolute_path)):
        path = _json_path(err.absolute_path)
        errors.append(f"{input_path}:{path}: {err.message}")
    if errors:
        raise ValueError("\n".join(errors))

def is_mlc_backend(name: str) -> bool:
    normalized = name.strip().lower()
    return normalized in {"mlc", "mlc_llm", "mlc-server", "mlc_server"}


def validate_mlc_managed_model_ref(model_ref: str) -> None:
    if not model_ref.startswith("HF://"):
        return
    suffix = model_ref[len("HF://") :].strip()
    if "/" not in suffix:
        raise ValueError(
            f"invalid MLC HF model reference `{model_ref}`: expected `HF://<org>/<model>`. "
            "Example: `HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC`"
        )
    org, model = suffix.split("/", 1)
    if not org or not model:
        raise ValueError(
            f"invalid MLC HF model reference `{model_ref}`: expected `HF://<org>/<model>`. "
            "Example: `HF://mlc-ai/Qwen2.5-7B-Instruct-q4f16_1-MLC`"
        )


def validate_mlc_model_lib_path(model_lib: str | None) -> str | None:
    if model_lib is None:
        return None
    normalized = model_lib.strip()
    if not normalized:
        return None
    model_lib_path = Path(normalized).expanduser().resolve()
    if not model_lib_path.exists() or not model_lib_path.is_file():
        raise ValueError(f"invalid --mlc-model-lib path (file not found): {model_lib_path}")
    return str(model_lib_path)


def probe_mlc_serve_capabilities(
    server_bin: str,
    server_bin_args: str,
    timeout_sec: float = 20.0,
) -> dict[str, Any]:
    cmd = split_args(server_bin) + split_args(server_bin_args) + ["--help"]
    commandline = " ".join(shlex.quote(part) for part in cmd)
    result: dict[str, Any] = {
        "probe_ok": False,
        "probe_returncode": None,
        "commandline": commandline,
        "help_excerpt": "",
        "supports_opt": False,
        "supports_device": False,
        "supports_model_lib": False,
    }
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
    except FileNotFoundError as exc:
        result["probe_error"] = str(exc)
        return result
    except subprocess.TimeoutExpired:
        result["probe_error"] = f"timed out after {timeout_sec}s"
        return result

    help_text = f"{proc.stdout}\n{proc.stderr}".strip()
    normalized = help_text.lower()
    result["probe_returncode"] = int(proc.returncode)
    result["probe_ok"] = bool(help_text) and ("usage:" in normalized or proc.returncode == 0)
    if help_text:
        excerpt_lines = help_text.splitlines()[:40]
        result["help_excerpt"] = "\n".join(excerpt_lines)
    result["supports_opt"] = "--opt" in normalized
    result["supports_device"] = "--device" in normalized
    result["supports_model_lib"] = "--model-lib" in normalized
    return result


def resolve_mlc_runtime_args(
    args: argparse.Namespace,
    capabilities: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    probe_ok = bool(capabilities.get("probe_ok", False))
    supports_opt = bool(capabilities.get("supports_opt", False)) if probe_ok else True
    supports_device = bool(capabilities.get("supports_device", False)) if probe_ok else True
    supports_model_lib = bool(capabilities.get("supports_model_lib", False)) if probe_ok else True

    downgraded_flags: list[str] = []
    effective_opt: str | None = None
    effective_device: str | None = None
    effective_model_lib: str | None = None

    requested_opt = str(getattr(args, "mlc_opt", "") or "").strip()
    if requested_opt:
        if supports_opt:
            effective_opt = requested_opt
        else:
            downgraded_flags.append("mlc_opt")

    requested_device = str(getattr(args, "mlc_device", "") or "").strip()
    if requested_device:
        if supports_device:
            effective_device = requested_device
        else:
            downgraded_flags.append("mlc_device")

    requested_model_lib = str(getattr(args, "mlc_model_lib", "") or "").strip()
    if requested_model_lib:
        if supports_model_lib:
            effective_model_lib = validate_mlc_model_lib_path(requested_model_lib)
        else:
            downgraded_flags.append("mlc_model_lib")

    runtime_effective = {
        "effective_opt": effective_opt,
        "effective_device": effective_device,
        "effective_model_lib": effective_model_lib,
    }
    return runtime_effective, downgraded_flags


def is_mlc_jit_compile_failure(text: str) -> bool:
    normalized = text.lower()
    signatures = [
        "vuid-standalonespirv-memorysemantics-10866",
        "spv_success",
        "tvm.error.internalerror",
        "mlc_llm compile",
    ]
    return any(signature in normalized for signature in signatures)


def _build_mlc_compile_overrides(args: argparse.Namespace) -> str:
    source = build_mlc_overrides(args)
    mapped: dict[str, int] = {}
    if "max_num_sequence" in source:
        mapped["max_batch_size"] = int(source["max_num_sequence"])
    if "prefill_chunk_size" in source:
        mapped["prefill_chunk_size"] = int(source["prefill_chunk_size"])
    if "max_total_seq_length" in source:
        mapped["context_window_size"] = int(source["max_total_seq_length"])
    parts: list[str] = []
    for key in ["max_batch_size", "context_window_size", "prefill_chunk_size"]:
        if key in mapped:
            parts.append(f"{key}={mapped[key]}")
    return ";".join(parts)


def build_mlc_compile_command(
    *,
    args: argparse.Namespace,
    model_ref: str,
    output_path: Path,
    mlc_runtime_effective: dict[str, Any] | None = None,
) -> list[str]:
    cmd: list[str] = []
    cmd.extend(split_args(args.server_bin))
    base_args = split_args(args.server_bin_args)
    replaced = False
    for token in base_args:
        if token == "serve" and not replaced:
            cmd.append("compile")
            replaced = True
            continue
        cmd.append(token)
    if not replaced:
        cmd.append("compile")

    runtime = mlc_runtime_effective or {}
    device = str(runtime.get("effective_device") or "auto").strip() or "auto"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd.extend(
        [
            model_ref,
            "--device",
            device,
            "--opt",
            "O0",
            "--output",
            str(output_path),
        ]
    )
    overrides = _build_mlc_compile_overrides(args)
    if overrides:
        cmd.extend(["--overrides", overrides])
    return cmd


def run_mlc_compile_attempt(
    *,
    command: list[str],
    compile_dir: Path,
    attempt_name: str,
    env_overrides: dict[str, str] | None = None,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    compile_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = compile_dir / f"compile_{attempt_name}_stdout.log"
    stderr_path = compile_dir / f"compile_{attempt_name}_stderr.log"
    commandline = " ".join(shlex.quote(part) for part in command)

    env = os.environ.copy()
    safe_env_overrides: dict[str, str] = {}
    if env_overrides:
        for key, value in env_overrides.items():
            env_key = str(key).strip()
            env_val = str(value)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", env_key):
                raise ValueError(f"invalid env override key: {env_key!r}")
            if "\x00" in env_val or "\n" in env_val or "\r" in env_val:
                raise ValueError(f"invalid env override value for key: {env_key!r}")
            safe_env_overrides[env_key] = env_val
        env.update(safe_env_overrides)

    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=timeout_sec,
        )
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
        returncode = int(proc.returncode)
        timed_out = False
    except subprocess.TimeoutExpired as timeout_exc:
        stdout_text = timeout_exc.stdout or ""
        stderr_text = timeout_exc.stderr or ""
        if timeout_sec is not None:
            stderr_text = f"{stderr_text}\n[perflab] mlc compile timed out after {timeout_sec}s"
        else:
            stderr_text = f"{stderr_text}\n[perflab] mlc compile timed out"
        returncode = 124
        timed_out = True

    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    return {
        "attempt_name": attempt_name,
        "commandline": commandline,
        "returncode": returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "env_overrides": safe_env_overrides,
        "timed_out": timed_out,
        "timeout_sec": timeout_sec,
    }


def run_mlc_compile_with_retry(
    *,
    command: list[str],
    compile_dir: Path,
) -> dict[str, Any]:
    compile_timeout_sec = 1200.0
    timeout_raw = os.environ.get("PERFLAB_MLC_COMPILE_TIMEOUT_SEC", "").strip()
    if timeout_raw:
        try:
            parsed = float(timeout_raw)
            if parsed > 0:
                compile_timeout_sec = parsed
        except ValueError:
            pass

    def _extract_output_arg(cmd: list[str]) -> tuple[int | None, Path | None]:
        for flag in ("--output", "-o"):
            if flag in cmd:
                idx = cmd.index(flag)
                if idx + 1 < len(cmd):
                    return idx + 1, Path(cmd[idx + 1]).expanduser().resolve()
        return None, None

    def _is_ascii_only(value: str) -> bool:
        try:
            value.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    def _prepare_ascii_safe_command(cmd: list[str]) -> tuple[list[str], Path | None, Path | None]:
        output_idx, requested_output = _extract_output_arg(cmd)
        if output_idx is None or requested_output is None:
            return list(cmd), None, None
        if _is_ascii_only(str(requested_output)):
            return list(cmd), requested_output, None

        staging_root = Path(tempfile.gettempdir()) / "perflab_mlc_compile_ascii"
        staging_root.mkdir(parents=True, exist_ok=True)
        staged_name = f"{requested_output.stem}_{uuid4().hex}{requested_output.suffix or '.dll'}"
        staged_output = (staging_root / staged_name).resolve()

        rewritten = list(cmd)
        rewritten[output_idx] = str(staged_output)
        return rewritten, requested_output, staged_output

    def _read_text(path_str: str) -> str:
        path = Path(path_str)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    compile_command, requested_output_path, staged_output_path = _prepare_ascii_safe_command(command)

    attempts: list[dict[str, Any]] = []
    first_attempt = run_mlc_compile_attempt(
        command=compile_command,
        compile_dir=compile_dir,
        attempt_name="o0_default",
        env_overrides=None,
        timeout_sec=compile_timeout_sec,
    )
    attempts.append(first_attempt)
    selected_compile_strategy: str | None = None
    used_validation_bypass = False
    jit_signature_matched = False
    final_attempt = first_attempt

    if first_attempt["returncode"] == 0:
        selected_compile_strategy = "o0_default"
    else:
        first_text = _read_text(first_attempt["stdout_log"]) + "\n" + _read_text(first_attempt["stderr_log"])
        jit_signature_matched = is_mlc_jit_compile_failure(first_text)
        if jit_signature_matched:
            second_attempt = run_mlc_compile_attempt(
                command=compile_command,
                compile_dir=compile_dir,
                attempt_name="o0_disable_shader_validation",
                env_overrides={"TVM_VULKAN_DISABLE_SHADER_VALIDATION": "1"},
                timeout_sec=compile_timeout_sec,
            )
            attempts.append(second_attempt)
            final_attempt = second_attempt
            if second_attempt["returncode"] == 0:
                selected_compile_strategy = "o0_disable_shader_validation"
                used_validation_bypass = True

    success = int(final_attempt["returncode"]) == 0
    output_relocated = False
    effective_output_path = requested_output_path
    if success and staged_output_path is not None and requested_output_path is not None:
        # Keep runtime loading on the ASCII-safe staged path; non-ASCII paths may fail in downstream dynamic loading.
        effective_output_path = staged_output_path
        try:
            requested_output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_output_path, requested_output_path)
            output_relocated = True
        except OSError:
            output_relocated = False
    elif success and staged_output_path is not None:
        effective_output_path = staged_output_path
    result = {
        "success": success,
        "attempts": attempts,
        "selected_compile_strategy": selected_compile_strategy,
        "used_validation_bypass": used_validation_bypass,
        "final_returncode": int(final_attempt["returncode"]),
        "requested_output_path": str(requested_output_path) if requested_output_path else None,
        "staged_output_path": str(staged_output_path) if staged_output_path else None,
        "effective_output_path": str(effective_output_path) if effective_output_path else None,
        "output_relocated": output_relocated,
    }
    if not success:
        raise RuntimeError(
            f"mlc compile fallback failed after {len(attempts)} attempt(s) "
            f"(last code {final_attempt['returncode']}, jit_signature_matched={jit_signature_matched}). "
            f"See {final_attempt['stderr_log']}"
        )
    return result


def detect_global_git_https_rewrite() -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "config", "--global", "--get-regexp", "url"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    hits: list[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        normalized = line.lower()
        if "gitclone.com" not in normalized:
            continue
        if ".insteadof" not in normalized:
            continue
        if "https://" not in normalized:
            continue
        hits.append(line)
    return hits


def enforce_no_blocking_git_https_rewrite() -> None:
    hits = detect_global_git_https_rewrite()
    if not hits:
        return
    details = "\n".join(f"- {line}" for line in hits)
    raise ValueError(
        "blocking global git https rewrite detected (gitclone mirror).\n"
        f"{details}\n"
        "Fix commands:\n"
        "  git config --global --unset-all url.https://gitclone.com/.insteadof\n"
        "  git config --global --get-regexp url"
    )

def ping_url(url: str, timeout_sec: float = 2.0) -> bool:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return int(resp.status) == 200
    except urllib.error.HTTPError as exc:
        return int(exc.code) == 200
    except urllib.error.URLError:
        return False

def _read_log_tail(path: Path | None, max_lines: int = 40) -> str:
    if path is None:
        return ""
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def wait_for_server_ready(
    base_url: str,
    timeout_sec: int,
    interval_sec: float,
    managed_proc: subprocess.Popen[str] | None = None,
    managed_stderr_path: Path | None = None,
) -> None:
    target = base_url.rstrip("/")
    endpoints = ["/health", "/v1/models"]
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if managed_proc is not None:
            returncode = managed_proc.poll()
            if returncode is not None:
                stderr_tail = _read_log_tail(managed_stderr_path)
                details = (
                    f"\n--- server_stderr.log (tail) ---\n{stderr_tail}"
                    if stderr_tail
                    else ""
                )
                raise RuntimeError(f"managed server exited before ready (code {returncode}){details}")
        for endpoint in endpoints:
            if ping_url(f"{target}{endpoint}"):
                return
        time.sleep(max(0.1, interval_sec))

    joined = ", ".join(f"{target}{ep}" for ep in endpoints)
    raise TimeoutError(f"server health check timed out after {timeout_sec}s: {joined}")

def stop_managed_server(proc: subprocess.Popen[str] | None, timeout_sec: float = 8.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_sec)

def build_server_command(
    args: argparse.Namespace,
    model_ref: str,
    mlc_runtime_effective: dict[str, Any] | None = None,
) -> list[str]:
    cmd: list[str] = []
    cmd.extend(split_args(args.server_bin))
    cmd.extend(split_args(args.server_bin_args))
    if is_mlc_backend(args.backend):
        runtime = mlc_runtime_effective or {}
        effective_model_lib = runtime.get("effective_model_lib")
        effective_device = runtime.get("effective_device")
        effective_opt = runtime.get("effective_opt")
        mlc_model_lib = validate_mlc_model_lib_path(
            str(effective_model_lib) if effective_model_lib else None
        )
        cmd.extend(
            [
                model_ref,
                "--mode",
                args.mlc_mode,
                "--host",
                args.server_host,
                "--port",
                str(args.server_port),
            ]
        )
        if effective_device:
            cmd.extend(["--device", str(effective_device)])
        if effective_opt:
            cmd.extend(["--opt", str(effective_opt)])
        if mlc_model_lib is not None:
            cmd.extend(["--model-lib", mlc_model_lib])
        overrides = build_mlc_overrides(args)
        overrides_arg = build_mlc_overrides_arg(overrides)
        if overrides_arg:
            cmd.extend(["--overrides", overrides_arg])
    else:
        cmd.extend(["-m", model_ref, "--host", args.server_host, "--port", str(args.server_port)])
    cmd.extend(split_args(args.server_extra_args))
    return cmd

def make_repro_commands(
    out_dir: Path,
    run_json: dict[str, Any],
) -> list[str]:
    commands = [
        f"python harness/benchctl.py doctor --out {shlex.quote(str(out_dir))}",
        run_json["commandline"],
        f"python harness/benchctl.py validate --input {shlex.quote(str(out_dir / 'metrics.jsonl'))}",
        f"python harness/benchctl.py report --input {shlex.quote(str(out_dir))} --out {shlex.quote(str(out_dir))}",
    ]
    return commands

def maybe_sha256_model(model_path: Path) -> str:
    if model_path.exists() and model_path.is_file():
        return sha256_file(model_path)
    return "model-sha256-unknown"


def _expand_command_placeholders(command: str, env: dict[str, str]) -> str:
    pattern = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")
    return pattern.sub(lambda m: env.get(m.group(1), m.group(0)), command)


def _extract_set_env_prefix(command: str) -> tuple[dict[str, str], str]:
    """
    Parse Windows-style command prefixes like:
      set "KEY=value" && set FOO=bar && <actual command>
    and return (env_overrides, remaining_command).
    """
    env_updates: dict[str, str] = {}
    remaining = command.strip()
    pattern = re.compile(
        r'^\s*set\s+"?([A-Za-z_][A-Za-z0-9_]*)=(.*?)"?\s*&&\s*(.+)$',
        re.IGNORECASE | re.DOTALL,
    )
    while True:
        match = pattern.match(remaining)
        if not match:
            break
        key = match.group(1).strip()
        value = match.group(2)
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"invalid set-prefix env value for key: {key!r}")
        env_updates[key] = value
        remaining = match.group(3).strip()
    return env_updates, remaining


def _tokenize_tool_command(command: str, env: dict[str, str]) -> list[str]:
    expanded = _expand_command_placeholders(command, env)
    try:
        args = shlex.split(expanded, posix=False)
    except ValueError as exc:
        raise ValueError(f"invalid tool command quoting: {exc}") from exc
    if not args:
        raise ValueError("tool command is empty after parsing")
    return args


def _redact_sensitive_text(value: str) -> str:
    text = str(value)
    patterns = [
        re.compile(r"(?i)(--?(?:api[-_]?key|token|password|secret)\s*[= ]\s*)([^\s\"']+)"),
        re.compile(r"(?i)(authorization:\s*bearer\s+)([^\s\"']+)"),
    ]
    for pattern in patterns:
        text = pattern.sub(r"\1***", text)
    return text


def _format_command_for_meta(args: list[str]) -> str:
    return _redact_sensitive_text(" ".join(shlex.quote(part) for part in args))


def _redact_command_args(args: list[str]) -> list[str]:
    sensitive_flags = {
        "--api-key",
        "--apikey",
        "--token",
        "--password",
        "--secret",
    }
    redacted: list[str] = []
    mask_next = False
    for part in args:
        lowered = part.lower()
        if mask_next:
            redacted.append("***")
            mask_next = False
            continue
        if lowered in sensitive_flags:
            redacted.append(part)
            mask_next = True
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            if key.lower() in sensitive_flags and value:
                redacted.append(f"{key}=***")
                continue
        redacted.append(_redact_sensitive_text(part))
    return redacted


def _redact_env_overrides(env_overrides: dict[str, str]) -> dict[str, str]:
    sensitive_key_pattern = re.compile(r"(?i)(token|password|secret|api[_-]?key|authorization)")
    redacted: dict[str, str] = {}
    for key, value in env_overrides.items():
        key_text = str(key)
        if sensitive_key_pattern.search(key_text):
            redacted[key_text] = "***"
            continue
        redacted[key_text] = _redact_sensitive_text(str(value))
    return redacted


def _strip_shell_control_tokens(args: list[str]) -> list[str]:
    control_tokens = {"&&", "||", "|", ";"}
    stripped: list[str] = []
    for token in args:
        if token in control_tokens:
            break
        stripped.append(token)
    return stripped


def run_tool_command(
    command: str,
    raw_dir: Path,
    base_url: str,
    model: str,
    tool_output_jsonl: Path | None,
    aiperf_artifact_dir: Path | None,
    aiperf_records_jsonl: Path | None,
) -> None:
    env = os.environ.copy()
    env["PERFLAB_SERVER_BASE_URL"] = base_url.rstrip("/")
    env["PERFLAB_OPENAI_BASE_URL"] = f"{base_url.rstrip('/')}/v1"
    env["PERFLAB_MODEL"] = model
    if tool_output_jsonl is not None:
        env["PERFLAB_TOOL_OUTPUT_JSONL"] = str(tool_output_jsonl)
    if aiperf_artifact_dir is not None:
        env["PERFLAB_AIPERF_ARTIFACT_DIR"] = str(aiperf_artifact_dir)
    if aiperf_records_jsonl is not None:
        env["PERFLAB_AIPERF_RECORDS_JSONL"] = str(aiperf_records_jsonl)

    timeout_sec: float | None = None
    timeout_raw = env.get("PERFLAB_TOOL_TIMEOUT_SEC", "").strip()
    if timeout_raw:
        try:
            parsed = float(timeout_raw)
            if parsed > 0:
                timeout_sec = parsed
        except ValueError:
            timeout_sec = None

    expanded_command = _expand_command_placeholders(command, env).strip()
    command_env_overrides, expanded_command = _extract_set_env_prefix(expanded_command)
    if command_env_overrides:
        env.update(command_env_overrides)
    if not expanded_command:
        raise ValueError("tool command is empty")
    try:
        parsed_args = _tokenize_tool_command(expanded_command, env)
    except ValueError:
        parsed_args = [expanded_command]
    parsed_args = _strip_shell_control_tokens(parsed_args)
    if not parsed_args:
        raise ValueError("tool command is empty after control-token stripping")
    command_meta = _redact_sensitive_text(expanded_command)
    command_args_meta = _redact_command_args(parsed_args)
    command_env_overrides_meta = _redact_env_overrides(command_env_overrides)
    start_time = now_utc_iso()
    try:
        proc = subprocess.run(
            expanded_command,
            shell=False,
            text=True,
            capture_output=True,
            check=False,
            env=env,
            timeout=timeout_sec,
            encoding="utf-8",
            errors="replace",
        )
        end_time = now_utc_iso()
        (raw_dir / "tool_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (raw_dir / "tool_stderr.log").write_text(proc.stderr, encoding="utf-8")
        write_json(
            raw_dir / "tool_run_meta.json",
            {
                "command": command_meta,
                "command_args": command_args_meta,
                "returncode": proc.returncode,
                "timed_out": False,
                "timeout_sec": timeout_sec,
                "started_at_utc": start_time,
                "ended_at_utc": end_time,
                "server_base_url": env["PERFLAB_SERVER_BASE_URL"],
                "openai_base_url": env["PERFLAB_OPENAI_BASE_URL"],
                "model": env["PERFLAB_MODEL"],
                "tool_output_jsonl": env.get("PERFLAB_TOOL_OUTPUT_JSONL"),
                "aiperf_artifact_dir": env.get("PERFLAB_AIPERF_ARTIFACT_DIR"),
                "aiperf_records_jsonl": env.get("PERFLAB_AIPERF_RECORDS_JSONL"),
                "command_env_overrides": command_env_overrides_meta,
            },
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"tool command failed (code {proc.returncode}). "
                f"See {raw_dir / 'tool_stdout.log'} and {raw_dir / 'tool_stderr.log'}"
            )
    except subprocess.TimeoutExpired as exc:
        end_time = now_utc_iso()
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        (raw_dir / "tool_stdout.log").write_text(stdout_text, encoding="utf-8")
        (raw_dir / "tool_stderr.log").write_text(stderr_text, encoding="utf-8")
        write_json(
            raw_dir / "tool_run_meta.json",
            {
                "command": command_meta,
                "command_args": command_args_meta,
                "returncode": 124,
                "timed_out": True,
                "timeout_sec": timeout_sec,
                "started_at_utc": start_time,
                "ended_at_utc": end_time,
                "server_base_url": env["PERFLAB_SERVER_BASE_URL"],
                "openai_base_url": env["PERFLAB_OPENAI_BASE_URL"],
                "model": env["PERFLAB_MODEL"],
                "tool_output_jsonl": env.get("PERFLAB_TOOL_OUTPUT_JSONL"),
                "aiperf_artifact_dir": env.get("PERFLAB_AIPERF_ARTIFACT_DIR"),
                "aiperf_records_jsonl": env.get("PERFLAB_AIPERF_RECORDS_JSONL"),
                "command_env_overrides": command_env_overrides_meta,
            },
        )
        timeout_msg = f"{timeout_sec}s" if timeout_sec else "configured timeout"
        raise RuntimeError(
            f"tool command timed out after {timeout_msg}. "
            f"See {raw_dir / 'tool_stdout.log'} and {raw_dir / 'tool_stderr.log'}"
        ) from exc

def run_llama_bench_command(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-bench command failed (code {proc.returncode}). "
            f"See {stdout_path} and {stderr_path}"
        )

def build_llama_bench_command(
    *,
    args: argparse.Namespace,
    model_path: Path,
    repetitions: int,
    threads: int,
    batch: int,
    ubatch: int,
    ngl: int,
    prompt_tokens: int,
    output_tokens: int,
    test_kind: str,
) -> list[str]:
    cmd = split_args(args.bench_bin)
    cmd.extend(split_args(args.bench_bin_args))
    cmd.extend(
        [
            "-m",
            str(model_path),
            "-o",
            "jsonl",
            "-r",
            str(repetitions),
            "-t",
            str(threads),
            "-b",
            str(batch),
            "-ub",
            str(ubatch),
            "-ngl",
            str(ngl),
        ]
    )

    if test_kind == "pp":
        cmd.extend(["-p", str(prompt_tokens), "-n", "0"])
    elif test_kind == "tg":
        cmd.extend(["-p", "0", "-n", str(output_tokens)])
    elif test_kind == "pg":
        cmd.extend(["-pg", f"{prompt_tokens},{output_tokens}"])
    else:
        raise ValueError(f"unsupported llama-bench test kind: {test_kind}")

    cmd.extend(split_args(args.bench_extra_args))
    return cmd
