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


RUN_ORDER = [
    "r1_baseline",
    "r2_flash_only",
    "r3_compile_only",
    "r4_flash_compile",
    "r5_phi3_confirm",
]
DEFAULT_REQUIRED_BACKENDS = ["llama", "mlc", "torch_rocm"]


def now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check release readiness gates.")
    parser.add_argument("--c06-meta", required=True, help="C06 rerun torch_push_pipeline_meta.json")
    parser.add_argument("--fourtrack-before-meta", required=True, help="Main-track before pipeline_meta.json")
    parser.add_argument("--fourtrack-after-meta", required=True, help="Main-track after pipeline_meta.json")
    parser.add_argument("--three-point-csv", required=True, help="three_point_metrics.csv")
    parser.add_argument("--repeat-stability-json", required=True, help="repeat_stability.json")
    parser.add_argument("--quality-report-json", required=True, help="quality gate report json")
    parser.add_argument(
        "--main-track-required-backends",
        default=",".join(DEFAULT_REQUIRED_BACKENDS),
        help="Comma-separated required backends for main-track gate.",
    )
    parser.add_argument(
        "--non-ort-release-gate-json",
        default="",
        help="Optional non-ORT release gate json (expects pass=true).",
    )
    parser.add_argument(
        "--mlflow-evidence-json",
        default="",
        help="Optional MLflow evidence json (expects mlflow_used=true).",
    )
    parser.add_argument(
        "--gpuz-bandwidth-gate-json",
        default="",
        help="Optional GPU-Z bandwidth gate json (expects pass=true).",
    )
    parser.add_argument(
        "--ort-release-gate-json",
        default="",
        help="Legacy optional ORT release gate json (compatibility path).",
    )
    parser.add_argument(
        "--input-path-output-mode",
        default="relative",
        choices=["relative", "basename", "raw"],
        help="How to serialize input paths in output json.",
    )
    parser.add_argument("--out-json", required=True, help="Output readiness json path")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid json: {path}: {exc}") from exc


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _normalize_backend(name: str) -> str:
    n = (name or "").strip().lower()
    mapping = {
        "llama_cpp": "llama",
        "mlc_llm": "mlc",
        "ort_dml": "ort",
        "llama": "llama",
        "mlc": "mlc",
        "ort": "ort",
        "torch_rocm": "torch_rocm",
    }
    return mapping.get(n, n)


def _check_c06(meta: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    runs = meta.get("runs") or {}
    for run_name in RUN_ORDER:
        run = runs.get(run_name)
        if not isinstance(run, dict):
            errors.append(f"missing run: {run_name}")
            continue
        stage_b = run.get("stage_b") or {}
        hard = (
            run.get("eligible") is True
            and str(run.get("torch_device") or "").lower() == "cuda"
            and run.get("fallback_triggered") is False
            and run.get("runtime_device_fallback") is False
            and stage_b.get("ttft_p50_ms") is not None
            and stage_b.get("tps_mean") is not None
        )
        if not hard:
            errors.append(f"hard gate failed: {run_name}")
    phi3 = meta.get("phi3_confirmation") or {}
    if phi3.get("passed") is not True:
        errors.append("phi3_confirmation.passed != true")
    return (len(errors) == 0, errors)


def _check_maintrack(meta: dict[str, Any], label: str, required_backends: list[str]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if meta.get("artifacts_ready") is not True:
        errors.append(f"{label}: artifacts_ready != true")
    progress = meta.get("progress") or {}
    if str(progress.get("state") or "").lower() != "completed":
        errors.append(f"{label}: progress.state != completed")
    stage_a = (meta.get("stage_results") or {}).get("stage_a") or []
    success_backends = set()
    for row in stage_a:
        if str(row.get("status") or "").lower() != "success":
            continue
        success_backends.add(_normalize_backend(str(row.get("backend") or "")))
    for backend in required_backends:
        if backend not in success_backends:
            errors.append(f"{label}: backend has no stage_a success: {backend}")
    return (len(errors) == 0, errors)


def _check_three_point_csv(path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]
    except FileNotFoundError:
        return False, [f"missing file: {path}"]
    required_cols = {"run_name", "C11_ttft", "C11_tps", "B0_ttft", "B0_tps"}
    if not required_cols.issubset(set(reader.fieldnames or [])):
        errors.append("three-point csv missing required columns")
    names = {str(row.get("run_name") or "").strip() for row in rows}
    for run_name in RUN_ORDER:
        if run_name not in names:
            errors.append(f"three-point csv missing run: {run_name}")
    return (len(errors) == 0, errors)


def _check_repeat_stability(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if payload.get("all_hard") is not True:
        errors.append("repeat_stability.all_hard != true")
    if payload.get("all_tps_95") is not True:
        errors.append("repeat_stability.all_tps_95 != true")
    if payload.get("repeat_stability_overall") is not True:
        errors.append("repeat_stability.repeat_stability_overall != true")
    return (len(errors) == 0, errors)


def _check_quality(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    required_true = ["ruff_pass", "tests_pass", "ci_contract_pass"]
    for key in required_true:
        if _to_bool(payload.get(key)) is not True:
            errors.append(f"quality gate failed: {key}")
    return (len(errors) == 0, errors)


def _check_simple_pass(payload: dict[str, Any], field_name: str, gate_name: str) -> tuple[bool, list[str]]:
    if payload.get(field_name) is True:
        return True, []
    return False, [f"{gate_name}.{field_name} != true"]


def _check_mlflow_evidence(payload: dict[str, Any]) -> tuple[bool, list[str]]:
    if payload.get("mlflow_used") is True:
        return True, []
    return False, ["mlflow_evidence.mlflow_used != true"]


def _parse_required_backends(raw: str) -> list[str]:
    items = [it.strip().lower() for it in str(raw).split(",") if it.strip()]
    if not items:
        return DEFAULT_REQUIRED_BACKENDS[:]
    return items


def _format_output_path(path: Path, *, mode: str, base_dir: Path) -> str:
    resolved = path.resolve()
    if mode == "raw":
        return str(resolved)
    if mode == "basename":
        return resolved.name
    try:
        return str(resolved.relative_to(base_dir.resolve())).replace("\\", "/")
    except ValueError:
        return resolved.name


def main() -> int:
    args = parse_args()
    out_json_path = Path(args.out_json).resolve()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    required_backends = _parse_required_backends(args.main_track_required_backends)

    try:
        c06_meta = _read_json(Path(args.c06_meta).resolve())
        before_meta = _read_json(Path(args.fourtrack_before_meta).resolve())
        after_meta = _read_json(Path(args.fourtrack_after_meta).resolve())
        repeat_stability = _read_json(Path(args.repeat_stability_json).resolve())
        quality_report = _read_json(Path(args.quality_report_json).resolve())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    non_ort_gate_payload: dict[str, Any] | None = None
    if str(args.non_ort_release_gate_json).strip():
        path = Path(args.non_ort_release_gate_json).resolve()
        try:
            non_ort_gate_payload = _read_json(path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    mlflow_evidence_payload: dict[str, Any] | None = None
    if str(args.mlflow_evidence_json).strip():
        path = Path(args.mlflow_evidence_json).resolve()
        try:
            mlflow_evidence_payload = _read_json(path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    gpuz_gate_payload: dict[str, Any] | None = None
    if str(args.gpuz_bandwidth_gate_json).strip():
        path = Path(args.gpuz_bandwidth_gate_json).resolve()
        try:
            gpuz_gate_payload = _read_json(path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    ort_gate_payload: dict[str, Any] | None = None
    if str(args.ort_release_gate_json).strip():
        path = Path(args.ort_release_gate_json).resolve()
        try:
            ort_gate_payload = _read_json(path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    checks: list[dict[str, Any]] = []
    ready = True

    ok_c06, err_c06 = _check_c06(c06_meta)
    checks.append({"name": "c06_push_gate", "pass": ok_c06, "errors": err_c06})
    ready = ready and ok_c06

    ok_before, err_before = _check_maintrack(before_meta, "maintrack_before", required_backends)
    checks.append({"name": "maintrack_before_gate", "pass": ok_before, "errors": err_before})
    ready = ready and ok_before

    ok_after, err_after = _check_maintrack(after_meta, "maintrack_after", required_backends)
    checks.append({"name": "maintrack_after_gate", "pass": ok_after, "errors": err_after})
    ready = ready and ok_after

    ok_csv, err_csv = _check_three_point_csv(Path(args.three_point_csv).resolve())
    checks.append({"name": "three_point_csv_gate", "pass": ok_csv, "errors": err_csv})
    ready = ready and ok_csv

    ok_repeat, err_repeat = _check_repeat_stability(repeat_stability)
    checks.append({"name": "repeat_stability_gate", "pass": ok_repeat, "errors": err_repeat})
    ready = ready and ok_repeat

    ok_quality, err_quality = _check_quality(quality_report)
    checks.append({"name": "quality_gate", "pass": ok_quality, "errors": err_quality})
    ready = ready and ok_quality

    if non_ort_gate_payload is not None:
        ok_non_ort, err_non_ort = _check_simple_pass(non_ort_gate_payload, "pass", "non_ort_release_gate")
        checks.append({"name": "non_ort_release_gate", "pass": ok_non_ort, "errors": err_non_ort})
        ready = ready and ok_non_ort

    if mlflow_evidence_payload is not None:
        ok_mlflow, err_mlflow = _check_mlflow_evidence(mlflow_evidence_payload)
        checks.append({"name": "mlflow_evidence_gate", "pass": ok_mlflow, "errors": err_mlflow})
        ready = ready and ok_mlflow

    if gpuz_gate_payload is not None:
        ok_gpuz, err_gpuz = _check_simple_pass(gpuz_gate_payload, "pass", "gpuz_bandwidth_gate")
        checks.append({"name": "gpuz_bandwidth_gate", "pass": ok_gpuz, "errors": err_gpuz})
        ready = ready and ok_gpuz

    if ort_gate_payload is not None:
        ok_ort, err_ort = _check_simple_pass(ort_gate_payload, "pass", "ort_release_gate")
        checks.append({"name": "ort_release_gate", "pass": ok_ort, "errors": err_ort})
        ready = ready and ok_ort

    payload: dict[str, Any] = {
        "generated_at_utc": now_utc_iso(),
        "ready": ready,
        "inputs_redacted": args.input_path_output_mode != "raw",
        "checks": checks,
        "inputs": {
            "c06_meta": _format_output_path(
                Path(args.c06_meta), mode=args.input_path_output_mode, base_dir=out_json_path.parent
            ),
            "fourtrack_before_meta": _format_output_path(
                Path(args.fourtrack_before_meta), mode=args.input_path_output_mode, base_dir=out_json_path.parent
            ),
            "fourtrack_after_meta": _format_output_path(
                Path(args.fourtrack_after_meta), mode=args.input_path_output_mode, base_dir=out_json_path.parent
            ),
            "three_point_csv": _format_output_path(
                Path(args.three_point_csv), mode=args.input_path_output_mode, base_dir=out_json_path.parent
            ),
            "repeat_stability_json": _format_output_path(
                Path(args.repeat_stability_json), mode=args.input_path_output_mode, base_dir=out_json_path.parent
            ),
            "quality_report_json": _format_output_path(
                Path(args.quality_report_json), mode=args.input_path_output_mode, base_dir=out_json_path.parent
            ),
            "main_track_required_backends": required_backends,
        },
    }
    if non_ort_gate_payload is not None:
        payload["inputs"]["non_ort_release_gate_json"] = _format_output_path(
            Path(args.non_ort_release_gate_json), mode=args.input_path_output_mode, base_dir=out_json_path.parent
        )
    if mlflow_evidence_payload is not None:
        payload["inputs"]["mlflow_evidence_json"] = _format_output_path(
            Path(args.mlflow_evidence_json), mode=args.input_path_output_mode, base_dir=out_json_path.parent
        )
    if gpuz_gate_payload is not None:
        payload["inputs"]["gpuz_bandwidth_gate_json"] = _format_output_path(
            Path(args.gpuz_bandwidth_gate_json), mode=args.input_path_output_mode, base_dir=out_json_path.parent
        )
    if ort_gate_payload is not None:
        payload["inputs"]["ort_release_gate_json"] = _format_output_path(
            Path(args.ort_release_gate_json), mode=args.input_path_output_mode, base_dir=out_json_path.parent
        )

    out_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"release_readiness_json: {out_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
