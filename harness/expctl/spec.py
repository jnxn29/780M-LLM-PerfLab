from __future__ import annotations

from pathlib import Path
from typing import Any, cast

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from expctl.types import ExpSpec

ALLOWED_COMMANDS = {"servebench", "enginebench", "compare-runs"}


def _as_dict(value: Any, *, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a map/object")
    return value


def _load_matrix(raw: dict[str, Any]) -> dict[str, list[Any]]:
    matrix: dict[str, list[Any]] = {}
    for key, value in raw.items():
        if isinstance(value, list):
            if not value:
                raise ValueError(f"matrix.{key} must not be empty")
            matrix[key] = value
            continue
        matrix[key] = [value]
    return matrix


def load_exp_spec(path: Path) -> ExpSpec:
    if yaml is None:
        raise RuntimeError("pyyaml is required for expctl. Install from environment.yml.")
    if not path.exists():
        raise FileNotFoundError(f"spec not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("spec root must be an object")
    version = str(raw.get("version") or "")
    if version != "exp_spec_v1":
        raise ValueError("spec.version must be exp_spec_v1")
    name = str(raw.get("name") or path.stem)
    task = _as_dict(raw.get("task"), field="task")
    command = str(task.get("command") or "")
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"task.command must be one of {sorted(ALLOWED_COMMANDS)}")
    base_args = _as_dict(task.get("base_args"), field="task.base_args")
    matrix = _load_matrix(_as_dict(raw.get("matrix"), field="matrix"))
    resources_raw = _as_dict(raw.get("resources"), field="resources")
    tracking_raw = _as_dict(raw.get("tracking"), field="tracking")
    mlflow_raw = _as_dict(tracking_raw.get("mlflow"), field="tracking.mlflow")
    profiling_raw = _as_dict(raw.get("profiling"), field="profiling")
    rgp_raw = _as_dict(profiling_raw.get("rgp"), field="profiling.rgp")
    gates_raw = _as_dict(raw.get("gates"), field="gates")
    spec: ExpSpec = cast(
        ExpSpec,
        {
            "version": "exp_spec_v1",
            "name": name,
            "task": {"command": command, "base_args": base_args},
            "matrix": matrix,
            "resources": {
                "enabled": bool(resources_raw.get("enabled", True)),
                "interval_sec": float(resources_raw.get("interval_sec", 5.0)),
            },
            "tracking": {
                "mlflow": {
                    "enabled": bool(mlflow_raw.get("enabled", False)),
                    "required": bool(mlflow_raw.get("required", False)),
                    "uri": (str(mlflow_raw["uri"]) if mlflow_raw.get("uri") else None),
                    "experiment_name": str(mlflow_raw.get("experiment_name", name)),
                }
            },
            "profiling": {
                "rgp": {
                    "enabled": bool(rgp_raw.get("enabled", False)),
                    "root_glob": str(rgp_raw.get("root_glob", "raw_profiles/rgp/*.rgp")),
                }
            },
            "gates": {
                "min_success_ratio": float(gates_raw.get("min_success_ratio", 1.0)),
                "require_artifacts_ready": bool(gates_raw.get("require_artifacts_ready", True)),
            },
        },
    )
    return spec

