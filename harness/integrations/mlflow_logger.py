from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def log_mlflow_run(
    *,
    enabled: bool,
    required: bool,
    uri: str | None,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts_dir: Path,
) -> tuple[bool, str | None]:
    if not enabled:
        return False, None
    try:
        mlflow = importlib.import_module("mlflow")
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch
        if required:
            raise RuntimeError("mlflow is required but unavailable") from exc
        return False, "mlflow_unavailable"

    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        for key, value in params.items():
            mlflow.log_param(str(key), str(value))
        for key, value in metrics.items():
            mlflow.log_metric(str(key), float(value))
        if artifacts_dir.exists():
            mlflow.log_artifacts(str(artifacts_dir))
    return True, None

