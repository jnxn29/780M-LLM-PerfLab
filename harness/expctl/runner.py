from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from expctl.execute import build_cli_args, collect_artifacts, read_summary_metrics, redact_cli_args
from expctl.gates import evaluate_gates
from expctl.matrix import build_run_plan
from integrations.mlflow_logger import log_mlflow_run
from profiling.rgp_evidence import collect_rgp_evidence
from resources.sampler import ResourceSampler
from run_db import init_db, insert_artifacts, insert_event, list_completed_runs, upsert_run_end, upsert_run_start

ROOT = Path(__file__).resolve().parents[2]
BENCHCTL = ROOT / "harness" / "benchctl.py"


def _ensure_child(base: Path, child: Path) -> Path:
    b = base.resolve()
    c = child.resolve()
    if not str(c).startswith(str(b)):
        raise ValueError(f"path escapes output root: {c}")
    return c


def _render_report(name: str, rows: list[dict[str, Any]], gates: dict[str, Any]) -> str:
    lines = [f"# Experiment Report: {name}", "", f"- total: {gates['total_count']}", f"- success: {gates['success_count']}", f"- success_ratio: {gates['success_ratio']}", f"- passed: {gates['passed']}", "", "| run_id | status | exit_code | blocker |", "| --- | --- | ---: | --- |"]
    for row in rows:
        lines.append(f"| {row['run_id']} | {row['status']} | {row['exit_code']} | {row.get('blocker','')} |")
    return "\n".join(lines) + "\n"


def run_experiment(*, spec: dict[str, Any], spec_path: Path, out_root: Path, run_db_path: Path, resume: bool, fail_fast: bool, max_workers: int) -> dict[str, Any]:
    del spec_path
    init_db(run_db_path)
    run_plan = build_run_plan(spec["matrix"])
    done = list_completed_runs(run_db_path) if resume else set()
    rows: list[dict[str, Any]] = []
    if max_workers != 1:
        insert_event(run_db_path, "expctl", "warn", f"max_workers={max_workers} ignored in v1; running serial")
    with ResourceSampler(out_root / "resource.jsonl", spec["resources"]["enabled"], spec["resources"]["interval_sec"]):
        for item in run_plan:
            run_id = item["run_id"]
            run_dir = _ensure_child(out_root, out_root / "runs" / run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            if run_id in done:
                rows.append({"run_id": run_id, "status": "skipped", "exit_code": 0, "artifacts_ready": True})
                continue
            args = build_cli_args(spec["task"]["base_args"], item["params"], run_dir)
            argv = [sys.executable, str(BENCHCTL), spec["task"]["command"], *args]
            upsert_run_start(run_db_path, run_id, run_dir, item["params"])
            insert_event(run_db_path, run_id, "info", " ".join(redact_cli_args(argv)))
            proc = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, check=False)
            (run_dir / "expctl_stdout.log").write_text(proc.stdout or "", encoding="utf-8")
            (run_dir / "expctl_stderr.log").write_text(proc.stderr or "", encoding="utf-8")
            artifacts = collect_artifacts(spec["task"]["command"], run_dir)
            insert_artifacts(run_db_path, run_id, artifacts)
            artifacts_ready = all(bool(v.get("exists")) for v in artifacts.values())
            status = "success" if proc.returncode == 0 and artifacts_ready else "failed"
            blocker = None if status == "success" else ("artifacts_missing" if proc.returncode == 0 else "subprocess_failed")
            upsert_run_end(run_db_path, run_id, status, proc.returncode, blocker)
            metrics = read_summary_metrics(run_dir / "summary.csv")
            ml = spec["tracking"]["mlflow"]
            try:
                used_mlflow, ml_error = log_mlflow_run(enabled=ml["enabled"], required=ml["required"], uri=ml["uri"], experiment_name=ml["experiment_name"], run_name=run_id, params=item["params"], metrics=metrics, artifacts_dir=run_dir)
            except Exception as exc:
                if ml["required"]:
                    raise
                used_mlflow, ml_error = False, f"mlflow_runtime_error:{type(exc).__name__}"
            rgp_meta = collect_rgp_evidence(enabled=spec["profiling"]["rgp"]["enabled"], root_glob=spec["profiling"]["rgp"]["root_glob"], base_dir=run_dir)
            rows.append({"run_id": run_id, "status": status, "exit_code": proc.returncode, "artifacts_ready": artifacts_ready, "blocker": blocker, "metrics": metrics, "mlflow_used": used_mlflow, "mlflow_error": ml_error, "rgp": rgp_meta})
            if fail_fast and status != "success":
                break
    gates = evaluate_gates(run_results=rows, min_success_ratio=spec["gates"]["min_success_ratio"], require_artifacts_ready=spec["gates"]["require_artifacts_ready"])
    return {
        "exit_code": 0 if gates["passed"] else 1,
        "meta": {"name": spec["name"], "task": spec["task"]["command"], "run_count": len(rows), "runs": rows, "gates": gates},
        "report_md": _render_report(spec["name"], rows, gates),
    }
