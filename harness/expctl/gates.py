from __future__ import annotations

from typing import Any


def evaluate_gates(
    *,
    run_results: list[dict[str, Any]],
    min_success_ratio: float,
    require_artifacts_ready: bool,
) -> dict[str, Any]:
    total = len(run_results)
    success_count = sum(1 for row in run_results if row.get("status") == "success")
    ratio = (success_count / total) if total else 0.0
    pass_ratio = ratio >= min_success_ratio
    missing_artifacts = 0
    if require_artifacts_ready:
        missing_artifacts = sum(
            1 for row in run_results if row.get("status") == "success" and not row.get("artifacts_ready", False)
        )
    passed = pass_ratio and missing_artifacts == 0
    reasons: list[str] = []
    if not pass_ratio:
        reasons.append("min_success_ratio_not_met")
    if missing_artifacts > 0:
        reasons.append("artifacts_not_ready")
    return {
        "passed": passed,
        "success_count": success_count,
        "total_count": total,
        "success_ratio": round(ratio, 6),
        "missing_artifacts": missing_artifacts,
        "reasons": reasons,
    }

