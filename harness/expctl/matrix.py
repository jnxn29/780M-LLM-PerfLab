from __future__ import annotations

import itertools
import re
from typing import Any


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_")
    return (text or "na")[:24]


def expand_matrix(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]
    keys = sorted(matrix.keys())
    values = [matrix[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def build_run_id(idx: int, item: dict[str, Any]) -> str:
    if not item:
        return f"exp_{idx:03d}"
    parts = [f"{k}_{_slug(v)}" for k, v in sorted(item.items())]
    return f"exp_{idx:03d}__" + "__".join(parts)


def build_run_plan(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    rows = expand_matrix(matrix)
    plan: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        plan.append({"run_id": build_run_id(idx, row), "params": row})
    return plan

