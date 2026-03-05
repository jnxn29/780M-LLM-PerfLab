from __future__ import annotations

from pathlib import Path
from typing import Any

from core import read_jsonl_with_lineno

def _read_int(payload: dict[str, Any], key: str, *, path: Path, lineno: int) -> int:
    if key not in payload:
        raise ValueError(f"{path}:{lineno}:{key}: missing field")
    try:
        return int(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{lineno}:{key}: must be an integer") from exc


def _read_float(payload: dict[str, Any], key: str, *, path: Path, lineno: int) -> float:
    if key not in payload:
        raise ValueError(f"{path}:{lineno}:{key}: missing field")
    try:
        return float(payload[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{lineno}:{key}: must be a number") from exc


def _classify_test(n_prompt: int, n_gen: int) -> str:
    if n_prompt > 0 and n_gen == 0:
        return "pp"
    if n_prompt == 0 and n_gen > 0:
        return "tg"
    if n_prompt > 0 and n_gen > 0:
        return "pg"
    raise ValueError("unsupported test row: expected pp/tg/pg classification")


def extract_llama_bench_metric(input_path: Path, expected_test: str) -> dict[str, float]:
    if expected_test not in {"pp", "tg", "pg"}:
        raise ValueError("expected_test must be one of: pp, tg, pg")
    if not input_path.exists():
        raise FileNotFoundError(f"llama-bench output not found: {input_path}")

    rows = read_jsonl_with_lineno(input_path)
    if not rows:
        raise ValueError(f"{input_path}: no rows found")

    matched: list[dict[str, float]] = []
    for lineno, payload in rows:
        n_prompt = _read_int(payload, "n_prompt", path=input_path, lineno=lineno)
        n_gen = _read_int(payload, "n_gen", path=input_path, lineno=lineno)
        avg_ts = _read_float(payload, "avg_ts", path=input_path, lineno=lineno)
        stddev_ts = _read_float(payload, "stddev_ts", path=input_path, lineno=lineno)

        test_kind = _classify_test(n_prompt, n_gen)
        if test_kind == expected_test:
            matched.append(
                {
                    "mean": avg_ts,
                    "stddev": stddev_ts,
                    "n_prompt": float(n_prompt),
                    "n_gen": float(n_gen),
                }
            )

    if not matched:
        raise ValueError(f"{input_path}: no `{expected_test}` rows found")
    if len(matched) > 1:
        raise ValueError(f"{input_path}: multiple `{expected_test}` rows found")

    return matched[0]
