from __future__ import annotations

from typing import Any


def build_mlc_overrides(args: Any) -> dict[str, int]:
    overrides: dict[str, int] = {}
    if getattr(args, "mlc_max_num_sequence", None) is not None:
        overrides["max_num_sequence"] = int(args.mlc_max_num_sequence)
    if getattr(args, "mlc_max_total_seq_length", None) is not None:
        overrides["max_total_seq_length"] = int(args.mlc_max_total_seq_length)
    if getattr(args, "mlc_prefill_chunk_size", None) is not None:
        overrides["prefill_chunk_size"] = int(args.mlc_prefill_chunk_size)
    return overrides


def build_mlc_overrides_arg(overrides: dict[str, int]) -> str:
    keys = ["max_num_sequence", "max_total_seq_length", "prefill_chunk_size"]
    parts: list[str] = []
    for key in keys:
        if key in overrides:
            parts.append(f"{key}={overrides[key]}")
    return ";".join(parts)
