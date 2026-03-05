from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from core import read_jsonl_with_lineno

TIME_TO_MS = {
    "ns": 1e-6,
    "us": 1e-3,
    "ms": 1.0,
    "s": 1000.0,
}

FALLBACK_KEYS = [
    "ttft_from_time_to_first_output_token",
    "ttft_from_request_latency",
    "itl_from_inter_chunk_latency",
    "itl_from_tps",
    "tps_from_output_tokens_and_request_latency",
    "rps_from_request_latency",
    "prompt_tokens_from_input_token_count",
    "output_tokens_from_output_token_count",
]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _metric_value(metrics: dict[str, Any], key: str, *, path: Path, lineno: int, required: bool) -> tuple[float, str | None]:
    if key not in metrics or metrics[key] is None:
        if required:
            raise ValueError(f"{path}:{lineno}:metrics.{key}: missing field")
        return 0.0, None

    raw_metric = metrics[key]
    if isinstance(raw_metric, dict):
        if "value" not in raw_metric:
            if required:
                raise ValueError(f"{path}:{lineno}:metrics.{key}.value: missing field")
            return 0.0, str(raw_metric.get("unit") or "")
        raw_value = raw_metric["value"]
        unit = raw_metric.get("unit")
    else:
        raw_value = raw_metric
        unit = None

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{lineno}:metrics.{key}.value: must be a number") from exc
    return value, str(unit) if unit is not None else None


def _metric_value_any(
    metrics: dict[str, Any],
    keys: Iterable[str],
    *,
    path: Path,
    lineno: int,
    required_path: str,
) -> tuple[float, str | None, str]:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            value, unit = _metric_value(metrics, key, path=path, lineno=lineno, required=True)
            return value, unit, key
    raise ValueError(f"{path}:{lineno}:{required_path}: missing field")


def _to_time_ms(value: float, unit: str | None, *, path: Path, lineno: int, key: str) -> float:
    if unit is None:
        raise ValueError(f"{path}:{lineno}:metrics.{key}.unit: missing time unit")

    normalized = unit.strip().lower()
    if normalized not in TIME_TO_MS:
        raise ValueError(f"{path}:{lineno}:metrics.{key}.unit: unsupported time unit `{unit}`")
    return value * TIME_TO_MS[normalized]


def _request_id(metadata: dict[str, Any], lineno: int) -> str:
    req_id = str(metadata.get("x_request_id") or "").strip()
    if req_id:
        return req_id
    if "session_num" in metadata:
        return str(metadata["session_num"])
    return str(lineno)


def _concurrency(payload: dict[str, Any], metadata: dict[str, Any], default_concurrency: int) -> int:
    candidates = [
        payload.get("concurrency"),
        payload.get("n_concurrency"),
        metadata.get("concurrency"),
        metadata.get("n_concurrency"),
        metadata.get("target_concurrency"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            value = int(candidate)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return default_concurrency


def _timestamp_utc(metadata: dict[str, Any], *, path: Path, lineno: int) -> str:
    start_ns = metadata.get("request_start_ns")
    if start_ns is None:
        return _now_utc_iso()

    try:
        start_ns_int = int(start_ns)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}:{lineno}:metadata.request_start_ns: must be integer nanoseconds") from exc

    try:
        dt = datetime.fromtimestamp(start_ns_int / 1_000_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise ValueError(f"{path}:{lineno}:metadata.request_start_ns: invalid timestamp") from exc
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _parse_inter_chunk_latency_ms(metrics: dict[str, Any], *, path: Path, lineno: int) -> tuple[float, bool]:
    if "inter_chunk_latency" not in metrics or metrics["inter_chunk_latency"] is None:
        return 0.0, False

    raw_metric = metrics["inter_chunk_latency"]
    unit: str | None = None
    values_raw: Any = raw_metric
    if isinstance(raw_metric, dict):
        if "value" not in raw_metric:
            raise ValueError(f"{path}:{lineno}:metrics.inter_chunk_latency.value: missing field")
        values_raw = raw_metric["value"]
        unit_raw = raw_metric.get("unit")
        unit = str(unit_raw) if unit_raw is not None else None

    if not isinstance(values_raw, list):
        raise ValueError(f"{path}:{lineno}:metrics.inter_chunk_latency.value: must be a number list")
    if not values_raw:
        raise ValueError(f"{path}:{lineno}:metrics.inter_chunk_latency.value: must not be empty")

    values: list[float] = []
    for idx, item in enumerate(values_raw):
        try:
            values.append(float(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{lineno}:metrics.inter_chunk_latency.value[{idx}]: must be a number"
            ) from exc

    return _to_time_ms(_mean(values), unit, path=path, lineno=lineno, key="inter_chunk_latency"), True


def adapt_aiperf_profile_export_jsonl(
    input_path: Path,
    output_path: Path,
    *,
    default_concurrency: int = 1,
) -> dict[str, Any]:
    if default_concurrency < 1:
        raise ValueError("default_concurrency must be >= 1")
    if not input_path.exists():
        raise FileNotFoundError(f"aiperf input not found: {input_path}")

    source_rows = read_jsonl_with_lineno(input_path)
    if not source_rows:
        raise ValueError(f"{input_path}: no rows found")

    converted: list[dict[str, Any]] = []
    skipped_rows = 0
    fallback_counts: dict[str, int] = {key: 0 for key in FALLBACK_KEYS}

    for lineno, payload in source_rows:
        if payload.get("error") not in (None, "", {}, []):
            skipped_rows += 1
            continue

        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{input_path}:{lineno}:metadata: must be an object")

        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{input_path}:{lineno}:metrics: must be an object")

        request_latency_ms = 0.0
        has_request_latency = False
        if "request_latency" in metrics and metrics["request_latency"] is not None:
            latency_raw, latency_unit = _metric_value(
                metrics,
                "request_latency",
                path=input_path,
                lineno=lineno,
                required=True,
            )
            request_latency_ms = _to_time_ms(
                latency_raw,
                latency_unit,
                path=input_path,
                lineno=lineno,
                key="request_latency",
            )
            has_request_latency = True

        ttft_ms = 0.0
        if "time_to_first_token" in metrics and metrics["time_to_first_token"] is not None:
            ttft_raw, ttft_unit = _metric_value(
                metrics,
                "time_to_first_token",
                path=input_path,
                lineno=lineno,
                required=True,
            )
            ttft_ms = _to_time_ms(ttft_raw, ttft_unit, path=input_path, lineno=lineno, key="time_to_first_token")
        elif "time_to_first_output_token" in metrics and metrics["time_to_first_output_token"] is not None:
            ttft_raw, ttft_unit = _metric_value(
                metrics,
                "time_to_first_output_token",
                path=input_path,
                lineno=lineno,
                required=True,
            )
            ttft_ms = _to_time_ms(
                ttft_raw,
                ttft_unit,
                path=input_path,
                lineno=lineno,
                key="time_to_first_output_token",
            )
            fallback_counts["ttft_from_time_to_first_output_token"] += 1
        elif has_request_latency:
            ttft_ms = request_latency_ms
            fallback_counts["ttft_from_request_latency"] += 1
        else:
            raise ValueError(
                f"{input_path}:{lineno}:metrics.time_to_first_token: missing field "
                "(fallback requires metrics.request_latency)"
            )

        prompt_tokens_raw, _ = _metric_value(
            metrics,
            "input_sequence_length",
            path=input_path,
            lineno=lineno,
            required=False,
        )
        if prompt_tokens_raw == 0.0 and ("input_token_count" in metrics):
            prompt_tokens_raw, _ = _metric_value(
                metrics,
                "input_token_count",
                path=input_path,
                lineno=lineno,
                required=False,
            )
            fallback_counts["prompt_tokens_from_input_token_count"] += 1

        output_tokens_raw, _ = _metric_value(
            metrics,
            "output_sequence_length",
            path=input_path,
            lineno=lineno,
            required=False,
        )
        if output_tokens_raw == 0.0 and ("output_token_count" in metrics):
            output_tokens_raw, _ = _metric_value(
                metrics,
                "output_token_count",
                path=input_path,
                lineno=lineno,
                required=False,
            )
            fallback_counts["output_tokens_from_output_token_count"] += 1

        tps = 0.0
        if "output_token_throughput_per_user" in metrics and metrics["output_token_throughput_per_user"] is not None:
            tps, _ = _metric_value(
                metrics,
                "output_token_throughput_per_user",
                path=input_path,
                lineno=lineno,
                required=True,
            )
        elif "output_token_throughput" in metrics and metrics["output_token_throughput"] is not None:
            tps, _ = _metric_value(
                metrics,
                "output_token_throughput",
                path=input_path,
                lineno=lineno,
                required=True,
            )
        elif has_request_latency and request_latency_ms > 0.0 and output_tokens_raw > 0.0:
            tps = output_tokens_raw / (request_latency_ms / 1000.0)
            fallback_counts["tps_from_output_tokens_and_request_latency"] += 1
        else:
            raise ValueError(
                f"{input_path}:{lineno}:metrics.output_token_throughput: missing field "
                "(fallback requires output_token_count/output_sequence_length + request_latency)"
            )

        itl_ms = 0.0
        if "inter_token_latency" in metrics and metrics["inter_token_latency"] is not None:
            itl_raw, itl_unit = _metric_value(
                metrics,
                "inter_token_latency",
                path=input_path,
                lineno=lineno,
                required=True,
            )
            itl_ms = _to_time_ms(
                itl_raw,
                itl_unit,
                path=input_path,
                lineno=lineno,
                key="inter_token_latency",
            )
        else:
            inter_chunk_ms, has_inter_chunk = _parse_inter_chunk_latency_ms(
                metrics,
                path=input_path,
                lineno=lineno,
            )
            if has_inter_chunk:
                itl_ms = inter_chunk_ms
                fallback_counts["itl_from_inter_chunk_latency"] += 1
            elif tps > 0.0:
                itl_ms = 1000.0 / tps
                fallback_counts["itl_from_tps"] += 1
            else:
                raise ValueError(
                    f"{input_path}:{lineno}:metrics.inter_token_latency: missing field "
                    "(fallback requires inter_chunk_latency or output throughput)"
                )

        if "request_throughput" in metrics:
            rps, _ = _metric_value(metrics, "request_throughput", path=input_path, lineno=lineno, required=True)
        elif has_request_latency:
            if request_latency_ms <= 0.0:
                raise ValueError(f"{input_path}:{lineno}:metrics.request_latency.value: must be > 0 for rps fallback")
            rps = 1000.0 / request_latency_ms
            fallback_counts["rps_from_request_latency"] += 1
        else:
            raise ValueError(
                f"{input_path}:{lineno}:metrics.request_throughput: missing field "
                "(fallback requires metrics.request_latency)"
            )

        converted.append(
            {
                "request_id": _request_id(metadata, lineno),
                "timestamp_utc": _timestamp_utc(metadata, path=input_path, lineno=lineno),
                "concurrency": _concurrency(payload, metadata, default_concurrency),
                "prompt_tokens": max(0, int(round(prompt_tokens_raw))),
                "output_tokens": max(0, int(round(output_tokens_raw))),
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "tps": tps,
                "rps": rps,
            }
        )

    if not converted:
        raise ValueError(f"{input_path}: no valid rows after filtering error rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in converted:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    return {
        "input_rows": len(source_rows),
        "output_rows": len(converted),
        "skipped_rows": skipped_rows,
        "fallback_counts": fallback_counts,
    }
