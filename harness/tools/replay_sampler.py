#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import error path
    raise RuntimeError("pyyaml is required for replay_sampler.py") from exc

MAX_WORKERS_CAP = 16


def now_utc_iso(ts: float | None = None) -> str:
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_workload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"workload file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"workload root must be an object: {path}")

    prompt_tokens = int(payload.get("prompt_tokens", 200))
    output_tokens = int(payload.get("output_tokens", 200))
    repetitions = int(payload.get("repetitions", 1))
    raw_concurrency = payload.get("concurrency", [1])
    if isinstance(raw_concurrency, int):
        concurrency_list = [int(raw_concurrency)]
    elif isinstance(raw_concurrency, list):
        concurrency_list = [int(v) for v in raw_concurrency]
    else:
        raise ValueError("workload.concurrency must be int or list[int]")

    if prompt_tokens < 1 or output_tokens < 1:
        raise ValueError("prompt_tokens/output_tokens must be >= 1")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if not concurrency_list or any(v < 1 for v in concurrency_list):
        raise ValueError("concurrency must contain positive integers")

    return {
        "name": str(payload.get("name", path.stem)),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "repetitions": repetitions,
        "concurrency": concurrency_list,
    }


def build_prompt(prompt_tokens: int) -> str:
    # Keep request payload short for runtime stability; token fields are
    # recorded from usage when available and fall back to workload defaults.
    return (
        "PerfLab streaming benchmark request. "
        f"target_prompt_tokens={max(1, int(prompt_tokens))}."
    )


def _parse_stream_response(
    response: Any,
    *,
    request_id: str,
    started_at: float,
    fallback_prompt_tokens: int,
    fallback_output_tokens: int,
) -> dict[str, Any]:
    token_timestamps: list[float] = []
    usage_prompt_tokens: int | None = None
    usage_output_tokens: int | None = None

    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue

        data = line[5:].strip()
        if data == "[DONE]":
            break

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{request_id}: invalid SSE JSON chunk: {data}") from exc

        if isinstance(payload.get("error"), dict):
            raise RuntimeError(f"{request_id}: server error chunk: {payload['error']}")

        usage = payload.get("usage")
        if isinstance(usage, dict):
            try:
                if usage.get("prompt_tokens") is not None:
                    usage_prompt_tokens = int(usage["prompt_tokens"])
                if usage.get("completion_tokens") is not None:
                    usage_output_tokens = int(usage["completion_tokens"])
            except (TypeError, ValueError):
                pass

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            continue
        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str) and content:
            token_timestamps.append(time.perf_counter())

    finished_at = time.perf_counter()
    total_duration = max(finished_at - started_at, 1e-6)
    first_token_at = token_timestamps[0] if token_timestamps else None

    if first_token_at is None:
        ttft_ms = total_duration * 1000.0
        decode_duration = total_duration
    else:
        ttft_ms = max(0.0, (first_token_at - started_at) * 1000.0)
        decode_duration = max(finished_at - first_token_at, 1e-6)

    if len(token_timestamps) >= 2:
        gaps = [token_timestamps[i] - token_timestamps[i - 1] for i in range(1, len(token_timestamps))]
        itl_ms = (sum(gaps) / len(gaps)) * 1000.0
    else:
        itl_ms = 0.0

    output_tokens = usage_output_tokens
    if output_tokens is None or output_tokens <= 0:
        output_tokens = len(token_timestamps)
    if output_tokens <= 0:
        output_tokens = fallback_output_tokens

    prompt_tokens = usage_prompt_tokens
    if prompt_tokens is None or prompt_tokens <= 0:
        prompt_tokens = fallback_prompt_tokens

    tps = float(output_tokens) / decode_duration

    return {
        "request_id": request_id,
        "timestamp_utc": now_utc_iso(started_at),
        "prompt_tokens": int(prompt_tokens),
        "output_tokens": int(output_tokens),
        "ttft_ms": round(ttft_ms, 3),
        "itl_ms": round(itl_ms, 3),
        "tps": round(tps, 3),
    }


def sample_one_request(
    *,
    base_url: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    request_timeout_sec: float,
    request_id: str,
    fallback_prompt_tokens: int,
    fallback_output_tokens: int,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    body = {
        "model": model,
        "stream": True,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=request_timeout_sec) as response:
            status = int(getattr(response, "status", 200))
            if status != 200:
                raise RuntimeError(f"{request_id}: HTTP {status}")
            row = _parse_stream_response(
                response,
                request_id=request_id,
                started_at=started_at,
                fallback_prompt_tokens=fallback_prompt_tokens,
                fallback_output_tokens=fallback_output_tokens,
            )
            return row
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{request_id}: HTTPError {exc.code}: {body_text}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{request_id}: request failed: {exc.reason}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows replay sampler for servebench replay mode")
    parser.add_argument("--workload", required=True, help="Workload YAML path")
    parser.add_argument("--output-jsonl", default=os.getenv("PERFLAB_TOOL_OUTPUT_JSONL", ""), help="Replay JSONL output path")
    parser.add_argument("--base-url", default=os.getenv("PERFLAB_SERVER_BASE_URL", ""), help="Server base URL")
    parser.add_argument("--model", default=os.getenv("PERFLAB_MODEL", ""), help="Model id/path for OpenAI request")
    parser.add_argument("--request-timeout-sec", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic prompt generation")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_jsonl = str(args.output_jsonl).strip()
    if not output_jsonl:
        raise ValueError("missing output path: provide --output-jsonl or PERFLAB_TOOL_OUTPUT_JSONL")
    base_url = str(args.base_url).strip()
    if not base_url:
        raise ValueError("missing base URL: provide --base-url or PERFLAB_SERVER_BASE_URL")
    model = str(args.model).strip()
    if not model:
        raise ValueError("missing model: provide --model or PERFLAB_MODEL")

    random.seed(args.seed)
    workload = load_workload(Path(args.workload))
    prompt = build_prompt(workload["prompt_tokens"])
    timeout = float(args.request_timeout_sec)
    if timeout <= 0:
        raise ValueError("--request-timeout-sec must be > 0")

    rows: list[dict[str, Any]] = []
    for concurrency in workload["concurrency"]:
        for rep_idx in range(workload["repetitions"]):
            batch_started = time.perf_counter()
            effective_workers = max(1, min(int(concurrency), MAX_WORKERS_CAP))
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = []
                for req_idx in range(concurrency):
                    request_id = f"c{concurrency}-rep{rep_idx + 1}-req{req_idx + 1}"
                    futures.append(
                        executor.submit(
                            sample_one_request,
                            base_url=base_url,
                            model=model,
                            prompt_text=prompt,
                            max_tokens=int(workload["output_tokens"]),
                            request_timeout_sec=timeout,
                            request_id=request_id,
                            fallback_prompt_tokens=int(workload["prompt_tokens"]),
                            fallback_output_tokens=int(workload["output_tokens"]),
                        )
                    )

                batch_rows = [future.result() for future in futures]
            batch_elapsed = max(time.perf_counter() - batch_started, 1e-6)
            batch_rps = round(concurrency / batch_elapsed, 3)
            for row in batch_rows:
                row["concurrency"] = int(concurrency)
                row["rps"] = batch_rps
                rows.append(row)

    if not rows:
        raise RuntimeError("no replay rows generated")

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    print(f"[replay-sampler] wrote {output_path}")
    print(f"[replay-sampler] rows={len(rows)} workload={workload['name']} concurrency={workload['concurrency']} repetitions={workload['repetitions']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
