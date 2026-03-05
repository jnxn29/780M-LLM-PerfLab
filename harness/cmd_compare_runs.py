from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cmd_compare import (
    build_comparison_metric_rows,
    read_serving_summary,
    write_comparison_outputs,
)
from core import commandline_from_argv, now_utc_iso, read_json, write_json

FALLBACK_KEYS = [
    "ttft_from_time_to_first_output_token",
    "itl_from_inter_chunk_latency",
    "itl_from_tps",
    "tps_from_output_tokens_and_request_latency",
    "rps_from_request_latency",
    "prompt_tokens_from_input_token_count",
    "output_tokens_from_output_token_count",
]


def _read_run_json_if_exists(run_dir: Path) -> dict[str, Any]:
    run_path = run_dir / "run.json"
    if not run_path.exists():
        return {}
    payload = read_json(run_path)
    if isinstance(payload, dict):
        return payload
    return {}


def _resolve_label(run_dir: Path, explicit_label: str | None) -> str:
    if explicit_label:
        return explicit_label
    run_payload = _read_run_json_if_exists(run_dir)
    backend = run_payload.get("backend")
    if isinstance(backend, dict):
        name = backend.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return run_dir.name


def _collect_repro_commands(
    *,
    out_dir: Path,
    baseline_run: Path,
    candidate_run: Path,
    baseline_run_payload: dict[str, Any],
    candidate_run_payload: dict[str, Any],
    strict_aiperf_observed: bool,
    strict_aiperf_allow_fallback_keys: list[str],
) -> list[str]:
    command = (
        "python harness/benchctl.py compare-runs "
        f"--baseline-run {baseline_run} --candidate-run {candidate_run} --out {out_dir}"
    )
    if strict_aiperf_observed:
        command += " --strict-aiperf-observed"
    if strict_aiperf_allow_fallback_keys:
        command += " --strict-aiperf-allow-fallback "
        command += ",".join(strict_aiperf_allow_fallback_keys)

    commands = [command]
    baseline_cmd = baseline_run_payload.get("commandline")
    candidate_cmd = candidate_run_payload.get("commandline")
    if isinstance(baseline_cmd, str) and baseline_cmd.strip():
        commands.append(baseline_cmd.strip())
    if isinstance(candidate_cmd, str) and candidate_cmd.strip():
        commands.append(candidate_cmd.strip())
    return commands


def _validate_fallback_counts(meta_path: Path, side: str) -> dict[str, int]:
    payload = read_json(meta_path)
    if not isinstance(payload, dict):
        raise ValueError(f"{side}: invalid aiperf adapt meta (root must be object): {meta_path}")
    fallback_counts = payload.get("fallback_counts")
    if not isinstance(fallback_counts, dict):
        raise ValueError(f"{side}: missing fallback_counts in aiperf adapt meta: {meta_path}")

    parsed: dict[str, int] = {}
    for key in FALLBACK_KEYS:
        if key not in fallback_counts:
            raise ValueError(f"{side}: missing fallback_counts.{key} in {meta_path}")
        value = fallback_counts[key]
        try:
            count = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{side}: fallback_counts.{key} must be integer in {meta_path}") from exc
        if count < 0:
            raise ValueError(f"{side}: fallback_counts.{key} must be >= 0 in {meta_path}")
        parsed[key] = count
    return parsed


def _strict_aiperf_gate(
    *,
    baseline_run: Path,
    candidate_run: Path,
    allowed_fallback_keys: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": True,
        "pass": False,
        "allowed_fallback_keys": allowed_fallback_keys,
        "failure_reason": None,
        "baseline_fallback_counts": None,
        "candidate_fallback_counts": None,
        "allowed_hits": {"baseline": {}, "candidate": {}},
        "disallowed_hits": {"baseline": {}, "candidate": {}},
    }

    baseline_meta = baseline_run / "raw_tool_output" / "aiperf_adapt_meta.json"
    candidate_meta = candidate_run / "raw_tool_output" / "aiperf_adapt_meta.json"
    if not baseline_meta.exists():
        reason = f"baseline: missing aiperf_adapt_meta.json: {baseline_meta}"
        result["failure_reason"] = reason
        raise ValueError(reason)
    if not candidate_meta.exists():
        reason = f"candidate: missing aiperf_adapt_meta.json: {candidate_meta}"
        result["failure_reason"] = reason
        raise ValueError(reason)

    baseline_counts = _validate_fallback_counts(baseline_meta, "baseline")
    candidate_counts = _validate_fallback_counts(candidate_meta, "candidate")
    result["baseline_fallback_counts"] = baseline_counts
    result["candidate_fallback_counts"] = candidate_counts

    allowed_key_set = set(allowed_fallback_keys)
    baseline_allowed = {k: v for k, v in baseline_counts.items() if v > 0 and k in allowed_key_set}
    candidate_allowed = {k: v for k, v in candidate_counts.items() if v > 0 and k in allowed_key_set}
    baseline_disallowed = {k: v for k, v in baseline_counts.items() if v > 0 and k not in allowed_key_set}
    candidate_disallowed = {k: v for k, v in candidate_counts.items() if v > 0 and k not in allowed_key_set}
    result["allowed_hits"] = {"baseline": baseline_allowed, "candidate": candidate_allowed}
    result["disallowed_hits"] = {
        "baseline": baseline_disallowed,
        "candidate": candidate_disallowed,
    }

    if baseline_disallowed or candidate_disallowed:
        parts: list[str] = []
        if baseline_disallowed:
            baseline_hits = [f"{k}={v}" for k, v in baseline_disallowed.items()]
            parts.append("baseline fallback hits: " + ", ".join(baseline_hits))
        if candidate_disallowed:
            candidate_hits = [f"{k}={v}" for k, v in candidate_disallowed.items()]
            parts.append("candidate fallback hits: " + ", ".join(candidate_hits))
        reason = " ; ".join(parts) or "disallowed fallback hits found"
        result["failure_reason"] = reason
        raise ValueError(reason)

    result["pass"] = True
    return result


def _append_strict_section(report_path: Path, strict_info: dict[str, Any]) -> None:
    text = report_path.read_text(encoding="utf-8")
    lines = [
        "## Strict AIPerf Observed Gate",
        f"- enabled: `{bool(strict_info.get('enabled', False))}`",
        f"- pass: `{bool(strict_info.get('pass', False))}`",
        f"- allowed_fallback_keys: `{','.join(strict_info.get('allowed_fallback_keys', []))}`",
        f"- allowed_hits: `{json.dumps(strict_info.get('allowed_hits', {}), ensure_ascii=False)}`",
        f"- disallowed_hits: `{json.dumps(strict_info.get('disallowed_hits', {}), ensure_ascii=False)}`",
    ]
    failure_reason = strict_info.get("failure_reason")
    if failure_reason:
        lines.append(f"- failure_reason: `{failure_reason}`")
    report_path.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _parse_allowed_fallback_keys(raw: str) -> list[str]:
    if not raw.strip():
        return []
    parsed: list[str] = []
    for key in raw.split(","):
        normalized = key.strip()
        if not normalized:
            continue
        if normalized not in parsed:
            parsed.append(normalized)
    unknown = [k for k in parsed if k not in FALLBACK_KEYS]
    if unknown:
        raise ValueError(f"unknown strict allow fallback keys: {','.join(unknown)}")
    return parsed


def cmd_compare_runs(args: argparse.Namespace) -> int:
    baseline_run = Path(args.baseline_run)
    candidate_run = Path(args.candidate_run)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_run.exists():
        raise FileNotFoundError(f"baseline run dir not found: {baseline_run}")
    if not candidate_run.exists():
        raise FileNotFoundError(f"candidate run dir not found: {candidate_run}")

    run_id = args.run_id or (
        f"compare_runs_{now_utc_iso().replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')}"
    )

    baseline_summary = read_serving_summary(baseline_run / "summary.csv")
    candidate_summary = read_serving_summary(candidate_run / "summary.csv")
    rows, overall = build_comparison_metric_rows(
        baseline_summary=baseline_summary,
        candidate_summary=candidate_summary,
    )

    baseline_label = _resolve_label(baseline_run, args.baseline_label)
    candidate_label = _resolve_label(candidate_run, args.candidate_label)
    if baseline_label == candidate_label:
        raise ValueError("baseline and candidate labels must be different")

    baseline_run_payload = _read_run_json_if_exists(baseline_run)
    candidate_run_payload = _read_run_json_if_exists(candidate_run)
    allowed_fallback_keys = _parse_allowed_fallback_keys(args.strict_aiperf_allow_fallback)
    if allowed_fallback_keys and not args.strict_aiperf_observed:
        raise ValueError("--strict-aiperf-allow-fallback requires --strict-aiperf-observed")
    repro_commands = _collect_repro_commands(
        out_dir=out_dir,
        baseline_run=baseline_run,
        candidate_run=candidate_run,
        baseline_run_payload=baseline_run_payload,
        candidate_run_payload=candidate_run_payload,
        strict_aiperf_observed=bool(args.strict_aiperf_observed),
        strict_aiperf_allow_fallback_keys=allowed_fallback_keys,
    )

    strict_info: dict[str, Any] = {
        "enabled": bool(args.strict_aiperf_observed),
        "pass": None,
        "allowed_fallback_keys": allowed_fallback_keys,
        "baseline_fallback_counts": None,
        "candidate_fallback_counts": None,
        "allowed_hits": {"baseline": {}, "candidate": {}},
        "disallowed_hits": {"baseline": {}, "candidate": {}},
        "failure_reason": None,
    }

    if args.strict_aiperf_observed:
        strict_info = _strict_aiperf_gate(
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            allowed_fallback_keys=allowed_fallback_keys,
        )

    comparison_summary, comparison_report = write_comparison_outputs(
        out_dir=out_dir,
        run_id=run_id,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        baseline_run_dir=baseline_run,
        candidate_run_dir=candidate_run,
        rows=rows,
        overall=overall,
        source_label="compare_runs",
        source_value=f"{baseline_run} vs {candidate_run}",
        source_sha256=None,
        repro_commands=repro_commands,
    )

    _append_strict_section(comparison_report, strict_info)

    write_json(
        out_dir / "compare_meta.json",
        {
            "run_id": run_id,
            "timestamp_utc": now_utc_iso(),
            "commandline": commandline_from_argv(),
            "inputs": {
                "baseline_run": str(baseline_run),
                "candidate_run": str(candidate_run),
            },
            "runs": {
                "baseline": {"label": baseline_label, "dir": str(baseline_run)},
                "candidate": {"label": candidate_label, "dir": str(candidate_run)},
            },
            "commands": repro_commands,
            "outputs": {
                "comparison_summary_csv": str(comparison_summary),
                "comparison_report_md": str(comparison_report),
            },
            "overall": overall,
            "strict_aiperf_observed": strict_info,
        },
    )

    print(f"[compare-runs] wrote {comparison_summary}")
    print(f"[compare-runs] wrote {comparison_report}")
    print(f"[compare-runs] wrote {out_dir / 'compare_meta.json'}")
    return 0
