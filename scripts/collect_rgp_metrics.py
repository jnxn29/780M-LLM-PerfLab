#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

READ_CANDIDATES = [
    "dram_read_bytes",
    "dram read bytes",
    "dramreadbytes",
    "vram_read_bytes",
    "gpu_read_bytes",
    "read_bytes",
]
WRITE_CANDIDATES = [
    "dram_write_bytes",
    "dram write bytes",
    "dramwritebytes",
    "vram_write_bytes",
    "gpu_write_bytes",
    "write_bytes",
]
TOKEN_CANDIDATES = [
    "completion_tokens",
    "output_tokens",
    "generated_tokens",
    "token_count",
    "tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and summarize RGP DRAM metrics from CSV exports.")
    parser.add_argument(
        "--input-root",
        default="reports/rgp_raw",
        help="Directory containing raw RGP CSV files.",
    )
    parser.add_argument(
        "--out-root",
        default="reports/perf_timeline",
        help="Directory to write aggregated RGP summary/report artifacts.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_name(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {normalize_name(name): name for name in columns}
    for candidate in candidates:
        hit = normalized.get(normalize_name(candidate))
        if hit:
            return hit
    return None


def parse_file_stem(stem: str) -> tuple[str, str, str]:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "unknown", "unknown"
    profile = parts[-1]
    backend = parts[-2]
    snapshot = "_".join(parts[:-2])
    return snapshot, backend, profile


def safe_sum(frame: pd.DataFrame, column: str | None) -> float:
    if not column or column not in frame.columns:
        return float("nan")
    series = pd.to_numeric(frame[column], errors="coerce")
    return float(series.fillna(0).sum())


def plot_empty(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bytes_per_token(summary: pd.DataFrame, out_path: Path) -> None:
    rows = summary.copy()
    rows = rows.loc[rows["status"] == "ok"]
    rows = rows.loc[pd.to_numeric(rows["dram_bytes_per_token"], errors="coerce") > 0]
    if rows.empty:
        plot_empty(out_path, "RGP DRAM Bytes / Token", "No valid rows")
        return

    rows = rows.sort_values("dram_bytes_per_token", ascending=False)
    labels = rows.apply(lambda r: f"{r['backend']}:{r['profile_id']}", axis=1).tolist()
    values = rows["dram_bytes_per_token"].tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_title("RGP DRAM Bytes per Token")
    ax.set_xlabel("Profile")
    ax.set_ylabel("bytes / token")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_read_write(summary: pd.DataFrame, out_path: Path) -> None:
    rows = summary.copy()
    rows = rows.loc[pd.to_numeric(rows["dram_total_bytes"], errors="coerce") > 0]
    if rows.empty:
        plot_empty(out_path, "RGP DRAM Read/Write", "No valid read/write rows")
        return

    rows = rows.sort_values("dram_total_bytes", ascending=False)
    labels = rows.apply(lambda r: f"{r['backend']}:{r['profile_id']}", axis=1).tolist()
    read_values = rows["dram_read_bytes"].fillna(0).tolist()
    write_values = rows["dram_write_bytes"].fillna(0).tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, read_values, label="dram_read_bytes")
    ax.bar(labels, write_values, bottom=read_values, label="dram_write_bytes")
    ax.set_title("RGP DRAM Read/Write Bytes")
    ax.set_xlabel("Profile")
    ax.set_ylabel("bytes")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def build_report(summary: pd.DataFrame, report_path: Path) -> None:
    lines: list[str] = []
    lines.append("# RGP DRAM Metrics Report")
    lines.append("")
    lines.append(f"generated_at_utc: `{now_utc()}`")
    lines.append("")
    lines.append(f"- files_total: {len(summary)}")
    lines.append(f"- files_ok: {int((summary['status'] == 'ok').sum()) if not summary.empty else 0}")
    lines.append(
        f"- files_missing_columns: {int((summary['status'] == 'rgp_columns_missing').sum()) if not summary.empty else 0}"
    )
    lines.append("")

    if summary.empty:
        lines.append("- no RGP CSV files found under `reports/rgp_raw/`")
    else:
        lines.append("## Rows")
        lines.append("")
        lines.append(
            "| snapshot | backend | profile | status | dram_read_bytes | dram_write_bytes | dram_bytes_per_token | missing_columns |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | --- |")
        for _, row in summary.iterrows():
            read_val = row.get("dram_read_bytes")
            write_val = row.get("dram_write_bytes")
            bpt_val = row.get("dram_bytes_per_token")
            read_text = "n/a" if pd.isna(read_val) else f"{float(read_val):.0f}"
            write_text = "n/a" if pd.isna(write_val) else f"{float(write_val):.0f}"
            bpt_text = "n/a" if pd.isna(bpt_val) else f"{float(bpt_val):.3f}"
            lines.append(
                "| "
                + f"{row.get('snapshot_id','')} | {row.get('backend','')} | {row.get('profile_id','')} | "
                + f"{row.get('status','')} | {read_text} | {write_text} | {bpt_text} | "
                + f"{row.get('missing_columns','')} |"
            )
        lines.append("")

    lines.append("## Charts")
    lines.append("")
    lines.append("- `rgp_bytes_per_token.png`")
    lines.append("- `rgp_dram_read_write.png`")
    lines.append("")
    lines.append("![](rgp_bytes_per_token.png)")
    lines.append("")
    lines.append("![](rgp_dram_read_write.png)")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_rows(input_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted(input_root.glob("*.csv")):
        snapshot_id, backend, profile_id = parse_file_stem(csv_path.stem)
        base_row: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "backend": backend,
            "profile_id": profile_id,
            "source_file": str(csv_path),
            "status": "ok",
            "dram_read_bytes": float("nan"),
            "dram_write_bytes": float("nan"),
            "dram_total_bytes": float("nan"),
            "token_count": float("nan"),
            "dram_bytes_per_token": float("nan"),
            "missing_columns": "",
            "generated_at_utc": now_utc(),
        }
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            base_row["status"] = "rgp_columns_missing"
            base_row["missing_columns"] = f"read_error:{exc.__class__.__name__}"
            rows.append(base_row)
            continue

        columns = [str(c) for c in frame.columns]
        read_col = find_column(columns, READ_CANDIDATES)
        write_col = find_column(columns, WRITE_CANDIDATES)
        token_col = find_column(columns, TOKEN_CANDIDATES)

        missing: list[str] = []
        if not read_col:
            missing.append("dram_read_bytes")
        if not write_col:
            missing.append("dram_write_bytes")
        if not token_col:
            missing.append("token_count")

        read_bytes = safe_sum(frame, read_col)
        write_bytes = safe_sum(frame, write_col)
        total_bytes = read_bytes + write_bytes if not pd.isna(read_bytes) and not pd.isna(write_bytes) else float("nan")
        token_count = safe_sum(frame, token_col)
        bytes_per_token = float("nan")
        if not pd.isna(total_bytes) and not pd.isna(token_count) and token_count > 0:
            bytes_per_token = total_bytes / token_count

        base_row["dram_read_bytes"] = read_bytes
        base_row["dram_write_bytes"] = write_bytes
        base_row["dram_total_bytes"] = total_bytes
        base_row["token_count"] = token_count
        base_row["dram_bytes_per_token"] = bytes_per_token
        if missing:
            base_row["status"] = "rgp_columns_missing"
            base_row["missing_columns"] = ",".join(missing)

        rows.append(base_row)
    return rows


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(input_root=input_root) if input_root.exists() else []
    summary = pd.DataFrame(rows)

    summary_path = out_root / "rgp_summary.csv"
    report_path = out_root / "rgp_report.md"
    chart_bpt = out_root / "rgp_bytes_per_token.png"
    chart_rw = out_root / "rgp_dram_read_write.png"

    if summary.empty:
        summary = pd.DataFrame(
            columns=[
                "snapshot_id",
                "backend",
                "profile_id",
                "source_file",
                "status",
                "dram_read_bytes",
                "dram_write_bytes",
                "dram_total_bytes",
                "token_count",
                "dram_bytes_per_token",
                "missing_columns",
                "generated_at_utc",
            ]
        )

    summary.to_csv(summary_path, index=False, encoding="utf-8")
    plot_bytes_per_token(summary=summary, out_path=chart_bpt)
    plot_read_write(summary=summary, out_path=chart_rw)
    build_report(summary=summary, report_path=report_path)

    latest = {
        "generated_at_utc": now_utc(),
        "rows": int(len(summary)),
        "status_counts": summary["status"].value_counts().to_dict() if not summary.empty else {},
        "summary_path": str(summary_path),
        "report_path": str(report_path),
    }
    (out_root / "rgp_latest_snapshot.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[rgp] wrote: {summary_path}")
    print(f"[rgp] wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
