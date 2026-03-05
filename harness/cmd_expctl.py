from __future__ import annotations

import argparse
from pathlib import Path

from core import commandline_from_argv, now_utc_iso, write_json
from expctl.runner import run_experiment
from expctl.spec import load_exp_spec


def _safe_path(raw: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if ".." in path.parts:
        raise ValueError(f"unsafe path: {raw}")
    return path


def cmd_expctl(args: argparse.Namespace) -> int:
    spec_path = _safe_path(args.spec)
    out_root = _safe_path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_db_path = (
        _safe_path(args.run_db)
        if args.run_db
        else (out_root / "run_db.sqlite").resolve()
    )

    spec = load_exp_spec(spec_path)
    result = run_experiment(
        spec=spec,
        spec_path=spec_path,
        out_root=out_root,
        run_db_path=run_db_path,
        resume=bool(args.resume),
        fail_fast=bool(args.fail_fast),
        max_workers=int(args.max_workers),
    )
    write_json(
        out_root / "experiment_meta.json",
        {
            "generated_at_utc": now_utc_iso(),
            "commandline": commandline_from_argv(),
            "spec_path": str(spec_path),
            "out_root": str(out_root),
            "run_db": str(run_db_path),
            **result["meta"],
        },
    )
    (out_root / "exp_report.md").write_text(result["report_md"], encoding="utf-8")
    print(f"[expctl] wrote {out_root / 'experiment_meta.json'}")
    print(f"[expctl] wrote {out_root / 'exp_report.md'}")
    return int(result["exit_code"])

