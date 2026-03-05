from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from core import now_utc_iso


def _conn(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path) -> None:
    with _conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, status TEXT, start_utc TEXT, end_utc TEXT,
              exit_code INTEGER, out_dir TEXT, params_json TEXT, blocker TEXT
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              run_id TEXT, key TEXT, path TEXT, artifact_exists INTEGER, size_bytes INTEGER
            );
            CREATE TABLE IF NOT EXISTS events (
              run_id TEXT, ts_utc TEXT, level TEXT, message TEXT
            );
            """
        )


def upsert_run_start(db_path: Path, run_id: str, out_dir: Path, params: dict[str, Any]) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs(run_id,status,start_utc,out_dir,params_json) VALUES(?,?,?,?,?)",
            (run_id, "running", now_utc_iso(), str(out_dir), json.dumps(params, ensure_ascii=False)),
        )


def upsert_run_end(db_path: Path, run_id: str, status: str, exit_code: int, blocker: str | None) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "UPDATE runs SET status=?, end_utc=?, exit_code=?, blocker=? WHERE run_id=?",
            (status, now_utc_iso(), int(exit_code), blocker, run_id),
        )


def insert_artifacts(db_path: Path, run_id: str, data: dict[str, dict[str, Any]]) -> None:
    with _conn(db_path) as conn:
        conn.execute("DELETE FROM artifacts WHERE run_id=?", (run_id,))
        for key, item in data.items():
            conn.execute(
                "INSERT INTO artifacts(run_id,key,path,artifact_exists,size_bytes) VALUES(?,?,?,?,?)",
                (run_id, key, item.get("path"), int(bool(item.get("exists"))), int(item.get("size_bytes") or 0)),
            )


def insert_event(db_path: Path, run_id: str, level: str, message: str) -> None:
    with _conn(db_path) as conn:
        conn.execute(
            "INSERT INTO events(run_id,ts_utc,level,message) VALUES(?,?,?,?)",
            (run_id, now_utc_iso(), level, message),
        )


def list_completed_runs(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    with _conn(db_path) as conn:
        rows = conn.execute("SELECT run_id FROM runs WHERE status='success'").fetchall()
    return {str(row[0]) for row in rows}
