from __future__ import annotations
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable
import json

log = logging.getLogger("eden.storage")

DEFAULT_DB = Path("data/backtests/backtests.sqlite").resolve()

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS runs (
      id TEXT PRIMARY KEY,
      created_at TEXT,
      symbol TEXT,
      timeframe TEXT,
      strategy TEXT,
      params_json TEXT,
      status TEXT,
      metrics_json TEXT,
      results_path TEXT,
      env_snapshot TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metrics (
      run_id TEXT,
      metric_name TEXT,
      metric_value REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS trades (
      run_id TEXT,
      timestamp TEXT,
      side TEXT,
      qty REAL,
      entry REAL,
      exit REAL,
      pnl REAL,
      tags_json TEXT
    );
    """
]


def _connect(db_path: Path | str = DEFAULT_DB):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    return con


def init_db(db_path: Path | str = DEFAULT_DB):
    con = _connect(db_path)
    try:
        cur = con.cursor()
        for ddl in SCHEMA:
            cur.execute(ddl)
        con.commit()
    finally:
        con.close()


def insert_run(run_id: str, payload: Dict[str, Any], db_path: Path | str = DEFAULT_DB):
    con = _connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO runs (id, created_at, symbol, timeframe, strategy, params_json, status, metrics_json, results_path, env_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                payload.get("created_at"),
                payload.get("symbol"),
                payload.get("timeframe"),
                payload.get("strategy"),
                json.dumps(payload.get("params", {})),
                payload.get("status"),
                json.dumps(payload.get("metrics", {})),
                payload.get("results_path"),
                json.dumps(payload.get("env_snapshot", {})),
            ),
        )
        con.commit()
    finally:
        con.close()


def insert_metrics(run_id: str, metrics: Dict[str, float], db_path: Path | str = DEFAULT_DB):
    con = _connect(db_path)
    try:
        cur = con.cursor()
        for k, v in metrics.items():
            cur.execute("INSERT INTO metrics (run_id, metric_name, metric_value) VALUES (?, ?, ?)", (run_id, k, float(v)))
        con.commit()
    finally:
        con.close()


def insert_trades(run_id: str, trades: Iterable[Dict[str, Any]], db_path: Path | str = DEFAULT_DB):
    con = _connect(db_path)
    try:
        cur = con.cursor()
        for t in trades:
            cur.execute(
                "INSERT INTO trades (run_id, timestamp, side, qty, entry, exit, pnl, tags_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    str(t.get("timestamp") or t.get("open_time")),
                    t.get("side"),
                    float(t.get("qty", 0.0)),
                    float(t.get("entry", t.get("entry_price", 0.0))),
                    float(t.get("exit", t.get("exit_price", 0.0))),
                    float(t.get("pnl", 0.0)),
                    json.dumps(t.get("tags", {})),
                ),
            )
        con.commit()
    finally:
        con.close()
