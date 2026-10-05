"""SQLite 存储层。

数据库文件路径由环境变量 BLAST_DB 指定（部署时指向挂载卷，默认 /data/blast.db）。
只做持久化与按条件查询，业务规则在 service 层。时间戳统一用 UTC ISO 字符串；
并发入库顺序以 measurements.id（AUTOINCREMENT，按提交先后）为准。
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

DB_PATH = os.environ.get("BLAST_DB", "/data/blast.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    lithology     TEXT NOT NULL DEFAULT '',
    decay         REAL NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS designs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id       INTEGER NOT NULL REFERENCES zones(id),
    name          TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    status        TEXT NOT NULL DEFAULT 'planned',   -- planned / blasted
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS design_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id     INTEGER NOT NULL REFERENCES designs(id),
    version       INTEGER NOT NULL,
    params        TEXT NOT NULL,                     -- JSON
    created_at    TEXT NOT NULL,
    UNIQUE(design_id, version)
);

CREATE TABLE IF NOT EXISTS factor_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id       INTEGER NOT NULL REFERENCES zones(id),
    version_index INTEGER NOT NULL,                  -- 分区内单调递增（仅比较用，撤回不重置）
    factor        REAL NOT NULL,
    evidence_id   INTEGER REFERENCES measurements(id),
    decay         REAL NOT NULL,
    created_at    TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,        -- 当前有效链上为 1
    chain         TEXT NOT NULL DEFAULT 'current',   -- current / withdrawn
    superseded_by INTEGER REFERENCES factor_versions(id),
    note          TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS measurements (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id     INTEGER NOT NULL REFERENCES designs(id),
    zone_id       INTEGER NOT NULL REFERENCES zones(id),
    points        TEXT NOT NULL,                     -- JSON 原始筛分数据
    fit           TEXT NOT NULL,                     -- JSON 拟合结果
    usable        INTEGER NOT NULL,
    x50           REAL,
    uniformity    REAL,
    g             REAL,                              -- 该次设计几何因子
    qweight       REAL,                              -- 拟合质量权重
    single_factor REAL,                              -- a_k = x50/g
    factor_version_id INTEGER REFERENCES factor_versions(id),
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'active',    -- active / withdrawn
    withdrawn_at  TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id         INTEGER NOT NULL REFERENCES designs(id),
    design_version    INTEGER NOT NULL,
    factor_version_id INTEGER NOT NULL REFERENCES factor_versions(id),
    zone_id           INTEGER NOT NULL,
    rock_factor       REAL NOT NULL,
    x50               REAL NOT NULL,
    uniformity        REAL NOT NULL,
    powder_factor     REAL NOT NULL,
    oversize_m        REAL NOT NULL,
    oversize_pct      REAL NOT NULL,
    oversize_limit    REAL NOT NULL,
    oversize_pass     INTEGER NOT NULL,
    passing_table     TEXT NOT NULL,                 -- JSON
    kind              TEXT NOT NULL DEFAULT 'manual',-- manual / batch
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fv_zone ON factor_versions(zone_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_fv_active ON factor_versions(zone_id, active);
CREATE INDEX IF NOT EXISTS idx_meas_zone ON measurements(zone_id, id);
CREATE INDEX IF NOT EXISTS idx_pred_design ON predictions(design_id, id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------- 小工具 ----------
def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def loads(s: str | None) -> Any:
    return json.loads(s) if s is not None else None


def insert(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()) -> int:
    cur = conn.execute(sql, list(params))
    return int(cur.lastrowid)


def get(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()):
    return conn.execute(sql, list(params)).fetchone()


def query(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = ()):
    return conn.execute(sql, list(params)).fetchall()
