"""SQLite 持久化层。

数据库文件放在挂载卷（环境变量 BLAST_DB_PATH，默认 /data/blast.db）。

并发模型：SQLite 单文件 + 进程内一把可重入锁。所有写事务都在
``lock`` 下串行提交，保证"两次实测同时入库同一分区"的结果与按入库
顺序逐条处理完全一致（SQL 本身也通过 seq 列把顺序固化下来）。
读取不需要长事务，快照读即可。
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DB_PATH = os.environ.get("BLAST_DB_PATH", "/data/blast.db")


def utcnow_iso() -> str:
    """UTC 时间戳，定长 ISO 串，可按字典序排序。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00"


SCHEMA = """
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    lithology TEXT NOT NULL DEFAULT '',
    baseline_factor REAL NOT NULL,
    baseline_weight REAL NOT NULL DEFAULT 0.0,
    forgetting REAL NOT NULL DEFAULT 0.90,
    created_at TEXT NOT NULL
);

-- 分区折叠状态（每行一个分区，随每次入库原地更新）
CREATE TABLE IF NOT EXISTS zone_state (
    zone_id INTEGER PRIMARY KEY REFERENCES zones(id),
    fold_w REAL NOT NULL DEFAULT 0,
    fold_wl REAL NOT NULL DEFAULT 0,
    obs_count INTEGER NOT NULL DEFAULT 0,
    last_blast_id INTEGER
);

-- 系数版本：每次引起系数变化的事件（实测入库 / 撤回重推）都追加一行
CREATE TABLE IF NOT EXISTS factor_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    version_no INTEGER NOT NULL,           -- 分区内递增（含被取代行）
    factor REAL NOT NULL,
    event_type TEXT NOT NULL,              -- baseline / measurement / withdrawal
    blast_id INTEGER,                      -- 触发放点
    measurement_id INTEGER,
    superseded INTEGER NOT NULL DEFAULT 0, -- 撤回重推后旧行置 1
    lineage TEXT NOT NULL DEFAULT 'active',-- active / withdrawn / rebuilt
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_fv_zone ON factor_versions(zone_id, version_no);

CREATE TABLE IF NOT EXISTS designs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    implemented INTEGER NOT NULL DEFAULT 0,
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS design_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id INTEGER NOT NULL REFERENCES designs(id),
    version_no INTEGER NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(design_id, version_no)
);

-- 预测绑定 设计版本 + 系数版本（不可变快照）
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id INTEGER NOT NULL REFERENCES designs(id),
    design_version INTEGER NOT NULL,
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    factor_version_id INTEGER NOT NULL REFERENCES factor_versions(id),
    result_json TEXT NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 1, -- 该设计最新预测=1
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_design ON predictions(design_id, id);

CREATE TABLE IF NOT EXISTS blasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_id INTEGER NOT NULL REFERENCES designs(id),
    design_version INTEGER NOT NULL,
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    factor_version_id INTEGER NOT NULL REFERENCES factor_versions(id),
    fired_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    blast_id INTEGER NOT NULL REFERENCES blasts(id),
    zone_id INTEGER NOT NULL REFERENCES zones(id),
    seq_in_zone INTEGER NOT NULL,          -- 入库顺序（并发时由写锁确定）
    bins_json TEXT NOT NULL,
    fit_json TEXT NOT NULL,
    fit_usable INTEGER NOT NULL,
    implied_factor REAL,
    factor_version_id INTEGER REFERENCES factor_versions(id),
    withdrawn INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(zone_id, seq_in_zone)
);
"""


class NotFound(KeyError):
    """记录不存在。"""


class Storage:
    """薄 SQLite 封装：连接持有、写锁、行映射与少量领域查询。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=10000")
        with self._conn:
            self._conn.executescript(SCHEMA)

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    # ---- 基础工具 ----
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, tuple(params))

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        cur = self._conn.execute(sql, tuple(params))
        return cur.fetchone()

    def query_all(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        cur = self._conn.execute(sql, tuple(params))
        return cur.fetchall()

    def get_one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row:
        row = self.query_one(sql, params)
        if row is None:
            raise NotFound(sql)
        return row

    # ---- 分区 ----
    def create_zone(
        self,
        name: str,
        baseline_factor: float,
        forgetting: float = 0.90,
        baseline_weight: float = 0.0,
        lithology: str = "",
    ) -> int:
        ts = utcnow_iso()
        cur = self.execute(
            """INSERT INTO zones(name, lithology, baseline_factor,
                                 baseline_weight, forgetting, created_at)
               VALUES (?,?,?,?,?,?)""",
            (name, lithology, baseline_factor, baseline_weight, forgetting, ts),
        )
        zone_id = cur.lastrowid
        self.execute(
            "INSERT INTO zone_state(zone_id) VALUES (?)", (zone_id,)
        )
        v = self.execute(
            """INSERT INTO factor_versions(zone_id, version_no, factor,
                  event_type, note, created_at)
               VALUES (?,?,?,?, '基线系数', ?)""",
            (zone_id, 1, baseline_factor, "baseline", ts),
        )
        return zone_id

    def get_zone(self, zone_id: int) -> sqlite3.Row:
        return self.get_one("SELECT * FROM zones WHERE id=?", (zone_id,))

    def get_zone_by_name(self, name: str) -> sqlite3.Row:
        return self.get_one("SELECT * FROM zones WHERE name=?", (name,))

    def list_zones(self) -> list[sqlite3.Row]:
        return self.query_all("SELECT * FROM zones ORDER BY id")

    def get_state(self, zone_id: int) -> sqlite3.Row:
        return self.get_one("SELECT * FROM zone_state WHERE zone_id=?", (zone_id,))

    def update_state(
        self,
        zone_id: int,
        fold_w: float,
        fold_wl: float,
        obs_count: int,
        last_blast_id: int | None,
    ) -> None:
        self.execute(
            """UPDATE zone_state SET fold_w=?, fold_wl=?, obs_count=?,
                                     last_blast_id=?
               WHERE zone_id=?""",
            (fold_w, fold_wl, obs_count, last_blast_id, zone_id),
        )

    # ---- 系数版本 ----
    def next_factor_version_no(self, zone_id: int) -> int:
        row = self.query_one(
            "SELECT COALESCE(MAX(version_no),0)+1 AS v FROM factor_versions WHERE zone_id=?",
            (zone_id,),
        )
        return int(row["v"])

    def add_factor_version(
        self,
        zone_id: int,
        factor: float,
        event_type: str,
        blast_id: int | None = None,
        measurement_id: int | None = None,
        lineage: str = "active",
        note: str = "",
        created_at: str | None = None,
    ) -> int:
        ts = created_at or utcnow_iso()
        no = self.next_factor_version_no(zone_id)
        cur = self.execute(
            """INSERT INTO factor_versions(zone_id, version_no, factor,
                  event_type, blast_id, measurement_id, lineage, note, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (zone_id, no, factor, event_type, blast_id, measurement_id,
             lineage, note, ts),
        )
        return cur.lastrowid

    def active_factor_version(self, zone_id: int) -> sqlite3.Row:
        """当前生效（最新未取代）版本。"""
        return self.get_one(
            """SELECT * FROM factor_versions
               WHERE zone_id=? AND superseded=0
               ORDER BY version_no DESC LIMIT 1""",
            (zone_id,),
        )

    def factor_version(self, version_id: int) -> sqlite3.Row:
        return self.get_one("SELECT * FROM factor_versions WHERE id=?", (version_id,))

    def factor_history(
        self, zone_id: int, include_superseded: bool = False
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM factor_versions WHERE zone_id=?"
        if not include_superseded:
            sql += " AND superseded=0"
        sql += " ORDER BY version_no"
        return self.query_all(sql, (zone_id,))

    def factor_version_as_of(self, zone_id: int, ts: str) -> sqlite3.Row | None:
        """查询在 ts 时刻"当时能看到的最新版本"（被取代的行也参与）。"""
        return self.query_one(
            """SELECT * FROM factor_versions
               WHERE zone_id=? AND created_at <= ?
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (zone_id, ts),
        )

    # ---- 设计 ----
    def create_design(self, name: str, zone_id: int, params_json: str) -> int:
        ts = utcnow_iso()
        cur = self.execute(
            "INSERT INTO designs(name, zone_id, created_at) VALUES (?,?,?)",
            (name, zone_id, ts),
        )
        design_id = cur.lastrowid
        self.execute(
            """INSERT INTO design_versions(design_id, version_no, params_json, created_at)
               VALUES (?,1,?,?)""",
            (design_id, params_json, ts),
        )
        return design_id

    def get_design(self, design_id: int) -> sqlite3.Row:
        return self.get_one("SELECT * FROM designs WHERE id=?", (design_id,))

    def get_design_version(
        self, design_id: int, version_no: int | None = None
    ) -> sqlite3.Row:
        if version_no is None:
            d = self.get_design(design_id)
            version_no = d["current_version"]
        return self.get_one(
            """SELECT * FROM design_versions
               WHERE design_id=? AND version_no=?""",
            (design_id, version_no),
        )

    def add_design_version(self, design_id: int, params_json: str) -> int:
        ts = utcnow_iso()
        d = self.get_design(design_id)
        no = int(d["current_version"]) + 1
        self.execute(
            """INSERT INTO design_versions(design_id, version_no, params_json, created_at)
               VALUES (?,?,?,?)""",
            (design_id, no, params_json, ts),
        )
        self.execute(
            "UPDATE designs SET current_version=? WHERE id=?", (no, design_id)
        )
        return no

    def list_open_designs(self) -> list[sqlite3.Row]:
        return self.query_all(
            "SELECT * FROM designs WHERE implemented=0 ORDER BY id"
        )

    def mark_implemented(self, design_id: int) -> None:
        self.execute("UPDATE designs SET implemented=1 WHERE id=?", (design_id,))

    # ---- 预测 ----
    def add_prediction(
        self,
        design_id: int,
        design_version: int,
        zone_id: int,
        factor_version_id: int,
        result_json: str,
    ) -> int:
        ts = utcnow_iso()
        cur = self.execute(
            """INSERT INTO predictions(design_id, design_version, zone_id,
                  factor_version_id, result_json, created_at)
               VALUES (?,?,?,?,?,?)""",
            (design_id, design_version, zone_id, factor_version_id,
             result_json, ts),
        )
        return cur.lastrowid

    def latest_prediction(self, design_id: int) -> sqlite3.Row | None:
        return self.query_one(
            "SELECT * FROM predictions WHERE design_id=? ORDER BY id DESC LIMIT 1",
            (design_id,),
        )

    def list_predictions(self, design_id: int) -> list[sqlite3.Row]:
        return self.query_all(
            "SELECT * FROM predictions WHERE design_id=? ORDER BY id", (design_id,)
        )

    # ---- 爆破实施 ----
    def create_blast(
        self,
        design_id: int,
        design_version: int,
        zone_id: int,
        factor_version_id: int,
        fired_at: str,
    ) -> int:
        ts = utcnow_iso()
        cur = self.execute(
            """INSERT INTO blasts(design_id, design_version, zone_id,
                  factor_version_id, fired_at, created_at)
               VALUES (?,?,?,?,?,?)""",
            (design_id, design_version, zone_id, factor_version_id,
             fired_at, ts),
        )
        return cur.lastrowid

    def get_blast(self, blast_id: int) -> sqlite3.Row:
        return self.get_one("SELECT * FROM blasts WHERE id=?", (blast_id,))

    def list_blasts(self) -> list[sqlite3.Row]:
        return self.query_all("SELECT * FROM blasts ORDER BY id")

    # ---- 实测 ----
    def next_measurement_seq(self, zone_id: int) -> int:
        row = self.query_one(
            "SELECT COALESCE(MAX(seq_in_zone),0)+1 AS s FROM measurements WHERE zone_id=?",
            (zone_id,),
        )
        return int(row["s"])

    def add_measurement(
        self,
        blast_id: int,
        zone_id: int,
        seq: int,
        bins_json: str,
        fit_json: str,
        fit_usable: bool,
        implied_factor: float | None,
        factor_version_id: int | None,
    ) -> int:
        ts = utcnow_iso()
        cur = self.execute(
            """INSERT INTO measurements(blast_id, zone_id, seq_in_zone,
                  bins_json, fit_json, fit_usable, implied_factor,
                  factor_version_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (blast_id, zone_id, seq, bins_json, fit_json,
             1 if fit_usable else 0, implied_factor, factor_version_id, ts),
        )
        return cur.lastrowid

    def get_measurement(self, measurement_id: int) -> sqlite3.Row:
        return self.get_one("SELECT * FROM measurements WHERE id=?", (measurement_id,))

    def list_measurements(
        self, zone_id: int | None = None, include_withdrawn: bool = True
    ) -> list[sqlite3.Row]:
        sql = "SELECT * FROM measurements WHERE 1=1"
        params: list[Any] = []
        if zone_id is not None:
            sql += " AND zone_id=?"
            params.append(zone_id)
        if not include_withdrawn:
            sql += " AND withdrawn=0"
        sql += " ORDER BY zone_id, seq_in_zone, id"
        return self.query_all(sql, params)

    def active_measurements(self, zone_id: int) -> list[sqlite3.Row]:
        """当前存活（未撤回、拟合可用）实测，按入库顺序。"""
        return self.query_all(
            """SELECT * FROM measurements
               WHERE zone_id=? AND withdrawn=0 AND fit_usable=1
               ORDER BY seq_in_zone""",
            (zone_id,),
        )

    def mark_measurement_withdrawn(self, measurement_id: int) -> None:
        self.execute(
            "UPDATE measurements SET withdrawn=1 WHERE id=?", (measurement_id,)
        )

    def supersede_factor_versions_from(self, zone_id: int, version_no: int) -> None:
        """把分区内 version_no 及之后的未取代版本标记为被取代。"""
        self.execute(
            """UPDATE factor_versions SET superseded=1, lineage='withdrawn'
               WHERE zone_id=? AND version_no>=? AND superseded=0""",
            (zone_id, version_no),
        )

    def replace_measurement_factor_version(
        self, measurement_id: int, factor_version_id: int
    ) -> None:
        self.execute(
            "UPDATE measurements SET factor_version_id=? WHERE id=?",
            (factor_version_id, measurement_id),
        )

    def close(self) -> None:
        self._conn.close()


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def loads(text: str) -> Any:
    return json.loads(text)
