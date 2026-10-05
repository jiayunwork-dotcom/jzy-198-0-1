"""业务编排层：设计管理、预测、实测入库拟合、系数标定/撤回、批量重预测。

并发：同一分区的实测入库按分区互斥锁串行化；SQLite 以 BEGIN IMMEDIATE 抢占写锁。
两次同时入库同一分区时，提交顺序决定 measurements.id，系数结果与“按入库顺序
依次处理”严格相同。
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict

from . import calibration, fragmentation, storage
from .fitting import FitResult, fit_curve, quality_weight
from .schemas import DesignParams, DesignUpdate, MeasurementUpload, ZoneCreate

DEFAULT_SIEVES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2]


class NotFound(LookupError):
    """引用不存在的分区或设计（接口层映射为 404）。"""


class Conflict(ValueError):
    """当前状态下不允许的操作（409）。"""


_zone_locks: dict[int, threading.Lock] = defaultdict(threading.Lock)
_create_lock = threading.Lock()


def zone_lock(zone_id: int) -> threading.Lock:
    with _create_lock:
        return _zone_locks[zone_id]


# ============================================================ 分区
def create_zone(conn, payload: ZoneCreate):
    now = storage.utcnow()
    with storage.transaction(conn):
        zone_id = storage.insert(
            conn,
            "INSERT INTO zones(name, lithology, decay, created_at) VALUES(?,?,?,?)",
            (payload.name, payload.lithology, payload.decay, now),
        )
        storage.insert(
            conn,
            """INSERT INTO factor_versions
               (zone_id, version_index, factor, evidence_id, decay, created_at,
                active, chain, note)
               VALUES(?,?,?,?,?,?,1,'current','initial factor')""",
            (zone_id, 0, payload.initial_factor, None, payload.decay, now),
        )
    return get_zone(conn, zone_id)


def get_zone(conn, zone_id: int):
    row = storage.get(conn, "SELECT * FROM zones WHERE id=?", (zone_id,))
    if row is None:
        raise NotFound(f"zone {zone_id} not found")
    fv = current_factor_version(conn, zone_id)
    return {
        "id": row["id"],
        "name": row["name"],
        "lithology": row["lithology"],
        "decay": row["decay"],
        "current_factor": fv["factor"],
        "current_version_id": fv["id"],
    }


def list_zones(conn):
    return [get_zone(conn, r["id"]) for r in storage.query(conn, "SELECT id FROM zones")]


def current_factor_version(conn, zone_id: int):
    row = storage.get(
        conn,
        """SELECT * FROM factor_versions
           WHERE zone_id=? AND active=1
           ORDER BY version_index DESC, id DESC LIMIT 1""",
        (zone_id,),
    )
    if row is None:  # 理论上不会发生：建区即有 v0
        raise NotFound(f"no active factor version for zone {zone_id}")
    return row


def factor_version(conn, version_id: int):
    row = storage.get(
        conn, "SELECT * FROM factor_versions WHERE id=?", (version_id,)
    )
    if row is None:
        raise NotFound(f"factor version {version_id} not found")
    return row


def factor_as_of(conn, zone_id: int, at_iso: str):
    """查询任意时刻某分区生效的系数：created_at <= at 的最新 active 版本。"""
    row = storage.get(
        conn,
        """SELECT * FROM factor_versions
           WHERE zone_id=? AND active=1 AND created_at<=?
           ORDER BY created_at DESC, id DESC LIMIT 1""",
        (zone_id, at_iso),
    )
    if row is None:
        raise NotFound(f"zone {zone_id} has no factor version as of {at_iso}")
    return row


def factor_history(conn, zone_id: int, include_superseded: bool = True):
    where = "zone_id=?" if include_superseded else "zone_id=? AND active=1"
    rows = storage.query(
        conn,
        f"SELECT * FROM factor_versions WHERE {where} ORDER BY id",
        (zone_id,),
    )
    if not rows:
        raise NotFound(f"zone {zone_id} not found")
    return rows


# ============================================================ 设计
def create_design(conn, payload: DesignParams):
    get_zone(conn, payload.zone_id)  # 不存在 -> 404
    now = storage.utcnow()
    with storage.transaction(conn):
        design_id = storage.insert(
            conn,
            """INSERT INTO designs(zone_id, name, current_version, status,
                                   created_at, updated_at)
               VALUES(?, ?, 1, 'planned', ?, ?)""",
            (payload.zone_id, payload.name, now, now),
        )
        storage.insert(
            conn,
            """INSERT INTO design_versions(design_id, version, params, created_at)
               VALUES(?, 1, ?, ?)""",
            (design_id, storage.dumps(payload.model_dump()), now),
        )
    return get_design(conn, design_id)


def _row_to_design(conn, row) -> dict:
    dv = storage.get(
        conn,
        "SELECT * FROM design_versions WHERE design_id=? AND version=?",
        (row["id"], row["current_version"]),
    )
    return {
        "id": row["id"],
        "version": row["current_version"],
        "name": row["name"],
        "zone_id": row["zone_id"],
        "status": row["status"],
        "params": storage.loads(dv["params"]),
        "version_created_at": dv["created_at"],
    }


def get_design(conn, design_id: int):
    row = storage.get(conn, "SELECT * FROM designs WHERE id=?", (design_id,))
    if row is None:
        raise NotFound(f"design {design_id} not found")
    return _row_to_design(conn, row)


def list_designs(conn, status: str | None = None):
    if status:
        rows = storage.query(
            conn, "SELECT * FROM designs WHERE status=? ORDER BY id", (status,)
        )
    else:
        rows = storage.query(conn, "SELECT * FROM designs ORDER BY id")
    return [_row_to_design(conn, r) for r in rows]


def update_design(conn, design_id: int, patch: DesignUpdate):
    current = get_design(conn, design_id)
    if current["status"] != "planned":
        raise Conflict("design already blasted; parameters are frozen")
    merged = {**current["params"], **patch.model_dump(exclude_none=True)}
    params = DesignParams(**merged)  # 复用同一套拒收校验
    now = storage.utcnow()
    new_version = current["version"] + 1
    with storage.transaction(conn):
        storage.insert(
            conn,
            """INSERT INTO design_versions(design_id, version, params, created_at)
               VALUES(?,?,?,?)""",
            (design_id, new_version, storage.dumps(params.model_dump()), now),
        )
        conn.execute(
            "UPDATE designs SET current_version=?, name=?, updated_at=? WHERE id=?",
            (new_version, params.name, now, design_id),
        )
    return get_design(conn, design_id)


def design_version_params(conn, design_id: int, version: int) -> dict:
    row = storage.get(
        conn,
        "SELECT params FROM design_versions WHERE design_id=? AND version=?",
        (design_id, version),
    )
    if row is None:
        raise NotFound(f"design {design_id} version {version} not found")
    return storage.loads(row["params"])


# ============================================================ 预测
def _do_predict(conn, design: dict, factor_row, kind: str = "manual",
                extra_sieves: list[float] | None = None):
    params = design["params"]
    fv = factor_row
    res = fragmentation.predict(params, fv["factor"])
    over_pass = res["oversize_pct"] <= params["oversize_limit_pct"]
    table_sizes = sorted(set(DEFAULT_SIEVES + [params["oversize_m"]] +
                             (extra_sieves or [])))
    table = {
        f"{s:g}": round(fragmentation.passing_fraction(s, res["x50"],
                                                       res["uniformity"]), 4)
        for s in table_sizes
    }
    now = storage.utcnow()
    with storage.transaction(conn):
        pid = storage.insert(
            conn,
            """INSERT INTO predictions(design_id, design_version, factor_version_id,
                  zone_id, rock_factor, x50, uniformity, powder_factor, oversize_m,
                  oversize_pct, oversize_limit, oversize_pass, passing_table,
                  kind, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                design["id"], design["version"], fv["id"], design["zone_id"],
                fv["factor"], res["x50"], res["uniformity"], res["powder_factor"],
                params["oversize_m"], res["oversize_pct"],
                params["oversize_limit_pct"], 1 if over_pass else 0,
                storage.dumps(table), kind, now,
            ),
        )
    return get_prediction(conn, pid)


def get_prediction(conn, prediction_id: int):
    row = storage.get(
        conn, "SELECT * FROM predictions WHERE id=?", (prediction_id,)
    )
    if row is None:
        raise NotFound(f"prediction {prediction_id} not found")
    return _prediction_row(row)


def latest_prediction(conn, design_id: int):
    row = storage.get(
        conn,
        "SELECT * FROM predictions WHERE design_id=? ORDER BY id DESC LIMIT 1",
        (design_id,),
    )
    return _prediction_row(row) if row else None


def _prediction_row(row) -> dict:
    return {
        "id": row["id"],
        "design_id": row["design_id"],
        "design_version": row["design_version"],
        "factor_version_id": row["factor_version_id"],
        "rock_factor": row["rock_factor"],
        "x50": row["x50"],
        "uniformity": row["uniformity"],
        "powder_factor": row["powder_factor"],
        "oversize_pct": row["oversize_pct"],
        "oversize_pass": bool(row["oversize_pass"]),
        "passing_table": storage.loads(row["passing_table"]),
        "kind": row["kind"],
        "created_at": row["created_at"],
    }


def predict_design(conn, design_id: int, factor_version_id: int | None = None,
                   extra_sieves: list[float] | None = None):
    design = get_design(conn, design_id)
    if factor_version_id is None:
        fv = current_factor_version(conn, design["zone_id"])
    else:
        fv = factor_version(conn, factor_version_id)
        if fv["zone_id"] != design["zone_id"]:
            raise Conflict("factor version does not belong to design's zone")
    return _do_predict(conn, design, fv, "manual", extra_sieves)


# ============================================================ 实测
def _fit_from_payload(payload: MeasurementUpload) -> FitResult:
    sizes = [p.size_m for p in payload.points]
    pcts = [p.passing_pct for p in payload.points]
    opens = [p.open for p in payload.points]
    return fit_curve(sizes, pcts, opens)


def _geometry_factor(params: dict) -> float:
    """g_k：X50=A*g_k 中的几何/炸药部分（不含 A），单位米。"""
    q = fragmentation.powder_factor(
        params["burden"], params["spacing"],
        params["bench_height"], params["charge_kg"],
    )
    return (
        0.01
        * q ** (-0.8)
        * params["charge_kg"] ** (1.0 / 6.0)
        * (1.15 / params["explosive_rws"]) ** (19.0 / 30.0)
    )


def _next_version_index(conn, zone_id: int) -> int:
    row = storage.get(
        conn,
        "SELECT COALESCE(MAX(version_index),-1)+1 AS vi FROM factor_versions WHERE zone_id=?",
        (zone_id,),
    )
    return int(row["vi"])


def _active_observations(conn, zone_id: int):
    rows = storage.query(
        conn,
        """SELECT id, single_factor, qweight FROM measurements
           WHERE zone_id=? AND status='active' AND usable=1
             AND single_factor IS NOT NULL ORDER BY id""",
        (zone_id,),
    )
    return [
        calibration.Observation(
            observation=r["single_factor"], qweight=r["qweight"],
            evidence_id=r["id"],
        )
        for r in rows
    ]


def upload_measurement(conn, payload: MeasurementUpload) -> dict:
    design = get_design(conn, payload.design_id)  # 404 if missing
    zone_id = design["zone_id"]
    get_zone(conn, zone_id)
    fit = _fit_from_payload(payload)

    lock = zone_lock(zone_id)
    lock.acquire()
    try:
        with storage.transaction(conn):
            dup = storage.get(
                conn,
                """SELECT id FROM measurements
                   WHERE design_id=? AND status='active'""",
                (design["id"],),
            )
            if dup is not None:
                raise Conflict(
                    f"design {design['id']} already has an active measurement"
                )

            now = storage.utcnow()
            # 先落实测（拟合结果始终保存，便于追溯坏曲线）
            meas_id = storage.insert(
                conn,
                """INSERT INTO measurements(design_id, zone_id, points, fit,
                       usable, x50, uniformity, g, qweight, single_factor,
                       factor_version_id, created_at, status)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?, 'active')""",
                (
                    design["id"], zone_id,
                    storage.dumps([p.model_dump() for p in payload.points]),
                    storage.dumps(_fit_json(fit)),
                    1 if fit.usable else 0,
                    fit.x50 if fit.usable else None,
                    fit.uniformity if fit.usable else None,
                    None, None, None, None, now,
                ),
            )

            fv_id = None
            factor_value = None
            if fit.usable:
                params = design_version_params(conn, design["id"], design["version"])
                g = _geometry_factor(params)
                qw = quality_weight(fit)
                single = fit.x50 / g
                # 增量更新（权威定义与全量折叠一致；decay=1 时逐位相同）
                zone_row = storage.get(
                    conn, "SELECT decay FROM zones WHERE id=?", (zone_id,)
                )
                decay = float(zone_row["decay"])
                obs = _active_observations(conn, zone_id)  # 不含本条
                s0, s1 = calibration.moments(obs, decay)
                s0, s1 = calibration.fold_state(s0, s1, single, qw, decay)
                factor_value = calibration.fold_value(s0, s1)

                vi = _next_version_index(conn, zone_id)
                fv_id = storage.insert(
                    conn,
                    """INSERT INTO factor_versions(zone_id, version_index, factor,
                           evidence_id, decay, created_at, active, chain, note)
                       VALUES(?,?,?,?,?,?,1,'current','measurement')""",
                    (zone_id, vi, factor_value, meas_id, decay, now),
                )
                conn.execute(
                    """UPDATE measurements SET g=?, qweight=?, single_factor=?,
                          factor_version_id=? WHERE id=?""",
                    (g, qw, single, fv_id, meas_id),
                )

            # 设计进入“已实施”
            conn.execute(
                "UPDATE designs SET status='blasted', updated_at=? WHERE id=?",
                (now, design["id"]),
            )
        return get_measurement(conn, meas_id)
    finally:
        lock.release()


def _fit_json(fit: FitResult) -> dict:
    def num(v):
        return v if isinstance(v, (int, float)) and math.isfinite(v) else None

    return {
        "x50": num(fit.x50),
        "uniformity": num(fit.uniformity),
        "usable": fit.usable,
        "r2": num(fit.r2),
        "rmse_y": num(fit.rmse_y),
        "max_p_residual_pct": num(fit.max_p_residual_pct),
        "residuals_pct": fit.residuals_pct,
        "included": fit.included,
        "bimodal": fit.bimodal,
        "reasons": fit.reasons,
    }


def get_measurement(conn, measurement_id: int) -> dict:
    row = storage.get(
        conn, "SELECT * FROM measurements WHERE id=?", (measurement_id,)
    )
    if row is None:
        raise NotFound(f"measurement {measurement_id} not found")
    fit = storage.loads(row["fit"])
    fv = None
    if row["factor_version_id"] is not None:
        fv = factor_version(conn, row["factor_version_id"])
    design_row = storage.get(
        conn, "SELECT current_version FROM designs WHERE id=?",
        (row["design_id"],),
    )
    return {
        "id": row["id"],
        "design_id": row["design_id"],
        "design_version": design_row["current_version"],
        "zone_id": row["zone_id"],
        "status": row["status"],
        "fit": fit,
        "factor_version_id": row["factor_version_id"],
        "rock_factor": fv["factor"] if fv else None,
        "created_at": row["created_at"],
        "withdrawn_at": row["withdrawn_at"],
        "points": storage.loads(row["points"]),
    }


def list_measurements(conn, zone_id: int | None = None):
    if zone_id is not None:
        rows = storage.query(
            conn,
            "SELECT id FROM measurements WHERE zone_id=? ORDER BY id",
            (zone_id,),
        )
    else:
        rows = storage.query(conn, "SELECT id FROM measurements ORDER BY id")
    return [get_measurement(conn, r["id"]) for r in rows]


# ============================================================ 撤回
def withdraw_measurement(conn, measurement_id: int) -> dict:
    """撤回一条实测及其引发的系数更新；其后的版本在剔除该条数据基础上重推。

    做法（全量重推，口径权威；与增量在线路径定义一致）：
      1. 该实测置 withdrawn（原始筛分与拟合结果保留可查）；
      2. 撤回点之前的系数版本原值保留（它们不依赖被撤回数据）；
      3. 撤回点及其后、在当前有效链上的版本整体置 active=0/chain='withdrawn'，
         对仍保留的有效实测按 id 顺序逐条“前缀全量折叠”，生成重推新版本，
         measurements.factor_version_id 回填到新版本；
      4. 被撤回实测自身的旧版本无对应重推版本，仅置撤回链。
    撤回前后版本都在 factor_versions 表里，可按 id / 时刻查询。
    """
    existing = get_measurement(conn, measurement_id)
    zone_id = existing["zone_id"]
    if existing["status"] != "active":
        raise Conflict(f"measurement {measurement_id} is not active")

    lock = zone_lock(zone_id)
    lock.acquire()
    try:
        with storage.transaction(conn):
            now = storage.utcnow()
            conn.execute(
                """UPDATE measurements SET status='withdrawn', withdrawn_at=?
                   WHERE id=?""",
                (now, measurement_id),
            )

            target = storage.get(
                conn, "SELECT * FROM measurements WHERE id=?", (measurement_id,)
            )
            start_fv = target["factor_version_id"]
            decay = float(
                storage.get(conn, "SELECT decay FROM zones WHERE id=?", (zone_id,))[
                    "decay"
                ]
            )

            # 撤回后保留的全部有效观测（按入库顺序）
            kept = storage.query(
                conn,
                """SELECT * FROM measurements
                   WHERE zone_id=? AND status='active' AND usable=1 ORDER BY id""",
                (zone_id,),
            )

            if start_fv is not None:
                cut_index = storage.get(
                    conn,
                    "SELECT version_index FROM factor_versions WHERE id=?",
                    (start_fv,),
                )["version_index"]

                # 先快照当前链上 cut 起的旧版本（必须在插入重推版本之前，
                # 否则会把新版本也圈进来误作废）
                old_chain = storage.query(
                    conn,
                    """SELECT * FROM factor_versions
                       WHERE zone_id=? AND active=1 AND version_index>=?
                       ORDER BY id""",
                    (zone_id, cut_index),
                )

                # 各保留观测原属版本的 version_index；只重推 cut 之后的观测，
                # cut 之前的版本（不依赖被撤回数据）原样保留。
                def _orig_index(meas_row) -> int:
                    return storage.get(
                        conn,
                        "SELECT version_index FROM factor_versions WHERE id=?",
                        (meas_row["factor_version_id"],),
                    )["version_index"]

                # 前缀全量折叠：cut 之前的观测也进入计算（保持衰减语义），
                # 但不为它们生成新版本
                reissue_by_evidence: dict[int, int] = {}
                obs_prefix: list[calibration.Observation] = []
                for r in kept:
                    obs_prefix.append(
                        calibration.Observation(
                            observation=r["single_factor"],
                            qweight=r["qweight"],
                            evidence_id=r["id"],
                        )
                    )
                    if _orig_index(r) < cut_index:
                        continue
                    factor_value = calibration.fit_all(obs_prefix, decay)
                    vi = _next_version_index(conn, zone_id)
                    new_id = storage.insert(
                        conn,
                        """INSERT INTO factor_versions(zone_id, version_index,
                               factor, evidence_id, decay, created_at, active,
                               chain, note)
                           VALUES(?,?,?,?,?,?,1,'current',
                                  'reissued after measurement withdraw')""",
                        (zone_id, vi, factor_value, r["id"], decay, now),
                    )
                    reissue_by_evidence[r["id"]] = new_id

                # 旧版本整体作废；有重推对应的挂 superseded_by
                for r in old_chain:
                    repl = reissue_by_evidence.get(r["evidence_id"])
                    conn.execute(
                        """UPDATE factor_versions SET active=0, chain='withdrawn',
                               superseded_by=COALESCE(?, superseded_by)
                           WHERE id=?""",
                        (repl, r["id"]),
                    )

                # cut 之后保留实测的版本指针改挂重推版本
                for r in kept:
                    if r["id"] in reissue_by_evidence:
                        conn.execute(
                            "UPDATE measurements SET factor_version_id=? WHERE id=?",
                            (reissue_by_evidence[r["id"]], r["id"]),
                        )
        return get_measurement(conn, measurement_id)
    finally:
        lock.release()


# ============================================================ 批量重新预测
def batch_repredict(conn, zone_id: int | None = None) -> dict:
    """系数更新后对所有尚未实施的设计重新预测，返回大块合格性翻转清单。"""
    zones = (
        [get_zone(conn, zone_id)]
        if zone_id is not None
        else list_zones(conn)
    )
    flips = []
    total = 0
    for z in zones:
        fv = current_factor_version(conn, z["id"])
        designs = [
            _row_to_design(conn, r)
            for r in storage.query(
                conn,
                "SELECT * FROM designs WHERE zone_id=? AND status='planned' ORDER BY id",
                (z["id"],),
            )
        ]
        for design in designs:
            old = latest_prediction(conn, design["id"])
            params = design["params"]
            res = fragmentation.predict(params, fv["factor"])
            new_pass = res["oversize_pct"] <= params["oversize_limit_pct"]
            _do_predict(conn, design, fv, kind="batch")
            total += 1
            if old is not None and bool(old["oversize_pass"]) != bool(new_pass):
                flips.append({
                    "design_id": design["id"],
                    "name": design["name"],
                    "old_factor_version_id": old["factor_version_id"],
                    "new_factor_version_id": fv["id"],
                    "old_oversize_pct": old["oversize_pct"],
                    "new_oversize_pct": res["oversize_pct"],
                    "old_pass": bool(old["oversize_pass"]),
                    "new_pass": bool(new_pass),
                    "flipped": True,
                })
    return {"zone_id": zone_id, "re_predicted": total, "flips": flips}
