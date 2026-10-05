"""领域服务层：分区、设计版本、预测、爆破实施、实测入库与撤回、批量重预测。

本层只管业务编排，SQL 在 storage.py，数值算法在 fragmentation /
sieving / calibration。
"""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timezone

from . import calibration
from .fragmentation import (
    DesignParams,
    Prediction,
    ValidationError,
    implied_rock_factor,
    predict as predict_design_params,
)
from .sieving import FitResult, SieveValidationError, fit_curve
from .storage import NotFound, Storage, dumps, loads


class ConflictError(ValueError):
    """状态冲突（如对已实施设计做修改）。"""


# ---------------------------------------------------------------- 分区

def create_zone(
    store: Storage,
    name: str,
    baseline_factor: float,
    forgetting: float = calibration.DEFAULT_FORGETTING,
    baseline_weight: float = calibration.DEFAULT_BASELINE_WEIGHT,
    lithology: str = "",
) -> dict:
    if not name or not str(name).strip():
        raise ValidationError("分区名称不能为空")
    if not math.isfinite(baseline_factor) or baseline_factor <= 0:
        raise ValidationError("基线岩石系数必须为正数")
    if not (0.0 < forgetting <= 1.0):
        raise ValidationError("遗忘因子必须在 (0,1] 内")
    if not math.isfinite(baseline_weight) or baseline_weight < 0:
        raise ValidationError("基线权重不能为负")
    with store.lock:
        try:
            zone_id = store.create_zone(
                name.strip(), baseline_factor, forgetting,
                baseline_weight, lithology,
            )
        except Exception as exc:  # UNIQUE 冲突
            raise ConflictError(f"分区 {name!r} 已存在") from exc
        return _zone_dict(store.get_zone(zone_id))


def get_zone(store: Storage, zone_id: int) -> dict:
    return _zone_dict(store.get_zone(zone_id))


def list_zones(store: Storage) -> list[dict]:
    return [_zone_dict(r) for r in store.list_zones()]


def current_factor(store: Storage, zone_id: int) -> dict:
    store.get_zone(zone_id)
    row = store.active_factor_version(zone_id)
    state = store.get_state(zone_id)
    out = _factor_version_dict(row)
    out["observations"] = state["obs_count"]
    return out


def factor_history(store: Storage, zone_id: int, include_superseded: bool = False) -> list[dict]:
    store.get_zone(zone_id)
    return [_factor_version_dict(r) for r in
            store.factor_history(zone_id, include_superseded)]


def factor_as_of(store: Storage, zone_id: int, ts: str) -> dict:
    store.get_zone(zone_id)
    _parse_ts(ts)
    row = store.factor_version_as_of(zone_id, ts)
    if row is None:
        raise NotFound(f"分区 {zone_id} 在 {ts} 之前没有任何系数版本")
    return _factor_version_dict(row)


# ---------------------------------------------------------------- 设计

def params_from_dict(data: dict) -> DesignParams:
    try:
        p = DesignParams(
            hole_diameter_mm=float(data["hole_diameter_mm"]),
            burden_m=float(data["burden_m"]),
            spacing_m=float(data["spacing_m"]),
            bench_height_m=float(data["bench_height_m"]),
            hole_depth_m=float(data["hole_depth_m"]),
            stemming_m=float(data["stemming_m"]),
            explosive_mass_kg=float(data["explosive_mass_kg"]),
            explosive_rws=float(data.get("explosive_rws", 100.0)),
            hole_deviation_m=float(data.get("hole_deviation_m", 0.0)),
            oversize_sieve_m=float(data.get(
                "oversize_sieve_m", 0.60)),
            oversize_limit_pct=float(data.get(
                "oversize_limit_pct", 5.0)),
        )
    except KeyError as exc:
        raise ValidationError(f"缺少设计参数：{exc.args[0]}") from exc
    p.validate()
    return p


def create_design(store: Storage, name: str, zone_id: int, params: dict) -> dict:
    if not name or not str(name).strip():
        raise ValidationError("设计名称不能为空")
    p = params_from_dict(params)
    with store.lock:
        store.get_zone(zone_id)  # 不存在 -> NotFound
        try:
            design_id = store.create_design(name.strip(), zone_id, dumps(asdict(p)))
        except Exception as exc:
            raise ConflictError(f"设计 {name!r} 已存在") from exc
    return get_design(store, design_id)


def update_design(store: Storage, design_id: int, params: dict) -> dict:
    p = params_from_dict(params)
    with store.lock:
        d = store.get_design(design_id)
        if d["implemented"]:
            raise ConflictError("设计已实施爆破，不能再修改；如需调整请新建设计")
        no = store.add_design_version(design_id, dumps(asdict(p)))
    return get_design(store, design_id)


def get_design(store: Storage, design_id: int) -> dict:
    d = store.get_design(design_id)
    dv = store.get_design_version(design_id, d["current_version"])
    return {
        "id": d["id"],
        "name": d["name"],
        "zone_id": d["zone_id"],
        "implemented": bool(d["implemented"]),
        "current_version": d["current_version"],
        "created_at": d["created_at"],
        "params": loads(dv["params_json"]),
    }


def list_designs(store: Storage) -> list[dict]:
    return [get_design(store, r["id"]) for r in
            store.query_all("SELECT id FROM designs ORDER BY id")]


# ---------------------------------------------------------------- 预测

def _prediction_to_dict(pred: Prediction) -> dict:
    return asdict(pred)


def predict(
    store: Storage, design_id: int, design_version: int | None = None
) -> dict:
    """用设计指定版本与该分区当前系数版本做预测并入库（绑定双版本）。"""
    with store.lock:
        d = store.get_design(design_id)
        if design_version is None:
            design_version = d["current_version"]
        dv = store.get_design_version(design_id, design_version)
        params = params_from_dict(loads(dv["params_json"]))
        fv = store.active_factor_version(d["zone_id"])
        pred = predict_design_params(params, fv["factor"])
        pid = store.add_prediction(
            design_id, design_version, d["zone_id"], fv["id"],
            dumps(_prediction_to_dict(pred)),
        )
        store.execute(
            "UPDATE predictions SET is_current=0 WHERE design_id=? AND id<>?",
            (design_id, pid),
        )
        return {
            "prediction_id": pid,
            "design_id": design_id,
            "design_version": design_version,
            "factor_version": _factor_version_dict(fv),
            "prediction": _prediction_to_dict(pred),
        }


def bulk_repredict(store: Storage) -> dict:
    """系数更新后，对所有尚未实施的设计用最新设计版本+最新系数重预测。

    返回每个发生变化的设计，以及其中大块率合格状态翻转的清单
    （合格→超标 / 超标→合格）。
    """
    with store.lock:
        results: list[dict] = []
        for d in store.list_open_designs():
            design_id = d["id"]
            dv = store.get_design_version(design_id, d["current_version"])
            params = params_from_dict(loads(dv["params_json"]))
            fv = store.active_factor_version(d["zone_id"])
            prev_row = store.latest_prediction(design_id)
            prev = loads(prev_row["result_json"]) if prev_row else None
            if prev_row is not None and (
                prev_row["design_version"] == d["current_version"]
                and prev_row["factor_version_id"] == fv["id"]
            ):
                continue  # 输入未变，结果必然相同，跳过
            new_pred = predict_design_params(params, fv["factor"])
            pid = store.add_prediction(
                design_id, d["current_version"], d["zone_id"], fv["id"],
                dumps(_prediction_to_dict(new_pred)),
            )
            store.execute(
                "UPDATE predictions SET is_current=0 WHERE design_id=? AND id<>?",
                (design_id, pid),
            )
            entry = {
                "design_id": design_id,
                "name": d["name"],
                "new_prediction_id": pid,
                "previous": _prediction_to_dict(
                    _prediction_from_dict(prev)) if prev else None,
                "current": _prediction_to_dict(new_pred),
                "flipped": False,
                "flip_direction": None,
            }
            if prev is not None:
                was_ok = bool(prev["oversize_ok"])
                now_ok = new_pred.oversize_ok
                if was_ok != now_ok:
                    entry["flipped"] = True
                    entry["flip_direction"] = (
                        "pass_to_fail" if was_ok and not now_ok else "fail_to_pass"
                    )
            results.append(entry)
        flips = [r for r in results if r["flipped"]]
        return {
            "repredicted": len(results),
            "flipped": len(flips),
            "flips": flips,
            "details": results,
        }


def passing_for_prediction(
    store: Storage, prediction_id: int | None, sieve_m: float,
    x50_m: float | None = None, uniformity_n: float | None = None,
) -> dict:
    """给定筛孔计算通过率。

    优先用显式传入的 X50/n；否则按 prediction_id 取已入库预测。
    """
    from .fragmentation import passing_fraction

    if x50_m is not None and uniformity_n is not None:
        if x50_m <= 0 or uniformity_n <= 0 or sieve_m < 0:
            raise ValidationError("X50、n 必须为正，筛孔不能为负")
        x50, n = float(x50_m), float(uniformity_n)
    elif prediction_id is not None:
        row = store.query_one(
            "SELECT * FROM predictions WHERE id=?", (prediction_id,)
        )
        if row is None:
            raise NotFound(f"预测 {prediction_id} 不存在")
        result = loads(row["result_json"])
        x50, n = result["x50_m"], result["uniformity_n"]
    else:
        raise ValidationError("必须提供 prediction_id 或 x50_m 与 uniformity_n")
    p = passing_fraction(x50, n, sieve_m)
    return {
        "sieve_m": sieve_m,
        "passing_fraction": p,
        "passing_pct": p * 100.0,
        "retained_pct": (1.0 - p) * 100.0,
        "x50_m": x50,
        "uniformity_n": n,
    }


def _prediction_from_dict(d: dict) -> Prediction:
    return Prediction(**d)


# ---------------------------------------------------------------- 爆破

def fire_blast(
    store: Storage,
    design_id: int,
    fired_at: str | None = None,
    design_version: int | None = None,
) -> dict:
    """登记一次爆破实施；冻结当时的设计版本与系数版本。"""
    ts = fired_at or _now_iso()
    _parse_ts(ts)
    with store.lock:
        d = store.get_design(design_id)
        if d["implemented"]:
            raise ConflictError("该设计已登记爆破实施")
        version = design_version or d["current_version"]
        store.get_design_version(design_id, version)
        fv = store.active_factor_version(d["zone_id"])
        blast_id = store.create_blast(
            design_id, version, d["zone_id"], fv["id"], ts
        )
        store.mark_implemented(design_id)
        b = store.get_blast(blast_id)
        return _blast_dict(store, b)


def get_blast(store: Storage, blast_id: int) -> dict:
    return _blast_dict(store, store.get_blast(blast_id))


def list_blasts(store: Storage) -> list[dict]:
    return [_blast_dict(store, r) for r in store.list_blasts()]


# ---------------------------------------------------------------- 实测

def upload_measurement(
    store: Storage, blast_id: int, bins: list[dict]
) -> dict:
    """爆后图像筛分入库：拟合 → 反推 → 分区系数增量更新。

    全流程在 storage 写锁内完成；两次并发入库因此严格按持锁顺序
    串行，最终系数与逐条顺序入库一致。
    """
    with store.lock:
        blast = store.get_blast(blast_id)
        zone = store.get_zone(blast["zone_id"])
        existing = store.query_one(
            "SELECT id FROM measurements WHERE blast_id=? AND withdrawn=0",
            (blast_id,),
        )
        if existing is not None:
            raise ConflictError(
                f"爆破 {blast_id} 已有有效实测；如需更正请先撤回原实测"
            )

        fit = fit_curve(bins)  # 非法数据抛 SieveValidationError
        fit_json = dumps(_fit_to_json(fit))

        implied: float | None = None
        # 反推必须使用该次爆破当时冻结的系数版本（而不是入库时的
        # 当前版本），否则系数在爆破与实测之间更新过就会算错
        fv_blast = store.factor_version(blast["factor_version_id"])
        if fit.usable:
            params = _blast_params(store, blast)
            a_used = fv_blast["factor"]
            from .fragmentation import predict_x50

            predicted_x50 = predict_x50(params, a_used)
            implied = implied_rock_factor(
                params, fit.x50_m, predicted_x50, a_used
            )
            state = store.get_state(zone["id"])
            fold = calibration.Fold(
                w=state["fold_w"], wl=state["fold_wl"], count=state["obs_count"]
            )
            new_factor = calibration.incremental_update(
                fold, implied, zone["forgetting"],
                zone["baseline_factor"], zone["baseline_weight"],
            )
            seq = store.next_measurement_seq(zone["id"])
            mid = store.add_measurement(
                blast_id, zone["id"], seq, dumps(bins), fit_json,
                True, implied, None,
            )
            fv_id = store.add_factor_version(
                zone["id"], new_factor, "measurement",
                blast_id=blast_id, measurement_id=mid,
                note=f"实测 #{mid} 反推系数 {implied:.6g}",
            )
            store.replace_measurement_factor_version(mid, fv_id)
            store.update_state(
                zone["id"], fold.w, fold.wl, fold.count, blast_id
            )
        else:
            seq = store.next_measurement_seq(zone["id"])
            mid = store.add_measurement(
                blast_id, zone["id"], seq, dumps(bins), fit_json,
                False, None, None,
            )
        return {
            "measurement_id": mid,
            "blast_id": blast_id,
            "zone_id": zone["id"],
            "fit": _fit_to_json(fit),
            "implied_factor": implied,
            "factor_version": _factor_version_dict(
                store.factor_version(
                    store.get_measurement(mid)["factor_version_id"]
                )
            ) if fit.usable else _factor_version_dict(
                store.active_factor_version(zone["id"])
            ),
            "coefficient_updated": bool(fit.usable),
        }


def get_measurement(store: Storage, measurement_id: int) -> dict:
    m = store.get_measurement(measurement_id)
    out = dict(m)
    out["withdrawn"] = bool(m["withdrawn"])
    out["fit_usable"] = bool(m["fit_usable"])
    out["bins"] = loads(m["bins_json"])
    out["fit"] = loads(m["fit_json"])
    if m["factor_version_id"]:
        out["factor_version"] = _factor_version_dict(
            store.factor_version(m["factor_version_id"])
        )
    return out


def withdraw_measurement(store: Storage, measurement_id: int) -> dict:
    """撤回一条录错的实测，并级联重推此后的系数版本。

    步骤（同一写锁/事务语义内）：
    1. 标记实测 withdrawn；
    2. 它产生的系数版本及其后的活动版本全部置 superseded；
    3. 用剔除该条后的全部存活实测，按原入库顺序重放折叠；
       先补一条 withdrawal 版本（记录撤回当时的系数），随后为每条
       仍存活的后续实测追加 rebuilt 版本并重挂实测外键；
    4. 分区折叠状态重置为重放结果。

    被取代的旧版本保留在库中（include_superseded 可查），
    撤回前后的历史都能查询。
    """
    with store.lock:
        m = store.get_measurement(measurement_id)
        if m["withdrawn"]:
            raise ConflictError(f"实测 {measurement_id} 已撤回")
        zone = store.get_zone(m["zone_id"])

        # 该实测产生过版本（拟合不可用时没有级联）
        trigger_version_no: int | None = None
        if m["factor_version_id"] is not None:
            trigger_version_no = store.factor_version(
                m["factor_version_id"]
            )["version_no"]

        store.mark_measurement_withdrawn(measurement_id)
        if trigger_version_no is not None:
            store.supersede_factor_versions_from(
                zone["id"], trigger_version_no
            )

        # 剔除后按原入库顺序重放
        survivors = store.active_measurements(zone["id"])
        fold = calibration.Fold()
        last_blast: int | None = None
        rebuilt_rows: list[dict] = []
        # 撤回时刻系数 = 重放到"被撤条目之前"的水平（幸存者中
        # seq 小于被撤者的部分）
        for sm in survivors:
            implied = float(sm["implied_factor"])
            # 该幸存者是否位于被撤点之后（需要补 rebuilt 版本）
            is_after = (
                trigger_version_no is not None and sm["seq_in_zone"] > m["seq_in_zone"]
            )
            if is_after and not rebuilt_rows:
                # 先落一条 withdrawal 版本，固定撤回发生时的系数
                factor_at_withdrawal = calibration.fold_to_factor(
                    fold, zone["baseline_factor"], zone["baseline_weight"]
                )
                wv_id = store.add_factor_version(
                    zone["id"], factor_at_withdrawal, "withdrawal",
                    blast_id=m["blast_id"], measurement_id=measurement_id,
                    lineage="rebuilt",
                    note=f"撤回实测 #{measurement_id}，剔除后重推",
                )
                rebuilt_rows.append({"version_id": wv_id, "measurement_id": None})
            calibration.incremental_update(
                fold, implied, zone["forgetting"],
                zone["baseline_factor"], zone["baseline_weight"],
            )
            last_blast = sm["blast_id"]
            if is_after:
                fv_id = store.add_factor_version(
                    zone["id"],
                    calibration.fold_to_factor(
                        fold, zone["baseline_factor"], zone["baseline_weight"]
                    ),
                    "measurement",
                    blast_id=sm["blast_id"], measurement_id=sm["id"],
                    lineage="rebuilt",
                    note=f"撤回 #{measurement_id} 后重推（原实测 #{sm['id']}）",
                )
                store.replace_measurement_factor_version(sm["id"], fv_id)
                rebuilt_rows.append({"version_id": fv_id, "measurement_id": sm["id"]})

        if trigger_version_no is not None and not rebuilt_rows:
            # 被撤的是最后一条：只补 withdrawal 版本记录当时系数
            factor_at_withdrawal = calibration.fold_to_factor(
                fold, zone["baseline_factor"], zone["baseline_weight"]
            )
            store.add_factor_version(
                zone["id"], factor_at_withdrawal, "withdrawal",
                blast_id=m["blast_id"], measurement_id=measurement_id,
                lineage="rebuilt",
                note=f"撤回实测 #{measurement_id}（末条），系数回到剔除后水平",
            )

        store.update_state(
            zone["id"], fold.w, fold.wl, fold.count, last_blast
        )
        current = store.active_factor_version(zone["id"])
        return {
            "withdrawn_measurement_id": measurement_id,
            "zone_id": zone["id"],
            "replayed_observations": fold.count,
            "factor_version": _factor_version_dict(current),
            "history": [
                _factor_version_dict(r)
                for r in store.factor_history(zone["id"])
            ],
        }


def rebuild_zone_factor(store: Storage, zone_id: int) -> dict:
    """管理用：用该分区全部存活观测做一次全量重拟合（对比用）。

    不产生新版本（全量与折叠在数学上应一致），只返回全量结果与
    当前折叠结果及相对差。
    """
    with store.lock:
        zone = store.get_zone(zone_id)
        state = store.get_state(zone_id)
        active = store.active_measurements(zone_id)
        factors = [float(r["implied_factor"]) for r in active]
        cal = calibration.full_refit(
            factors, zone["baseline_factor"],
            zone["baseline_weight"], zone["forgetting"],
        )
        fold = calibration.Fold(
            w=state["fold_w"], wl=state["fold_wl"], count=state["obs_count"]
        )
        fold_factor = calibration.fold_to_factor(
            fold, zone["baseline_factor"], zone["baseline_weight"]
        )
        rel = (
            abs(cal.factor - fold_factor) / fold_factor
            if fold_factor else float("nan")
        )
        return {
            "zone_id": zone_id,
            "full_refit_factor": cal.factor,
            "incremental_factor": fold_factor,
            "relative_difference": rel,
            "n_observations": cal.n_observations,
            "forgetting": zone["forgetting"],
        }


# ---------------------------------------------------------------- 辅助

def _blast_params(store: Storage, blast) -> DesignParams:
    dv = store.get_design_version(blast["design_id"], blast["design_version"])
    return params_from_dict(loads(dv["params_json"]))


def _fit_to_json(fit: FitResult) -> dict:
    d = asdict(fit)
    return _sanitize_nan(d)


def _sanitize_nan(obj):
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "+00:00"


def _parse_ts(ts: str) -> datetime:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        dt = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise ValidationError(f"时间戳格式无法解析：{ts!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _zone_dict(r) -> dict:
    return {
        "id": r["id"],
        "name": r["name"],
        "lithology": r["lithology"],
        "baseline_factor": r["baseline_factor"],
        "baseline_weight": r["baseline_weight"],
        "forgetting": r["forgetting"],
        "created_at": r["created_at"],
    }


def _factor_version_dict(r) -> dict:
    return {
        "id": r["id"],
        "zone_id": r["zone_id"],
        "version_no": r["version_no"],
        "factor": r["factor"],
        "event_type": r["event_type"],
        "blast_id": r["blast_id"],
        "measurement_id": r["measurement_id"],
        "superseded": bool(r["superseded"]),
        "lineage": r["lineage"],
        "note": r["note"],
        "created_at": r["created_at"],
    }


def _blast_dict(store: Storage, b) -> dict:
    fv = store.factor_version(b["factor_version_id"])
    return {
        "id": b["id"],
        "design_id": b["design_id"],
        "design_version": b["design_version"],
        "zone_id": b["zone_id"],
        "factor_version": _factor_version_dict(fv),
        "fired_at": b["fired_at"],
        "created_at": b["created_at"],
    }
