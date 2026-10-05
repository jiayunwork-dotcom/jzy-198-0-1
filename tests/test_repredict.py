"""批量重新预测：系数更新后对未实施设计重预测，列出大块率翻转清单。"""

from __future__ import annotations

import pytest

from app import services
from tests.conftest import TYPICAL_PARAMS, synthetic_bins


# 临界设计：A=7 时大块率约 4.87%（合格），A>=8 即超标（见模型扫描）
CRITICAL_PARAMS = dict(TYPICAL_PARAMS)
CRITICAL_PARAMS.update(
    burden_m=4.0,
    spacing_m=6.0,
    explosive_mass_kg=550.0,
    stemming_m=4.0,
    hole_deviation_m=0.2,
    oversize_sieve_m=1.0,
    oversize_limit_pct=5.0,
)


def _zone(store, baseline, forgetting=1.0, name="Z"):
    return services.create_zone(
        store, name, baseline_factor=baseline, forgetting=forgetting
    )["id"]


def _new_design(store, zone, name, params=None):
    return services.create_design(store, name, zone, params or CRITICAL_PARAMS)["id"]


def _force_factor_via_measurement(store, zone, target_factor, n=1.5):
    """通过一次"预测 0.40 / 实测等比放大"的闭环把分区系数推到 target。

    利用单次反推恒等式 A_new = A_used * X50m / X50pred。
    """
    did = _new_design(store, zone, f"calib-{target_factor}-{id(store)}")
    bid = services.fire_blast(store, did)["id"]
    # 取当前系数下的预测 X50，再按目标比例造实测曲线
    from app.storage import loads
    blast = store.get_blast(bid)
    dv = store.get_design_version(blast["design_id"], blast["design_version"])
    params = services.params_from_dict(loads(dv["params_json"]))
    from app.fragmentation import predict_x50
    fv = store.factor_version(blast["factor_version_id"])
    xpred = predict_x50(params, fv["factor"])
    xmeas = xpred * target_factor / fv["factor"]
    out = services.upload_measurement(store, bid, synthetic_bins(xmeas, n))
    assert out["coefficient_updated"]
    return out


def test_bulk_repredict_lists_pass_to_fail(store):
    zone = _zone(store, baseline=7.0)
    # 三个未实施设计，初始都在 A=7 下预测（临界设计合格）
    d_critical = _new_design(store, zone, "临界孔网")
    d_other = _new_design(store, zone, "另一个临界孔网")
    pred_c = services.predict(store, d_critical)
    pred_o = services.predict(store, d_other)
    assert pred_c["prediction"]["oversize_ok"] is True
    assert pred_o["prediction"]["oversize_ok"] is True

    # 系数尚未变化：批量重预测应跳过（无新预测产生）
    same = services.bulk_repredict(store)
    assert same["repredicted"] == 0
    assert same["flipped"] == 0

    # 一次实测把系数从 7 推到 8.4（>8 即超标）
    _force_factor_via_measurement(store, zone, 8.4)
    result = services.bulk_repredict(store)
    assert result["repredicted"] == 2
    assert result["flipped"] == 2
    dirs = {f["design_id"]: f["flip_direction"] for f in result["flips"]}
    assert dirs[d_critical] == "pass_to_fail"
    assert dirs[d_other] == "pass_to_fail"
    for f in result["flips"]:
        assert f["previous"]["oversize_ok"] is True
        assert f["current"]["oversize_ok"] is False
        assert f["current"]["oversize_pct"] > 5.0
        assert f["current"]["rock_factor"] == pytest.approx(8.4)


def test_bulk_repredict_lists_fail_to_pass_when_factor_drops(store):
    """实测显示岩石比预想更软（系数下降）时，超标的设计转为合格。"""
    # 基线高（临界设计在 A=10 下超标）
    zone = _zone(store, baseline=10.0, name="高系数区")
    did = _new_design(store, zone, "超标设计")
    pred = services.predict(store, did)
    assert pred["prediction"]["oversize_ok"] is False

    # 一次实测把系数拉回 7（合格）
    _force_factor_via_measurement(store, zone, 7.0)
    result = services.bulk_repredict(store)
    assert result["flipped"] == 1
    flip = result["flips"][0]
    assert flip["design_id"] == did
    assert flip["flip_direction"] == "fail_to_pass"
    assert flip["previous"]["oversize_ok"] is False
    assert flip["current"]["oversize_ok"] is True


def test_bulk_repredict_skips_implemented(store):
    """已实施设计不参与批量重预测。"""
    zone = _zone(store, baseline=7.0)
    d_open = _new_design(store, zone, "未实施")
    d_fired = _new_design(store, zone, "已实施")
    services.predict(store, d_open)
    services.predict(store, d_fired)
    services.fire_blast(store, d_fired)
    _force_factor_via_measurement(store, zone, 10.0)
    result = services.bulk_repredict(store)
    touched = {e["design_id"] for e in result["details"]}
    assert d_open in touched
    assert d_fired not in touched


def test_bulk_repredict_uses_latest_design_version(store):
    """设计被修改（新版本）后，批量重预测用最新版本参数。"""
    zone = _zone(store, baseline=7.0)
    did = _new_design(store, zone, "可调整孔网")
    services.predict(store, did)  # v1 预测
    # 修改设计为更密的孔网（装药量/体积更大 -> 更细，保持合格）
    improved = dict(CRITICAL_PARAMS)
    improved.update(burden_m=3.5, spacing_m=5.0, explosive_mass_kg=550.0)
    services.update_design(store, did, improved)
    _force_factor_via_measurement(store, zone, 9.0)
    result = services.bulk_repredict(store)
    entry = next(e for e in result["details"] if e["design_id"] == did)
    assert entry["current"] is not None
    # 新预测单耗对应更密孔网
    q_improved = 550.0 / (3.5 * 5.0 * 12.0)
    assert entry["current"]["powder_factor_kg_m3"] == pytest.approx(q_improved, rel=1e-9)
