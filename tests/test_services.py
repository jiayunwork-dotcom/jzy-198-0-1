"""端到端服务测试：闭环标定、撤回重推、并发入库、批量重预测、版本查询。"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import services
from app.calibration import full_refit
from app.fragmentation import predict_x50
from app.storage import loads
from tests.conftest import TYPICAL_PARAMS, rr_passing, synthetic_bins


def _make_zone(store, baseline=10.0, forgetting=1.0, name="北区", bw=0.0):
    return services.create_zone(
        store, name, baseline_factor=baseline, forgetting=forgetting,
        baseline_weight=bw, lithology="花岗岩"
    )["id"]


def _make_design(store, zone_id, name="D", **param_over):
    params = dict(TYPICAL_PARAMS)
    params.update(param_over)
    return services.create_design(store, name, zone_id, params)["id"]


def _fire(store, design_id, fired_at=None):
    return services.fire_blast(store, design_id, fired_at=fired_at)["id"]


def _upload_curve(store, blast_id, x50, n=1.5):
    bins = synthetic_bins(x50, n)
    return services.upload_measurement(store, blast_id, bins)


def test_end_to_end_ingestion_updates_factor(store):
    zone = _make_zone(store, baseline=7.0, forgetting=1.0)
    d1 = _make_design(store, zone, "D1")
    b1 = _fire(store, d1)
    res = _upload_curve(store, b1, x50=0.48)
    assert res["coefficient_updated"] is True
    # 单次反推（默认基线权重 0，首个实测反推值即为新区系数）：
    # A=7*X50实测/X50预测
    fv = res["factor_version"]
    factor = fv["factor"]
    from app.storage import loads as _loads
    blast = store.get_blast(b1)
    dv = store.get_design_version(blast["design_id"], blast["design_version"])
    p = services.params_from_dict(_loads(dv["params_json"]))
    xpred = predict_x50(p, 7.0)
    assert factor == pytest.approx(7.0 * 0.48 / xpred, rel=1e-12)
    # 分区当前系数与历史一致
    cur = services.current_factor(store, zone)
    assert cur["factor"] == pytest.approx(factor, rel=1e-12)
    assert cur["observations"] == 1
    hist = services.factor_history(store, zone)
    assert [h["version_no"] for h in hist] == [1, 2]
    assert hist[0]["event_type"] == "baseline"
    assert hist[1]["event_type"] == "measurement"


def test_multiple_uploads_match_full_refit(store):
    """rho=1：若干条入库后，增量系数与全量重拟合一致（1e-9）。"""
    zone = _make_zone(store, baseline=8.0, forgetting=1.0, bw=1.0)
    implied = []
    for i, xm in enumerate([0.35, 0.42, 0.50, 0.31, 0.38]):
        did = _make_design(store, zone, f"D{i}")
        bid = _fire(store, did)
        out = _upload_curve(store, bid, xm)
        implied.append(out["implied_factor"])
    rebuild = services.rebuild_zone_factor(store, zone)
    assert rebuild["relative_difference"] < 1e-9
    assert rebuild["full_refit_factor"] == pytest.approx(
        full_refit(implied, 8.0, 1.0, 1.0).factor, rel=1e-12
    )


def test_withdraw_middle_and_replay(store):
    """撤回中间一条：由它引起的更新撤回，其后版本在剔除基础上重推。"""
    zone = _make_zone(store, baseline=9.0, forgetting=1.0)
    uploads = []
    for i, xm in enumerate([0.30, 0.45, 0.40]):
        did = _make_design(store, zone, f"D{i}")
        bid = _fire(store, did)
        out = _upload_curve(store, bid, xm)
        uploads.append((bid, out))

    a1, a2, a3 = [u[1]["implied_factor"] for u in uploads]
    f3 = services.current_factor(store, zone)["factor"]
    expected_all = full_refit([a1, a2, a3], 9.0, 0.0, 1.0).factor
    assert f3 == pytest.approx(expected_all, rel=1e-12)

    # 撤回第 2 条
    mid2 = [
        r["id"] for r in store.query_all(
            "SELECT id FROM measurements ORDER BY id"
        )
    ][1]
    wd = services.withdraw_measurement(store, mid2)
    factor_after = wd["factor_version"]["factor"]
    expected_remaining = full_refit([a1, a3], 9.0, 0.0, 1.0).factor
    assert factor_after == pytest.approx(expected_remaining, rel=1e-12)

    # 存活实测的版本外键指向重推版本
    m1 = store.query_one(
        "SELECT factor_version_id FROM measurements WHERE withdrawn=0 ORDER BY id LIMIT 1"
    )
    m3 = store.query_one(
        "SELECT factor_version_id FROM measurements WHERE withdrawn=0 ORDER BY id DESC LIMIT 1"
    )
    fv1 = store.factor_version(m1["factor_version_id"])
    fv3 = store.factor_version(m3["factor_version_id"])
    assert fv1["lineage"] == "active"           # 撤点之前的版本不动
    assert fv3["lineage"] == "rebuilt"          # 之后的被重推
    assert fv3["superseded"] == 0

    # 当前活动历史：baseline(v1), v2(m1), withdrawal(v_new), rebuilt(a3)
    active = services.factor_history(store, zone)
    lineages = [h["lineage"] for h in active]
    assert lineages.count("active") >= 2
    assert "rebuilt" in lineages
    # 被取代的旧行仍可查
    full_hist = services.factor_history(store, zone, include_superseded=True)
    assert any(h["superseded"] for h in full_hist)

    # 再撤回第一条：只剩 a3
    mid1 = store.query_one(
        "SELECT id FROM measurements ORDER BY id LIMIT 1"
    )["id"]
    wd2 = services.withdraw_measurement(store, mid1)
    assert wd2["factor_version"]["factor"] == pytest.approx(
        full_refit([a3], 9.0, 0.0, 1.0).factor, rel=1e-12
    )


def test_withdraw_last_then_factor_reverts(store):
    zone = _make_zone(store, baseline=10.0, forgetting=1.0)
    ids = []
    for i, xm in enumerate([0.34, 0.40]):
        did = _make_design(store, zone, f"D{i}")
        bid = _fire(store, did)
        out = _upload_curve(store, bid, xm)
        ids.append(out["measurement_id"])
    factor_before_last = services.factor_history(store, zone)[1]["factor"]
    wd = services.withdraw_measurement(store, ids[1])
    assert wd["factor_version"]["factor"] == pytest.approx(
        factor_before_last, rel=1e-12
    )
    # 再加入新实测，序列在撤回后继续工作
    did = _make_design(store, zone, "Dnew")
    bid = _fire(store, did)
    out = _upload_curve(store, bid, 0.36)
    assert out["coefficient_updated"]


def test_bad_curve_measurement_does_not_update_factor(store):
    zone = _make_zone(store, baseline=10.0, forgetting=1.0)
    did = _make_design(store, zone)
    bid = _fire(store, did)

    def bimodal(d):
        return 0.45 * rr_passing(0.08, 1.6, d) + 0.55 * rr_passing(0.55, 1.9, d)

    from tests.conftest import synthetic_bins_custom_passing
    bins = synthetic_bins_custom_passing(bimodal)
    out = services.upload_measurement(store, bid, bins)
    assert out["coefficient_updated"] is False
    assert out["fit"]["usable"] is False
    assert services.current_factor(store, zone)["factor"] == pytest.approx(10.0)
    # 不可用实测可以撤回（无系数级联），不报错
    services.withdraw_measurement(store, out["measurement_id"])


def test_concurrent_uploads_same_zone_equal_sequential(store):
    """两次实测同时入库同一分区：最终系数 == 按实际入库(seq)顺序串行处理。

    带遗忘因子时加权几何均值与顺序有关，因此"一致"的定义是：
    无论锁被谁先拿到，结果都必须与该持锁顺序下逐条调用的结果相同。
    本测试构造 N 个爆破，用线程屏障让所有上传同时发起，重复多次
    以覆盖不同的持锁先后。
    """
    import os
    import tempfile
    from app.storage import Storage

    n = 6
    x_targets = [0.33, 0.47, 0.28, 0.41, 0.36, 0.52]
    tmp = tempfile.mkdtemp()

    def fresh_db(tag):
        s = Storage(os.path.join(tmp, f"{tag}.db"))
        zid = services.create_zone(
            s, tag, baseline_factor=8.0, forgetting=0.85
        )["id"]
        bids = []
        for i in range(n):
            did = _make_design(s, zid, f"{tag}{i}")
            bids.append(_fire(s, did))
        return s, zid, bids

    all_bins = [synthetic_bins(x, 1.5) for x in x_targets]
    counter = {"i": 0}
    seen_orders = set()

    def sequential_factor(order_bids, order_bins):
        counter["i"] += 1
        s, zid, _ = fresh_db(f"seq{counter['i']}")
        for bid, bins in zip(order_bids, order_bins):
            services.upload_measurement(s, bid, bins)
        f = services.current_factor(s, zid)["factor"]
        s.close()
        return f

    for trial in range(8):
        s, zid, bids = fresh_db(f"par{trial}")
        barrier = threading.Barrier(n)

        def upload(idx):
            barrier.wait()
            return services.upload_measurement(s, bids[idx], all_bins[idx])

        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(upload, range(n)))

        rows = s.query_all(
            """SELECT blast_id, seq_in_zone FROM measurements
               ORDER BY seq_in_zone"""
        )
        blast_to_idx = {bid: i for i, bid in enumerate(bids)}
        order_idx = tuple(blast_to_idx[r["blast_id"]] for r in rows)
        seen_orders.add(order_idx)

        # 按同样的持锁顺序在独立库里串行处理，结果必须逐位相同
        ordered_bids = [bids[i] for i in order_idx]
        ordered_bins = [all_bins[i] for i in order_idx]
        expected = sequential_factor(ordered_bids, ordered_bins)
        actual = services.current_factor(s, zid)["factor"]
        assert actual == pytest.approx(expected, rel=1e-12)

        # seq 必然是 1..n 各一次
        assert [r["seq_in_zone"] for r in rows] == list(range(1, n + 1))
        s.close()

    # 多次并发应至少观察到两种持锁先后，否则屏障没有真正制造竞争
    assert len(seen_orders) >= 2, seen_orders


def test_design_and_factor_versions_bound_to_prediction(store):
    zone = _make_zone(store, baseline=7.0, forgetting=1.0)
    did = _make_design(store, zone)
    pred1 = services.predict(store, did)
    assert pred1["design_version"] == 1
    assert pred1["factor_version"]["version_no"] == 1
    bid = _fire(store, did)
    _upload_curve(store, bid, 0.50)
    # 爆破后设计锁定
    with pytest.raises(services.ConflictError):
        services.update_design(store, did, TYPICAL_PARAMS)
    # 新建一个设计，预测绑定新系数版本
    did2 = _make_design(store, zone, "D2")
    pred2 = services.predict(store, did2)
    assert pred2["factor_version"]["id"] != pred1["factor_version"]["id"]


def test_factor_as_of_query(store):
    from app.storage import NotFound
    zone = _make_zone(store, baseline=7.0, forgetting=1.0)
    did = _make_design(store, zone)
    bid = _fire(store, did)
    zone_created = services.get_zone(store, zone)["created_at"]
    # 分区刚创建、首条实测尚未入库：只看得到基线版本
    before = services.factor_as_of(store, zone, zone_created)
    assert before["version_no"] == 1
    assert before["factor"] == pytest.approx(7.0)
    _upload_curve(store, bid, 0.44)
    # 同一时刻仍是基线（点查询不被未来更新影响）
    before2 = services.factor_as_of(store, zone, zone_created)
    assert before2["version_no"] == 1
    # 未来能看到最新
    late = services.factor_as_of(store, zone, "2099-01-01T00:00:00+00:00")
    assert late["version_no"] >= 2
    # 分区诞生之前 -> 404
    with pytest.raises(NotFound):
        services.factor_as_of(store, zone, "2000-01-01T00:00:00+00:00")


def test_blast_records_factor_version_used_at_time(store):
    """某次爆破当时用的是哪一版系数可查。"""
    zone = _make_zone(store, baseline=7.0, forgetting=1.0)
    did = _make_design(store, zone)
    bid = _fire(store, did)
    blast = services.get_blast(store, bid)
    assert blast["factor_version"]["version_no"] == 1
    _upload_curve(store, bid, 0.44)
    # 历史记录不变
    blast2 = services.get_blast(store, bid)
    assert blast2["factor_version"]["version_no"] == 1
