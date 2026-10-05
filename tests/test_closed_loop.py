"""闭环主体测试：
  * 增量入库与全量重拟合一致（不衰减，相对差 1e-9 以内）；
  * 撤回实测 -> 系数链重推，撤回前后历史都可查；
  * 两次实测并发入库同一分区，结果与按入库顺序依次处理一致；
  * 系数更新后批量重新预测，列出大块合格性翻转清单。
"""

import threading

import numpy as np
import pytest

from tests.conftest import SIEVE_SIZES, rr_curve


def _zone(client, decay, factor=7.0):
    r = client.post("/zones", json={
        "name": "测试区", "lithology": "安山岩",
        "initial_factor": factor, "decay": decay,
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _design(client, zone_id, payload, **over):
    body = {**payload, "zone_id": zone_id, **over}
    r = client.post("/designs", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload_curve(client, design_id, x50, n=1.4, noise=0.0, seed=0):
    points = rr_curve(SIEVE_SIZES, x50, n, noise=noise, seed=seed)
    r = client.post("/measurements",
                    json={"design_id": design_id, "points": points})
    assert r.status_code == 201, r.text
    return r.json()


def _db_observations(conn, zone_id):
    from app import calibration
    rows = conn.execute(
        """SELECT single_factor, qweight, id FROM measurements
           WHERE zone_id=? AND status='active' AND usable=1 ORDER BY id""",
        (zone_id,),
    ).fetchall()
    return [
        calibration.Observation(r["single_factor"], r["qweight"], r["id"])
        for r in rows
    ]


# ----------------------------------------------------------- 增量 vs 全量
def test_incremental_upload_matches_full_refit_decay_one(
        client, conn, standard_design_payload):
    """decay=1：连续入库 6 次，接口算出的当前系数与全量重拟合相对差 < 1e-9。"""
    from app import calibration

    zone = _zone(client, decay=1.0, factor=7.0)
    rng = np.random.default_rng(0)

    # 每次设计参数略变（g 不同），实测中值块度也不同
    for k in range(6):
        d = _design(
            client, zone, standard_design_payload,
            burden=round(5.8 + 0.15 * k, 2),
            charge_kg=float(300 + 10 * k),
            name=f"爆区{k}",
        )
        # 让实测反推系数落在 6.5~9.5 之间
        # x50_meas = a_target * g，g 由设计参数决定
        from app.service import _geometry_factor, design_version_params
        params = design_version_params(conn, d, 1)
        g = _geometry_factor(params)
        a_target = 6.5 + rng.uniform(0, 3.0)
        _upload_curve(client, d, a_target * g, n=1.3 + 0.05 * k)

        # 每一步都核对增量口径与全量口径
        factor_now = client.get(f"/zones/{zone}").json()["current_factor"]
        full = calibration.fit_all(_db_observations(conn, zone), decay=1.0)
        assert calibration.relative_difference(factor_now, full) < 1e-9

    # 版本链：初始 + 6
    hist = client.get(f"/zones/{zone}/factor-history").json()
    assert [v["version_index"] for v in hist] == [0, 1, 2, 3, 4, 5, 6]
    assert all(v["active"] for v in hist)


def test_bad_curve_not_calibrated_but_design_blasted(
        client, conn, standard_design_payload):
    """双峰坏曲线：标记不可用、不出系数新版本，但设计仍标记为已实施。"""
    zone = _zone(client, decay=1.0)
    d = _design(client, zone, standard_design_payload)
    LN2 = np.log(2.0)
    sizes = np.array([0.02, 0.04, 0.08, 0.16, 0.25, 0.4, 0.63, 1.0, 1.5])

    def rr(x, x50, n):
        return 1 - np.exp(-LN2 * (x / x50) ** n)

    p = 100 * (0.5 * rr(sizes, 0.12, 1.6) + 0.5 * rr(sizes, 0.75, 1.6))
    points = [{"size_m": float(s), "passing_pct": float(v)}
              for s, v in zip(sizes, p)]
    r = client.post("/measurements",
                    json={"design_id": d, "points": points})
    assert r.status_code == 201
    m = r.json()
    assert m["fit"]["usable"] is False
    assert m["fit"]["bimodal"] is True
    assert m["factor_version_id"] is None
    # 系数没有新版本
    hist = client.get(f"/zones/{zone}/factor-history").json()
    assert len(hist) == 1
    # 设计已实施
    assert client.get(f"/designs/{d}").json()["status"] == "blasted"


# ----------------------------------------------------------- 撤回重推
def test_withdraw_reissues_versions(client, conn, standard_design_payload):
    """3 次入库后撤回第 2 条：其后版本重推；第 1 条版本保留；历史全可查。"""
    from app import calibration
    from app.service import _geometry_factor, design_version_params

    zone = _zone(client, decay=0.9, factor=7.0)
    targets = [8.0, 9.5, 7.2]
    m_ids = []
    for k, a_target in enumerate(targets):
        d = _design(client, zone, standard_design_payload,
                    burden=6.0 + 0.1 * k, name=f"爆区{k}")
        params = design_version_params(conn, d, 1)
        m = _upload_curve(client, d, a_target * _geometry_factor(params),
                          n=1.4)
        m_ids.append(m["id"])

    before = client.get(f"/zones/{zone}").json()["current_factor"]
    # 模拟人工口径：三次有效观测全量折叠
    all_obs = _db_observations(conn, zone)
    assert before == pytest.approx(calibration.fit_all(all_obs, 0.9), rel=1e-12)

    # 撤回第 2 条（id = m_ids[1]）
    w = client.post(f"/measurements/{m_ids[1]}/withdraw")
    assert w.status_code == 200, w.text
    wm = w.json()
    assert wm["status"] == "withdrawn"
    assert wm["withdrawn_at"] is not None

    # 新系数 = 剔除第 2 条后的全量折叠
    kept = [o for o in _db_observations(conn, zone)
            if o.evidence_id != m_ids[1]]
    after = client.get(f"/zones/{zone}").json()["current_factor"]
    assert after == pytest.approx(calibration.fit_all(kept, 0.9), rel=1e-12)
    assert after != before

    # 第 1 条引发的版本仍 active 且数值不变；第 3 条挂到重推版本
    m1 = client.get(f"/measurements/{m_ids[0]}").json()
    m3 = client.get(f"/measurements/{m_ids[2]}").json()
    fv1 = client.get(f"/zones/{zone}/factor",
                     params={"version_id": m1["factor_version_id"]}).json()
    assert fv1["active"] is True
    fv3 = client.get(f"/zones/{zone}/factor",
                     params={"version_id": m3["factor_version_id"]}).json()
    assert "withdraw" in fv3["note"]
    assert fv3["active"] is True

    # 撤回前的旧版本（第 2、3 条引发的）仍可按 id 查到，但已在撤回链上
    old_hist = client.get(f"/zones/{zone}/factor-history").json()
    withdrawn_versions = [v for v in old_hist if v["chain"] == "withdrawn"]
    assert len(withdrawn_versions) == 2
    for v in withdrawn_versions:
        assert v["active"] is False
        assert v["superseded_by"] is not None or v["evidence_id"] == m_ids[1]

    # active_only 视图只返回有效链
    active_hist = client.get(
        f"/zones/{zone}/factor-history", params={"active_only": True}
    ).json()
    assert all(v["active"] for v in active_hist)
    assert len(active_hist) == 3  # 初始 + 第1条 + 第3条重推

    # 已撤回实测的拟合结果仍可查
    fit_q = client.get(f"/measurements/{m_ids[1]}").json()
    assert "fit" in fit_q and fit_q["fit"]["usable"]

    # 不能重复撤回
    again = client.post(f"/measurements/{m_ids[1]}/withdraw")
    assert again.status_code == 409

    # 撤回重推后再入库一条，链继续推进且与全量一致
    d4 = _design(client, zone, standard_design_payload, burden=6.3, name="爆区3")
    params = design_version_params(conn, d4, 1)
    _upload_curve(client, d4, 8.7 * _geometry_factor(params), n=1.4)
    final = client.get(f"/zones/{zone}").json()["current_factor"]
    obs_now = _db_observations(conn, zone)
    expected_ids = [m_ids[0], m_ids[2]]
    newest_meas = conn.execute(
        "SELECT MAX(id) AS m FROM measurements WHERE zone_id=?", (zone,)
    ).fetchone()["m"]
    expected_ids.append(newest_meas)
    assert [o.evidence_id for o in obs_now] == expected_ids
    assert final == pytest.approx(calibration.fit_all(obs_now, 0.9), rel=1e-12)


# ----------------------------------------------------------- 并发入库
def test_concurrent_uploads_sequential_consistency(
        client, conn, standard_design_payload):
    """两次（实际 8 次）实测并发入同一分区，最终系数 == 按 measurements.id
    顺序依次处理的全量结果。"""
    from app import calibration
    from app.service import _geometry_factor, design_version_params

    zone = _zone(client, decay=1.0, factor=7.0)
    n_threads = 8
    design_ids = []
    targets = []
    for k in range(n_threads):
        d = _design(client, zone, standard_design_payload,
                    burden=round(5.9 + 0.08 * k, 2),
                    charge_kg=float(290 + 8 * k), name=f"并发爆区{k}")
        design_ids.append(d)
        params = design_version_params(conn, d, 1)
        targets.append((6.8 + 0.35 * k) * _geometry_factor(params))

    barrier = threading.Barrier(n_threads)
    results: list[int] = []
    errors: list[Exception] = []

    def worker(i):
        try:
            barrier.wait()
            m = _upload_curve(client, design_ids[i], targets[i], n=1.4)
            results.append(m["id"])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert len(results) == n_threads

    # 按入库顺序（measurements.id）全量折叠应与当前系数完全一致
    final = client.get(f"/zones/{zone}").json()["current_factor"]
    seq = calibration.fit_all(_db_observations(conn, zone), decay=1.0)
    assert calibration.relative_difference(final, seq) < 1e-9

    # 顺序无关性（decay=1）：与按设计编号顺序的折叠也一致
    rows = conn.execute(
        """SELECT single_factor, qweight, id FROM measurements
           WHERE zone_id=? AND usable=1 ORDER BY design_id""",
        (zone,),
    ).fetchall()
    obs_by_design = [
        calibration.Observation(r["single_factor"], r["qweight"], r["id"])
        for r in rows
    ]
    assert calibration.relative_difference(
        final, calibration.fit_all(obs_by_design, 1.0)) < 1e-9


def test_concurrent_uploads_decay_ordered_by_id(
        client, conn, standard_design_payload):
    """decay<1 时结果依赖顺序；并发后系数仍须等于按 id 顺序的折叠。"""
    from app import calibration
    from app.service import _geometry_factor, design_version_params

    zone = _zone(client, decay=0.85, factor=7.0)
    n_threads = 6
    design_ids, targets = [], []
    for k in range(n_threads):
        d = _design(client, zone, standard_design_payload,
                    burden=round(6.0 + 0.1 * k, 2), name=f"d{k}")
        design_ids.append(d)
        params = design_version_params(conn, d, 1)
        targets.append((7.5 + 0.4 * k) * _geometry_factor(params))

    barrier = threading.Barrier(n_threads)
    errors = []

    def worker(i):
        try:
            barrier.wait()
            _upload_curve(client, design_ids[i], targets[i])
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    final = client.get(f"/zones/{zone}").json()["current_factor"]
    seq = calibration.fit_all(_db_observations(conn, zone), decay=0.85)
    assert calibration.relative_difference(final, seq) < 1e-9


# ----------------------------------------------------------- 批量重新预测
FLIP_DESIGN = dict(
    diameter_mm=250.0, burden=6.5, spacing=8.0, bench_height=12.0,
    hole_depth=13.5, stemming=4.0, charge_kg=300.0, explosive_rws=1.0,
    deviation=0.0, oversize_m=0.6, oversize_limit_pct=15.0,
)


def test_batch_repredict_flip_list(client, conn):
    """系数从 7 更新到约 8.4：跨门槛设计由合格变超标；撤回后又恢复。"""
    from app.service import _geometry_factor, design_version_params

    zone = _zone(client, decay=1.0, factor=7.0)
    # 两个未实施设计：临界设计 + 一个远离门槛的安全设计
    d_edge = _design(client, zone, FLIP_DESIGN, name="临界爆区")
    safe = dict(FLIP_DESIGN)
    safe.update(burden=5.5, spacing=6.5, charge_kg=380.0,
                oversize_limit_pct=25.0, name="安全爆区")
    d_safe = _design(client, zone, safe)

    p_edge = client.post(f"/designs/{d_edge}/predict").json()
    p_safe = client.post(f"/designs/{d_safe}/predict").json()
    assert p_edge["oversize_pass"] is True
    assert p_safe["oversize_pass"] is True

    # 一次实测把分区系数标定到 8.4
    d_blast = _design(client, zone, FLIP_DESIGN, name="已实施爆区")
    params = design_version_params(conn, d_blast, 1)
    meas = _upload_curve(
        client, d_blast, 8.4 * _geometry_factor(params), n=1.5
    )
    assert meas["rock_factor"] == pytest.approx(8.4, abs=1e-6)

    # 批量重预测（只重算尚未实施的）
    rep = client.post("/repredict", params={"zone_id": zone})
    assert rep.status_code == 200, rep.text
    report = rep.json()
    assert report["re_predicted"] == 2
    assert len(report["flips"]) == 1
    flip = report["flips"][0]
    assert flip["design_id"] == d_edge
    assert flip["old_pass"] is True and flip["new_pass"] is False
    assert flip["new_oversize_pct"] > 15.0
    # 安全设计不翻转
    flipped_ids = {f["design_id"] for f in report["flips"]}
    assert d_safe not in flipped_ids

    # 已实施的设计不参与重预测
    assert d_blast not in flipped_ids

    # 撤回该实测 -> 系数回 7 -> 再批量，临界设计超标变合格
    client.post(f"/measurements/{meas['id']}/withdraw")
    rep2 = client.post("/repredict", params={"zone_id": zone}).json()
    back = [f for f in rep2["flips"] if f["design_id"] == d_edge]
    assert len(back) == 1
    assert back[0]["old_pass"] is False and back[0]["new_pass"] is True
