"""HTTP 接口测试：拒收规则、闭环 API、版本查询。"""

from __future__ import annotations

import pytest

from tests.conftest import TYPICAL_PARAMS, synthetic_bins


def _zone(client, name="北区", baseline=7.0, forgetting=1.0):
    r = client.post("/zones", json={
        "name": name, "baseline_factor": baseline,
        "forgetting": forgetting, "lithology": "花岗岩",
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _design(client, zone_id, name="D", **over):
    params = dict(TYPICAL_PARAMS)
    params.update(over)
    r = client.post("/designs", json={"name": name, "zone_id": zone_id, "params": params})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _blast(client, design_id):
    r = client.post("/blasts", json={"design_id": design_id})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------- 健康检查

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_passing_reference_endpoint(client):
    # 参考值：X50=0.30、n=1.5、0.60 m -> 85.9%
    r = client.get("/passing", params={
        "sieve_m": 0.60, "x50_m": 0.30, "uniformity_n": 1.5})
    assert r.status_code == 200
    body = r.json()
    assert body["passing_pct"] == pytest.approx(85.9, abs=0.05)
    assert body["retained_pct"] == pytest.approx(14.1, abs=0.05)


def test_passing_requires_arguments(client):
    r = client.get("/passing", params={"sieve_m": 0.6})
    assert r.status_code == 400
    r = client.get("/passing", params={
        "sieve_m": 0.6, "prediction_id": 999})
    assert r.status_code == 404


# ---------------------------------------------------------------- 拒收

@pytest.mark.parametrize("field,bad", [
    ("hole_diameter_mm", 0),
    ("burden_m", -1),
    ("spacing_m", 0),
    ("bench_height_m", -2),
    ("explosive_mass_kg", 0),
])
def test_reject_non_positive_params(client, field, bad):
    z = _zone(client)
    params = dict(TYPICAL_PARAMS)
    params[field] = bad
    r = client.post("/designs", json={"name": "bad", "zone_id": z, "params": params})
    assert r.status_code == 400
    assert field in r.json()["error"] or "正数" in r.json()["error"]


def test_reject_stemming_longer_than_hole(client):
    z = _zone(client)
    params = dict(TYPICAL_PARAMS, stemming_m=14.0, hole_depth_m=13.5)
    r = client.post("/designs", json={"name": "bad", "zone_id": z, "params": params})
    assert r.status_code == 400
    assert "堵塞" in r.json()["error"]


def test_reject_non_positive_rws(client):
    z = _zone(client)
    r = client.post("/designs", json={
        "name": "bad", "zone_id": z,
        "params": dict(TYPICAL_PARAMS, explosive_rws=0),
    })
    assert r.status_code == 400


def test_reject_unknown_zone_and_design(client):
    r = client.post("/designs", json={
        "name": "x", "zone_id": 999, "params": TYPICAL_PARAMS})
    assert r.status_code == 404
    r = client.post("/designs/999/predict", json={})
    assert r.status_code == 404
    r = client.post("/blasts", json={"design_id": 999})
    assert r.status_code == 404


def test_reject_non_monotonic_sieve(client):
    z = _zone(client)
    d = _design(client, z)
    b = _blast(client, d)
    # 出现负占比 -> 累计不单调
    bins = [
        {"lower": None, "upper": 0.1, "retained_pct": 30},
        {"lower": 0.1, "upper": 0.3, "retained_pct": -10},
        {"lower": 0.3, "upper": None, "retained_pct": 80},
    ]
    r = client.post("/measurements", json={"blast_id": b, "bins": bins})
    assert r.status_code == 400


def test_reject_passing_out_of_0_100(client):
    z = _zone(client)
    d = _design(client, z)
    b = _blast(client, d)
    # 各档合计 130%（等价于累计通过率越出 100）
    bins = [
        {"lower": None, "upper": 0.1, "retained_pct": 60},
        {"lower": 0.1, "upper": None, "retained_pct": 70},
    ]
    r = client.post("/measurements", json={"blast_id": b, "bins": bins})
    assert r.status_code == 400


def test_reject_measurement_for_unknown_blast(client):
    bins = synthetic_bins(0.3, 1.5)
    r = client.post("/measurements", json={"blast_id": 4242, "bins": bins})
    assert r.status_code == 404


def test_reject_duplicate_blast(client):
    z = _zone(client)
    d = _design(client, z)
    assert client.post("/blasts", json={"design_id": d}).status_code == 201
    r = client.post("/blasts", json={"design_id": d})
    assert r.status_code == 409


def test_update_implemented_design_rejected(client):
    z = _zone(client)
    d = _design(client, z)
    _blast(client, d)
    r = client.patch(f"/designs/{d}", json={"params": TYPICAL_PARAMS})
    assert r.status_code == 409


# ---------------------------------------------------------------- 闭环

def test_full_loop_api(client):
    z = _zone(client, baseline=7.0, forgetting=1.0)
    d = _design(client, z, name="台阶12-北")
    # 另一个尚未实施的设计：系数更新前先预测一次
    d2 = _design(client, z, name="台阶12-北-2")
    client.post(f"/designs/{d2}/predict")

    # 预测（绑定 v1 基线系数）
    r = client.post(f"/designs/{d}/predict")
    assert r.status_code == 200
    pred = r.json()
    assert pred["factor_version"]["version_no"] == 1
    assert "oversize_pct" in pred["prediction"]
    assert 0 < pred["prediction"]["oversize_pct"] < 100

    # 实施
    b = _blast(client, d)

    # 上传一条与参考场景接近的实测
    bins = synthetic_bins(0.48, 1.5)
    r = client.post("/measurements", json={"blast_id": b, "bins": bins})
    assert r.status_code == 201
    m = r.json()
    assert m["coefficient_updated"] is True
    assert m["fit"]["usable"] is True
    new_factor = m["factor_version"]["factor"]

    # 分区当前系数与历史
    r = client.get(f"/zones/{z}/factor")
    assert r.json()["factor"] == pytest.approx(new_factor)
    hist = client.get(f"/zones/{z}/factor/history").json()
    assert [h["version_no"] for h in hist] == [1, 2]

    # 实测详情
    r = client.get(f"/measurements/{m['measurement_id']}")
    assert r.status_code == 200
    assert r.json()["fit"]["x50_m"] == pytest.approx(0.48, rel=1e-6)

    # 新建未实施设计 -> 批量重预测（d2 的预测绑定的是旧系数版本）
    r = client.post("/designs/repredict")
    assert r.status_code == 200
    assert r.json()["repredicted"] >= 1

    # 撤回实测 -> 系数回到基线
    r = client.post(f"/measurements/{m['measurement_id']}/withdraw", json={})
    assert r.status_code == 200
    cur = client.get(f"/zones/{z}/factor").json()
    assert cur["factor"] == pytest.approx(7.0)
    # 撤回前后的历史都可查
    full = client.get(f"/zones/{z}/factor/history?include_superseded=true").json()
    assert any(h["lineage"] == "rebuilt" for h in full)
    assert any(h["lineage"] == "active" for h in full)


def test_bad_curve_api_marked_unusable(client):
    from tests.conftest import synthetic_bins_custom_passing, rr_passing

    z = _zone(client)
    d = _design(client, z)
    b = _blast(client, d)

    def bimodal(x):
        return 0.45 * rr_passing(0.08, 1.6, x) + 0.55 * rr_passing(0.55, 1.9, x)

    bins = synthetic_bins_custom_passing(bimodal)
    r = client.post("/measurements", json={"blast_id": b, "bins": bins})
    assert r.status_code == 201
    body = r.json()
    assert body["fit"]["usable"] is False
    assert body["coefficient_updated"] is False
    assert body["factor_version"]["factor"] == pytest.approx(7.0)


def test_design_versions_and_predictions_listed(client):
    z = _zone(client)
    d = _design(client, z)
    client.post(f"/designs/{d}/predict")
    client.patch(f"/designs/{d}", json={"params": dict(
        TYPICAL_PARAMS, explosive_mass_kg=500.0)})
    client.post(f"/designs/{d}/predict")
    r = client.get(f"/designs/{d}")
    assert r.json()["current_version"] == 2
    preds = client.get(f"/designs/{d}/predictions").json()
    assert [p["design_version"] for p in preds] == [1, 2]
    assert preds[-1]["is_current"] is True
