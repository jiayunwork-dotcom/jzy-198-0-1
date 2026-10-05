"""接口层测试：拒收规则、设计版本化、预测与系数绑定、历史/时刻查询。"""

import pytest


def _make_zone(client, decay=1.0, factor=7.0, name="北帮花岗岩区"):
    r = client.post("/zones", json={
        "name": name, "lithology": "花岗岩",
        "initial_factor": factor, "decay": decay,
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_design(client, payload, zone_id):
    body = {**payload, "zone_id": zone_id}
    r = client.post("/designs", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_reference_via_api_prediction(client, standard_design_payload):
    """直接验证预测接口的通过率表在参考参数下的一致性（模型函数另测85.9%）。"""
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    r = client.post(f"/designs/{d['id']}/predict",
                    params={"extra_sieves": [0.6]})
    assert r.status_code == 201, r.text
    pred = r.json()
    assert pred["design_version"] == 1
    assert pred["factor_version_id"] == 1
    assert pred["rock_factor"] == 7.0
    assert pred["passing_table"]["0.6"] > 0  # 表可查任意筛孔
    assert 0 < pred["x50"] < 1.0


def test_predict_binds_versions_and_is_immutable(client, standard_design_payload):
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    p1 = client.post(f"/designs/{d['id']}/predict").json()
    # 之后设计改了：新预测绑定设计版本 2，旧预测仍记录版本 1
    client.patch(f"/designs/{d['id']}", json={"burden": 5.5})
    p2 = client.post(f"/designs/{d['id']}/predict").json()
    assert p1["design_version"] == 1 and p2["design_version"] == 2
    assert p1["id"] != p2["id"]
    hist = client.get(f"/designs/{d['id']}/predictions").json()
    assert [h["design_version"] for h in hist] == [2, 1]


# ---------------- 拒收 ----------------
@pytest.mark.parametrize("field,bad", [
    ("diameter_mm", 0), ("diameter_mm", -250),
    ("burden", -6.0), ("spacing", 0), ("bench_height", -12),
    ("charge_kg", 0),
])
def test_reject_non_positive(client, standard_design_payload, field, bad):
    z = _make_zone(client)
    body = {**standard_design_payload, "zone_id": z, field: bad}
    r = client.post("/designs", json=body)
    assert r.status_code == 422, r.text


def test_reject_stemming_longer_than_hole(client, standard_design_payload):
    z = _make_zone(client)
    body = {**standard_design_payload, "zone_id": z,
            "hole_depth": 13.5, "stemming": 14.0}
    r = client.post("/designs", json=body)
    assert r.status_code == 422
    assert "stemming" in r.text


def test_reject_non_positive_rws(client, standard_design_payload):
    z = _make_zone(client)
    body = {**standard_design_payload, "zone_id": z, "explosive_rws": 0}
    assert client.post("/designs", json=body).status_code == 422


def test_reject_unknown_zone(client, standard_design_payload):
    body = {**standard_design_payload, "zone_id": 999}
    assert client.post("/designs", json=body).status_code == 404


def test_reject_unknown_design_predict(client):
    assert client.post("/designs/999/predict").status_code == 404


def test_reject_non_monotone_sieve(client, standard_design_payload):
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    points = [
        {"size_m": 0.1, "passing_pct": 20.0},
        {"size_m": 0.2, "passing_pct": 40.0},
        {"size_m": 0.4, "passing_pct": 35.0},  # 回落
        {"size_m": 0.8, "passing_pct": 90.0},
    ]
    r = client.post("/measurements",
                    json={"design_id": d["id"], "points": points})
    assert r.status_code == 422
    assert "non-decreasing" in r.text


def test_reject_pct_out_of_range(client, standard_design_payload):
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    points = [
        {"size_m": 0.1, "passing_pct": -1.0},
        {"size_m": 0.2, "passing_pct": 40.0},
        {"size_m": 0.4, "passing_pct": 101.0},
    ]
    assert client.post("/measurements", json={
        "design_id": d["id"], "points": points}).status_code == 422


def test_reject_open_bin_wrong_boundary(client, standard_design_payload):
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    points = [
        {"size_m": 0.05, "passing_pct": 5.0, "open": True},  # 最细开档必须0%
        {"size_m": 0.2, "passing_pct": 40.0},
        {"size_m": 1.2, "passing_pct": 100.0, "open": True},
    ]
    assert client.post("/measurements", json={
        "design_id": d["id"], "points": points}).status_code == 422


def test_reject_duplicate_sizes(client, standard_design_payload):
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    points = [
        {"size_m": 0.2, "passing_pct": 10.0},
        {"size_m": 0.2, "passing_pct": 40.0},
        {"size_m": 0.8, "passing_pct": 90.0},
    ]
    assert client.post("/measurements", json={
        "design_id": d["id"], "points": points}).status_code == 422


def test_design_frozen_after_blasting(client, standard_design_payload):
    from tests.conftest import rr_curve, SIEVE_SIZES
    z = _make_zone(client)
    d = _make_design(client, standard_design_payload, z)
    client.post("/measurements", json={
        "design_id": d["id"],
        "points": rr_curve(SIEVE_SIZES, 0.30, 1.4),
    })
    r = client.patch(f"/designs/{d['id']}", json={"burden": 5.0})
    assert r.status_code == 409


def test_factor_as_of_and_history(client, standard_design_payload):
    from tests.conftest import rr_curve, SIEVE_SIZES
    z = _make_zone(client, factor=7.0)
    # 初始版本
    hist0 = client.get(f"/zones/{z}/factor-history").json()
    assert len(hist0) == 1 and hist0[0]["version_index"] == 0
    # 一次实测产生新版本
    d = _make_design(client, standard_design_payload, z)
    up = client.post("/measurements", json={
        "design_id": d["id"], "points": rr_curve(SIEVE_SIZES, 0.42, 1.4)}).json()
    v1_id = up["factor_version_id"]
    assert v1_id == 2
    # 按时刻查：初始版本时刻拿到初始系数，当前拿到新版本
    init_created = hist0[0]["created_at"]
    early = client.get(f"/zones/{z}/factor", params={"at": init_created})
    assert early.status_code == 200 and early.json()["factor"] == 7.0
    # 分区建立之前的时刻无版本可查
    before_all = client.get(
        f"/zones/{z}/factor", params={"at": "2000-01-01T00:00:00"}
    )
    assert before_all.status_code == 404
    now = client.get(f"/zones/{z}/factor").json()
    assert now["id"] == v1_id
    # 某次预测可回看当时绑定的系数版本
    d2 = _make_design(client, standard_design_payload, z)
    p = client.post(f"/designs/{d2['id']}/predict").json()
    assert p["factor_version_id"] == v1_id
