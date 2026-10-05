"""额外边界：坏曲线撤回后可对同一设计补录纠正数据；撤回不影响别的分区。"""

import numpy as np

from tests.conftest import SIEVE_SIZES, rr_curve


def _zone(client, decay=1.0, factor=7.0):
    r = client.post("/zones", json={
        "name": "区", "initial_factor": factor, "decay": decay})
    assert r.status_code == 201
    return r.json()["id"]


def _bimodal_points():
    LN2 = np.log(2.0)
    sizes = np.array([0.02, 0.04, 0.08, 0.16, 0.25, 0.4, 0.63, 1.0, 1.5])

    def rr(x, x50, n):
        return 1 - np.exp(-LN2 * (x / x50) ** n)

    p = 100 * (0.5 * rr(sizes, 0.12, 1.6) + 0.5 * rr(sizes, 0.75, 1.6))
    return [{"size_m": float(s), "passing_pct": float(v)}
            for s, v in zip(sizes, p)]


def test_bad_curve_withdraw_then_corrected_upload(client, standard_design_payload):
    z = _zone(client)
    d = client.post("/designs",
                    json={**standard_design_payload, "zone_id": z}).json()["id"]

    # 坏曲线：不标定
    bad = client.post("/measurements",
                      json={"design_id": d, "points": _bimodal_points()}).json()
    assert bad["fit"]["usable"] is False
    assert bad["factor_version_id"] is None

    # 撤回坏曲线（不引发任何系数版本变化）
    w = client.post(f"/measurements/{bad['id']}/withdraw")
    assert w.status_code == 200
    hist = client.get(f"/zones/{z}/factor-history").json()
    assert len(hist) == 1 and hist[0]["version_index"] == 0

    # 同一设计补录一条正确曲线 -> 正常标定出新版本
    from app.service import _geometry_factor, design_version_params
    from app import storage
    conn = storage.connect()
    params = design_version_params(conn, d, 1)
    x50_meas = 8.2 * _geometry_factor(params)
    conn.close()
    good = client.post("/measurements", json={
        "design_id": d,
        "points": rr_curve(SIEVE_SIZES, x50_meas, 1.4),
    })
    assert good.status_code == 201, good.text
    g = good.json()
    assert g["fit"]["usable"] is True
    assert g["rock_factor"] is not None
    assert client.get(f"/zones/{z}").json()["current_version_id"] == \
        g["factor_version_id"]

    # 同一设计不能再有第二条 active 实测
    dup = client.post("/measurements", json={
        "design_id": d,
        "points": rr_curve(SIEVE_SIZES, x50_meas, 1.4),
    })
    assert dup.status_code == 409


def test_withdraw_does_not_touch_other_zone(client, standard_design_payload):
    from app.service import _geometry_factor, design_version_params
    from app import storage

    z1 = _zone(client, factor=7.0)
    z2 = _zone(client, factor=6.0)
    conn = storage.connect()

    ids1, ids2 = [], []
    for k, (z, ids) in enumerate([(z1, ids1), (z2, ids2)]):
        for j in range(2):
            d = client.post("/designs", json={
                **standard_design_payload, "zone_id": z,
                "burden": 6.0 + 0.1 * j, "name": f"z{k}-{j}",
            }).json()["id"]
            params = design_version_params(conn, d, 1)
            m = client.post("/measurements", json={
                "design_id": d,
                "points": rr_curve(SIEVE_SIZES,
                                   (7.5 + 0.3 * j) * _geometry_factor(params), 1.4),
            }).json()
            ids.append(m["id"])

    before_z2 = client.get(f"/zones/{z2}").json()["current_factor"]
    client.post(f"/measurements/{ids1[0]}/withdraw")
    after_z2 = client.get(f"/zones/{z2}").json()["current_factor"]
    assert after_z2 == before_z2
    conn.close()


def test_withdraw_unknown_measurement_404(client):
    assert client.post("/measurements/999/withdraw").status_code == 404
