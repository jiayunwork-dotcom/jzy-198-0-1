"""FastAPI 接口层：只暴露 HTTP 接口。

路由：
  POST   /zones                         建岩石分区
  GET    /zones                         分区列表
  GET    /zones/{id}                    分区与当前系数
  GET    /zones/{id}/factor-history     按分区的系数历史（含撤回链）
  GET    /zones/{id}/factor             当前/任意时刻系数（?at=ISO）

  POST   /designs                       建爆破设计
  GET    /designs                       设计列表（?status=planned/blasted）
  GET    /designs/{id}                  设计详情
  PATCH  /designs/{id}                  修改（生成新版本；已实施的冻结）

  POST   /designs/{id}/predict          预测（绑定设计版本与系数版本）
  GET    /designs/{id}/predictions      该设计全部预测（最新在前）
  GET    /predictions/{id}              某次预测（含当时绑定的系数版本）
  POST   /repredict                     批量重新预测（?zone_id=，缺省全矿）

  POST   /measurements                  实测上传（拟合+标定+出系数新版本）
  GET    /measurements                  实测列表
  GET    /measurements/{id}             实测与拟合结果
  POST   /measurements/{id}/withdraw    撤回（系数链重推）
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import service, storage
from .schemas import (
    DesignParams,
    DesignUpdate,
    MeasurementUpload,
    ZoneCreate,
)


def get_conn():
    return storage.connect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = storage.connect()
    storage.init_db(conn)
    conn.close()
    yield


app = FastAPI(title="Blasting Fragmentation Closed-Loop Service", version="1.0")


@app.exception_handler(service.NotFound)
async def not_found_handler(request, exc: service.NotFound):
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(service.Conflict)
async def conflict_handler(request, exc: service.Conflict):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


# ---------------- 分区与系数 ----------------
@app.post("/zones", status_code=201)
def create_zone(payload: ZoneCreate):
    conn = get_conn()
    try:
        return service.create_zone(conn, payload)
    finally:
        conn.close()


@app.get("/zones")
def list_zones():
    conn = get_conn()
    try:
        return service.list_zones(conn)
    finally:
        conn.close()


@app.get("/zones/{zone_id}")
def get_zone(zone_id: int):
    conn = get_conn()
    try:
        return service.get_zone(conn, zone_id)
    finally:
        conn.close()


@app.get("/zones/{zone_id}/factor")
def get_factor(zone_id: int, at: str | None = None, version_id: int | None = None):
    """当前系数；?at=2026-.. 查询任意时刻；?version_id= 精确取某版（可含撤回版）。"""
    conn = get_conn()
    try:
        if version_id is not None:
            row = service.factor_version(conn, version_id)
            if row["zone_id"] != zone_id:
                raise HTTPException(404, "version does not belong to this zone")
        elif at is not None:
            row = service.factor_as_of(conn, zone_id, at)
        else:
            row = service.current_factor_version(conn, zone_id)
        return _fv(row)
    finally:
        conn.close()


@app.get("/zones/{zone_id}/factor-history")
def factor_history(zone_id: int, active_only: bool = False):
    conn = get_conn()
    try:
        rows = service.factor_history(conn, zone_id,
                                      include_superseded=not active_only)
        return [_fv(r) for r in rows]
    finally:
        conn.close()


def _fv(row) -> dict:
    return {
        "id": row["id"],
        "zone_id": row["zone_id"],
        "version_index": row["version_index"],
        "factor": row["factor"],
        "evidence_id": row["evidence_id"],
        "decay": row["decay"],
        "created_at": row["created_at"],
        "active": bool(row["active"]),
        "chain": row["chain"],
        "superseded_by": row["superseded_by"],
        "note": row["note"],
    }


# ---------------- 设计 ----------------
@app.post("/designs", status_code=201)
def create_design(payload: DesignParams):
    conn = get_conn()
    try:
        return service.create_design(conn, payload)
    finally:
        conn.close()


@app.get("/designs")
def list_designs(status: str | None = Query(None, pattern="^(planned|blasted)$")):
    conn = get_conn()
    try:
        return service.list_designs(conn, status)
    finally:
        conn.close()


@app.get("/designs/{design_id}")
def get_design(design_id: int):
    conn = get_conn()
    try:
        return service.get_design(conn, design_id)
    finally:
        conn.close()


@app.patch("/designs/{design_id}")
def update_design(design_id: int, patch: DesignUpdate):
    conn = get_conn()
    try:
        return service.update_design(conn, design_id, patch)
    finally:
        conn.close()


# ---------------- 预测 ----------------
@app.post("/designs/{design_id}/predict", status_code=201)
def predict(
    design_id: int,
    factor_version_id: int | None = None,
    extra_sieves: list[float] | None = Query(default=None),
):
    conn = get_conn()
    try:
        return service.predict_design(
            conn, design_id, factor_version_id, extra_sieves
        )
    finally:
        conn.close()


@app.get("/designs/{design_id}/predictions")
def design_predictions(design_id: int):
    conn = get_conn()
    try:
        service.get_design(conn, design_id)
        rows = storage.query(
            conn,
            "SELECT * FROM predictions WHERE design_id=? ORDER BY id DESC",
            (design_id,),
        )
        return [service._prediction_row(r) for r in rows]
    finally:
        conn.close()


@app.get("/predictions/{prediction_id}")
def get_prediction(prediction_id: int):
    conn = get_conn()
    try:
        return service.get_prediction(conn, prediction_id)
    finally:
        conn.close()


@app.post("/repredict")
def repredict(zone_id: int | None = None):
    conn = get_conn()
    try:
        return service.batch_repredict(conn, zone_id)
    finally:
        conn.close()


# ---------------- 实测 ----------------
@app.post("/measurements", status_code=201)
def upload_measurement(payload: MeasurementUpload):
    conn = get_conn()
    try:
        return service.upload_measurement(conn, payload)
    finally:
        conn.close()


@app.get("/measurements")
def list_measurements(zone_id: int | None = None):
    conn = get_conn()
    try:
        return service.list_measurements(conn, zone_id)
    finally:
        conn.close()


@app.get("/measurements/{measurement_id}")
def get_measurement(measurement_id: int):
    conn = get_conn()
    try:
        return service.get_measurement(conn, measurement_id)
    finally:
        conn.close()


@app.post("/measurements/{measurement_id}/withdraw")
def withdraw_measurement(measurement_id: int):
    conn = get_conn()
    try:
        return service.withdraw_measurement(conn, measurement_id)
    finally:
        conn.close()


@app.get("/health")
def health():
    return {"status": "ok"}
