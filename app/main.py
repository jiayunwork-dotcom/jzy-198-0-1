"""FastAPI 入口：只暴露 HTTP 接口。

服务进程本身无状态，状态全部在 SQLite（默认 /data/blast.db，挂载卷）。
错误约定：
  400 — 业务参数不合法（含题目列出的各类拒收）；
  404 — 引用不存在的分区/设计/爆破/实测/版本；
  409 — 状态冲突（重复实施、重复上传等）。
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import schemas as sv
from . import services
from .fragmentation import ValidationError
from .sieving import SieveValidationError
from .storage import DEFAULT_DB_PATH, NotFound, Storage, loads, utcnow_iso

app = FastAPI(title="露天矿爆破块度闭环管理服务", version="1.0.0")

_store: Storage | None = None


def get_store() -> Storage:
    global _store
    if _store is None:
        _store = Storage(DEFAULT_DB_PATH)
    return _store


def init_with(db_path: str) -> Storage:
    """测试用：显式指定数据库文件。"""
    global _store
    if _store is not None:
        _store.close()
    _store = Storage(db_path)
    return _store


@app.exception_handler(ValidationError)
@app.exception_handler(SieveValidationError)
async def _validation_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.exception_handler(NotFound)
async def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(services.ConflictError)
async def _conflict_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=409, content={"error": str(exc)})


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "time": utcnow_iso()}


# ---------------------------------------------------------------- 分区

@app.post("/zones", status_code=201)
def create_zone(body: sv.ZoneCreate) -> dict:
    return services.create_zone(
        get_store(), body.name, body.baseline_factor,
        body.forgetting, body.baseline_weight, body.lithology,
    )


@app.get("/zones")
def list_zones() -> list[dict]:
    return services.list_zones(get_store())


@app.get("/zones/{zone_id}")
def get_zone(zone_id: int) -> dict:
    return services.get_zone(get_store(), zone_id)


@app.get("/zones/{zone_id}/factor")
def current_factor(zone_id: int) -> dict:
    return services.current_factor(get_store(), zone_id)


@app.get("/zones/{zone_id}/factor/history")
def factor_history(zone_id: int, include_superseded: bool = False) -> list[dict]:
    return services.factor_history(get_store(), zone_id, include_superseded)


@app.get("/zones/{zone_id}/factor/as-of")
def factor_as_of(zone_id: int, at: str) -> dict:
    return services.factor_as_of(get_store(), zone_id, at)


@app.post("/zones/{zone_id}/factor/rebuild")
def rebuild_factor(zone_id: int) -> dict:
    return services.rebuild_zone_factor(get_store(), zone_id)


# ---------------------------------------------------------------- 设计 / 预测

@app.post("/designs", status_code=201)
def create_design(body: sv.DesignCreate) -> dict:
    return services.create_design(
        get_store(), body.name, body.zone_id, body.params.model_dump()
    )


@app.patch("/designs/{design_id}")
def update_design(design_id: int, body: sv.DesignUpdate) -> dict:
    return services.update_design(get_store(), design_id, body.params.model_dump())


@app.get("/designs")
def list_designs() -> list[dict]:
    return services.list_designs(get_store())


@app.get("/designs/{design_id}")
def get_design(design_id: int) -> dict:
    return services.get_design(get_store(), design_id)


@app.post("/designs/{design_id}/predict")
def predict(design_id: int, body: sv.PredictRequest | None = None) -> dict:
    version = body.design_version if body else None
    return services.predict(get_store(), design_id, version)


@app.get("/designs/{design_id}/predictions")
def list_predictions(design_id: int) -> list[dict]:
    store = get_store()
    store.get_design(design_id)
    out = []
    for r in store.list_predictions(design_id):
        out.append({
            "id": r["id"],
            "design_version": r["design_version"],
            "factor_version_id": r["factor_version_id"],
            "is_current": bool(r["is_current"]),
            "created_at": r["created_at"],
            "result": loads(r["result_json"]),
        })
    return out


@app.post("/designs/repredict")
def bulk_repredict() -> dict:
    return services.bulk_repredict(get_store())


@app.get("/passing")
def passing(
    sieve_m: float,
    prediction_id: int | None = None,
    x50_m: float | None = None,
    uniformity_n: float | None = None,
) -> dict:
    """任意筛孔通过率：引用某预测，或直接给 X50、n。"""
    return services.passing_for_prediction(
        get_store(), prediction_id, sieve_m, x50_m, uniformity_n
    )


# ---------------------------------------------------------------- 爆破

@app.post("/blasts", status_code=201)
def fire_blast(body: sv.BlastCreate) -> dict:
    return services.fire_blast(
        get_store(), body.design_id, body.fired_at, body.design_version
    )


@app.get("/blasts")
def list_blasts() -> list[dict]:
    return services.list_blasts(get_store())


@app.get("/blasts/{blast_id}")
def get_blast(blast_id: int) -> dict:
    return services.get_blast(get_store(), blast_id)


# ---------------------------------------------------------------- 实测

@app.post("/measurements", status_code=201)
def upload_measurement(body: sv.MeasurementUpload) -> dict:
    bins = [b.model_dump(by_alias=True) for b in body.bins]
    # by_alias 会带 None 键，剔除以便 parse_bins 识别开区间
    bins = [
        {k: v for k, v in b.items() if v is not None}
        for b in bins
    ]
    return services.upload_measurement(get_store(), body.blast_id, bins)


@app.get("/measurements/{measurement_id}")
def get_measurement(measurement_id: int) -> dict:
    return services.get_measurement(get_store(), measurement_id)


@app.post("/measurements/{measurement_id}/withdraw")
def withdraw_measurement(measurement_id: int, body: sv.WithdrawRequest | None = None) -> dict:
    return services.withdraw_measurement(get_store(), measurement_id)
