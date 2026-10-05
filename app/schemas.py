"""HTTP 请求模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ZoneCreate(BaseModel):
    name: str
    baseline_factor: float = 10.0
    baseline_weight: float = 0.0
    forgetting: float = 0.90
    lithology: str = ""


class DesignParamsModel(BaseModel):
    hole_diameter_mm: float
    burden_m: float
    spacing_m: float
    bench_height_m: float
    hole_depth_m: float
    stemming_m: float
    explosive_mass_kg: float
    explosive_rws: float = 100.0
    hole_deviation_m: float = 0.0
    oversize_sieve_m: float = 0.60
    oversize_limit_pct: float = 5.0


class DesignCreate(BaseModel):
    name: str
    zone_id: int
    params: DesignParamsModel


class DesignUpdate(BaseModel):
    params: DesignParamsModel


class PredictRequest(BaseModel):
    design_version: int | None = None


class BlastCreate(BaseModel):
    design_id: int
    design_version: int | None = None
    fired_at: str | None = None


class SieveBinModel(BaseModel):
    # 开区间："小于 0.05" -> lower_m=null；"大于 1.2" -> upper_m=null
    lower_m: float | None = Field(default=None, alias="lower")
    upper_m: float | None = Field(default=None, alias="upper")
    retained_pct: float

    model_config = {"populate_by_name": True}


class MeasurementUpload(BaseModel):
    blast_id: int
    bins: list[SieveBinModel]


class WithdrawRequest(BaseModel):
    reason: str | None = None
