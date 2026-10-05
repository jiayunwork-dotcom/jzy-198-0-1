"""请求/响应模型与业务输入校验。

题目点名的拒收规则在这里集中表达（FastAPI/Pydantic 层返回 422）：
  * 孔径、孔距、排距、台阶高度、装药量非正；
  * 堵塞长度大于孔深；
  * 炸药相对威力不为正；
  * 筛分累计通过率不单调或越出 [0,100]；
引用不存在的分区/设计在接口层返回 404。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class ZoneCreate(BaseModel):
    name: str
    lithology: str = ""
    initial_factor: float = 7.0
    decay: float = Field(
        0.9, ge=0.0, le=1.0,
        description="旧数据指数衰减因子；1.0 表示不衰减（增量与全量严格一致）",
    )


class DesignParams(BaseModel):
    name: str
    zone_id: int
    diameter_mm: float
    burden: float            # 最小抵抗线/排距，m
    spacing: float           # 孔距，m
    bench_height: float      # 台阶高度，m
    hole_depth: float        # 孔深，m
    stemming: float          # 堵塞长度，m
    charge_kg: float         # 单孔装药量，kg
    explosive_rws: float = 1.0
    deviation: float = 0.0   # 钻孔偏差（亚钻），m
    oversize_m: float = 0.8  # 大块界定尺寸，m
    oversize_limit_pct: float = 10.0  # 大块率合格上限，%

    @model_validator(mode="after")
    def _check(self):
        positive_fields = {
            "diameter_mm": self.diameter_mm,
            "burden": self.burden,
            "spacing": self.spacing,
            "bench_height": self.bench_height,
            "hole_depth": self.hole_depth,
            "stemming": self.stemming,
            "charge_kg": self.charge_kg,
            "oversize_m": self.oversize_m,
        }
        for fname, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{fname} must be positive, got {value}")
        if self.stemming > self.hole_depth:
            raise ValueError(
                f"stemming ({self.stemming}) must not exceed hole_depth "
                f"({self.hole_depth})"
            )
        if self.explosive_rws <= 0:
            raise ValueError(
                f"explosive_rws must be positive, got {self.explosive_rws}"
            )
        if self.deviation < 0:
            raise ValueError("deviation must be non-negative")
        if self.deviation >= min(self.burden, self.bench_height):
            raise ValueError(
                "deviation must be smaller than both burden and bench_height"
            )
        if not (0 < self.oversize_limit_pct < 100):
            raise ValueError("oversize_limit_pct must lie in (0, 100)")
        # Cunningham 均匀性指数的几何一致性
        if 2.2 - 14.0 * self.burden / self.diameter_mm <= 0:
            raise ValueError(
                "burden/diameter ratio too large for a positive uniformity index "
                "(14*B/D must stay below 2.2)"
            )
        if self.hole_depth - self.stemming <= 0:
            raise ValueError("charge length (hole_depth - stemming) must be positive")
        return self


class DesignUpdate(BaseModel):
    name: str | None = None
    diameter_mm: float | None = None
    burden: float | None = None
    spacing: float | None = None
    bench_height: float | None = None
    hole_depth: float | None = None
    stemming: float | None = None
    charge_kg: float | None = None
    explosive_rws: float | None = None
    deviation: float | None = None
    oversize_m: float | None = None
    oversize_limit_pct: float | None = None


class SievePoint(BaseModel):
    size_m: float
    passing_pct: float
    open: bool = False  # True=开区间边界档（“大于x”/“小于x”）

    @field_validator("size_m")
    @classmethod
    def _size_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("sieve size must be positive")
        return v


class MeasurementUpload(BaseModel):
    design_id: int
    points: list[SievePoint]

    @model_validator(mode="after")
    def _check_curve(self):
        if len(self.points) < 3:
            raise ValueError("at least 3 sieve points are required")
        sizes = [p.size_m for p in self.points]
        if any(sizes[i + 1] <= sizes[i] for i in range(len(sizes) - 1)):
            raise ValueError("sieve sizes must be strictly increasing and unique")
        prev = -0.0
        for p in self.points:
            if not (0.0 <= p.passing_pct <= 100.0):
                raise ValueError(
                    f"passing_pct {p.passing_pct} out of range [0, 100]"
                )
            if p.passing_pct < prev - 1e-9:
                raise ValueError("cumulative passing must be non-decreasing")
            prev = p.passing_pct
        open_flags = [p.open for p in self.points]
        if open_flags[0]:
            if self.points[0].passing_pct != 0.0:
                raise ValueError(
                    "finest open bin ('smaller than ...') must have passing 0%"
                )
        if open_flags[-1]:
            if self.points[-1].passing_pct != 100.0:
                raise ValueError(
                    "coarsest open bin ('larger than ...') must have passing 100%"
                )
        return self


# ---------- 响应模型 ----------
class ZoneOut(BaseModel):
    id: int
    name: str
    lithology: str
    decay: float
    current_factor: float
    current_version_id: int


class DesignOut(BaseModel):
    id: int
    version: int
    name: str
    zone_id: int
    status: str
    params: dict
    version_created_at: str


class PredictionOut(BaseModel):
    id: int
    design_id: int
    design_version: int
    factor_version_id: int
    rock_factor: float
    x50: float
    uniformity: float
    powder_factor: float
    oversize_pct: float
    oversize_pass: bool
    passing_table: dict[str, float]


class FitOut(BaseModel):
    x50: float | None
    uniformity: float | None
    usable: bool
    r2: float | None
    rmse_y: float | None
    max_p_residual_pct: float | None
    residuals_pct: list[float]
    included: list[bool]
    bimodal: bool
    reasons: list[str]


class MeasurementOut(BaseModel):
    id: int
    design_id: int
    zone_id: int
    status: str
    fit: FitOut
    factor_version_id: int | None
    rock_factor: float | None
    created_at: str
    withdrawn_at: str | None


class FactorVersionOut(BaseModel):
    id: int
    zone_id: int
    version_index: int
    factor: float
    evidence_id: int | None
    created_at: str
    active: bool
    superseded_by: int | None
    chain: str
    note: str


class RepredictRow(BaseModel):
    design_id: int
    name: str
    old_factor_version_id: int
    new_factor_version_id: int
    old_oversize_pct: float
    new_oversize_pct: float
    old_pass: bool
    new_pass: bool
    flipped: bool


class RepredictReport(BaseModel):
    zone_id: int | None
    re_predicted: int
    flips: list[RepredictRow]
