"""爆堆图像筛分数据的解析、校验与 Rosin-Rammler 拟合。

实测数据以"分档（bin）"形式给出：每档记录该档内的矿块质量占比
（retained_pct），两端可以是开区间——

    {"lower": null, "upper": 0.05, "retained_pct": 3}   # 小于 0.05 m
    {"lower": 1.2,  "upper": null, "retained_pct": 2}   # 大于 1.2 m

拟合目标为 Rosin-Rammler 累计通过率分布

    P(d) = 1 - exp( -(d/Xc)**n )

两边做两次对数变换即可线性化：

    y = ln(-ln(1-P)) = n * ln d - n * ln Xc
                  = m*x + b,   m=n, x=ln d

于是"拟合"就是对 (x, y) 做一次带权最小二乘（正规方程由本模块直接
推导计算，不调用 scipy / curve_fit 等现成拟合或优化库）。

加权与开区间的处理决定见各函数的 docstring；拟合在变换后的线性
坐标空间中进行，并把残差反映射回通过率空间报告。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# 拟合质量阈值（判定为"不可用"、不参与标定）
MIN_INTERIOR_POINTS = 3       # 参与线性拟合的内部点数下限
MIN_WEIGHTED_R2 = 0.97        # 变换空间加权决定系数下限
MAX_P_RESIDUAL_PP = 6.0       # 通过率空间最大残差（百分点）
# 残差结构判据（抓系统性弯曲/双峰，放过随机小噪声）：
# 符号游程数达到该阈值，且存在长度 >= SIGN_RUN_MIN_BLOCK 的同号连续段。
# 双峰的典型残差是 - ++ -- +（同号成段）；随机噪声是 +-+-+-（长段为1）。
MIN_SIGN_RUNS = 4
SIGN_RUN_MIN_BLOCK = 2
# 归一化容差：图像筛分各档占比之和允许的舍入误差
MASS_TOLERANCE_PCT = 1.0
# 开区间（边界不可见）点的权重折扣
OPEN_BIN_WEIGHT_FACTOR = 0.5


class SieveValidationError(ValueError):
    """筛分数据不合法。"""


@dataclass(frozen=True)
class SieveBin:
    """一个筛档：lower < d <= upper 之间的矿块质量占比（百分数）。"""

    retained_pct: float
    lower_m: float | None
    upper_m: float | None

    @property
    def is_open_fine(self) -> bool:
        return self.lower_m is None

    @property
    def is_open_coarse(self) -> bool:
        return self.upper_m is None


@dataclass(frozen=True)
class FittingPoint:
    """累计通过率曲线上的一个拟合点 (d, P)。"""

    sieve_m: float
    passing_pct: float          # 该筛孔的累计通过率（百分数）
    weight: float
    open_boundary: bool         # 是否来自开区间（只有一个边界可见）


@dataclass
class PointResidual:
    sieve_m: float
    observed_pct: float
    predicted_pct: float
    residual_pp: float         # 观测-预测（百分点）
    used_in_fit: bool


@dataclass
class FitResult:
    """拟合输出。usable=False 时不参与岩石系数标定。"""

    x50_m: float
    uniformity_n: float
    characteristic_m: float
    usable: bool
    reject_reason: str | None
    weighted_r2: float
    max_abs_residual_pp: float
    rmse_pp: float
    sign_runs: int
    sign_blocks: int            # 最长同号连续段长度
    n_interior_points: int
    residuals: list[PointResidual] = field(default_factory=list)


def parse_bins(
    raw: list[dict],
) -> list[SieveBin]:
    """校验并排序原始筛档。

    校验内容：
    * 至少两档；占比为非负有限数；
    * 每档至少给一个边界；闭区间须 lower < upper 且都为正；
    * 至多一个下端开（"小于 x"）和一个上端开（"大于 x"）；
    * 各档按尺寸顺序两两恰好相接（相邻 a.upper == b.lower），首档
      若不开则下端必须为 0；
    * 占比之和为 100%（容差 MASS_TOLERANCE_PCT，自动按比例归一）。
    """
    if not isinstance(raw, list) or len(raw) < 2:
        raise SieveValidationError("筛分数据至少需要两个筛档")

    bins: list[SieveBin] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SieveValidationError("每个筛档必须是对象")
        lower = item.get("lower_m", item.get("lower"))
        upper = item.get("upper_m", item.get("upper"))
        frac = item.get("retained_pct", item.get("pct"))
        if frac is None or not isinstance(frac, (int, float)) or not math.isfinite(frac):
            raise SieveValidationError("筛档占比必须是有限数值")
        if frac < 0:
            raise SieveValidationError(f"筛档占比不能为负（{frac}）")
        lower = _validate_boundary(lower, "下限")
        upper = _validate_boundary(upper, "上限")
        if lower is None and upper is None:
            raise SieveValidationError("筛档不能上下都开口")
        if lower is not None and upper is not None:
            if lower < 0 or upper <= 0:
                raise SieveValidationError("筛孔尺寸必须为正数")
            if lower >= upper:
                raise SieveValidationError(f"筛档下限({lower})必须小于上限({upper})")
        else:
            # 开区间唯一可见的边界必须严格为正
            edge = upper if lower is None else lower
            if edge <= 0:
                raise SieveValidationError("开区间边界筛孔尺寸必须为正数")
        bins.append(SieveBin(float(frac), lower, upper))

    # 按尺寸排序：用可见的最小边界代表该档
    def bin_key(b: SieveBin) -> float:
        return b.upper_m if b.upper_m is not None else b.lower_m  # type: ignore[return-value]

    bins.sort(key=bin_key)

    n_open_fine = sum(b.is_open_fine for b in bins)
    n_open_coarse = sum(b.is_open_coarse for b in bins)
    if n_open_fine > 1 or n_open_coarse > 1:
        raise SieveValidationError("同一端只能有一个开区间")
    if n_open_fine and bins[0].upper_m is None:
        raise SieveValidationError("下端开区间必须是最细的一档")
    if n_open_coarse and bins[-1].lower_m is None:
        raise SieveValidationError("上端开区间必须是最粗的一档")

    # 相邻档必须恰好相接；首档下端要么为 0 要么开口
    first = bins[0]
    if not first.is_open_fine and (first.lower_m != 0.0):
        raise SieveValidationError(
            f"最细档必须从 0 开始或开口，收到下限 {first.lower_m}"
        )
    for a, b in zip(bins, bins[1:]):
        if a.upper_m is None or b.lower_m is None:
            raise SieveValidationError("开区间只能出现在最细或最粗一档")
        if abs(a.upper_m - b.lower_m) > 1e-9:
            raise SieveValidationError(
                f"筛档必须连续相接：{a.upper_m} 与 {b.lower_m} 不衔接"
            )

    total = sum(b.retained_pct for b in bins)
    if total <= 0:
        raise SieveValidationError("筛档占比之和必须为正")
    if abs(total - 100.0) > MASS_TOLERANCE_PCT:
        raise SieveValidationError(
            f"各档累计占比之和必须为 100%（容差 {MASS_TOLERANCE_PCT} 个百分点），"
            f"实际为 {total:.3f}%"
        )
    if abs(total - 100.0) > 1e-9:
        scale = 100.0 / total
        bins = [SieveBin(b.retained_pct * scale, b.lower_m, b.upper_m) for b in bins]
    return bins


def _validate_boundary(v: object, label: str) -> float | None:
    if v is None:
        return None
    if not isinstance(v, (int, float)) or not math.isfinite(v):
        raise SieveValidationError(f"筛档{label}必须是有限数值或 null")
    return float(v)


def bins_to_points(bins: list[SieveBin]) -> list[FittingPoint]:
    """把质量分档转成累计通过率曲线上的点。

    累计通过率约定为"P(d)=尺寸不超过 d 的物料占比"。自最细档起累加：

    * 闭区间档 (a,b]：在其**上边界** b 给出一个点，P=截至该档的累计
      占比，权重=该档占比（质量越大的档对分布形状的信息量越大，
      在加权最小二乘中权重越高）；
    * 最细开区间档 (−inf,b]：其全部物料都 <= b，因此它在 b 处同样
      贡献一个累计点，但"小于 b"内部形状未知，权重打
      OPEN_BIN_WEIGHT_FACTOR 折；
    * 最粗开区间档 (a,+inf)：其全部物料都 > a，在 a 处不产生新点
      （P(a) 与上一档上边界相同），它仅说明尾部存在；这部分质量
      并入 a 处那个点，同样按折扣加权。

    另外筛掉 P≈0 与 P≈100 的点：双对数变换在两端发散，无法参与
    线性拟合；但它们的预测残差仍在通过率空间单独报告。
    """
    points: list[FittingPoint] = []
    cum = 0.0
    for i, b in enumerate(bins):
        if b.is_open_coarse:
            # 不产生新点；把开粗档的质量并入前一个边界点
            if not points:
                raise SieveValidationError("最粗开区间之前必须存在闭档")
            prev = points[-1]
            points[-1] = FittingPoint(
                prev.sieve_m,
                prev.passing_pct,
                prev.weight + b.retained_pct * OPEN_BIN_WEIGHT_FACTOR,
                True,
            )
            continue

        cum += b.retained_pct
        edge = b.upper_m  # 非粗开档必有上边界
        assert edge is not None
        if b.is_open_fine:
            w = b.retained_pct * OPEN_BIN_WEIGHT_FACTOR
            open_flag = True
        else:
            w = float(b.retained_pct)
            open_flag = False
        # 零占比档（如曲线终点 P=100 的空尾档）仍报告该边界点，
        # 但权重为 0，不进线性拟合
        if points and abs(points[-1].sieve_m - edge) < 1e-12:
            prev = points[-1]
            points[-1] = FittingPoint(
                edge,
                cum,
                prev.weight + w,
                prev.open_boundary or open_flag,
            )
        else:
            points.append(FittingPoint(edge, min(cum, 100.0), w, open_flag))
    return points


def _weighted_linear_fit(
    x: list[float], y: list[float], w: list[float]
) -> tuple[float, float]:
    """带截距的加权一元最小二乘，正规方程由 NumPy 直接构造求解。

    返回 (斜率 m, 截距 b)。不调用任何现成曲线拟合/优化库，
    只做数组运算。
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    wa = np.asarray(w, dtype=np.float64)
    sw = float(wa.sum())
    if sw <= 0:
        raise SieveValidationError("有效拟合点不足，无法求解最小二乘")
    swx = float(np.dot(wa, xa))
    swy = float(np.dot(wa, ya))
    swxx = float(np.dot(wa, xa * xa))
    swxy = float(np.dot(wa, xa * ya))
    det = sw * swxx - swx * swx
    if abs(det) < 1e-300:
        raise SieveValidationError("正规方程奇异，无法求解最小二乘")
    m = (sw * swxy - swx * swy) / det
    b = (swy - m * swx) / sw
    return m, b


def _rr_transform(passing_pct: float) -> float:
    """y = ln(-ln(1-P))，P 为百分数。"""
    p = passing_pct / 100.0
    if p <= 0.0 or p >= 1.0:
        raise ValueError("Rosin-Rammler 变换要求 0 < P < 1")
    return math.log(-math.log(1.0 - p))


def _rr_passing(x: float, m: float, b: float) -> float:
    """由变换空间直线 (m,b) 在筛孔 x（米）处反算通过率（百分数）。"""
    y = m * math.log(x) + b
    return (1.0 - math.exp(-math.exp(y))) * 100.0


def _weighted_r2(x: list[float], y: list[float], w: list[float], m: float, b: float) -> float:
    """变换空间的加权决定系数（NumPy 数组运算）。"""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    wa = np.asarray(w, dtype=np.float64)
    sw = float(wa.sum())
    if sw <= 0:
        return 0.0
    ybar = float(np.dot(wa, ya)) / sw
    ss_tot = float(np.dot(wa, (ya - ybar) ** 2))
    ss_res = float(np.dot(wa, (ya - (m * xa + b)) ** 2))
    if ss_tot <= 1e-300:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _sign_run_structure(vals: list[float]) -> tuple[int, int]:
    """统计残差符号结构，返回 (游程数, 最长同号连续段长度)。

    与真分布一致时残差符号应随机交替、最长同号段长度为 1；
    双峰/明显弯曲会出现 ++、-- 之类同号成段（曲线在某区间系统性
    偏离直线）。忽略零残差。
    """
    runs = 0
    longest = 0
    last = 0
    block = 0
    for v in vals:
        s = 1 if v > 1e-9 else -1 if v < -1e-9 else 0
        if s == 0:
            continue
        if s != last:
            runs += 1
            last = s
            block = 1
        else:
            block += 1
        longest = max(longest, block)
    return runs, longest


def fit_curve(raw_bins: list[dict]) -> FitResult:
    """从原始筛档拟合 X50 与均匀性指数 n，并做质量判定。

    拟合空间：双对数线性化空间 (ln d, ln(-ln(1-P)))。原因——
    Rosin-Rammler 在该空间是严格直线，加权最小二乘有闭式解，
    无须迭代或任何优化库；而且 n（斜率）正是题目要求的均匀性指数，
    Xc 由截距直接给出。

    权重：各档质量占比（信息量大的粗档权重大），开区间档打五折。

    质量判定（任一不满足则 usable=False，拒绝参与标定）：
      1) 有效内部点 >= MIN_INTERIOR_POINTS；
      2) 变换空间加权 R^2 >= MIN_WEIGHTED_R2；
      3) 通过率空间最大绝对残差 <= MAX_P_RESIDUAL_PP 个百分点；
      4) 残差符号出现"多游程且有同号连续段"（>=MIN_SIGN_RUNS 个
         游程且最长段 >=SIGN_RUN_MIN_BLOCK），抓系统性弯曲/双峰；
         纯交替（+-+）的小残差视为随机噪声放行。
    """
    bins = parse_bins(raw_bins)
    points = bins_to_points(bins)

    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    interior_flags: list[bool] = []
    for pt in points:
        if 1e-6 < pt.passing_pct < 100.0 - 1e-6 and pt.weight > 0:
            xs.append(math.log(pt.sieve_m))
            ys.append(_rr_transform(pt.passing_pct))
            ws.append(pt.weight)
            interior_flags.append(True)
        else:
            interior_flags.append(False)

    n_interior = len(xs)
    residuals: list[PointResidual] = []
    reject: str | None = None
    if n_interior < MIN_INTERIOR_POINTS:
        reject = f"有效内部拟合点仅 {n_interior} 个（<{MIN_INTERIOR_POINTS}）"
        m = b = n = xc = x50 = float("nan")
        r2 = float("nan")
        for pt in points:
            residuals.append(
                PointResidual(pt.sieve_m, pt.passing_pct, float("nan"),
                              float("nan"), False)
            )
        return FitResult(
            x50_m=x50, uniformity_n=n, characteristic_m=xc, usable=False,
            reject_reason=reject, weighted_r2=r2, max_abs_residual_pp=float("nan"),
            rmse_pp=float("nan"), sign_runs=0, sign_blocks=0,
            n_interior_points=n_interior, residuals=residuals,
        )

    m, b = _weighted_linear_fit(xs, ys, ws)
    if m <= 0:
        reject = f"拟合斜率（均匀性指数）非正：{m:.4g}"
    n = m
    xc = math.exp(-b / m) if m > 0 else float("nan")
    x50 = xc * math.log(2.0) ** (1.0 / n) if m > 0 else float("nan")
    r2 = _weighted_r2(xs, ys, ws, m, b)

    interior_resid_pp: list[float] = []
    max_abs = 0.0
    sq_sum = 0.0
    idx = 0
    for pt, is_interior in zip(points, interior_flags):
        pred = _rr_passing(pt.sieve_m, m, b)
        res = pt.passing_pct - pred
        residuals.append(
            PointResidual(pt.sieve_m, pt.passing_pct, pred, res, is_interior)
        )
        max_abs = max(max_abs, abs(res))
        sq_sum += res * res
        if is_interior:
            interior_resid_pp.append(res)
        idx += 1
    rmse = math.sqrt(sq_sum / len(points))
    runs, blocks = _sign_run_structure(interior_resid_pp)

    reasons: list[str] = []
    if reject:
        reasons.append(reject)
    if math.isfinite(r2) and r2 < MIN_WEIGHTED_R2:
        reasons.append(f"加权R²={r2:.4f}低于阈值{MIN_WEIGHTED_R2}（曲线明显偏离Rosin-Rammler，可能双峰）")
    if max_abs > MAX_P_RESIDUAL_PP:
        reasons.append(
            f"通过率最大残差{max_abs:.2f}个百分点超过{MAX_P_RESIDUAL_PP}"
        )
    structured = runs >= MIN_SIGN_RUNS and blocks >= SIGN_RUN_MIN_BLOCK
    if structured:
        reasons.append(
            f"残差呈系统性结构（游程{runs}、同号连续段{blocks}），"
            "曲线疑似双峰/多峰"
        )

    return FitResult(
        x50_m=x50,
        uniformity_n=n,
        characteristic_m=xc,
        usable=not reasons,
        reject_reason="；".join(reasons) if reasons else None,
        weighted_r2=r2,
        max_abs_residual_pp=max_abs,
        rmse_pp=rmse,
        sign_runs=runs,
        sign_blocks=blocks,
        n_interior_points=n_interior,
        residuals=residuals,
    )
