# 露天矿爆破块度闭环服务

把「爆破设计 → 块度预测 → 爆后图像筛分实测 → 岩石系数标定」串成闭环的后端服务。
Python 3.12 + FastAPI + NumPy + SQLite；块度模型、筛分拟合、系数标定均为自写
闭式算法，不依赖任何曲线拟合/优化库。

## 模型与方法

### 1. 块度预测（Kuz-Ram，`app/fragmentation.py`）

* 中值块度（米）

      X50 = 0.01 · A · q^(-0.8) · Q^(1/6) · (1.15/RWS)^(19/30)

  A 岩石系数（分区标定）；q = Q/(B·S·H) 单耗；Q 单孔药量；RWS 炸药相对威力。
  因此：**单耗越高 X50 越小**，且 **X50 与 A 严格成正比**（单次反推
  A_new = A_used · X_meas / X_pred，如 7、0.40、0.48 → 8.4）。
* Rosin-Rammler 通过率（由 X50 与均匀性指数 n 参数化）

      P(x) = 100 · (1 − exp(−ln2 · (x/X50)^n))

  参考：X50=0.30 m、n=1.5 时 0.60 m 通过率 ≈ **85.9%**。
* 均匀性指数 n 用 Cunningham 1987 式（孔径、抵抗线、台阶高度、装药长度、钻孔偏差）。
* 大块率 = 100 − P(大块界定尺寸)，与设计里的合格上限比较。

### 2. 筛分曲线拟合（`app/fitting.py`，自写闭式加权线性回归）

* 在线性化空间 `y=ln(−ln(1−P))` 对 `z=ln(x)` 做一元加权最小二乘，斜率即 n，
  截距还回 X50；闭式公式，无迭代、无优化库。
* 权重用 delta 方法 `w=((1−P)·ln(1−P))²`：图像筛分百分率噪声在 0/100% 附近
  被线性化急剧放大，该权重自动压低两端、突出中段。
* 开区间档（"大于1.2m""小于0.05m"）累计值固定为 100%/0%，线性化后为 ±∞，
  不进入回归，仅作范围校验。
* 质量输出：加权 R²、线性化空间 RMSE、通过率空间逐档残差与最大偏差；
  另在「按对数间距归一的分档质量密度」上检测双峰（两峰夹明显谷）。
  R² < 0.985、最大残差 > 5 个百分点、n 不物理或双峰时 `usable=false`，
  **不参与标定**（原始曲线与拟合结果仍入库可查）。

### 3. 系数标定（`app/calibration.py` + `app/service.py`）

每次可用实测反推单次系数 `a_k = X50_k / g_k`（g_k 是该次设计的几何/炸药因子），
分区系数为带权平均 `A = Σw_k a_k / Σw_k`，权重 = 拟合质量(R²) × 时间衰减。

* **采用：增量更新 + 指数遗忘（可配），撤回时全量重推的组合方案。**
  * 正常入库只维护两个累加矩 S0、S1（旧数据 `decay^age` 降权，默认 decay=0.9，
    可在分区上设为 1.0 关闭遗忘，以跟踪开采推进中岩体变化），代价 O(1)，
    不需要每次扫全历史；
  * 撤回是罕见操作，此时对剩余观测做一次全量折叠（O(N)），保证口径权威。
  * decay=1（不衰减）时增量与全量在同一顺序下**数学完全相同**，测试按
    相对差 < 1e-9 断言（见 `test_calibration.py`、`test_closed_loop.py`）。
  * 全量重拟合每次 O(N) 且要重算所有版本；纯增量代价 O(1) 但无法直接撤回。
    本组合取两者长处：在线 O(1)、撤回 O(N) 且语义清晰。

每次更新生成一个**系数版本**（不可变），可查任意时刻生效版本
（`GET /zones/{id}/factor?at=...`）、某次预测绑定的版本，以及含撤回链的
完整历史（`active`/`chain`/`superseded_by`）。

### 4. 版本化、撤回、并发、批量重预测

* 设计是具名对象、引用分区；修改产生新设计版本，已实施（有实测）的设计冻结。
  每条预测同时绑定**设计版本**与**系数版本**。
* 实测撤回：该实测置 withdrawn，它引发的系数版本及其后版本整体进入
  `withdrawn` 链；保留的实测按入库顺序前缀全量折叠重推新版本并回填指针；
  撤回点之前的版本原值保留。撤回前后历史都可查询。
* 并发：同分区实测入库按分区互斥锁串行，SQLite 用 `BEGIN IMMEDIATE`
  抢占写锁；提交次序即 `measurements.id` 次序，最终系数与「按入库顺序
  依次处理」逐位一致（8 线程并发测试覆盖 decay=1 与 decay<1）。
* `POST /repredict` 对所有未实施设计用当前系数重新预测，报告大块合格性
  翻转清单（合格↔超标）。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/zones` | 建分区（初始系数、decay） |
| GET | `/zones`，`/zones/{id}` | 分区与当前系数 |
| GET | `/zones/{id}/factor?at=&version_id=` | 当前/任意时刻/指定版本系数 |
| GET | `/zones/{id}/factor-history` | 版本历史（`active_only` 可筛） |
| POST | `/designs` | 建设计（引用分区） |
| GET/PATCH | `/designs[/{id}]` | 列表/详情/修改（新版本，已实施冻结） |
| POST | `/designs/{id}/predict?extra_sieves=` | 预测（绑定两版本，返回通过率表） |
| GET | `/designs/{id}/predictions`，`/predictions/{id}` | 预测历史 |
| POST | `/repredict?zone_id=` | 批量重预测与翻转清单 |
| POST | `/measurements` | 实测上传（校验→拟合→标定→新版本） |
| GET | `/measurements[/{id}]` | 实测与拟合结果 |
| POST | `/measurements/{id}/withdraw` | 撤回并重推系数链 |
| GET | `/health` | 健康检查 |

拒收（422/404/409）：孔径、孔距、排距、台阶高度、装药量非正；堵塞 > 孔深；
RWS 非正；筛分通过率不单调或越出 [0,100]；筛孔不递增/重复；开区间边界值不是
0%/100%；引用不存在分区（404）或设计（404）；重复实测、修改已实施设计、
重复撤回（409）。

## 运行

```bash
docker build -t blast-frag .
docker run -p 8080:8080 -v /mnt/blastdata:/data blast-frag
# SQLite 文件位于挂载卷：/data/blast.db（可用 BLAST_DB 覆盖）
```

本地：

```bash
pip install -r requirements.txt
BLAST_DB=/data/blast.db uvicorn app.main:app --host 0.0.0.0 --port 8080
pytest -q
```

## 模块

`fragmentation.py` Kuz-Ram 与 RR；`fitting.py` 筛分拟合与质量/双峰判定；
`calibration.py` 加权矩的增量/全量折叠；`service.py` 业务编排与事务/并发；
`storage.py` SQLite；`schemas.py` 校验与模型；`main.py` HTTP 接口。
测试在 `tests/`，54 个用例覆盖题目要求的全部测试点。
