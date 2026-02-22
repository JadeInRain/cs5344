# CS5344 项目交接指南（Handover Guide）

本文档面向接手本框架的小组成员，说明已完成模块的设计理念、数据协议、特征逻辑、Pipeline 使用方式及后续开发接口。需要关注的文件包括 `adapter.py` `features.py` `model_pipeline.py` `main.py`其中后面两个py文件是你们需要在其基础上继续完善的。

---

## 0. 框架设计思路，说明后面为什么那样做

### 0.1 管道开发逻辑

**核心原因**：SCANIA 是车辆传感器数据，Backblaze 是硬盘 SMART 数据，虽然内容不同，但结构都是“资产ID + 时间戳 + 多维特征”。

所以设计一个适配层adapter，无论输入哪个数据集，都会被转化为统一的以下格式：

- `Asset_ID`: 唯一标识符（车辆 ID 或硬盘序列号）。
- `Time`:时序轴（运行步数或日期坐标）。
- `Features`: 标准化后的特征向量（读数、直方图或滚动统计量）。

### 0.2 数据预处理与标准化逻辑

**缺失值填补**：利用纵向数据的连续性。优先采用“前向填充”（Forward Fill）。例如硬盘今天没测数据，我们就假设它的健康状态和昨天一样。只有第一条数据也缺失时，才用全局均值。

**鲁棒标准化**：不使用普通的 StandardSclaer，而是使用 **RobustScaler**。原因：SCANIA 的数据中存在大量极端离群值。RobustScaler 使用中位数和四分位数进行缩放，能防止这些极端值把正常数据“挤压”到极小空间，保留设备异常时的关键脉冲信号。

### 0.3 纵向特征工程逻辑

**核心挑战**：单点的数据很难看出寿命。比如传感器读数 80，如果它是从 20 升上来的表示正在坏，如果它一直 80 则表示稳定 。

- **滑动窗口统计 (Rolling Window)**：针对每个 ID，计算过去一段时间（比如最近 5 步）的**均值**、**标准差**和**变化率（斜率diff）**。斜率增加通常预示着磨损加剧。
- **直方图特征压缩 (Histogram Dynamics)**：  
SCANIA 提供的是直方图桶（如变量 167 有 10 个桶）。
**方法**：计算直方图的“重心”变化。如果重心向高数值桶移动，说明设备负载或损耗在增加。这比直接把 10 个桶当成独立特征更具物理意义。
- **寿命标签生成 (RUL Discretization)**：
  - **逻辑**：根据故障点反推。对于已损坏设备，计算 $RUL=T_{failure}-t$；对于没坏的设备（右删失），统一标记为安全（标签 0）。然后根据 PPT 给出的阈值表进行离散化分桶

### 0.4 管道解耦和工厂模式

- **工厂函数设计 (**`create_generic_pipeline`**)**：
  - 将预处理（Imputer + Scaler）封装为一个“标准化生产线”。它接收你们提供的任何 `estimator`（如随机森林、XGBoost）。
- **接口**：
  - 对于具体的“代价矩阵”和“损失评估”，留待你们实现。在 `model_pipeline.py` 中。

## 1. 项目概览

### 1.1 核心理念：一套代码跑通两个数据集

本框架的核心目标是**用同一套代码支持 SCANIA（卡车）与 Backblaze（硬盘）两个纵向 RUL 数据集**，避免为每个数据集维护独立脚本。实现方式包括：

- **统一标识符**：将 SCANIA 的 `vehicle_id` 与 Backblaze 的 `serial_number` 统一映射为 `Asset_ID，`将 SCANIA 的 `time_step` 与 Backblaze 的 `date` 映射为统一的 `TIME` 列。下游逻辑只依赖统一列名。 
- **统一特征接口**：`features` 模块按 `Asset_ID` 分组计算滚动统计与直方图重心，不依赖具体数据集列名。
- **统一管道接口**：`model_pipeline` 通过 `dataset_name`（`"scania"` / `"backblaze"`）切换预处理配置，同一工厂函数 `create_generic_pipeline(estimator, dataset_name, ...)` 可接入任意估计器。

因此，**切换数据集时只需修改 `main.py` 顶部的 `DATASET_NAME`**，数据加载、特征工程与管道构建会自动跳转到对应逻辑。

### 1.2 模块详细说明


| 模块                  | 职责                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `adapter.py`        | 本模块实现数据适配 - **标识符统一**：将 SCANIA 的 `vehicle_id` 与 Backblaze 的 `serial_number` 统一映射为 `Asset_ID`。- **时间轴对齐**：将 SCANIA 的 `time_step` 与 Backblaze 的 `date` 映射为统一的 `TIME` 列。- **主要接口**： - `UniversalAdapter.load_data()`：负责自动解析路径并执行初步重命名。 - `to_canonical()` / `from_canonical()`：实现原始数据与标准协议之间的双向转换。数据加载（`UniversalAdapter.load_data`）、列名统一（`to_canonical` / `from_canonical`）、每资产最后一行（`last_readout`） |
| `features.py`       | 本模块是特征工程，解决了**变长序列 (Variable-length sequences)** 的特征提取问题，重点是从时间维度捕捉损耗趋势。- **纵向滚动统计 (Rolling Stats)**： - 针对每个 `Asset_ID` 独立分组计算。 - 提供 `mean`（近期均值）、`std`（信号波动）和 `diff`（瞬时变化率/斜率）特征。- **直方图重心 (Histogram COM)**： - 专为 SCANIA 设计，将变量 167 的 10 个直方图桶压缩为 1 个物理意义明确的"重心"特征。- **⚠️ 关键逻辑约束**： - **计算顺序**：必须在调用 `last_readout`（取快照）**之前**提取特征。 - **原因**：一旦执行快照，每个资产仅剩一行数据，将导致无法计算任何时间序列趋势                     |
| `model_pipeline.py` | 预处理封装（SimpleImputer + RobustScaler + OneHotEncoder）、工厂函数 `create_generic_pipeline`、代价评估接口（待实现）                                                                                                                                                                                                                                                                                                       |
| `main.py`           | 端到端演示：数据缝合 → 特征工程 → 取快照 → 拆 X/y → 建管道 → fit/predict                                                                                                                                                                                                                                                                                                                                                  |


---

## 2. 数据协议

### 2.1 统一命名规范

下游代码**只应依赖**以下两个统一列名，不要再使用各数据集原始列名：


| 统一列名         | 常量                 | 含义            | SCANIA 原始列   | Backblaze 原始列   |
| ------------ | ------------------ | ------------- | ------------ | --------------- |
| **Asset_ID** | `adapter.ASSET_ID` | 资产/设备唯一标识     | `vehicle_id` | `serial_number` |
| **time**     | `adapter.TIME`     | 时间维度（读表时间或步长） | `time_step`  | `date`          |


- 数据加载后通过 `to_canonical(df, dataset)` 将原始列重命名为 `Asset_ID` 与 `TIME`。
- 需要写回或提交时，通过 `from_canonical(df, dataset)` 还原为 `vehicle_id`/`time_step` 或 `serial_number`/`date`。

### 2.2 使用示例

```python
from adapter import UniversalAdapter, to_canonical, from_canonical, ASSET_ID, TIME

adapter = UniversalAdapter()
df = adapter.load_data("scania")   # 已有 Asset_ID，仍有 time_step
df = to_canonical(df, "scania")    # time_step -> time
# 此后一律使用 df[ASSET_ID], df[TIME]

# 提交 / 写回时还原
df_raw = from_canonical(df, "scania")  # vehicle_id, time_step
```

---

## 3. 特征逻辑

### 3.1 滚动窗口（Rolling Window）

- **实现位置**：`features.add_rolling_stats(df, columns, window=5, time_col=TIME)`  
- **原理**：  
  - 按 **Asset_ID** 分组，组内按 **time** 排序。  
  - 对 `columns` 中每一列在组内做：  
    - **滚动均值**：`rolling(window, min_periods=1).mean()` → 列名 `{col}_rolling_mean`  
    - **滚动标准差**：`rolling(window, min_periods=1).std()` → `{col}_rolling_std`  
    - **一阶差分**：当前值减前值 → `{col}_diff`（表趋势）
  - 使用 `groupby(ASSET_ID)[col].transform(...)`，**严格在组内计算，不产生跨资产信息**，从而支持变长序列（每个资产观测数不同）。

### 3.2 直方图重心（Histogram COM，仅 SCANIA）

- **实现位置**：`features.add_histogram_center(df)`  
- **原理**：对列 `167_0` … `167_9`，将桶索引 0…9 视为位置、列值为权重，计算加权重心：  
**COM = Σ(i × v_i) / Σ(v_i)**，空和时为 NaN。  
- **输出列**：`167_hist_com`（单列标量特征）。

### 3.3 重要：先特征工程，后取快照！！！

**务必保持以下顺序，否则滚动特征会因“每资产仅一行”而全部为 NaN：**

1. 在**完整纵向表**（全量 ops / train_set，例如 SCANIA 约 112 万行）上：
  - 先 `add_rolling_stats(...)`  
  - 再（若 SCANIA）`add_histogram_center(...)`
2. 若需合并静态表（如 SCANIA 的 spec、tte），在**带趋势特征的纵向表**上 merge。
3. **最后**再调用 `last_readout(df, dataset)`，保留每个 **Asset_ID** 的最后一行，作为建模用的“快照”。

当前 `main.py` 中的 `_load_and_stitch_scania` 与 `_load_and_stitch_backblaze` 已按此顺序实现，后续若有新特征也应在**取快照之前**在纵向表上计算。

使用示例：

```python
from features import add_rolling_stats, add_histogram_center

# 1. 在全量历史数据上算趋势
df = add_rolling_stats(df, columns=["sensor_1"], window=5) 
if DATASET_NAME == "scania":
    df = add_histogram_center(df)

# 2. 算完后取“最后时刻”的快照用于训练
from adapter import last_readout
df_final = last_readout(df, DATASET_NAME)
```

### 3.4 变长序列的鲁棒性

- 所有时序/滚动计算均在 **Asset_ID** 组内完成，不同资产的观测数可以不同（变长序列）。  
- `min_periods=1` 保证即使某资产只有 1 条记录也能得到有效滚动均值（标准差/差分可能为 NaN，由后续 SimpleImputer 处理）。  
- 不做跨资产的聚合或归一化，避免信息泄露与分布偏移。

---

## 4. Pipeline 使用

### 4.1 预处理框架

- **数值列**：`SimpleImputer(strategy=config)` → **RobustScaler(quantile_range=config)`  
  - **RobustScaler** 使用中位数与四分位距（默认 25–75 分位）进行缩放，对传感器异常值不敏感，适合含噪声的工业/ SMART 数据。
- **分类列**：`SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`  
- 配置由 `get_preprocess_config(dataset_name)` 提供，可按数据集扩展（如不同 `quantile_range`）。

### 4.2 如何接入任意模型（Estimator）

使用 **create_generic_pipeline** 将任意分类器包装进上述预处理管道：

```python
from model_pipeline import create_generic_pipeline

# 从训练集得到列类型（与 main.py 一致）
num_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]

# 接入任意估计器
from sklearn.ensemble import RandomForestClassifier
# 或 from xgboost import XGBClassifier 等
estimator = RandomForestClassifier(n_estimators=200, random_state=0)

pipe = create_generic_pipeline(
    estimator,
    dataset_name="scania",  # 或 "backblaze"
    num_cols=num_cols,
    cat_cols=cat_cols,
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)
```

- 管道结构：`steps=[("preprocess", ColumnTransformer), ("model", estimator)]`  
- 超参调优可通过 `pipe.set_params(model__n_estimators=100)` 等形式访问 `model` 步参数。

---

## 5. 开发接口（你们需要做的）

实现完下面这两个后，理论上运行main.py就能输出结果

### 5.1 实现代价矩阵与代价评估函数（在`model_pipeline.py`）

以下函数在 `model_pipeline.py` 中**仅保留接口与 NotImplementedError，需要你们实现**：


| 函数 / 协议                                             | 说明                                                                  | 预期实现位置              |
| --------------------------------------------------- | ------------------------------------------------------------------- | ------------------- |
| `get_cost_matrix(dataset)`                          | 实现代价矩阵，返回一个Scania 5×5 / Backblaze 3×3 代价矩阵 C，C[i,j]=真实类 i 预测为 j 的代价 | `model_pipeline.py` |
| `total_cost(y_true, y_pred, cost=..., dataset=...)` | 总代价 = Σ cost[y_true[i], y_pred[i]]                                  | `model_pipeline.py` |
| `evaluate_cost_sensitive(y_true, y_pred, dataset)`  | 实现总代价计算逻辑。返回含 total_cost 等指标的字典                                     | `model_pipeline.py` |
| `CostSensitiveEvaluator`                            | Protocol，便于注入或 mock 评估器                                             | 可选                  |


在 `main.py` 中，fit/predict 之后已有注释占位，可在此处调用上述函数并输出指标：

```python
# 可选：由负责模型的成员在此处接入代价矩阵与 evaluate_cost_sensitive
# from model_pipeline import get_cost_matrix, evaluate_cost_sensitive
# cost = get_cost_matrix(name)
# metrics = evaluate_cost_sensitive(y_true, y_pred, name)
```

### 5.2 实现模型调优（在 `main.py`）

- **占位模型**：当前 `main.py` 使用 `RandomForestClassifier(n_estimators=50, random_state=0)` 仅作演示。  
- **如何调优**：  
  - 在 `main.py` 中替换 `estimator` 实例，进行超参数搜索（GridSearch）或尝试不同的集成算法。可以选择模型（如 XGBoost、LightGBM 等），也可以使用 `create_generic_pipeline(estimator, ...)` 传入已调参的估计器。
- 网格/随机搜索可对 `pipe` 使用 `GridSearchCV` / `RandomizedSearchCV`，参数前缀为 `model__`*。

---

## 6. 小结

- **一套代码两数据集**：通过 `Asset_ID` / `TIME` 统一命名和 `dataset_name` 分支实现。  
- **数据协议**：统一使用 `Asset_ID` 与 `time`（常量 `TIME`）；原始列名仅在与外部数据或提交文件交互时通过 `from_canonical` 还原。  
- **特征逻辑**：滚动窗口与直方图重心均在**按 Asset_ID 分组的纵向表**上计算；**必须先做特征工程，再 last_readout 取快照**。  
- **鲁棒性**：RobustScaler 用于抑制传感器异常值；变长序列通过“仅组内计算、不跨资产”保证无信息泄露。  
- **Pipeline**：通过 `create_generic_pipeline(estimator, dataset_name, num_cols, cat_cols)` 接入任意估计器。  
- **后续开发**：在 `model_pipeline.py` 中实现 `get_cost_matrix`、`total_cost`、`evaluate_cost_sensitive`，在 `main.py` 中接入评估逻辑并替换/调优估计器。

