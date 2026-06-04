# Wafer Map 与 DRAM Fail Bit Map 缺陷模式分类技术报告

> 本报告描述的是一个基于合成数据（synthetic data）的原型系统（prototype）。所有数据、图像、模型与指标均来自规则生成的样本，用于工程思路验证（engineering validation），不代表真实晶圆厂（fab）或量产（production）表现。

## 1. 项目背景 Project Background

半导体制造与测试中，缺陷空间分布（spatial defect distribution）通常包含大量工程信息。例如，晶圆级缺陷可能与设备状态、工艺窗口、污染、边缘效应或局部异常有关；DRAM 阵列中的 fail bit 分布则可能反映行译码、列译码、局部阵列、位线/字线相关问题。

本项目构建了一个轻量级原型系统，用于演示如何将缺陷图（defect map）转化为可解释特征（interpretable features），并用传统机器学习模型（classical machine learning）进行缺陷模式分类（defect pattern classification）。

项目目标不是宣称已经解决真实量产问题，而是展示一套可复现的 AI 工程流程：

- 合成数据构造（synthetic data generation）
- 图像/空间特征工程（feature engineering）
- 传统模型训练（classical model training）
- 指标与图表产出（metrics and visualization）
- 结果解释与局限性说明（interpretation and limitations）

## 2. 业务问题 Business Problem

在真实制造与测试场景中，工程团队希望从空间缺陷图中快速判断异常模式，辅助以下工作：

- 识别潜在工艺异常（process excursion）
- 辅助失效分析（failure analysis）
- 支持良率改善（yield improvement）
- 优先定位设备、制程或测试环节的问题
- 将人工经验沉淀为可复用的数字化分析流程（digital workflow）

本项目用合成数据模拟这些模式，验证一个基本问题：

> 如果缺陷模式具有明确的空间结构，是否可以用可解释特征和传统模型对其进行分类？

答案在合成数据上是可行的，但这只能说明工程方法可跑通，不能直接推导到真实产线表现。

## 3. Wafer Map 与 DRAM Fail Bit Map 的区别

### Wafer Map

Wafer map 是晶圆级（wafer-level）视角。网格中的每个有效单元代表一个 die 位置。由于晶圆是圆形的，矩形网格中会存在晶圆外区域（outside-wafer area）。本项目使用：

- `-1`: 晶圆外区域
- `0`: 正常 die
- `1`: 缺陷 die

所有特征计算都会忽略 `-1` 区域，避免把晶圆外区域误当成正常 die。

### DRAM Fail Bit Map

DRAM fail bit map 是存储阵列级（memory-array-level）视角。网格中的每个单元代表一个 bit 或 address 位置。它是完整矩形阵列，因此所有单元都是有效区域。本项目使用：

- `0`: 正常 bit
- `1`: fail bit

DRAM 场景中，row failure、column failure、block failure 与阵列结构具有更直接的对应关系。

### 为什么不能默认混合建模

Wafer map 的空间含义是 die 在晶圆上的位置；DRAM fail bit map 的空间含义是 bit/address 在一个 memory array 中的位置。两者虽然都可以表示为二维矩阵，但工程语义不同。因此，本项目默认分别训练 wafer 模型与 DRAM 模型，不把两类图混在一个模型里。

## 4. 合成数据构造 Synthetic Data Generation

本项目使用规则生成的合成数据，分别构造：

- `data/synthetic/wafer_maps/wafer_maps.npz`
- `data/synthetic/dram_fail_bitmaps/dram_fail_bitmaps.npz`

Wafer map 默认大小为 `64x64`，使用圆形 wafer mask。DRAM fail bit map 默认大小为 `128x128`，使用完整矩形阵列。

合成数据的核心原则是让每类缺陷具有清晰的空间特征：

- center defect 聚集在中心
- edge defect 聚集在边缘
- ring defect 形成环状
- scratch defect 形成线状划痕
- cluster defect 形成局部团簇
- row failure 形成长横向模式
- column failure 形成长纵向模式
- block failure 形成局部矩形块
- random defect 随机稀疏分布

重要说明：这些模式来自人工规则，不来自真实设备或真实测试数据。

## 5. 缺陷模式体系 Defect Pattern Taxonomy

本项目实现 9 类缺陷模式：

| 类别 | 英文名称 | 典型空间特征 |
|---|---|---|
| 随机缺陷 | random defect | 稀疏、无明显结构 |
| 中心缺陷 | center defect | 靠近图中心聚集 |
| 边缘缺陷 | edge defect | 靠近边缘聚集 |
| 环形缺陷 | ring defect | 沿某一半径形成环状 |
| 划痕缺陷 | scratch defect | 细长线状或斜线状 |
| 团簇缺陷 | cluster defect | 局部邻近点聚集 |
| 行失效 | row failure | 横向整行或近似整行失效 |
| 列失效 | column failure | 纵向整列或近似整列失效 |
| 块失效 | block failure | 连续矩形区域失效 |

## 6. 特征工程 Feature Engineering

本项目没有直接使用 CNN，而是先建立一组可解释特征，便于面试与工程讨论。主要特征包括：

- 缺陷密度（fail_density）
- 缺陷数量（fail_count）
- 有效区域面积（valid_area）
- 缺陷质心（centroid_y, centroid_x）
- 质心到中心距离（centroid_distance_to_center）
- 径向分布统计（radial_mean, radial_std, radial_q25, radial_q75）
- 边缘/中心缺陷比例（edge_fail_ratio, center_fail_ratio）
- 行列投影统计（row_fail_max, column_fail_max 等）
- 连通域数量（connected_component_count）
- 最大连通域面积（largest_component_area）
- 最大连通域外接框（largest_component_bbox_height, largest_component_bbox_width）
- 整体缺陷外接框与长宽比（bounding_box_height, bounding_box_width, aspect_ratio）

Wafer map 特征计算时，晶圆外区域 `-1` 被排除。DRAM fail bit map 特征计算时，完整矩形阵列均为有效区域。

## 7. 模型方法 Model Design

本项目实现了两个传统模型（classical models）：

1. Logistic Regression
   - 使用 `StandardScaler`
   - 使用 `max_iter=2000`
   - 使用 `class_weight="balanced"`

2. Random Forest
   - 使用 `RandomForestClassifier`
   - `n_estimators=200`
   - `random_state=42`
   - `class_weight="balanced"`

模型分别训练：

- wafer logistic_regression
- wafer random_forest
- dram logistic_regression
- dram random_forest

这样可以避免把 wafer-level 语义与 memory-array-level 语义混在一起。

## 8. 实验设计 Experimental Design

数据划分采用分层抽样（stratified split）：

- train
- validation
- test

默认设置：

- `test_size = 0.2`
- `validation_size = 0.2`，相对于剩余训练数据
- `random_seed = 42`

输出文件包括：

- split 文件：`data/splits/{map_type}_splits.json`
- 模型文件：`models/{map_type}_{model_name}.joblib`
- 指标文件：`reports/metrics/{map_type}_{model_name}_metrics.json`
- 混淆矩阵图：`reports/figures/{map_type}_{model_name}_confusion_matrix.png`
- 特征重要性/系数图：`reports/figures/*feature_importance.png` 或 `*coefficients.png`

评估指标（metrics）包括：

- accuracy
- macro_precision
- macro_recall
- macro_f1
- weighted_f1
- per-class precision / recall / f1
- confusion matrix

## 9. 实验结果 Results

以下结果为合成验证指标（synthetic validation metrics），不代表真实生产表现。

| Map Type | Model | Accuracy | Macro F1 |
|---|---:|---:|---:|
| wafer | logistic_regression | 0.967 | 0.967 |
| wafer | random_forest | 0.978 | 0.978 |
| dram | logistic_regression | 0.983 | 0.983 |
| dram | random_forest | 1.000 | 1.000 |

从结果看，传统特征加传统模型已经可以很好地区分规则生成的合成缺陷模式。Random Forest 在两个 map type 上表现略强，DRAM random_forest 达到 `1.000`，主要原因是合成数据中的 row、column、block 等结构非常清晰。

这个结果不能被解读为真实 DRAM 量产或真实晶圆检测上的性能。

## 10. 误差分析 Error Analysis

在合成数据中，仍然存在一些容易混淆的模式：

1. **wafer cluster defect 与 block failure 有时会混淆**
   - cluster defect 是局部团簇。
   - block failure 是局部矩形块。
   - 当 cluster 比较紧密、block 边界不够规则时，二者都表现为局部大连通域，模型可能依赖 `largest_component_area` 和 bounding box 特征进行区分。

2. **scratch defect 有时可能与 row failure 混淆**
   - scratch defect 是线状缺陷，可能接近水平。
   - row failure 是明显横向长条。
   - 当 scratch 接近水平且覆盖较长距离时，`row_fail_max` 和 horizontal projection 特征可能接近 row failure。

3. **DRAM random_forest 的 1.000 需要谨慎解释**
   - 这个结果主要因为合成类别边界清晰、噪声可控。
   - 真实测试图中可能存在混合模式、弱信号、噪声、工艺漂移、测试程序差异。

4. **这些结果不代表真实 production performance**
   - 没有使用真实 fab/test 数据。
   - 没有经过跨 lot、跨设备、跨时间验证。
   - 没有与 failure analysis 或 process engineer 反馈闭环。

## 11. 模型解释 Model Interpretation

本项目的优势之一是特征具有明确工程含义。

### radial_std

`radial_std` 描述缺陷到中心距离的离散程度。

- center defect: 径向距离集中，通常较低。
- ring defect: 距离集中在一个环带，可能 radial_std 较低但 radial_mean 特征明显。
- random defect: 距离分布更分散。

### connected_component_count

`connected_component_count` 衡量缺陷连通域数量。

- random defect: 往往多个小连通域。
- cluster defect: 通常较少连通域。
- block failure: 通常存在一个较大连通域。

### centroid_distance_to_center

`centroid_distance_to_center` 衡量缺陷重心偏离图中心的程度。

- center defect: 距离较小。
- edge defect: 距离较大。
- cluster defect: 取决于 cluster 位置。

### largest_component_area

`largest_component_area` 衡量最大连续缺陷区域面积。

- block failure: 通常较大。
- cluster defect: 中等或较大。
- random defect: 通常较小。

### row_fail_max

`row_fail_max` 表示某一行上的最大 fail 数。

- row failure: 该值非常高。
- scratch defect: 如果接近水平，也会升高。
- random defect: 通常较低。

### column_fail_max

`column_fail_max` 表示某一列上的最大 fail 数。

- column failure: 该值非常高。
- 竖直 scratch defect: 也可能升高。
- random defect: 通常较低。

### edge_fail_ratio

`edge_fail_ratio` 衡量缺陷落在边缘区域的比例。

- edge defect: 通常较高。
- ring defect: 如果环靠近边缘，也可能偏高。
- center defect: 通常较低。

### center_fail_ratio

`center_fail_ratio` 衡量缺陷落在中心区域的比例。

- center defect: 通常较高。
- edge defect: 通常较低。
- random defect: 取决于随机分布。

### 各类缺陷的解释逻辑

- center defect: `center_fail_ratio` 高，`centroid_distance_to_center` 低。
- edge defect: `edge_fail_ratio` 高，`centroid_distance_to_center` 可能较高。
- ring defect: 径向分布集中，`radial_mean` 与 `radial_std` 有区分度。
- scratch defect: bounding box 较长，aspect ratio 和 projection std 有帮助。
- cluster defect: 连通域数量较少，局部 `largest_component_area` 较明显。
- row failure: `row_fail_max` 高，horizontal projection 特征强。
- column failure: `column_fail_max` 高，vertical projection 特征强。
- block failure: `largest_component_area` 高，bbox height/width 都较明显。

## 12. 工程价值 Engineering Value

虽然本项目使用合成数据，但仍能体现实际工程价值：

- 将 wafer map 与 DRAM fail bit map 的语义区分清楚。
- 用可解释特征建立 baseline，而不是直接跳到黑盒模型。
- 输出可复现的模型、指标、split 文件和可视化。
- 为后续接入真实数据提供工程骨架。
- 适合作为 AI + process optimization + testing analytics 的原型展示。

在真实场景中，这类系统可以作为工程师的辅助分析工具，帮助快速筛选异常模式、建立问题优先级、支持数字化/智能化分析流程。

## 13. 局限性 Limitations

本项目的主要局限包括：

- 数据完全合成，不包含真实制造噪声。
- 缺陷模式由规则生成，类别边界比真实场景更清晰。
- 没有 lot、wafer、device、recipe、tester、temperature 等上下文信息。
- 没有真实 FA 结论或 process engineer 标注。
- 没有部署监控、数据漂移检测或在线反馈闭环。
- 当前没有 CNN，也没有比较深度学习方法。
- 指标较高不代表生产可用。

因此，本项目应被定位为 synthetic prototype 和工程思路验证，而不是 production-ready 系统。

## 14. 下一步优化 Future Work

后续可以从以下方向扩展：

1. 引入真实或脱敏的 wafer/test 数据，并建立严格的数据治理流程。
2. 设计更复杂的合成机制，如混合缺陷、弱信号、批次漂移和噪声。
3. 增加半监督或异常检测方法（anomaly detection）。
4. 对接更多 process/test metadata，构建多模态特征。
5. 在 baseline 稳定后，再引入 CNN 或轻量视觉模型作为可选模块。
6. 建立模型监控，包括 drift detection、误报/漏报分析和工程反馈闭环。
7. 与 failure analysis、工艺工程和测试工程团队共同定义可落地的标签体系。

最终目标不是单纯追求高分，而是构建可解释、可复现、可被工程团队信任的缺陷模式分析流程。
