# 新设计：2D → 3D → PARIS

本流程接收 RTL/门级网表、LEF、Liberty 和 SDC，生成双层分割候选，运行 Place-MoL、CTS、GR，自动形成 PARIS 输入，再调用选定模型排序。网页的“原始网表 / 工程”展示各阶段进度、日志、实际特征和最终排序。

当前提供 Nangate45 双层 Memory-on-Logic 配方：逻辑位于下层，存储宏按候选分配到上下层。其他工艺、逻辑跨层划分需要对应的工艺/布局配方。原始网表必须有匹配的宏库与时序约束。

## 工具路径

建立 `native.json`，全部路径由运行服务器配置：

```json
{
  "yosys": "/opt/yosys/yosys",
  "openroad": "/opt/OpenROAD/build/bin/openroad",
  "place_python": "/opt/place-env/bin/python",
  "feature_python": "/opt/paris-env/bin/python",
  "place_mol_root": "/opt/Open3DBench/Place-MoL",
  "flow_root": "/opt/OpenROAD-3D/flow"
}
```

`place_python` 需要已编译可用的 DREAMPlace CPU 算子；`feature_python` 安装本工具的依赖。OpenROAD 需要匹配此 OpenROAD-3D flow；本次实际验证使用 c8e73fe7。旧版本缺少 endpoint 专用 slack API 时仅在端点求和与 TNS 闭合后接受结果，不能强行回填。

## 准备 2D 设计

工程目录包含RTL/网表、LEF、Liberty、SDC及`PARIS_PROJECT.json`。清单中的`partition.count`控制候选预算，`partition.seed`控制随机种子，文件路径相对于工程根目录。

## 一条命令准备数据

```bash
paris project run --project ./my-source --settings ./native.json --output ./prepared
paris predict --model ./shared-model --dataset ./prepared/dataset --output ./ranking.csv
```

数据准备依次运行预处理、分割、物理流程、特征提取，止于 GR。各阶段保留日志、进度、检查点与哈希；已存在的输出目录不会覆盖。CPU 两个并行候选，每个 EDA 任务最多两线程。

主要产物：

| 目录 / 文件 | 内容 |
| --- | --- |
| `preprocess/linked/` | 综合网表、2D floorplan、宏结构 |
| `partition/candidates/` | 明确的上下层宏分配 |
| `physical/runs/` | 逐候选 3D 布局、CTS、GR 检查点与日志 |
| `dataset/features.parquet` | 候选结构、CTS/GR、端点摘要、物理统计 |
| `dataset/source_sink.parquet` | 实测 GR 线网 delay/slew/slack 与连接 |
| `dataset/net_load.parquet` | GR R/C 与负载 |
| `dataset/complete.json` | 来源回执与输出校验 |
| `ranking.csv` | 候选质量分数和排序 |

原版设计专用 PARIS4.0 和跨设计初始模型使用独立模型包。新设计首次推理选择允许新设计的共享模型；原版七设计组合模型和权重保持原样。取得足够的新设计训练数据后，可使用已有设计适配入口训练独立版本。

## 网页配置

```bash
paris project profile --settings ./native.json --output ./native-profile.json
paris ui --workspace ./workspace --root /data/projects \
  --model ./shared-model --preparation-profile ./native-profile.json
```

网页中导入工程目录，或上传包含 `PARIS_PROJECT.json` 的 ZIP，检查工程后启动数据准备。生成特征后点击“使用已生成特征”即可预测。流程配方及工具源码被哈希绑定；升级工具请生成新配方。

## DR 验证 / 训练标签

DR 是独立阶段，用于验证或自训练的数据预算，不是每次 PARIS 推理都执行：

```bash
paris project route --gr-run ./prepared/physical/runs/<candidate-id> \
  --route-iterations 1 --output ./dr/<candidate-id>
```

之后通过现有 `paris collect --stage labels --config project.yaml` 导入 DR 检查点，运行 OpenRCX 与 STA 并检查端点 TNS 闭合。GR 特征与 DR 标签分开保存，预测输入不读取目标 DR。完整标签表需与原有完整候选特征按 design/candidate_id 对齐后用于训练。

DR预算由`--route-iterations`指定。应在统一布线预算下比较标签；残余DRC不代表签核完成。

上游：[ChiPBench](https://github.com/MIRALab-USTC/ChiPBench)、[Open3DBench](https://github.com/lamda-bbo/Open3DBench)。

### 固定逻辑占主导的分割面积策略（0.6.1）

逻辑固定在下层时，两层不一定能达到接近相等的面积。流程先由源网表几何计算不可避免的面积不平衡下界 `I_min`，再采用 `I <= I_min + (1 - I_min) × 0.25` 生成候选；能平衡的设计仍使用原来的 0.25 上限。依据和参数保存在 `partition/AREA_POLICY.json`。候选特征保留真实面积不平衡值，完整宏几何、重叠检查和物理合法化照常执行。该规则只使用分割前信息，不读取 DR 或预测分数。

### 检查点重载与宏几何修复（0.6.2）

独立 L2 STA 报告先重读同一候选实际导出的 `gr_segments.txt`，再估计 global-routing 寄生，避免仅重读 ODB 丢失路由器状态。不会以 placement RC 替代 GR RC。

宏合法化除了检查邻近轨道，也尝试将发生重叠的宏移到相邻宏的分离边界，并按轨道/站点节距取整。上下层分配、完整宏尺寸和方向保持不变，移动后重新检查全体宏的边界与同层重叠；后续标准单元合法化照常执行。所有移动保存在几何审计回执中。

### 未定义常量的物理实现（0.6.3）

综合入口与上游 OpenROAD-3D `synth.tcl` 保持一致：ABC 后先执行 `setundef -zero`，再由 `hilomap` 映射为真实 LOGIC0/LOGIC1 tie 单元。`CONSTANT_POLICY.json` 记录未定义引脚值的处理前后数量；剩余未定义值、或带普通信号引脚且不可布线的电源/地网会在预处理阶段被拒绝。这样不会把未驱动的逻辑常量误当作可直接详细布线的地网。改变常量实现后须重新生成 GR、特征和封存预测。
