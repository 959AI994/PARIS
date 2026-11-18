# PARIS 使用说明

PARIS 用 CTS/GR 信息筛选候选，将完整 DR、寄生提取、STA 留给少量候选。最终报告选择的是已经真实验证的候选。工具离线运行，无遥测、无服务器账号依赖。

## 1. 安装与快速体验

```bash
python -m pip install paris_eda-0.2.0-py3-none-any.whl
paris --help
```

要使用达到历史七设计成绩的完整PARIS4.0，请加载独立模型包：

```bash
python -m pip install -r requirements-legacy.txt
paris model import --input PARIS4_original_seed17.tar.gz --output original-paris4 --sha256 <模型SHA256>
paris predict --model original-paris4 --dataset original-gr-inputs --output ranking.parquet
```

该入口保留原始候选模型、线网模型、阶段共识、ariane136增广分支和端点融合门控，不用native配方替换。完整特征表需满足模型schema；原生ODB采集器不会填零伪造缺失特征。七设计模型库分别为每个design排序；筛选与DR预算应针对同一design的候选池。详见 [原版集成与一致性验证](ORIGINAL_PARIS4.md)。

以下是可选的软件流程演示，使用独立的native自训练配方：

```bash
paris demo --output demo
paris train --algorithm paris4-native --dataset demo/history --output demo/model --jobs 2 --wire-trees 50
paris predict --model demo/model --dataset demo/new_candidates --output demo/ranking.csv
paris screen --predictions demo/ranking.csv --budget 5 --output demo/selected.csv
```

demo 是模拟数据，验证软件链路，不代表真实设计精度。正式训练默认使用900棵线网树。Python依赖由安装器解决；离线环境需要预先准备相应平台的依赖wheel。

## 2. 配置自己的设计

```bash
paris init --directory my-project
```

编辑 `my-project/project.yaml`：

- `openroad.binary`：用户自己的 OpenROAD 可执行文件，支持绝对路径或 PATH 中的名称。
- `gr_binary/dr_binary`：可选，分别指定GR和DR回放二进制；默认使用 binary。
- `liberty`：该设计精确、有序的标准单元及宏 Liberty 列表，避免把整个PDK目录中的所有库都读入。
- `setrc`、`rcx_rules`：该流程实际使用的RC设置和提取规则。
- `workspace`：用户选择的工作目录。已有非PARIS目录不会被覆盖。
- `selection.max_nets`：GR线网采样预算，不代表关键路径已经完全覆盖。

每个项目对应一个design及其候选，保证工艺、Liberty和约束上下文明确。候选清单提供已有 CTS/GR ODB、SDC、GR `route.segments` 和 `congestion.csv`。这些路由导出文件应来自该候选的原GR流程，工具不悄悄重跑另一个GR来替换输入。DR ODB/SDC可以暂时留空。

CSV中的路径相对CSV文件所在目录；YAML中的路径相对YAML。标识符保留前导零。文件路径可以包含空格。

```bash
paris doctor --config my-project/project.yaml --labels
paris collect --config my-project/project.yaml --stage gr
paris collect --config my-project/project.yaml --stage labels
paris dataset validate --dataset my-project/paris-work/dataset
```

`gr` 路径不读取DR。`labels` 只读取清单中已有DR检查点，不启动新详细布线。标签由真实DR几何重新RCX/STA取得，并核对端点负slack之和。当前版本支持单角 setup/max，不覆盖hold或完整signoff。

## 3. 用户自训练

本节是可移植的`paris4.native.1`，不代表重新训练原冻结PARIS4.0，也不继承原成绩。`train`默认后端名称为`paris4-native`；加载原始权重请使用`model import`。

```bash
paris train --algorithm paris4-native --dataset my-project/paris-work/dataset --output my-paris4 --jobs 2
```

每design至少24个有完整TNS和匹配wire标签的候选。程序先保留候选级测试集，再在history内部选融合配方，封存测试预测，最后按选定配方重训全量权重。`METRICS.csv` 是留出模型的成绩；全量权重不能再用训练过的候选报告独立精度。

训练得到：模型、schema、依赖版本、数据签名、分割清单、history CV、留出预测、模型卡和SHA256。成本分数/名次越小越好，它不是STA真值TNS。

导入已有表格时可用：

```bash
paris dataset import --features features.parquet --source-sink source_sink.parquet \
  --net-load net_load.parquet --labels measured_labels.parquet --output prepared
```

source-sink原表中的 `dr_wire_delay` 会被单独拆入 `wire_labels.parquet`；预测输入表移除DR字段。候选特征表必须预先是纯CTS/GR特征，不能混入标签。大型数据建议按设计处理并控制线网采样量；0.1版训练主要使用内存数据表。

## 4. 加载原有PARIS4.0权重

```bash
python -m pip install -r requirements-legacy.txt
paris model import --input PARIS4_full_refit_20261116.tar.gz \
  --output imported --sha256 <模型发布的完整SHA256>
paris predict --model imported --dataset compatible-gr-inputs --output ranking.csv
```

只导入可信模型：joblib会加载Python对象。兼容入口调用原模型自己的推理代码；它仍需要原来的完整特征定义。不能把native采集器的110列输入填零扩充成旧版863列。对用户新设计，推荐先用native采集与自训练链路；跨设计共享实验包也可通过兼容入口加载，但必须满足其schema，效果需要新设计验收。

## 5. 自动化与DR预算

在YAML中配置自己的 `flow.dr_command`，命令以参数数组执行，不经过shell。PARIS提供环境变量：

| 环境变量 | 含义 |
|---|---|
| `PARIS_OPENROAD_BIN` | 配置的OpenROAD路径 |
| `PARIS_DESIGN` / `PARIS_CANDIDATE_ID` | 当前候选 |
| `PARIS_GR_ODB` | 原GR检查点 |
| `PARIS_JOB_DIR` | 本次任务输出目录 |
| `PARIS_DR_RESULT` | 流程应写出的JSON路径 |

用户流程完成后写出JSON，例如 `{"dr_odb":"result.odb","dr_sdc":"result.sdc"}`。相对路径以任务目录为基准。PARIS再独立提取真实标签，而不直接相信流程输出的某个TNS数字。平台相关的详细布线、修复、填充步骤由用户的已验证流程负责，PARIS不猜测这些设置。

`paris run` 自动化流程使用native采集器及其自训练模型。原有冻结模型若需要额外的分区/PCA等输入，请用兼容数据表运行 `paris predict`，再通过 `paris screen` 和 `paris label` 接入验证，避免混淆特征profile。

```bash
paris run --config my-project/project.yaml --model my-paris4 --budget 10
# 无模型时：每个design先取得24个校准候选，再自训练和验证剩余候选。
paris run --config my-project/project.yaml --train --budget 40
```

预算是该workspace累计的新DR执行次数，包含校准和失败尝试。已有检查点、已完成缓存不重复运行；部分结果保留并停止，不能盲目重启。需要更多预算时明确增加总预算。工作目录有独占锁，防止两个命令并发写同一份数据。未配置DR hook时，工具可以生成候选清单并报告等待流程接入。

## 6. 替换模型与交付边界

后续PARIS4算法升级通过后端接口集成，数据和OpenROAD配置接口保持稳定。详见 `ALGORITHM_PLUGIN.md`。

公开源码不附真实训练数据、PDK、服务器路径和预训练权重。模型、数据、客户流程可独立交付。工具包没有上传到公共仓库或PyPI。当前可用性验收不等于已经证明减少某个百分比的DR调用且保持同等质量；应在外部设计上按相同预算与GR对照。

0.1.0安装包已通过20项测试；真实已有检查点的DR TNS与标准标签完全闭合，原冻结PARIS4权重的分数和次序兼容性通过。完整验收范围见 [ACCEPTANCE.json](ACCEPTANCE.json)。详细布线hook目前用模拟流程验证了调度、预算与失败保留；接入用户平台后仍需进行一次真实流程验收。
