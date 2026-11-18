# 输入与产物

## 原始设计

工程目录包含 `PARIS_PROJECT.json`，引用的文件采用相对路径：

```json
{
  "version": 1,
  "design": "my_design",
  "top": "top_module",
  "inputs": {
    "netlist": ["rtl/top.v"],
    "lef": ["pdk/technology.lef", "pdk/cells.lef", "macro/memory.lef"],
    "liberty": ["pdk/NangateOpenCellLibrary_typical.lib", "macro/memory.lib"],
    "sdc": ["constraints.sdc"]
  },
  "partition": {"count": 36, "seed": 17}
}
```

当前Nangate45 MoL配方将标准逻辑放在下层，按候选分配存储宏。宏必须有匹配的LEF/Liberty，SDC需包含有效时钟。其他工艺需要相应物理流程配方。

## 已有特征

| 文件 | 内容 |
|---|---|
| `features.parquet` | 每候选一行：分割/布局、CTS/GR与端点描述 |
| `source_sink.parquet` | GR连接、delay、slew、路径R/C、扇出等 |
| `net_load.parquet` | GR π-model R/C与负载 |
| `FEATURE_CONTEXT.json` | 工艺、流程、corner、特征版本与时间单位 |
| `labels.parquet` | 训练使用：design、candidate_id、实测final_tns |
| `PARIS_TRAINING.json` | 训练数据的工艺上下文与标签来源 |

主键为 `design,candidate_id`。训练和待预测候选分开；同一候选的特征只来自CTS/GR。

## `paris design`产物

`CALIBRATION_PLAN.json`在DR前固定训练/待预测候选及实际分割hash。`prepared/`保存所有物理阶段、GR表和来源回执；`dr/`保存标定候选的详细布线结果；`training/`保存训练数据；`model/`保存新权重；`prediction-input/`只含待预测候选；`predictions.csv`为最终排序。输出目录必须为新目录，失败输出保留。
