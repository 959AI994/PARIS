# PARIS

用 CTS/GR 特征预测详细布线后的 TNS，对 3D-IC 分割候选排序。每个 design 一个 XGBoost，提供网页、CLI、数据准备与模型迭代。

## 安装与启动

Python 3.10+；运行 EDA 流程使用 Linux。

```bash
git clone https://github.com/959AI994/PARIS.git
cd PARIS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -e .
python scripts/start_web.py --open
```

打开 `http://127.0.0.1:8765`。默认加载随包权重和真实候选示例。网页包括输入、处理、输出、模型迭代、评测和方法介绍。

## CLI 预测与训练

```bash
paris predict --model models/seven-designs \
  --dataset examples/gr-candidates --output work/ranking.csv

paris train --dataset my-history --output work/my-model
paris predict --model work/my-model \
  --dataset new-candidates --output work/new-ranking.csv
```

输入为 `features.parquet`、`source_sink.parquet`、`net_load.parquet` 和 `FEATURE_CONTEXT.json`；训练另需实测 `labels.parquet`。输出包含候选ID、预测TNS和排序。新design先用本设计样本训练；网页“PARIS迭代”支持累计样本、固定验证集比较和版本更新。

## 原始网表 → 3D候选 → 预测

先按 [EDA安装](docs/EDA_SETUP.md) 编译随包 Place-MoL，并配置 OpenROAD、Yosys：

```bash
python scripts/configure.py --openroad /opt/OpenROAD/build/bin/openroad \
  --yosys /opt/yosys/yosys
# 已有该design模型：从原始RTL到新候选预测
paris design --project examples/or1200-source --settings native.json \
  --model models/or1200 --candidates 2 --seed 73 --output work/or1200-run

# 新design：36个候选，24个DR标签用于标定，其余候选用于预测
paris design --project my-source --settings native.json \
  --candidates 36 --label-budget 24 --output work/my-design
```

随包提供真实or1200原始RTL与工艺输入，可直接运行上面的示例。当前内置 Nangate45 双层 Memory-on-Logic 流程。其他工艺通过配置匹配的流程和库接入。原始工程格式见 [输入与数据](docs/INPUTS.md)。配置完成后，网页也可从“原始网表 / 工程”执行数据准备。

## 复现五组评测

```bash
python evaluation/five-splits/reproduce.py --output work/evaluation
# 从保存的训练划分重新训练并核对结果
python evaluation/five-splits/reproduce.py --refit --output work/refit
```

每组1000训练、200测试；划分、特征、标签、35份权重与预期结果均在 `evaluation/five-splits/`。数据约21MB，真实候选示例约18MB。

## 目录

```text
src/paris_tool/       CLI、网页、模型与数据流程
models/              已训练权重
examples/            真实CTS/GR候选示例
configs/             工具路径配置示例
scripts/             安装、配置与网页启动
vendor/              使用的Open3DBench组件、Place-MoL与OpenROAD-3D
evaluation/         固定五组评测与复现脚本
docs/               EDA安装、输入格式、方法与验证记录
```

```bash
python -m pip install -e '.[dev]'
pytest -q
```

第三方源码与许可证见 [vendor/SOURCES.json](vendor/SOURCES.json) 和 [NOTICE](NOTICE)。
