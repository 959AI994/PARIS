# 完整原版 PARIS4.0

`paris4.original.seed17.1`封装原始七设计seed-17模型链。它与`paris4.native.1`自训练配方、全1204候选最终refit权重分别标识，不做自动替换。

保留的组成包括：

1. 原序列化CTS/GR候选基线集成、特征预处理与候选误差修正模型。
2. 原序列化XGBoost线网DR wire-delay残差模型；从目标GR线网重新预测并使用原聚合代码。
3. 原设计对应的CTS/GR阶段共识开关与权重。
4. ariane136原88+32历史候选的增广分支。
5. 原104维GR端点摘要模型族、按设计冻结的端点融合权重、小历史样本门控。

原始候选及线网权重直接复制并校验SHA256。历史上未单独序列化的端点模型、ariane136增广模型，使用当时原历史样本、原标签版本、原代码与已经固定的参数恢复，不重新搜索模型或融合权重。恢复权重必须通过目标分数回执逐例一致性检查才可交付。

## 使用

```bash
python -m pip install -r requirements-legacy.txt
paris model import --input PARIS4_original_seed17.tar.gz --output original-paris4 --sha256 MODEL_ARCHIVE_SHA256
paris predict --model original-paris4 --dataset original-gr-inputs --output ranking.parquet
```

输入目录包含`features.parquet`、`source_sink.parquet`、`net_load.parquet`。候选描述与104维端点摘要需符合每设计的`INPUT_SCHEMA.json`，线网表只含GR字段。推理拒绝DR/final标签字段、训练候选ID和未知design；不修改目标ID。预先存在的`phys_wire_`预测字段会被丢弃，由GR输入重新计算。模型包不包含目标预测缓存或目标真值。

输出保留原始`paris4_cost_rank`、`predicted_order`，并以`rank_score`供工具其余命令使用。成本分数越小越好。这些是批次相对排名分数，不是物理DR TNS。不同design分别排序；至少提供两个同design候选。

## 原始模型恢复

源码附带`export_original_seed17.py`，供有权访问原始研究工件者恢复模型库：

```bash
python scripts/export_original_seed17.py \
  --run RESEARCH_RUN_ROOT \
  --experiment ORIGINAL_EXPERIMENT_DIRECTORY \
  --runtime-bundle TRUSTED_FULL_REFIT_RUNTIME_BUNDLE \
  --tool-source . --output NEW_OUTPUT_DIRECTORY
```

该脚本依赖指定的原始研究目录结构，不是用户新数据集训练入口。原权重和数据不会随公开源码分发。新设计训练是独立任务，不能用恢复脚本伪造跨设计支持。

## 七设计回归验收

固定目标数量：ariane133 104、ariane136 60、bp 100、bp_be 30、bp_fe 26、bp_multi 103、swerv_wrapper 62，共485。通过已安装工具执行模型导出、SHA校验导入与真实GR表推理，然后对照原封存Parquet预测。

检查逐候选分数的float64二进制值、所有候选对的大小/并列关系、候选分支排名、端点分支排名。预测封存后再读取统一v2_flowlibs目标标签，核对逐设计及Macro Kendall tau-b。比较必须使用同一标签版本与并列处理；CSV通用解析器的浮点舍入可能改变极近分数关系，正式回归使用Parquet。

0.2.0真实485候选回归通过，所有上述不一致数均为0；集成前后Macro tau-b同为**0.7996579129442029**。已安装工具对完整批次推理约264秒，未计入特征生成时间。

有权访问原回执与标签时，可独立复核：

```bash
python scripts/verify_original_predictions.py \
  --predictions ranking.parquet --reference ORIGINAL_GUARDED_PREDICTIONS.parquet \
  --labels evaluation_labels.parquet --output NEW_PARITY_DIRECTORY
```

这是已探索目标池上的软件与算法一致性验收，不是新的独立精度验证，也不是2小时端到端性能验收。源数据历史线网采样含GR/DR连接资格筛选；在新目标上使用GR-only采样的分布变化仍需验证。
