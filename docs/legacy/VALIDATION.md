# 0.5.1 数据检查

54项测试通过；两套真实历史工程输入也通过全部GR弧有限性、非负物理量与非正GR/CTS TNS检查。原版四个预测核心文件保持逐字节一致。迭代模块仅按数据声明/数值与键检查，不把它宣称为外部物理流程闭合证明。

# 0.5 设计适配迭代

51项测试通过。使用真实历史bp_fe候选做两轮明确线网抽样的工程回归：38→42个累计候选，训练26→30，固定验证12个ID不变。900棵树线网模型、104维端点分支及原始候选/纠错实现均保留；两版权重不同并全部保留。预测入口分数/名次/并列关系逐位一致，私有工程归档导入后也一致。没有启用或发布工程权重。

两个版本在这12个已探索历史候选上的tau-b均为0.6667，GR为0.6061；新增4个训练候选后持平，不能宣称随数据增加已改善。这只是每候选8条线网弧的流程验证，不是新设计、全线网精度或独立验证成绩。

浏览器已核对模块入口、真实文件/角色检查、训练进度、同协议版本表和工程版本禁用状态，无控制台错误/水平溢出。原始四个预测核心文件与0.3发布包逐字节一致，原权重及两个485候选结果未改动。详见ITERATION.md与ACCEPTANCE.json。

# 0.4 输入与数据准备界面

39 项完整测试通过；最终增加 DEF 连接描述支持及模式切换修正后，19 项输入/界面专项回归通过。浏览器核对真实485候选/7设计、三表900个字段、字段搜索与真实值、缺失工程阻止启动、配置模板下载，控制台无错误。原版四个预测核心模块与0.3发布轮逐字节一致，两个既有485候选预测记录保留。

本轮没有执行新的原始网表→完整原版特征的物理端到端实验。当前服务器未绑定工艺 profile，界面显式阻止启动；测试夹具只证明编排协议与门禁，不作为物理正确性或精度证据。参见 RAW_INPUT.md 与 ACCEPTANCE.json。

# Validation scope

## 0.3.0 graphical workspace acceptance

The installed graphical tool passed 26 tests. A browser initiated full original
PARIS4 inference on the real seven-design seed-17 pool of 485 candidates.
The original inference Parquet file was **byte-identical** to the accepted CLI
output (SHA256 `633516f07ef4de7470a3dadc479e162ec418a1166dec07ea89466a7f7575ea8c`).
All candidate scores and pairwise tie/order relations matched the original
reference, with unchanged macro tau-b **0.7996579129442029**. No new DR was run.

Browser checks covered model/data inspection, server directory selection,
invalid-input recovery, actual job execution, design switching, full pagination,
candidate search, top-rank ties, CSV and Parquet downloads, and persistence after
service restart. Downloaded CSV retained 485 unique keys and exact float64
scores; the downloaded Parquet retained the original file hash. No application
console errors or horizontal page overflow were observed at the tested desktop
viewport. Small-screen layouts are provided but are not part of this recorded
browser acceptance. Filesystem confinement, import-pointer confinement,
session/Origin checks and original-only model gating are covered by tests.

The interface does not display a new accuracy estimate without ground truth.
GR rank movement is an input/output comparison, not a claim of improved true
quality. GUI timing begins with existing feature tables and excludes physical
feature generation. See [current acceptance](ACCEPTANCE.json) and
[the interface guide](GUI.md).

## 0.2.0 full original model regression

The installed tool imported the complete original seed-17 model bank and ran
new inference on all 485 real GR target candidates across seven designs.
All final score float64 bit patterns, every candidate-pair ordering/tie, and
both candidate-branch and endpoint-branch ranks matched the original sealed
Parquet predictions exactly. Original and integrated macro Kendall tau-b were
both **0.7996579129442029** under the same v2_flowlibs evaluation labels.
The inference took approximately 264 seconds with two CPU threads; this starts
from existing feature tables and is not an end-to-end physical-flow benchmark.
No additional detailed route was run. The model contains weights and decisions,
not stored target predictions. Target labels were opened after prediction seal
and score parity. See [the current acceptance record](ACCEPTANCE.json).

CSV re-parsing can round near-tied rank values and slightly change reported
tau-b (the older CSV summary gave approximately 0.7995). The authoritative
reference and this regression use original float64 Parquet scores and give
approximately 0.7997. No model or fusion weights were changed to obtain this.

## Previous 0.1.0 native profile acceptance

The 0.1.0 installed-wheel acceptance on Linux/Python 3.10 passed 20 tests.
The installed CLI also completed synthetic train/predict/screen, real saved
checkpoint extraction, cache reuse and original frozen-model compatibility.
The saved bp_fe checkpoint returned DR TNS **-5654.962183915332 ns**, exactly
matching its canonical label; 1,918 GR source/sink rows matched historical DR
wire labels. No new detailed route was launched. Three known feature rows with
test IDs produced bitwise-identical original PARIS4 scores and candidate order.
See [the previous acceptance record](ACCEPTANCE_0_1.json) for runtime and limits.
These are integration checks, not independent accuracy evidence.

The release tests cover candidate/wire key integrity, separation of labels and
features, candidate-disjoint self-training, serialization/export/import parity,
training-ID rejection, route-budget persistence, GR baseline budget fairness,
archive traversal rejection, CLI installation/resources and real OpenROAD
extraction where recorded in the accompanying acceptance report.

Synthetic demonstrations and renamed known feature rows are software tests,
never independent physical accuracy results. A real saved-checkpoint test
verifies extraction and TNS closure, not ranking generalization. No additional
detailed route is required by that test.

Existing PARIS4 same-design and shared-model experiment scores belong to their
own frozen splits, labels and feature profiles. They do not automatically become
the accuracy of this new portable retraining recipe. Imported frozen bundles
retain their original scope. Fresh weights trained on all available labels have
no remaining independent test cases among those training examples.

For industrial acceptance, compare PARIS and GR at equal **total** new-DR
budgets, charging calibration, failures, exploration and final verification.
Report the best measured feasible TNS/WNS, missed-best regret, number of actual
DR executions, and total elapsed/CPU time. Keep an untouched external design
or candidate pool. `paris benchmark` is only an offline replay of fixed
holdout predictions, not evidence of measured acceleration or signoff quality.
