# 原始设计输入与数据准备

输入页支持两种入口：已有完整特征、原始网表/工程。原始工程需要网表、几何库 LEF、时序库 Liberty 和 SDC 约束；Open3DBench 风格工程还可能需要初始 DEF、Place-MoL 配置。仅有网表不能确定工艺、时序约束、层间连接规则或布局方法。

**实现边界：**0.4 提供真实工程导入/检查、工艺流程编排、产物/哈希/日志、特征清单和完整模型兼容性门禁。它不内置一个适配任意 3D-IC/PDK 的“网表转原版全特征”转换器。管理员必须把自己已验证的预处理、分割、布局/CTS/GR 和全特征提取入口配置为服务器流程。未配置时界面明确阻止启动；不能把这视为原始网表端到端流程已经验证。

## 工程格式

网页可下载配置模板。把 `PARIS_PROJECT.json` 放在工程根目录，使用服务器目录或打包为 ZIP 上传：

```json
{
  "version": 1,
  "design": "my_design",
  "top": "my_top",
  "inputs": {
    "netlist": "design.v",
    "lef": ["tech.lef", "cells.lef"],
    "liberty": ["cells.lib"],
    "sdc": "design.sdc"
  },
  "partition": {"count": 32, "seed": 17}
}
```

可选 inputs：`def`、`place_config`、`candidates`（已有分割候选）。相对路径限定在工程内；绝对路径必须位于服务启动时指定的 `--root` 中。共享工艺库可由服务器 profile 的 `inputs` 提供默认路径，不必反复上传。项目输入优先，Liberty 列表顺序保留。

Open3DBench 风格工程可用含连接信息的 `def` 代替 `netlist`，但必须经实际链接流程核验，不把任意 DEF 都视为有效网表。

单独上传 `.v/.sv/.def` 时会创建待补全的工程配置，默认 top 为文件名，**不会推断它就是实际顶层**，也不会虚构工艺文件。请核实真实 top、补齐配套文件再检查。

ZIP 限制256 MiB、解包2 GiB、20000个文件，不允许链接、特殊文件或越界路径。大工程请选择服务器目录。上传只传给运行 PARIS 的本机/用户服务器，不使用第三方存储。

## 绑定实际工艺流程

```bash
paris ui --root /path/to/projects --root /path/to/models --root /path/to/pdk \
  --workspace /path/to/ui-work --model /path/to/original-model \
  --preparation-profile /path/to/trusted-preparation-profile.json
```

profile 是**服务器管理员提供的可信配置**，不从上传包加载，网页也不能提供执行命令。运行时使用数组参数、没有 shell 拼接。示例中的 `/opt/.../flow_adapter.py` 表示用户实际已有的流程入口；本工具不附带该脚本，也不宣称这份示例已完成物理适配。

```json
{
  "version": 1,
  "name": "My 3D flow / exact Liberty corner",
  "stops_at": "global_route",
  "inputs": {"lef": ["/path/to/pdk/tech.lef", "/path/to/pdk/cells.lef"], "liberty": ["/path/to/pdk/cells.lib"]},
  "stages": {
    "preprocess": {
      "argv": ["/opt/flow/bin/python", "/opt/flow/flow_adapter.py", "preprocess", "--manifest", "{manifest}", "--output", "{output}"],
      "artifacts": ["preprocess/linked_design.json"]
    },
    "partition": {
      "argv": ["/opt/flow/bin/python", "/opt/flow/flow_adapter.py", "partition", "--manifest", "{manifest}", "--output", "{output}"],
      "artifacts": ["partition/candidates.jsonl"]
    },
    "physical": {
      "argv": ["/opt/flow/bin/python", "/opt/flow/flow_adapter.py", "physical", "--manifest", "{manifest}", "--output", "{output}", "--stop-after", "global_route", "--workers", "{workers}"],
      "artifacts": ["physical/GR_RECEIPTS.json"],
      "timeout": 28800
    },
    "features": {
      "argv": ["/opt/flow/bin/python", "/opt/flow/flow_adapter.py", "features", "--manifest", "{manifest}", "--output", "{output}", "--dataset", "{dataset}"],
      "artifacts": ["dataset/features.parquet", "dataset/source_sink.parquet", "dataset/net_load.parquet"]
    }
  }
}
```

四阶段按固定顺序执行。每阶段还可设置 `cwd`、`timeout` 和 `tracked_files`（额外需要冻结校验的工具/配置文件）。替换变量仅有 `{project}`、`{manifest}`、`{output}`、`{dataset}`、`{design}`、`{workers}`。profile 可用 `designs` 限定其适配的设计。

`{manifest}` 是本次冻结的 `PROJECT.json`，包含：`project` 原始目录、`design`、`top`、`inputs` 文件映射、`partition` 参数、请求候选数 `candidates`、各文件 `hashes`、`workers:2`、`stops_at:global_route`。实际流程须读取它，而不是使用历史目标池或缓存预测替代新工程。

阶段退出0并生成约定文件后才算完成，保存产物大小/SHA-256；任一错误都保留部分输出。单个服务只运行一个数据准备或预测任务。工具、profile 和输入哈希变化会阻止后续阶段；可用空间低于60 GiB时停止新阶段。超时终止该阶段进程组并保留日志。

CPU 线程上限环境为2，`PARIS_CPU_WORKERS=2`，`PARIS_STOP_AFTER=global_route`。这些是编排接口，**不是对任意外部程序的沙箱保证**；流程维护者必须核实脚本遵循工艺、精确设计 Liberty、角点与单位，确实在GR停止、没有新增 DR。软件不能凭profile声明证明物理正确性。

## 进度与特征

每阶段有独立日志和真实状态。可选地，适配器原子写入 `{output}/<stage>/PROGRESS.json`：

```json
{"completed": 7, "total": 32, "unit": "候选"}
```

网页显示工具报告的计数；没有计数就显示阶段运行中，不编造百分比。

数据准备最终写入 `{dataset}` 三张表。特征面板读取真实 Parquet schema 与前3行示例，显示：表名、行列数、字段名、类型、字段分组、示例值、来源路径。分组只是按字段名归类，列数包含标识键，并不等于模型采用的特征数。它还可检查部分已生成的表。预处理/分割/物理阶段的任意约定产物在阶段面板按文件显示。

生成完成后调用与已有特征相同的完整原版模型检查。缺少原版候选特征、GR线网电气量、包含DR/目标字段、未知设计或候选集合错误，状态为“模型未兼容”，不会补零或替换为简化模型。通过后“使用已生成特征”才连接到原来的预测入口。

原版模型目前有七个已训练设计的权重。**支持新设计的数据准备，不等于这些权重已经支持新设计推理**；新设计需另行训练/验证适用模型。

## 验证范围

自动化测试使用明确标注的微型测试数据，验证上传约束、路径隔离、真实子进程编排、阶段回执、失败保留、并发锁与模型门禁；它们不构成物理流程或预测精度证据。现有485候选的特征浏览和原版预测回归另行记录。新的原始网表到完整原版输入，只有在配置具体流程并完成真实工程验证后才可宣称打通。
