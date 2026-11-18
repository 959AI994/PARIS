"""Observable, isolated raw-design preparation. Tool recipes are server-owned.

Uploaded projects contain data, never executable commands. An administrator
binds a trusted PDK/flow recipe when starting the service. No model or missing
feature is substituted if that recipe does not produce the original schema.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import threading
import time
import uuid
import zipfile

from .utils import digest, save

STAGES = {
    "preprocess": ("网表与工艺预处理", "解析、链接网表，检查单元库、约束和设计结构"),
    "partition": ("2D → 3D 分割候选", "生成上下层分配，适配宏库、层映射并核验分割统计"),
    "physical": ("布局、CTS 与 GR", "逐候选运行到 global route，保留检查点与日志"),
    "features": ("生成 PARIS 特征", "提取候选、端点摘要和 GR 线网电气特征"),
    "validate": ("模型输入检查", "核验特征字段与模型输入要求"),
}
INPUTS = {
    "netlist": "网表 / DEF 连接描述",
    "lef": "LEF 几何库",
    "liberty": "Liberty 时序库",
    "sdc": "SDC 时序约束",
}
TABLES = ("features", "source_sink", "net_load")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def local_file(root, value):
    p = (root / value).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError("工程文件不能指向工程目录之外：" + str(value))
    if not p.is_file():
        raise ValueError("文件不存在：" + str(value))
    return p


def family(name, table):
    if name in {
        "design",
        "candidate_id",
        "net_name",
        "driver_pin",
        "sink_pin",
        "transition",
        "corner",
    }:
        return "标识与连接键"
    if table == "net_load":
        return "GR R/C 与负载"
    if table == "source_sink":
        return "GR 线网 delay / slew / slack"
    if name.startswith("ep_"):
        return "GR 端点与路径摘要"
    if name.startswith("phys_artifact_"):
        return "GR 日志与物理描述"
    if name.startswith("rel_"):
        return "候选批次相对特征"
    if name.startswith(("rudy_", "prototype_")):
        return "布局与拥塞估计"
    if name.startswith(("path_", "critical_", "timing_")):
        return "时序关键性与跨层路径"
    if name.startswith(("grt_", "gr_", "congestion", "log_grt")):
        return "GR 时序、布线与拥塞"
    if name.startswith(("cts_", "l1_", "fast_l1_")):
        return "CTS 时序与物理摘要"
    if name.startswith(("hierarchy", "cut_", "die_", "dopp_", "area_", "macro_")):
        return "分割、拓扑与层级"
    if name.startswith("phys_wire_"):
        return "线网物理摘要"
    return "其他候选特征"


def catalog(dataset):
    """Read schema metadata, not millions of wire rows. Never invent features."""
    import pyarrow.parquet as pq

    root = Path(dataset).resolve()
    tables, groups = [], {}
    for name in TABLES:
        path = root / (name + ".parquet")
        if not path.exists():
            tables.append(dict(name=path.name, status="missing", columns=0, rows=0))
            continue
        local_file(root, path.name)
        q = pq.ParquetFile(path)
        batch = next(q.iter_batches(batch_size=3), None)
        sample = batch.to_pydict() if batch else {}
        columns = []
        for field in q.schema_arrow:
            group = family(field.name, name)
            values = []
            for v in sample.get(field.name, []):
                if v is None or (isinstance(v, float) and not math.isfinite(v)):
                    values.append("缺失")
                else:
                    values.append(str(v)[:100])
            item = dict(
                name=field.name,
                dtype=str(field.type),
                table=path.name,
                family=group,
                sample=values,
            )
            columns.append(item)
            groups.setdefault(group, []).append(item)
        tables.append(
            dict(
                name=path.name,
                status="complete",
                columns=len(columns),
                rows=q.metadata.num_rows,
                bytes=path.stat().st_size,
                fields=columns,
            )
        )
    return dict(
        dataset=str(root),
        tables=tables,
        families=[dict(name=k, count=len(v), fields=v) for k, v in groups.items()],
        columns=sum(t["columns"] for t in tables),
    )


class Preparation:
    def __init__(self, app, profile=None):
        self.app = app
        self.root = app.workspace / "preparations"
        self.root.mkdir(exist_ok=True)
        (app.workspace / "uploads").mkdir(exist_ok=True)
        self.jobs, self.active, self.worker = {}, None, None
        self.profile_path = Path(profile).resolve() if profile else None
        self.profile = self._profile() if profile else None
        self.profile_hash = digest(self.profile_path) if profile else None
        self.tool_hashes = {}
        if self.profile:
            for stage in self.profile["stages"].values():
                for value in stage["argv"] + stage.get("tracked_files", []):
                    p = Path(value)
                    if p.is_absolute() and p.is_file():
                        self.tool_hashes[str(p)] = digest(p)
        for f in self.root.glob("*/PREPARATION.json"):
            try:
                r = json.loads(f.read_text())
                if r["status"] in {"running", "queued"}:
                    r.update(
                        status="interrupted",
                        error="上次服务已结束，保留部分输出；不会自动重启。",
                    )
                    save(f, r)
                self.jobs[r["id"]] = r
            except (KeyError, ValueError, OSError):
                pass

    def _profile(self):
        p = json.loads(self.profile_path.read_text())
        if p.get("version") != 1 or p.get("stops_at") != "global_route":
            raise ValueError(
                "Preparation profile must use version=1, stops_at=global_route"
            )
        if set(p.get("stages", {})) != set(STAGES) - {"validate"}:
            raise ValueError(
                "Preparation profile requires preprocess/partition/physical/features"
            )
        for key, s in p["stages"].items():
            if (
                not isinstance(s.get("argv"), list)
                or not s["argv"]
                or not all(isinstance(v, str) for v in s["argv"])
            ):
                raise ValueError("Stage argv must be a nonempty string array")
            if not Path(s["argv"][0]).is_absolute():
                raise ValueError("Stage executable must be absolute")
            if not os.access(s["argv"][0], os.X_OK):
                raise ValueError(
                    "Stage executable is missing or not executable: " + s["argv"][0]
                )
            if not s.get("artifacts") or not isinstance(s["artifacts"], list):
                raise ValueError("Every stage requires declared output artifacts")
            for pattern in s["artifacts"]:
                if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                    raise ValueError(
                        "Stage artifacts must stay inside the new output directory"
                    )
            if float(s.get("timeout", 7200)) <= 0:
                raise ValueError("Stage timeout must be positive")
        return p

    def config(self):
        return dict(
            profile=self.profile.get("name", "工艺流程") if self.profile else None,
            configured=bool(self.profile),
            active=self.active,
            runs=[
                {
                    k: v
                    for k, v in r.items()
                    if k not in {"catalog", "artifacts", "model_inspection"}
                }
                for r in sorted(
                    self.jobs.values(), key=lambda r: r["created"], reverse=True
                )[:20]
            ],
            stages=[
                dict(id=k, title=v[0], description=v[1]) for k, v in STAGES.items()
            ],
            upload_limit=256 * 1024**2,
        )

    def inspect(self, project):
        root = self.app.path(project, directory=True)
        manifest = local_file(root, "PARIS_PROJECT.json")
        if manifest.stat().st_size > 32768:
            raise ValueError("PARIS_PROJECT.json 过大")
        p = json.loads(manifest.read_text())
        if (
            p.get("version") != 1
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", p.get("design", ""))
            or p["design"] in {".", ".."}
        ):
            raise ValueError("工程配置需要 version:1 和有效的 design 名称")
        if not isinstance(p.get("top"), str) or not p["top"].strip():
            raise ValueError("请在 PARIS_PROJECT.json 中指定 top 模块")
        if set(p) - {"version", "design", "top", "inputs", "partition", "description"}:
            raise ValueError("工程只接受数据与分割参数；执行命令只能由服务器配置")
        count = p.get("partition", {}).get("count", 2)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 2 <= count <= 10000
        ):
            raise ValueError("候选数 count 必须为 2–10000 的整数")
        files, checks, hashes = [], [], {"PARIS_PROJECT.json": digest(manifest)}
        inputs = dict(self.profile.get("inputs", {})) if self.profile else {}
        declared = p.get("inputs", {})
        if not isinstance(declared, dict):
            raise ValueError("inputs 必须为文件映射")
        inputs.update(declared)
        if not isinstance(inputs, dict):
            raise ValueError("inputs 必须为文件映射")
        if set(inputs) - {
            "netlist",
            "lef",
            "liberty",
            "sdc",
            "def",
            "place_config",
            "candidates",
        }:
            raise ValueError("inputs 只允许网表、工艺库、SDC、DEF、布局配置和已有候选")
        for key in dict.fromkeys([*INPUTS, *inputs]):
            values = inputs.get(key, [])
            values = [values] if isinstance(values, str) else values
            if not isinstance(values, list) or not all(
                isinstance(v, str) and v for v in values
            ):
                raise ValueError("输入文件必须为相对路径字符串或路径列表")
            errors = []
            if not values:
                errors.append("未提供")
            for value in values:
                try:
                    path = (
                        self.app.path(value)
                        if Path(value).is_absolute()
                        else local_file(root, value)
                    )
                    if not path.is_file():
                        raise ValueError("需要文件：" + value)
                    h = digest(path)
                    files.append(
                        dict(kind=key, name=value, bytes=path.stat().st_size, sha256=h)
                    )
                    hashes[value] = h
                except ValueError as e:
                    errors.append(str(e))
            checks.append(
                dict(
                    id=key,
                    label=INPUTS.get(key, key),
                    ok=not errors,
                    message="、".join(errors) if errors else f"{len(values)} 个文件",
                )
            )
        if not inputs.get("netlist") and any(
            c["id"] == "def" and c["ok"] for c in checks
        ):
            netlist_check = next(c for c in checks if c["id"] == "netlist")
            netlist_check.update(
                ok=True, message="使用已有 DEF 中的连接描述，需经实际流程链接检查"
            )
        missing = [r["label"] for r in checks if not r["ok"]]
        blockers = ["缺少或无效：" + "、".join(missing)] if missing else []
        if not self.profile:
            blockers.append("服务器尚未配置该工艺的预处理/分割/CTS/GR 特征生成流程")
        elif digest(self.profile_path) != self.profile_hash:
            blockers.append("服务器流程配置已变化，请重启服务后检查")
        elif self.profile.get("designs") and p["design"] not in self.profile["designs"]:
            blockers.append("当前工艺流程未配置该设计，请由管理员配置对应流程")
        return dict(
            project=str(root),
            design=p["design"],
            top=p["top"],
            candidates=count,
            checks=checks,
            files=files,
            hashes=hashes,
            ready=not blockers,
            blockers=blockers,
            profile=self.profile.get("name") if self.profile else None,
            inputs=inputs,
            partition=p.get("partition", {}),
        )

    def upload(self, stream, size, filename):
        if size < 1 or size > 256 * 1024**2:
            raise ValueError("上传限制为 256 MiB；更大的工程请选择服务器目录")
        suffix = Path(filename).suffix.lower()
        if suffix not in {".zip", ".v", ".sv", ".def"}:
            raise ValueError("请上传 .zip 工程包或 .v/.sv/.def 设计文件")
        if shutil.disk_usage(self.app.workspace).free < 60 * 1024**3:
            raise ValueError("可用磁盘低于60 GiB，停止新增上传")
        folder = self.app.workspace / "uploads" / uuid.uuid4().hex
        folder.mkdir()
        archive = folder / ("upload" + suffix)
        with archive.open("xb") as out:
            left = size
            while left:
                chunk = stream.read(min(left, 1024**2))
                if not chunk:
                    raise ValueError("上传中断，部分文件已保留")
                out.write(chunk)
                left -= len(chunk)
        project = folder / "project"
        project.mkdir()
        if suffix == ".zip":
            try:
                z = zipfile.ZipFile(archive)
            except zipfile.BadZipFile as e:
                raise ValueError("ZIP 文件损坏或格式无效") from e
            with z:
                items = z.infolist()
                if len(items) > 20000 or sum(i.file_size for i in items) > 2 * 1024**3:
                    raise ValueError("工程包展开超过 2 GiB 或 20000 个文件")
                for i in items:
                    name = PurePosixPath(i.filename)
                    mode = i.external_attr >> 16
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or "\\" in i.filename
                        or stat.S_ISLNK(mode)
                        or (stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR})
                    ):
                        raise ValueError("工程包含不安全路径、链接或特殊文件")
                z.extractall(project)
            matches = list(project.rglob("PARIS_PROJECT.json"))
            if len(matches) != 1:
                raise ValueError("工程包需要且只能包含一份 PARIS_PROJECT.json")
            project = matches[0].parent
        else:
            name = "design" + suffix
            shutil.copyfile(archive, project / name)
            design = (
                re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)[:80] or "design"
            )
            save(
                project / "PARIS_PROJECT.json",
                dict(
                    version=1,
                    design=design,
                    top=design,
                    inputs={"def" if suffix == ".def" else "netlist": name},
                    partition=dict(count=2, seed=17),
                    description="单网表导入；请补充真实 top、LEF、Liberty 与 SDC 后检查",
                ),
            )
        return self.inspect(project)

    def save(self, r):
        save(self.root / r["id"] / "PREPARATION.json", r)

    def state(self, rid):
        with self.app.lock:
            if rid not in self.jobs:
                raise ValueError("数据准备任务不存在")
            r = json.loads(json.dumps(self.jobs[rid]))
        progress = self.root / rid / "artifacts" / r.get("phase", "") / "PROGRESS.json"
        if progress.is_file() and progress.stat().st_size < 32768:
            try:
                p = json.loads(progress.read_text())
                done, total = p["completed"], p["total"]
                if (
                    isinstance(done, int)
                    and isinstance(total, int)
                    and 0 <= done <= total
                    and total > 0
                ):
                    r["progress"] = dict(
                        completed=done,
                        total=total,
                        unit=str(p.get("unit", "候选"))[:50],
                    )
            except (ValueError, KeyError, TypeError):
                pass
        return r

    def start(self, project, model):
        inspected = self.inspect(project)
        if not inspected["ready"]:
            raise ValueError("；".join(inspected["blockers"]))
        self.app.path(model)
        with self.app.lock:
            if self.active or self.app.active or self.app.iteration.active:
                raise ValueError("已有任务正在运行，请等待完成")
            rid = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                + "-"
                + uuid.uuid4().hex[:8]
            )
            folder = self.root / rid
            folder.mkdir()
            r = dict(
                id=rid,
                kind="preparation",
                created=stamp(),
                started_at=time.time(),
                finished_at=None,
                status="queued",
                phase="preprocess",
                steps={s: "pending" for s in STAGES},
                events=[],
                inspection=inspected,
                model=str(model),
                profile_sha256=self.profile_hash,
                tool_hashes=self.tool_hashes,
                output=str(folder),
                dataset=None,
                catalog=None,
                error=None,
            )
            self.jobs[rid] = r
            self.active = rid
            self.save(r)
            self.worker = threading.Thread(
                target=self.run, args=(rid,), name="PARIS-preparation", daemon=False
            )
            self.worker.start()
            return self.state(rid)

    def event(self, r, stage, status, message):
        with self.app.lock:
            r["steps"][stage] = status
            r["phase"] = stage
            r["events"].append(dict(time=stamp(), stage=stage, message=message))
            self.save(r)

    def run(self, rid):
        r = self.jobs[rid]
        root = self.root / rid
        output = root / "artifacts"
        dataset = output / "dataset"
        try:
            output.mkdir()
            with self.app.lock:
                r["status"] = "running"
            info = r["inspection"]
            original = self.inspect(info["project"])
            if original["hashes"] != info["hashes"] or not original["ready"]:
                raise ValueError("工程输入或流程配置已变化")
            frozen = root / "PROJECT.json"
            save(frozen, dict(**info, workers=2, stops_at="global_route"))
            replacements = {
                "project": info["project"],
                "output": str(output),
                "manifest": str(frozen),
                "design": info["design"],
                "workers": "2",
                "dataset": str(dataset),
            }

            def expand(value):
                for k, v in replacements.items():
                    value = value.replace("{" + k + "}", v)
                return value

            for stage in list(STAGES)[:-1]:
                s = self.profile["stages"][stage]
                with self.app.lock:
                    r["phase"] = stage
                if shutil.disk_usage(output).free < 60 * 1024**3:
                    raise ValueError("可用磁盘低于60 GiB，停止启动新阶段")
                if digest(self.profile_path) != self.profile_hash:
                    raise ValueError("流程配置哈希发生变化")
                if any(digest(p) != h for p, h in self.tool_hashes.items()):
                    raise ValueError("工具或流程代码哈希发生变化")
                if self.inspect(info["project"])["hashes"] != info["hashes"]:
                    raise ValueError("阶段启动前工程输入哈希已变化")
                self.event(r, stage, "running", STAGES[stage][1])
                (output / stage).mkdir(exist_ok=True)
                env = os.environ.copy()
                env.update(
                    {
                        k: "2"
                        for k in [
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                        ]
                    }
                )
                env.update(PARIS_STOP_AFTER="global_route", PARIS_CPU_WORKERS="2")
                cmd = [expand(v) for v in s["argv"]]
                with (root / (stage + ".log")).open("x") as log:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=expand(s.get("cwd", str(output))),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = proc.wait(timeout=float(s.get("timeout", 7200)))
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        raise ValueError(STAGES[stage][0] + "超时，保留部分输出")
                if code:
                    raise ValueError(
                        STAGES[stage][0] + f"退出码 {code}；请查看 {stage}.log"
                    )
                artifacts = {}
                for pattern in s["artifacts"]:
                    found = [p for p in output.glob(pattern) if p.is_file()]
                    if not found:
                        raise ValueError("阶段没有生成约定产物：" + pattern)
                    for p in found:
                        local_file(output, str(p.relative_to(output)))
                        artifacts[str(p.relative_to(output))] = dict(
                            bytes=p.stat().st_size, sha256=digest(p)
                        )
                r.setdefault("artifacts", {})[stage] = artifacts
                r["catalog"] = catalog(dataset) if dataset.is_dir() else None
                self.event(
                    r,
                    stage,
                    "complete",
                    f"已核验 {len(artifacts)} 个产物；日志和哈希已保存",
                )
            if self.inspect(info["project"])["hashes"] != info["hashes"]:
                raise ValueError("处理前后工程输入哈希不一致")
            self.event(r, "validate", "running", STAGES["validate"][1])
            r["dataset"] = str(dataset)
            try:
                checked = self.app.inspect(r["model"], dataset)
                if (
                    len(checked["designs"]) != 1
                    or checked["designs"][0]["name"] != info["design"]
                ):
                    raise ValueError("生成特征的设计与原始工程不一致")
                if checked["candidates"] > info["candidates"]:
                    raise ValueError("生成候选数超过工程配置的预算")
            except ValueError as e:
                r["status"] = "blocked"
                r["error"] = str(e)
                self.event(
                    r,
                    "validate",
                    "blocked",
                    "特征已生成，但尚不能交给所选完整模型：" + str(e),
                )
            else:
                r["model_inspection"] = checked
                r["status"] = "complete"
                self.event(
                    r,
                    "validate",
                    "complete",
                    "完整特征与模型兼容，可转入预测；本阶段不训练模型",
                )
        except Exception as e:
            r["status"] = "failed"
            r["error"] = str(e)
            self.event(r, r["phase"], "failed", "已停止；保留全部部分产物和日志")
        finally:
            with self.app.lock:
                r["finished_at"] = time.time()
                self.save(r)
                self.active = None

    def log(self, rid):
        r = self.state(rid)
        text = "\n".join(e["time"] + "  " + e["message"] for e in r["events"])
        for stage in STAGES:
            p = self.root / rid / (stage + ".log")
            if p.is_file():
                text += (
                    "\n\n" + stage + ".log\n" + p.read_text(errors="replace")[-12000:]
                )
        return text
