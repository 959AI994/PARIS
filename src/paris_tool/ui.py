"""Local PARIS workspace: input preparation, inference and design refits.

Model bundles own their algorithms and declared input schemas.
The loopback-only service exposes only explicit filesystem roots and writes
independent run directories inside its own workspace.
"""

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import hashlib
import html
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs
import uuid
import zipfile

from . import __version__, models
from .utils import digest, save, verify, project_lock

ORIGINAL = "paris4.original.seed17.1"
SHARED = "paris4.cross-design.1"
SINGLE = "paris.single-xgb.1"
KEY = ["design", "candidate_id"]
STEPS = ["validate", "predict", "export"]


def now():
    return datetime.now(timezone.utc).isoformat()


class App:
    def __init__(
        self, workspace, roots, model=None, dataset=None, preparation_profile=None
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        marker = self.workspace / "UI_WORKSPACE.json"
        if (
            self.workspace.exists()
            and any(self.workspace.iterdir())
            and not marker.exists()
        ):
            raise ValueError(
                "Choose an empty directory or an existing PARIS UI workspace"
            )
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            save(marker, dict(format="paris.ui.v1"))
        (self.workspace / "runs").mkdir(exist_ok=True)
        self.roots = list(dict.fromkeys(Path(p).expanduser().resolve() for p in roots))
        if not self.roots or any(not p.is_dir() for p in self.roots):
            raise ValueError("At least one existing browse root is required")
        if not any(self.workspace.is_relative_to(p) for p in self.roots):
            self.roots.append(self.workspace)
        self.defaults = dict(model=str(model or ""), dataset=str(dataset or ""))
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.inspect_lock = threading.Lock()
        self.jobs = {}
        self.active = None
        self.worker = None
        for p in (self.workspace / "runs").glob("*/RUN.json"):
            try:
                r = json.loads(p.read_text())
                if r["status"] in ["queued", "running"]:
                    r.update(
                        status="interrupted",
                        error="The previous UI session ended. Partial files were retained; this job was not restarted.",
                    )
                    save(p, r)
                self.jobs[r["id"]] = r
            except (ValueError, KeyError, OSError):
                pass
        from .preparation import Preparation

        self.preparation = Preparation(self, preparation_profile)
        self.single_model = False
        if model:
            try:
                self.single_model = verify(models.find_root(model)).get("algorithm") == SINGLE
            except (OSError, ValueError):
                pass
        if self.single_model:
            from .single_iteration import Iteration
        else:
            from .iteration import Iteration

        self.iteration = Iteration(self)

    def path(self, value, directory=False):
        p = Path(value).expanduser().resolve()
        if not any(p.is_relative_to(root) for root in self.roots):
            raise ValueError("该路径不在启动时指定的可访问目录中")
        if directory and not p.is_dir():
            raise ValueError("文件夹不存在：" + str(p))
        if not p.exists():
            raise ValueError("路径不存在：" + str(p))
        return p

    def browse(self, value=None, kind="dataset"):
        if not value:
            return dict(
                path=None,
                parent=None,
                can_select=False,
                entries=[
                    dict(name=str(p), path=str(p), directory=True, hint="可访问目录")
                    for p in self.roots
                ],
            )
        p = self.path(value, directory=True)
        entries = []
        for child in sorted(
            p.iterdir(), key=lambda f: (not f.is_dir(), f.name.lower())
        ):
            if child.name.startswith("."):
                continue
            try:
                resolved = child.resolve()
                if not any(resolved.is_relative_to(root) for root in self.roots):
                    continue
                if child.is_dir():
                    hint = ""
                    if (child / "IMPORT.json").is_file() or (
                        child / "MODEL_CARD.json"
                    ).is_file():
                        hint = "模型目录"
                    elif (child / "features.parquet").is_file():
                        hint = "特征数据目录"
                    elif (child / "PARIS_PROJECT.json").is_file():
                        hint = "原始设计工程"
                    entries.append(
                        dict(
                            name=child.name,
                            path=str(resolved),
                            directory=True,
                            hint=hint,
                        )
                    )
                elif kind in ["model", "iteration-reference"] and (
                    child.name.endswith(".tar.gz") or child.name.endswith(".tgz")
                ):
                    entries.append(
                        dict(
                            name=child.name,
                            path=str(resolved),
                            directory=False,
                            hint="模型包",
                        )
                    )
            except OSError:
                continue
        selectable = (
            (p / "PARIS_PROJECT.json").is_file()
            if kind == "project"
            else (
                (p / "features.parquet").is_file()
                if kind in ["dataset", "training"]
                else (p / "MODEL_CARD.json").is_file() or (p / "IMPORT.json").is_file()
            )
        )
        parent = (
            str(p.parent)
            if p.parent != p and any(p.parent.is_relative_to(r) for r in self.roots)
            else None
        )
        return dict(path=str(p), parent=parent, entries=entries, can_select=selectable)

    def inspect(self, model, dataset):
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq

        with self.inspect_lock:
            source = self.path(model)
            if source.is_file():
                if not (
                    source.name.endswith(".tar.gz") or source.name.endswith(".tgz")
                ):
                    raise ValueError("请选择可信的 PARIS 模型包或模型目录")
                checksum = digest(source)
                imported = self.workspace / "models" / checksum
                if not imported.exists():
                    imported.parent.mkdir(exist_ok=True)
                    models.import_bundle(source, imported, expected=checksum)
                declared = (
                    imported
                    / json.loads((imported / "IMPORT.json").read_text())["root"]
                )
                if not declared.resolve().is_relative_to(imported.resolve()):
                    raise ValueError("模型导入记录不能指向导入目录之外")
                root = models.find_root(imported)
            else:
                declared = source
                if (source / "IMPORT.json").is_file():
                    declared = (
                        source
                        / json.loads((source / "IMPORT.json").read_text())["root"]
                    )
                self.path(declared, directory=True)
                root = models.find_root(source)
            card = verify(root)
            if self.single_model and card.get("algorithm") != SINGLE:
                raise ValueError("请选择PARIS模型")
            if (
                card.get("algorithm")
                not in [ORIGINAL, "paris4.original.design-refit.1", SHARED, SINGLE]
                or card.get("status") != "complete"
            ):
                raise ValueError(
                    "此界面使用完整原版 PARIS4.0 模型；所选模型的 algorithm 必须为 "
                    + ORIGINAL
                )
            dataset = self.path(dataset, directory=True)
            if card.get("algorithm") == SINGLE:
                from .algorithms.single_xgb import features, validate_context

                validate_context(card, dataset)
                for name in ["features.parquet", "source_sink.parquet", "net_load.parquet"]:
                    self.path(dataset / name)
                schema = json.loads((root / "FEATURE_SCHEMA.json").read_text())["columns"]
                frame, _ = features(dataset, schema)
                counts = frame.groupby("design", sort=True).size().to_dict()
                if set(counts) - set(card["designs"]):
                    raise ValueError("请先为新design训练PARIS模型")
                for d, part in frame.groupby("design"):
                    if set(part.candidate_id) & set(card["models"][d]["training_ids"]):
                        raise ValueError("预测候选与模型训练候选重合")
                    if len(part) < 2:
                        raise ValueError("每个design至少需要两个候选")
                return dict(model=str(source), model_root=str(root), model_name=card["name"], algorithm=SINGLE,
                    dataset=str(dataset), candidates=len(frame), designs=[dict(name=d,n=int(n)) for d,n in counts.items()],
                    files=[dict(name=n, rows=pq.ParquetFile(dataset/n).metadata.num_rows,
                                columns=len(pq.ParquetFile(dataset/n).schema_arrow.names), bytes=(dataset/n).stat().st_size)
                           for n in ["features.parquet","source_sink.parquet","net_load.parquet"]],
                    model_sha256=digest(root/"SHA256.json"), output_workspace=str(self.workspace), package_version=__version__)
            if card.get("algorithm") == "paris4.original.design-refit.1":
                context_file = dataset / "FEATURE_CONTEXT.json"
                if not context_file.is_file():
                    raise ValueError(
                        "设计适配权重需要 FEATURE_CONTEXT.json 核对PDK、流程和特征版本"
                    )
                self.path(context_file)
                context = json.loads(context_file.read_text())
                for k in [
                    "pdk_id",
                    "flow_id",
                    "corner",
                    "feature_version",
                    "time_unit",
                ]:
                    if context.get(k) != card.get("context", {}).get(k):
                        raise ValueError("预测输入与适配权重的工艺上下文不一致：" + k)
            details = []
            schemas = {}
            for name in ["features", "source_sink", "net_load"]:
                p = dataset / (name + ".parquet")
                if not p.is_file():
                    raise ValueError("缺少输入文件：" + p.name)
                self.path(p)
                q = pq.ParquetFile(p)
                names = q.schema_arrow.names
                forbidden = [
                    c
                    for c in names
                    if c.startswith(("dr_", "final_", "target_", "l3_"))
                ]
                if forbidden:
                    raise ValueError(
                        p.name + " 含禁止的目标/DR字段：" + ", ".join(forbidden[:5])
                    )
                if not set(KEY) <= set(names):
                    raise ValueError(p.name + " 缺少 design / candidate_id")
                schemas[name] = set(names)
                details.append(
                    dict(
                        name=p.name,
                        rows=q.metadata.num_rows,
                        columns=len(names),
                        bytes=p.stat().st_size,
                    )
                )
            if "grt_l2_tns" not in schemas["features"]:
                raise ValueError("候选特征表缺少 GR TNS：grt_l2_tns")
            f = pd.read_parquet(
                dataset / "features.parquet", columns=KEY + ["grt_l2_tns"]
            )
            if f.empty or f.duplicated(KEY).any() or f[KEY].isna().any().any():
                raise ValueError("候选标识为空或重复")
            if not all(isinstance(v, str) for v in f[KEY].to_numpy().ravel()):
                raise ValueError("design 和 candidate_id 必须为字符串")
            if (
                "grt_l2_tns" not in f
                or not np.isfinite(f.grt_l2_tns).all()
                or (f.grt_l2_tns > 0).any()
            ):
                raise ValueError("GR TNS 缺失或无效；单位应为 ns，值应不大于0")
            counts = f.groupby("design", sort=True).size().to_dict()
            if any(n < 2 for n in counts.values()):
                raise ValueError("每个 design 至少需要两个候选才能进行批次排序")
            if card.get("algorithm") == SHARED:
                if card.get("unknown_design_names_allowed") is not True:
                    raise ValueError("共享模型未声明支持新设计")
                schema = json.loads((root / "INPUT_SCHEMA.json").read_text())
                needed = schema["numeric_columns"]
                missing = set(needed) - schemas["features"]
                if missing:
                    raise ValueError(
                        "共享模型特征缺失：" + ", ".join(sorted(missing)[:8])
                    )
                numeric = pd.read_parquet(dataset / "features.parquet", columns=needed)
                if any(not pd.api.types.is_numeric_dtype(numeric[c]) for c in needed):
                    raise ValueError("共享模型输入包含非数值特征")
                if np.isinf(numeric.to_numpy(float)).any():
                    raise ValueError("共享模型输入包含无穷值")
                return dict(
                    model=str(source),
                    model_root=str(root),
                    model_name=card["name"],
                    algorithm=SHARED,
                    dataset=str(dataset),
                    candidates=len(f),
                    designs=[dict(name=d, n=int(n)) for d, n in counts.items()],
                    files=details,
                    model_sha256=digest(root / "SHA256.json"),
                    output_workspace=str(self.workspace),
                    package_version=__version__,
                )
            if set(counts) - set(card["designs"]):
                raise ValueError(
                    "模型不支持这些 design："
                    + ", ".join(sorted(set(counts) - set(card["designs"])))
                )
            if any(n < 2 for n in counts.values()):
                raise ValueError("每个 design 至少需要两个候选才能进行批次排序")
            wire_features = set()
            for design in counts:
                schema = json.loads(
                    (root / "models" / design / "INPUT_SCHEMA.json").read_text()
                )
                needed = set(schema["candidate_columns"]) - set(
                    c for c in schema["candidate_columns"] if c.startswith("phys_wire_")
                )
                missing = needed - schemas["features"]
                if missing:
                    raise ValueError(
                        "原版完整特征缺失：" + ", ".join(sorted(missing)[:8])
                    )
                wire_features.update(schema["wire_features"])
            net_rc = {"gr_pi_r", "gr_pi_c1", "gr_pi_c2"}
            missing = (wire_features - net_rc - schemas["source_sink"]) | (
                net_rc - schemas["net_load"]
            )
            if missing:
                raise ValueError("GR线网电气特征缺失：" + ", ".join(sorted(missing)))
            for name, needed in [
                (
                    "source_sink",
                    {
                        "net_name",
                        "driver_pin",
                        "sink_pin",
                        "transition",
                        "corner",
                        "gr_slack",
                    },
                ),
                ("net_load", {"net_name", "driver_pin", "corner"}),
            ]:
                if needed - schemas[name]:
                    raise ValueError(name + " 缺少线网连接键或 GR slack")
            return dict(
                model=str(source),
                model_root=str(root),
                model_name=card["name"],
                algorithm=card["algorithm"],
                dataset=str(dataset),
                candidates=len(f),
                designs=[dict(name=d, n=int(n)) for d, n in counts.items()],
                files=details,
                model_sha256=digest(root / "SHA256.json"),
                output_workspace=str(self.workspace),
                package_version=__version__,
            )

    def _save(self, r):
        save(self.workspace / "runs" / r["id"] / "RUN.json", r)

    def _update(self, rid, **kw):
        with self.lock:
            self.jobs[rid].update(**kw)
            self._save(self.jobs[rid])

    def _event(self, rid, step, status, message):
        with self.lock:
            r = self.jobs[rid]
            r["steps"][step] = status
            r["events"].append(dict(time=now(), step=step, message=message))
            r["phase"] = step
            self._save(r)

    def start(self, model, dataset):
        with self.lock:
            if self.active or self.preparation.active or self.iteration.active:
                raise ValueError("已有任务正在运行，请等待完成")
            rid = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                + "-"
                + uuid.uuid4().hex[:8]
            )
            folder = self.workspace / "runs" / rid
            folder.mkdir()
            r = dict(
                id=rid,
                status="queued",
                phase="validate",
                model=str(model),
                dataset=str(dataset),
                created=now(),
                started_at=time.time(),
                finished_at=None,
                steps={s: "pending" for s in STEPS},
                events=[],
                error=None,
                output=str(folder),
            )
            self.jobs[rid] = r
            self.active = rid
            self._save(r)
            self.worker = threading.Thread(
                target=self._run, args=(rid,), name="PARIS-inference", daemon=False
            )
            self.worker.start()
            return dict(r)

    def _run(self, rid):
        import numpy as np
        import pandas as pd

        folder = self.workspace / "runs" / rid
        try:
            self._update(rid, status="running")
            self._event(rid, "validate", "running", "正在检查完整模型与三个 GR 输入表")
            r = self.jobs[rid]
            inspected = self.inspect(r["model"], r["dataset"])
            dataset = Path(inspected["dataset"])
            model = Path(inspected["model_root"])
            inputs = {
                n: digest(dataset / (n + ".parquet"))
                for n in ["features", "source_sink", "net_load"]
            }
            self._update(rid, inspection=inspected, input_hashes=inputs)
            self._event(
                rid,
                "validate",
                "complete",
                f"输入检查通过：{inspected['candidates']} 个候选，{len(inspected['designs'])} 个设计",
            )
            self._event(
                rid,
                "predict",
                "running",
                (
                    "调用PARIS，生成TNS预测与候选排序"
                    if inspected["algorithm"] == SINGLE
                    else "调用 PARIS4.0 跨设计共享模型"
                    if inspected["algorithm"] == SHARED
                    else "调用完整原版 PARIS4.0：GR线网预测、候选修正、阶段共识与端点融合"
                ),
            )
            env = os.environ.copy()
            env.update(
                OPENBLAS_NUM_THREADS="2",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                PYTHONDONTWRITEBYTECODE="1",
            )
            command = [
                sys.executable,
                "-m",
                "paris_tool.cli",
                "predict",
                "--model",
                str(model),
                "--dataset",
                str(dataset),
                "--output",
                str(folder / "ranking.parquet"),
            ]
            with (folder / "cli.log").open("x") as log:
                p = subprocess.Popen(
                    command,
                    cwd=folder,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                )
                self._update(rid, worker_pid=p.pid)
                code = p.wait()
            if code:
                inner = folder / "ranking.inference_logs/predict.log"
                msg = (
                    (
                        inner.read_text(errors="replace")[-12000:]
                        if inner.exists()
                        else ""
                    )
                    + "\n"
                    + (folder / "cli.log").read_text(errors="replace")[-2000:]
                )
                raise RuntimeError(msg.strip())
            self._event(rid, "predict", "complete", "模型推理完成，原始排序分数已保存")
            self._event(rid, "export", "running", "核对候选完整性，整理排序与导出文件")
            f = pd.read_parquet(folder / "ranking.parquet")
            feature = pd.read_parquet(dataset / "features.parquet")
            if (
                len(f) != len(feature)
                or f.duplicated(KEY).any()
                or set(map(tuple, f[KEY].to_numpy()))
                != set(map(tuple, feature[KEY].to_numpy()))
            ):
                raise ValueError("预测结果与输入候选集合不一致")
            if not np.isfinite(f.rank_score).all():
                raise ValueError("模型输出非有限分数")
            if any(digest(dataset / (n + ".parquet")) != h for n, h in inputs.items()):
                raise ValueError("输入在运行期间发生变化；结果保留但不作为成功输出")
            if digest(model / "SHA256.json") != inspected["model_sha256"]:
                raise ValueError("模型签名在运行期间变化")
            display = f.merge(
                feature[KEY + ["grt_l2_tns"]].rename(columns={"grt_l2_tns": "gr_tns"}),
                on=KEY,
                validate="one_to_one",
            )
            display["gr_order"] = (
                display.groupby("design")
                .gr_tns.rank(ascending=False, method="min")
                .astype(int)
            )
            display["rank_change"] = display.gr_order - display.predicted_order
            display["tie_count"] = (
                display.groupby(["design", "rank_score"])
                .candidate_id.transform("size")
                .astype(int)
            )
            display = display.sort_values(
                ["design", "predicted_order", "candidate_id"], kind="stable"
            )
            display.to_parquet(folder / "display.parquet", index=False)
            display.to_csv(folder / "ranking.csv", index=False)
            stats = []
            for design, g in display.groupby("design", sort=True):
                stats.append(
                    dict(
                        name=design,
                        n=len(g),
                        top_ties=int(g.predicted_order.eq(1).sum()),
                    )
                )
            self._update(
                rid,
                designs=stats,
                candidates=len(display),
                prediction_sha256=digest(folder / "ranking.parquet"),
            )
            self._event(
                rid, "export", "complete", "完整排序、CSV、Parquet 与运行记录已生成"
            )
            self._update(rid, status="complete", finished_at=time.time())
        except Exception as e:
            phase = self.jobs[rid]["phase"]
            self._event(rid, phase, "failed", "本次运行停止，部分输出与日志均已保留")
            self._update(rid, status="failed", finished_at=time.time(), error=str(e))
        finally:
            with self.lock:
                self.active = None

    def state(self, rid):
        with self.lock:
            if rid not in self.jobs:
                raise ValueError("运行不存在")
            return json.loads(json.dumps(self.jobs[rid]))

    def results(self, rid, design=None, query="", offset=0, limit=25):
        import pandas as pd

        r = self.state(rid)
        if r["status"] != "complete":
            raise ValueError("结果尚未完成")
        f = pd.read_parquet(self.workspace / "runs" / rid / "display.parquet")
        design = design or r["designs"][0]["name"]
        f = f[f.design.eq(design)]
        if query:
            f = f[f.candidate_id.str.contains(query, case=False, regex=False)]
        total = len(f)
        f = f.iloc[max(0, offset) : max(0, offset) + min(max(1, limit), 200)]
        names = [
            "design",
            "candidate_id",
            "predicted_order",
            "rank_score",
            "predicted_tns",
            "gr_tns",
            "gr_order",
            "rank_change",
            "tie_count",
            "candidate_branch_score",
            "endpoint_branch_score",
        ]
        result = []
        for row in f[[c for c in names if c in f]].to_dict("records"):
            row["score_exact"] = repr(float(row["rank_score"]))
            result.append(row)
        return dict(design=design, total=total, offset=offset, rows=result)

    def log(self, rid):
        r = self.state(rid)
        folder = self.workspace / "runs" / rid
        text = "\n".join(e["time"] + "  " + e["message"] for e in r["events"])
        for n in ["cli.log", "ranking.inference_logs/predict.log"]:
            p = folder / n
            if p.exists():
                tail = p.read_text(errors="replace")[-16000:]
                if tail.strip():
                    text += "\n\n" + n + "\n" + tail
        return text


class Handler(BaseHTTPRequestHandler):
    server_version = "PARIS"

    def log_message(self, *args):
        pass

    @property
    def app(self):
        return self.server.app

    def respond(
        self,
        status,
        data,
        content_type="application/json; charset=utf-8",
        filename=None,
    ):
        body = (
            data
            if isinstance(data, bytes)
            else json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'",
        )
        if filename:
            self.send_header(
                "Content-Disposition", 'attachment; filename="' + filename + '"'
            )
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, api=False):
        host = self.headers.get("Host", "")
        parsed = urlsplit("http://" + host)
        if parsed.hostname not in ["localhost", "127.0.0.1", "::1"]:
            self.respond(403, dict(error="Loopback Host required"))
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + host:
            self.respond(403, dict(error="Origin mismatch"))
            return False
        if api and not secrets.compare_digest(
            self.headers.get("X-PARIS-Token", ""), self.app.token
        ):
            self.respond(403, dict(error="Session token required"))
            return False
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        query = parse_qs(urlsplit(self.path).query)
        if not self.authorized(api=path.startswith("/api/")):
            return
        try:
            if path == "/":
                content = (
                    files("paris_tool").joinpath("resources/ui/index.html").read_text()
                )
                content = content.replace(
                    "__PARIS_TOKEN__", html.escape(self.app.token, quote=True)
                )
                return self.respond(200, content.encode(), "text/html; charset=utf-8")
            if path in ["/app.css", "/app.js"]:
                content = (
                    files("paris_tool")
                    .joinpath("resources/ui/" + path[1:])
                    .read_bytes()
                )
                return self.respond(
                    200,
                    content,
                    (
                        "text/css; charset=utf-8"
                        if path.endswith(".css")
                        else "text/javascript; charset=utf-8"
                    ),
                )
            if path == "/method-diagram.png":
                content = files("paris_tool").joinpath("resources/ui/method-diagram.png").read_bytes()
                return self.respond(200, content, "image/png")
            if path == "/api/config":
                with self.app.lock:
                    runs = sorted(
                        self.app.jobs.values(), key=lambda r: r["created"], reverse=True
                    )[:20]
                return self.respond(
                    200,
                    dict(
                        version=__version__,
                        defaults=self.app.defaults,
                        workspace=str(self.app.workspace),
                        roots=[str(p) for p in self.app.roots],
                        active=self.app.active,
                        runs=runs,
                        preparation=self.app.preparation.config(),
                        iteration=self.app.iteration.config(),
                        model_profile="single-xgb" if self.app.single_model else "legacy",
                    ),
                )
            if path == "/api/benchmark":
                content = files("paris_tool").joinpath("resources/benchmark.json").read_text()
                return self.respond(200, json.loads(content))
            if path == "/api/browse":
                return self.respond(
                    200,
                    self.app.browse(
                        query.get("path", [None])[0], query.get("kind", ["dataset"])[0]
                    ),
                )
            if path == "/api/features":
                from .preparation import catalog

                return self.respond(
                    200,
                    catalog(
                        self.app.path(query.get("dataset", [""])[0], directory=True)
                    ),
                )
            m = re.fullmatch(
                r"/api/iterations/([A-Za-z0-9-]+)(?:/(log|download))?", path
            )
            if m:
                rid, action = m.groups()
                if action == "download":
                    p = self.app.iteration.artifact(
                        rid, query.get("format", ["model"])[0]
                    )
                    return self.respond(
                        200,
                        p.read_bytes(),
                        "application/octet-stream",
                        "paris-" + p.name,
                    )
                return self.respond(
                    200,
                    (
                        dict(text=self.app.iteration.log(rid))
                        if action
                        else self.app.iteration.state(rid)
                    ),
                )
            m = re.fullmatch(r"/api/preparations/([A-Za-z0-9-]+)(?:/(log))?", path)
            if m:
                rid, action = m.groups()
                return self.respond(
                    200,
                    (
                        dict(text=self.app.preparation.log(rid))
                        if action
                        else self.app.preparation.state(rid)
                    ),
                )
            m = re.fullmatch(
                r"/api/runs/([A-Za-z0-9-]+)(?:/(results|log|download))?", path
            )
            if m:
                rid, action = m.groups()
                if not action:
                    return self.respond(200, self.app.state(rid))
                if action == "results":
                    return self.respond(
                        200,
                        self.app.results(
                            rid,
                            query.get("design", [None])[0],
                            query.get("q", [""])[0],
                            int(query.get("offset", [0])[0]),
                            int(query.get("limit", [25])[0]),
                        ),
                    )
                if action == "log":
                    return self.respond(200, dict(text=self.app.log(rid)))
                if action == "download":
                    r = self.app.state(rid)
                    kind = query.get("format", ["csv"])[0]
                    names = {
                        "csv": "ranking.csv",
                        "parquet": "ranking.parquet",
                        "receipt": "RUN.json",
                    }
                    if kind == "log":
                        return self.respond(
                            200,
                            self.app.log(rid).encode(),
                            "text/plain; charset=utf-8",
                            "paris-run.log",
                        )
                    if kind not in names or (
                        kind != "receipt" and r["status"] != "complete"
                    ):
                        raise ValueError("文件尚未生成")
                    p = self.app.workspace / "runs" / rid / names[kind]
                    return self.respond(
                        200,
                        p.read_bytes(),
                        "application/octet-stream",
                        "paris-" + p.name,
                    )
            self.respond(404, dict(error="Not found"))
        except (ValueError, OSError, KeyError) as e:
            self.respond(400, dict(error=str(e)))

    def do_POST(self):
        if not self.authorized(api=True):
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            path = urlsplit(self.path).path
            if path == "/api/projects/upload":
                query = parse_qs(urlsplit(self.path).query)
                return self.respond(
                    201,
                    self.app.preparation.upload(
                        self.rfile, size, query.get("filename", [""])[0]
                    ),
                )
            if size < 1 or size > 32768:
                raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(size))
            path = urlsplit(self.path).path
            if path == "/api/inspect":
                return self.respond(
                    200, self.app.inspect(body["model"], body["dataset"])
                )
            if path == "/api/runs":
                return self.respond(202, self.app.start(body["model"], body["dataset"]))
            if path == "/api/projects/inspect":
                return self.respond(200, self.app.preparation.inspect(body["project"]))
            if path == "/api/preparations":
                return self.respond(
                    202, self.app.preparation.start(body["project"], body["model"])
                )
            if path == "/api/iterations/inspect":
                return self.respond(
                    200,
                    self.app.iteration.inspect(
                        body["dataset"],
                        body["reference"],
                        body.get("design"),
                        body.get("template"),
                    ),
                )
            if path == "/api/iterations":
                return self.respond(
                    202,
                    self.app.iteration.start(
                        body["dataset"],
                        body["reference"],
                        body.get("design"),
                        body.get("template"),
                    ),
                )
            if path == "/api/iterations/activate":
                return self.respond(200, self.app.iteration.activate(body["id"]))
            self.respond(404, dict(error="Not found"))
        except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as e:
            self.respond(400, dict(error=str(e)))


def serve(
    workspace,
    roots,
    model=None,
    dataset=None,
    host="127.0.0.1",
    port=8765,
    open_browser=False,
    preparation_profile=None,
):
    if host not in ["127.0.0.1", "localhost"]:
        raise ValueError(
            "UI binds to loopback only. Use SSH forwarding for a remote machine."
        )
    app = App(workspace, roots, model, dataset, preparation_profile)
    with project_lock(app.workspace):
        server = ThreadingHTTPServer((host, port), Handler)
        server.app = app
        url = f"http://127.0.0.1:{server.server_address[1]}"
        print("PARIS UI " + url, flush=True)
        print(
            "Model and data stay on this machine. Ctrl+C stops the service after any current prediction finishes.",
            flush=True,
        )
        if open_browser:
            import webbrowser

            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            if app.worker and app.worker.is_alive():
                app.worker.join()
            if app.preparation.worker and app.preparation.worker.is_alive():
                app.preparation.worker.join()
            if app.iteration.worker and app.iteration.worker.is_alive():
                app.iteration.worker.join()
    return dict(status="stopped", url=url)
