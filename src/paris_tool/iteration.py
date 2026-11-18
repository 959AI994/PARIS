"""Design-specific cumulative refits, fixed development holdout, reversible releases."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import uuid
from . import models
from .utils import digest, save, verify, run
from .iteration_worker import ADAPTED, ORIGINAL

KEY = ["design", "candidate_id"]
FILES = [
    "features.parquet",
    "source_sink.parquet",
    "net_load.parquet",
    "labels.parquet",
    "wire_labels.parquet",
    "PARIS_TRAINING.json",
]
STEPS = {
    "snapshot": "冻结数据与固定划分",
    "train": "重训完整设计子模型",
    "predict": "封存验证预测",
    "evaluate": "验证并比较版本",
}
CONTEXT = [
    "pdk_id",
    "flow_id",
    "corner",
    "label_version",
    "feature_version",
    "time_unit",
    "label_source",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def stable(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def metric(pred, truth):
    import numpy as np
    from scipy.stats import kendalltau

    value = float(kendalltau(pred, -truth, variant="b").statistic)
    best = np.flatnonzero(np.asarray(pred) == np.min(pred))
    regrets = -np.asarray(truth)[best] - np.min(-np.asarray(truth))
    return dict(
        tau_b=value if math.isfinite(value) else None,
        top1_regret_ns=float(regrets.mean()),
        top1_ties=len(best),
        n=len(truth),
    )


def promotion_gate(candidate, comparisons):
    if candidate["n"] < 12 or candidate["tau_b"] is None:
        return False
    return all(
        m["tau_b"] is not None
        and candidate["tau_b"] >= m["tau_b"] - 1e-12
        and candidate["top1_regret_ns"] <= m["top1_regret_ns"] + 1e-9
        for m in comparisons
    )


class Iteration:
    def __init__(self, app):
        self.app = app
        self.root = app.workspace / "iterations"
        self.root.mkdir(exist_ok=True)
        self.jobs, self.active, self.worker = {}, None, None
        for p in self.root.glob("jobs/*/ITERATION.json"):
            try:
                r = json.loads(p.read_text())
                if r["status"] in {"queued", "running"}:
                    r.update(
                        status="interrupted",
                        error="上次服务已结束；保留全部产物，不自动重启。",
                    )
                    for k, v in r["steps"].items():
                        if v == "running":
                            r["steps"][k] = "interrupted"
                    save(p, r)
                self.jobs[r["id"]] = r
            except (ValueError, KeyError, OSError):
                pass

    def reference(self, value):
        source = self.app.path(value)
        if source.is_file():
            checksum = digest(source)
            imported = self.app.workspace / "models" / checksum
            if not imported.exists():
                imported.parent.mkdir(exist_ok=True)
                models.import_bundle(source, imported, expected=checksum)
            source = imported
        if (source / "IMPORT.json").is_file():
            self.app.path(
                source / json.loads((source / "IMPORT.json").read_text())["root"],
                directory=True,
            )
        root = models.find_root(source)
        card = verify(root)
        if (
            card.get("algorithm") != ORIGINAL
            or not (root / "runtime/experiment/v3/models.py").is_file()
        ):
            raise ValueError(
                "参考必须为含完整训练实现的原版 PARIS4.0 模型库；不调用 native 简化配方"
            )
        return root, card

    def inspect(self, dataset, reference, design=None, template=None):
        import numpy as np
        import pandas as pd
        import pyarrow.parquet as pq

        root = self.app.path(dataset, directory=True)
        base, card = self.reference(reference)
        for name in FILES:
            if not (root / name).is_file():
                raise ValueError("训练数据缺少：" + name)
            self.app.path(root / name)
        context = json.loads((root / "PARIS_TRAINING.json").read_text())
        if not isinstance(context, dict):
            raise ValueError("PARIS_TRAINING.json 必须为JSON对象")
        if context.get("version") != 1 or any(
            not isinstance(context.get(k), str) or not context[k].strip()
            for k in CONTEXT
        ):
            raise ValueError(
                "PARIS_TRAINING.json 需要 version:1、PDK/flow/corner/标签与特征版本、单位和标签来源"
            )
        if context["time_unit"] != "ns" or context["label_source"] != "measured_dr_sta":
            raise ValueError("只接受单位ns的实测 DR STA 标签，不能使用模型伪标签")
        schemas, files = {}, []
        for name in FILES[:-1]:
            q = pq.ParquetFile(root / name)
            schemas[name] = set(q.schema_arrow.names)
            if not set(KEY) <= schemas[name]:
                raise ValueError(name + " 缺少候选键")
            if name in FILES[:3] and any(
                c.startswith(("dr_", "final_", "target_", "l3_", "delta_"))
                for c in schemas[name]
            ):
                raise ValueError("GR特征不能混入DR或目标字段")
            files.append(
                dict(name=name, rows=q.metadata.num_rows, columns=len(schemas[name]))
            )
        f = pd.read_parquet(root / "features.parquet")
        y = pd.read_parquet(root / "labels.parquet")[KEY + ["final_tns"]]
        for c in ["fast_l1_tns", "grt_l2_tns", "ep_gr_log_cost"]:
            if c in f and not np.isfinite(f[c]).all():
                raise ValueError("缺失或非有限关键GR/CTS字段：" + c)
        for c in ["fast_l1_tns", "grt_l2_tns"]:
            if c in f and (f[c] > 0).any():
                raise ValueError("GR/CTS TNS必须是非正ns数值：" + c)
        for frame in [f, y]:
            if (
                frame.empty
                or frame.duplicated(KEY).any()
                or frame[KEY].isna().any().any()
                or not all(isinstance(v, str) for v in frame[KEY].to_numpy().ravel())
            ):
                raise ValueError("候选ID必须是唯一非空字符串")
        if not np.isfinite(y.final_tns).all() or (y.final_tns > 0).any():
            raise ValueError("无效的实测TNS标签")
        if set(map(tuple, y[KEY].to_numpy())) - set(map(tuple, f[KEY].to_numpy())):
            raise ValueError("标签候选缺少GR特征")
        counts = y.groupby("design", sort=True).size().to_dict()
        design = design or next(iter(counts))
        if (
            design not in counts
            or not re.fullmatch(r"[A-Za-z0-9_.-]+", design)
            or design in {".", ".."}
        ):
            raise ValueError("请选择有效的design")
        template = template or (
            design if design in card["designs"] else sorted(card["designs"])[0]
        )
        if template not in card["designs"]:
            raise ValueError("参考配方不存在")
        schema = json.loads(
            (base / "models" / template / "INPUT_SCHEMA.json").read_text()
        )
        needed = {
            c for c in schema["candidate_columns"] if not c.startswith("phys_wire_")
        }
        if needed - schemas["features.parquet"]:
            raise ValueError(
                "缺少完整原版候选特征："
                + ", ".join(sorted(needed - schemas["features.parquet"])[:8])
            )
        needed_wire = set(schema["wire_features"]) - {"gr_pi_r", "gr_pi_c1", "gr_pi_c2"}
        if (
            needed_wire - schemas["source_sink.parquet"]
            or {"gr_pi_r", "gr_pi_c1", "gr_pi_c2"} - schemas["net_load.parquet"]
        ):
            raise ValueError("缺少完整GR线网电气特征")
        pair = KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"]
        if (
            set(pair) - schemas["source_sink.parquet"]
            or set(KEY + ["net_name", "driver_pin", "corner"])
            - schemas["net_load.parquet"]
        ):
            raise ValueError("GR线网连接键不完整")
        if set(pair + ["dr_wire_delay"]) - schemas["wire_labels.parquet"]:
            raise ValueError("线网标签需要GR连接键和实测 dr_wire_delay")
        for name in ["source_sink.parquet", "net_load.parquet", "wire_labels.parquet"]:
            corners = pd.read_parquet(
                root / name,
                columns=["corner"],
                filters=[("design", "==", design)],
                dtype_backend="pyarrow",
            ).corner
            if corners.isna().any() or set(corners.unique()) != {context["corner"]}:
                raise ValueError(name + " 的实际角点与数据声明不一致")
        labels = y[y.design.eq(design)].sort_values("candidate_id")
        identities = labels.candidate_id.tolist()
        base_sha = digest(base / "SHA256.json")
        purpose = context.get("purpose", "development")
        if purpose not in ["development", "engineering_smoke"]:
            raise ValueError("purpose 必须为 development 或 engineering_smoke")
        line = stable(
            dict(
                design=design,
                purpose=purpose,
                context={k: context[k] for k in CONTEXT},
                template=template,
                reference_sha256=base_sha,
            )
        )[:16]
        protocol_path = self.root / "lines" / line / "PROTOCOL.json"
        prior = (
            json.loads(protocol_path.read_text()) if protocol_path.is_file() else None
        )
        validation = (
            prior["validation_ids"]
            if prior
            else sorted(identities, key=lambda i: stable([design, i, 17]))[
                : max(12, math.ceil(len(identities) * 0.2))
            ]
        )
        train = sorted(set(identities) - set(validation))
        blockers = []
        if set(validation) - set(identities):
            blockers.append("固定验证候选不能从新样本集中消失")
        if prior and set(prior["training_ids"]) - set(train):
            blockers.append("请选择包含旧训练样本和新增样本的累计数据集")
        if len(train) < 24 or len(validation) < 12:
            blockers.append("至少需要36个有标签候选（训练≥24，固定验证≥12）")
        values = {
            r.candidate_id: stable(float(r.final_tns)) for r in labels.itertuples()
        }
        if prior and any(values.get(k) != v for k, v in prior["label_hashes"].items()):
            blockers.append("已有标签发生变化；需使用新的 label_version 建立新协议")
        return dict(
            dataset=str(root),
            reference=str(base),
            reference_sha256=base_sha,
            reference_name=card["name"],
            design=design,
            template=template,
            templates=card["designs"],
            designs=[dict(name=k, n=int(v)) for k, v in counts.items()],
            line=line,
            context={k: context[k] for k in CONTEXT},
            purpose=purpose,
            files=files,
            labeled=len(identities),
            training=len(train),
            validation=len(validation),
            training_ids=train,
            validation_ids=validation,
            label_hashes=values,
            new_training=(
                len(set(train) - set(prior["training_ids"])) if prior else len(train)
            ),
            protocol_exists=bool(prior),
            ready=not blockers,
            blockers=blockers,
        )

    def config(self):
        with self.app.lock:
            lines = []
            for p in (self.root / "lines").glob("*/PROTOCOL.json"):
                v = json.loads(p.read_text())
                current = p.with_name("CURRENT.json")
                lines.append(
                    dict(
                        line=p.parent.name,
                        design=v["design"],
                        context=v["context"],
                        validation=len(v["validation_ids"]),
                        current=(
                            json.loads(current.read_text())
                            if current.is_file()
                            else None
                        ),
                    )
                )
            return dict(
                active=self.active,
                runs=[
                    self.state(r["id"])
                    for r in sorted(
                        self.jobs.values(), key=lambda r: r["created"], reverse=True
                    )[:30]
                ],
                lines=lines,
                steps=[dict(id=k, title=v) for k, v in STEPS.items()],
            )

    def save(self, r):
        save(self.root / "jobs" / r["id"] / "ITERATION.json", r)

    def event(self, r, phase, status, message):
        with self.app.lock:
            r["phase"] = phase
            r["steps"][phase] = status
            r["events"].append(dict(time=now(), message=message))
            self.save(r)

    def state(self, rid):
        with self.app.lock:
            if rid not in self.jobs:
                raise ValueError("迭代版本不存在")
            r = json.loads(json.dumps(self.jobs[rid]))
        p = self.root / "jobs" / rid / "TRAIN_STATUS.json"
        if p.is_file():
            try:
                r["training_progress"] = json.loads(p.read_text())
            except ValueError:
                pass
        return r

    def start(self, dataset, reference, design=None, template=None):
        info = self.inspect(dataset, reference, design, template)
        if not info["ready"]:
            raise ValueError("；".join(info["blockers"]))
        with self.app.lock:
            if self.active or self.app.active or self.app.preparation.active:
                raise ValueError("已有任务正在运行")
            rid = (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                + "-"
                + uuid.uuid4().hex[:8]
            )
            folder = self.root / "jobs" / rid
            folder.mkdir(parents=True)
            r = dict(
                id=rid,
                created=now(),
                started_at=time.time(),
                finished_at=None,
                status="queued",
                phase="snapshot",
                steps={k: "pending" for k in STEPS},
                events=[],
                inspection=info,
                output=str(folder),
                model=None,
                error=None,
                eligible=False,
                metrics=None,
            )
            self.jobs[rid], self.active = r, rid
            self.save(r)
            self.worker = threading.Thread(
                target=self.work, args=(rid,), daemon=False, name="PARIS-design-refit"
            )
            self.worker.start()
            return self.state(rid)

    def work(self, rid):
        import numpy as np
        import pandas as pd
        from .iteration_worker import gr_tables

        r = self.jobs[rid]
        out, info = self.root / "jobs" / rid, r["inspection"]
        dataset = Path(info["dataset"])
        try:
            r["status"] = "running"
            self.event(r, "snapshot", "running", "核验累计样本；固定验证候选不进入训练")
            if shutil.disk_usage(out).free < 60 * 1024**3:
                raise ValueError("可用空间低于60 GiB，停止新增训练")
            hashes = {n: digest(dataset / n) for n in FILES}
            fresh = self.inspect(
                info["dataset"], info["reference"], info["design"], info["template"]
            )
            for k in [
                "training_ids",
                "validation_ids",
                "context",
                "label_hashes",
                "reference_sha256",
            ]:
                if fresh[k] != info[k]:
                    raise ValueError("检查后、冻结前输入发生变化：" + k)
            line = self.root / "lines" / info["line"]
            line.mkdir(parents=True, exist_ok=True)
            protocol_path = line / "PROTOCOL.json"
            protocol = (
                json.loads(protocol_path.read_text())
                if protocol_path.exists()
                else dict(
                    design=info["design"],
                    context=info["context"],
                    validation_ids=info["validation_ids"],
                    training_ids=[],
                    label_hashes={},
                    validation_scope="reused development holdout; NOT independent test",
                )
            )
            gr = gr_tables(dataset, info["design"], info["validation_ids"])
            signatures = {}
            sort_keys = [
                KEY,
                KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"],
                KEY + ["net_name", "driver_pin", "corner"],
            ]
            for name, frame, keys in zip(FILES[:3], gr, sort_keys):
                for cid, part in frame.groupby("candidate_id", sort=True):
                    a = (
                        pd.util.hash_pandas_object(
                            part.sort_values(keys).reset_index(drop=True), index=False
                        )
                        .to_numpy()
                        .tobytes()
                    )
                    signatures[name + ":" + str(cid)] = hashlib.sha256(a).hexdigest()
            if (
                protocol.get("validation_gr_hashes")
                and signatures != protocol["validation_gr_hashes"]
            ):
                raise ValueError(
                    "固定验证候选的GR输入发生变化，请使用新的 feature_version/flow_id"
                )
            protocol.update(
                training_ids=info["training_ids"],
                label_hashes=info["label_hashes"],
                validation_gr_hashes=signatures,
            )
            save(protocol_path, protocol)
            filters = [
                ("design", "==", info["design"]),
                ("candidate_id", "in", info["training_ids"]),
            ]
            pd.read_parquet(dataset / "labels.parquet", filters=filters)[
                KEY + ["final_tns"]
            ].to_parquet(out / "training_labels.parquet", index=False)
            w = pd.read_parquet(
                dataset / "wire_labels.parquet",
                filters=filters,
                dtype_backend="pyarrow",
            )
            pair = KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"]
            if (
                w.empty
                or w.duplicated(pair).any()
                or not np.isfinite(w.dr_wire_delay.to_numpy(float)).all()
                or (w.dr_wire_delay < 0).any()
            ):
                raise ValueError("无效或重复的训练线网标签")
            w.to_parquet(out / "training_wire_labels.parquet", index=False)
            manifest = {
                k: info[k]
                for k in [
                    "dataset",
                    "design",
                    "template",
                    "reference",
                    "reference_sha256",
                    "training_ids",
                    "validation_ids",
                    "line",
                ]
            }
            manifest.update(
                id=rid,
                purpose=info["purpose"],
                training_labels=str(out / "training_labels.parquet"),
                training_wire_labels=str(out / "training_wire_labels.parquet"),
            )
            manifest["input_hashes"] = hashes
            manifest["context"] = info["context"]
            manifest["training_labels_sha256"] = digest(out / "training_labels.parquet")
            manifest["training_wire_labels_sha256"] = digest(
                out / "training_wire_labels.parquet"
            )
            save(out / "TRAIN.json", manifest)
            r["input_hashes"] = hashes
            self.event(
                r,
                "snapshot",
                "complete",
                f"冻结 {info['training']} 训练候选与 {info['validation']} 固定验证候选",
            )
            self.event(
                r,
                "train",
                "running",
                "原版完整架构按参考配方重训；2CPU，不使用验证标签",
            )
            env = os.environ.copy()
            env.update(
                PYTHONDONTWRITEBYTECODE="1",
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                NUMEXPR_NUM_THREADS="2",
            )
            cmd = [sys.executable, "-m", "paris_tool.iteration_worker"]
            run(
                cmd
                + ["fit", "--manifest", out / "TRAIN.json", "--output", out / "model"],
                out,
                env,
                86400,
                "fit",
            )
            model_hash = digest(out / "model/SHA256.json")
            r.update(model=str(out / "model"), model_sha256=model_hash)
            self.event(r, "train", "complete", "新权重已封存，验证标签未参与训练")
            self.event(r, "predict", "running", "仅读取固定验证候选GR，先封存预测")
            run(
                cmd
                + [
                    "predict",
                    "--manifest",
                    out / "TRAIN.json",
                    "--bundle",
                    out / "model",
                    "--output",
                    out / "validation_predictions.parquet",
                ],
                out,
                env,
                3600,
                "predict",
            )
            pred_hash = digest(out / "validation_predictions.parquet")
            r["prediction_sha256"] = pred_hash
            current = line / "CURRENT.json"
            parent = json.loads(current.read_text()) if current.exists() else None
            baseline = Path(parent["model"]) if parent else Path(info["reference"])
            run(
                cmd
                + [
                    "predict",
                    "--baseline",
                    "--manifest",
                    out / "TRAIN.json",
                    "--bundle",
                    baseline,
                    "--output",
                    out / "baseline_predictions.parquet",
                ],
                out,
                env,
                3600,
                "baseline",
            )
            self.event(r, "predict", "complete", "预测和当前版本对照已封存，之后才评分")
            self.event(
                r, "evaluate", "running", "同候选比较 Kendall τ-b 与 Top-1 真值遗憾"
            )
            truth = pd.read_parquet(
                dataset / "labels.parquet",
                filters=[
                    ("design", "==", info["design"]),
                    ("candidate_id", "in", info["validation_ids"]),
                ],
            )[KEY + ["final_tns"]]
            pred = pd.read_parquet(out / "validation_predictions.parquet")
            if (
                len(pred) != info["validation"]
                or pred.duplicated(KEY).any()
                or not np.isfinite(pred.paris4_cost_rank).all()
            ):
                raise ValueError("验证预测不完整")
            evaluation = truth.merge(pred, on=KEY, validate="one_to_one")
            if len(evaluation) != len(truth):
                raise ValueError("验证候选不匹配")
            evaluation = evaluation.merge(
                gr[0][KEY + ["grt_l2_tns"]], on=KEY, validate="one_to_one"
            )
            target = evaluation.final_tns.to_numpy(float)
            metrics = {
                "GR": metric(-evaluation.grt_l2_tns.to_numpy(float), target),
                "candidate": metric(
                    evaluation.paris4_cost_rank.to_numpy(float), target
                ),
            }
            if (out / "baseline_predictions.parquet").exists():
                old = pd.read_parquet(out / "baseline_predictions.parquet")[
                    KEY + ["paris4_cost_rank"]
                ].rename(columns={"paris4_cost_rank": "current_score"})
                evaluation = evaluation.merge(old, on=KEY, validate="one_to_one")
                if len(evaluation) != info["validation"]:
                    raise ValueError("当前版本对照候选不匹配")
                metrics["current"] = metric(
                    evaluation.current_score.to_numpy(float),
                    evaluation.final_tns.to_numpy(float),
                )
            else:
                if parent:
                    raise ValueError(
                        "当前版本不能在固定验证集上比较，停止本次验证而非退回GR门禁"
                    )
                r["baseline_note"] = json.loads(
                    (out / "baseline_predictions.skip.json").read_text()
                )["reason"]
            eligible = promotion_gate(
                metrics["candidate"],
                [metrics[k] for k in ["GR", "current"] if k in metrics],
            )
            if info["purpose"] == "engineering_smoke":
                eligible = False
                r["deployment_note"] = "工程冒烟验证版本，不允许作为部署模型启用"
            for n, h in hashes.items():
                if digest(dataset / n) != h:
                    raise ValueError("训练期间数据哈希变化：" + n)
            if (
                digest(out / "model/SHA256.json") != model_hash
                or digest(out / "validation_predictions.parquet") != pred_hash
            ):
                raise ValueError("模型/预测封存哈希变化")
            evaluation.to_csv(out / "VALIDATION_CANDIDATES.csv", index=False)
            save(
                out / "VALIDATION.json",
                dict(
                    metrics=metrics,
                    eligible=eligible,
                    validation_ids=info["validation_ids"],
                    scope=protocol["validation_scope"],
                    prediction_sha256=pred_hash,
                    model_sha256=model_hash,
                    gate="tau-b and mean tied-top1 regret no worse than GR and comparable current model",
                    predicted_tns_mae=None,
                    note="rank scores are not TNS; no TNS MAE claimed",
                ),
            )
            r.update(
                metrics=metrics,
                eligible=eligible,
                parent=parent,
                validation_sha256=digest(out / "VALIDATION.json"),
            )
            self.event(
                r,
                "evaluate",
                "complete",
                (
                    "工程流程验证完成；此版本不作部署或独立精度主张"
                    if info["purpose"] == "engineering_smoke"
                    else (
                        "验证通过，可手动启用；当前模型未改变"
                        if eligible
                        else "未通过启用条件，保留负结果与当前模型"
                    )
                ),
            )
            r["status"] = "complete"
        except Exception as e:
            r.update(status="failed", error=str(e))
            self.event(r, r["phase"], "failed", "停止任务；保留日志和全部部分输出")
        finally:
            with self.app.lock:
                r["finished_at"] = time.time()
                self.save(r)
                self.active = None

    def activate(self, rid):
        with self.app.lock:
            if self.active or self.app.active or self.app.preparation.active:
                raise ValueError("等待当前任务完成后启用/回退")
            r = self.state(rid)
            if r["status"] != "complete" or not r.get("eligible"):
                raise ValueError("只有验证通过的版本可启用；不能绕过验证")
            out = Path(r["output"])
            if (
                digest(out / "VALIDATION.json") != r["validation_sha256"]
                or digest(out / "model/SHA256.json") != r["model_sha256"]
            ):
                raise ValueError("模型或验证回执已变化")
            verify(out / "model")
            line = self.root / "lines" / r["inspection"]["line"]
            current = line / "CURRENT.json"
            old = json.loads(current.read_text()) if current.exists() else None
            new = dict(
                id=rid,
                model=str(out / "model"),
                model_sha256=r["model_sha256"],
                activated_at=now(),
                design=r["inspection"]["design"],
            )
            save(
                line / "activations" / (uuid.uuid4().hex + ".json"),
                dict(previous=old, current=new),
            )
            save(current, new)
            return new

    def log(self, rid):
        r = self.state(rid)
        out = Path(r["output"])
        text = "\n".join(e["time"] + "  " + e["message"] for e in r["events"])
        for n in ["fit.log", "predict.log", "baseline.log"]:
            if (out / n).exists():
                text += (
                    "\n\n" + n + "\n" + (out / n).read_text(errors="replace")[-16000:]
                )
        return text

    def artifact(self, rid, kind):
        r = self.state(rid)
        if r["status"] != "complete":
            raise ValueError("版本尚未完成")
        folder = self.root / "jobs" / rid
        if kind == "model":
            if not r["eligible"]:
                raise ValueError("未通过验证的版本不能作为部署权重导出")
            verify(folder / "model")
            if digest(folder / "VALIDATION.json") != r["validation_sha256"]:
                raise ValueError("验证回执已变化")
            p = folder / "model.tar.gz"
            with self.app.lock:
                if not p.exists():
                    models.export(folder / "model", p)
            return p
        names = {
            "validation": "VALIDATION.json",
            "candidates": "VALIDATION_CANDIDATES.csv",
        }
        if kind not in names:
            raise ValueError("未知导出类型")
        return folder / names[kind]
