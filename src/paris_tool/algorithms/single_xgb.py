"""One candidate-level XGBoost per design, with a GR log-cost anchor."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.model_selection import KFold
import xgboost as xgb

from ..candidate_features import aggregate, PARAMS, GRID
from ..utils import fresh, save, seal, verify, digest

ALGORITHM = "paris.single-xgb.1"
KEY = ["design", "candidate_id"]
META = {"cts_odb_size_bytes", "cts_odb_mtime_ns", "grt_odb_size_bytes", "grt_odb_mtime_ns", "feature_schema_version"}
GR_FILES = ["features.parquet", "source_sink.parquet", "net_load.parquet"]
CONTEXT = ["pdk_id", "flow_id", "corner", "feature_version", "time_unit"]


def features(dataset, columns=None, design=None, ids=None):
    root = Path(dataset)
    filters = []
    if design is not None:
        filters.append(("design", "==", design))
    if ids is not None:
        filters.append(("candidate_id", "in", list(ids)))
    tables = [pd.read_parquet(root / name, filters=filters or None) for name in GR_FILES]
    expected = None
    for table in tables:
        if not set(KEY) <= set(table) or table.empty or table[KEY].isna().any().any():
            raise ValueError("GR输入缺少候选键或为空")
        if not all(isinstance(v, str) for v in table[KEY].to_numpy().ravel()):
            raise ValueError("候选ID与design必须是字符串")
        if any(c.startswith(("dr_", "final_", "target_", "l3_", "phys_wire_", "delta_")) for c in table):
            raise ValueError("GR输入包含DR或目标字段")
        keys = set(map(tuple, table[KEY].to_numpy()))
        if expected is None:
            expected = keys
        if keys != expected:
            raise ValueError("三个GR输入表的候选集合不一致")
    frame, source, net = tables
    if frame.duplicated(KEY).any():
        raise ValueError("重复候选ID")
    for col in ["grt_l2_tns", "fast_l1_tns"]:
        if col not in frame or not np.isfinite(frame[col]).all() or (frame[col] > 0).any():
            raise ValueError("GR/CTS TNS必须为非正ns数值")
    combined = frame.merge(aggregate(source, net), on=KEY, validate="one_to_one")
    if columns is None:
        columns = sorted(c for c in combined if c not in KEY + ["candidate_hash"] and c not in META
                         and pd.api.types.is_numeric_dtype(combined[c]))
    missing = set(columns) - set(combined)
    if missing:
        raise ValueError("单XGBoost特征缺失：" + ", ".join(sorted(missing)[:8]))
    if any(not pd.api.types.is_numeric_dtype(combined[c]) for c in columns):
        raise ValueError("模型特征必须为数值")
    if np.isinf(combined[columns].to_numpy(float)).any():
        raise ValueError("模型特征不能含无穷值")
    return combined, list(columns)


def validate_context(card, dataset):
    if not card.get("context"):
        return
    p = Path(dataset) / "FEATURE_CONTEXT.json"
    context = json.loads(p.read_text()) if p.is_file() else {}
    for key in CONTEXT:
        if context.get(key) != card["context"].get(key):
            raise ValueError("模型与GR输入的工艺上下文不一致：" + key)


def infer(root, frame, allow_training=False):
    card = verify(root)
    schema = json.loads((Path(root) / "FEATURE_SCHEMA.json").read_text())["columns"]
    if set(frame.design) - set(card["designs"]):
        raise ValueError("请先为新design训练XGBoost模型")
    results = []
    for design, part in frame.groupby("design", sort=True):
        part = part.sort_values("candidate_id").reset_index(drop=True)
        info = card["models"][design]
        if not allow_training and set(part.candidate_id) & set(info["training_ids"]):
            raise ValueError("预测候选与该模型的训练候选重合")
        model = xgb.XGBRegressor(n_jobs=2, device="cpu")
        model.load_model(Path(root) / info["file"])
        residual = info.get("target_mean", 0.) + info.get("target_scale", 1.) * model.predict(part[schema].to_numpy(np.float32)).astype(float)
        score = np.log1p(-part.grt_l2_tns.to_numpy(float)) + residual
        out = part[KEY].copy()
        out["predicted_log_cost"] = score
        out["predicted_tns"] = -np.expm1(np.clip(score, 0., 50.))
        out["rank_score"] = pd.Series(score).rank(method="average").to_numpy() / len(part)
        out["predicted_order"] = pd.Series(score).rank(method="min").astype(int).to_numpy()
        # Backwards-compatible score key for version reports.
        out["paris4_cost_rank"] = out.rank_score
        results.append(out)
    return pd.concat(results, ignore_index=True)


def fit(dataset, output, design, training_ids, context=None, labels=None):
    if not isinstance(design, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", design) or design in {".", ".."}:
        raise ValueError("无效design")
    root = fresh(output)
    frame, schema = features(dataset, design=design, ids=training_ids)
    if set(frame.candidate_id) != set(training_ids) or len(training_ids) < 24:
        raise ValueError("训练至少需要24个匹配候选")
    y = pd.read_parquet(labels or Path(dataset) / "labels.parquet")[KEY + ["final_tns"]]
    # Training-only snapshots supplied by the iteration worker cannot contain validation labels.
    y = y[y.design.eq(design) & y.candidate_id.isin(training_ids)]
    frame = frame.merge(y, on=KEY, validate="one_to_one").sort_values("candidate_id").reset_index(drop=True)
    if len(frame) != len(training_ids) or not np.isfinite(frame.final_tns).all() or (frame.final_tns > 0).any():
        raise ValueError("训练TNS标签缺失或无效")
    x = frame[schema].to_numpy(np.float32)
    anchor = np.log1p(-frame.grt_l2_tns.to_numpy(float))
    truth = np.log1p(-frame.final_tns.to_numpy(float))
    folds = list(KFold(3, shuffle=True, random_state=17).split(x))
    fold_ids = [dict(training_ids=frame.iloc[tr].candidate_id.tolist(), validation_ids=frame.iloc[va].candidate_id.tolist()) for tr, va in folds]
    save(root / "FOLDS.json", fold_ids)
    rows, choices = [], []
    for j, params in enumerate(GRID):
        for k, (tr, va) in enumerate(folds):
            residual = truth[tr] - anchor[tr]
            mean, scale = float(residual.mean()), max(float(residual.std()), 1e-12)
            m = xgb.XGBRegressor(**PARAMS, **params)
            m.fit(x[tr], (residual - mean) / scale)
            score = anchor[va] + mean + scale * m.predict(x[va]).astype(float)
            tau = float(kendalltau(score, truth[va], variant="b").statistic)
            rows.append(dict(configuration=j, fold=k, tau_b=tau if np.isfinite(tau) else None,
                             log_cost_mae=float(np.abs(score - truth[va]).mean())))
        current = rows[-3:]
        choices.append(dict(configuration=j, tau_b=float(np.mean([z["tau_b"] if z["tau_b"] is not None else 0. for z in current])),
                            log_cost_mae=float(np.mean([z["log_cost_mae"] for z in current]))))
    pick = max(choices, key=lambda z: (round(z["tau_b"], 12), -z["log_cost_mae"], -z["configuration"]))
    residual = truth - anchor
    mean, scale = float(residual.mean()), max(float(residual.std()), 1e-12)
    model = xgb.XGBRegressor(**PARAMS, **GRID[pick["configuration"]])
    model.fit(x, (residual - mean) / scale)
    (root / "models").mkdir()
    model.save_model(root / "models" / (design + ".json"))
    save(root / "FEATURE_SCHEMA.json", dict(columns=schema, count=len(schema)))
    save(root / "TRAINING_PROTOCOL.json", dict(algorithm=ALGORITHM, training_ids=training_ids, params=PARAMS, grid=GRID,
        selection=pick, cv=rows, validation_labels_used=False, context=context, input_hashes={n:digest(Path(dataset) / n) for n in GR_FILES}))
    seal(root, dict(name="PARIS · " + design + " XGBoost", algorithm=ALGORITHM, backend="paris-single-xgb", status="complete",
        designs=[design], context={k:context[k] for k in CONTEXT} if context else None,
        learned_estimators_per_design=1, feature_count=len(schema),
        models={design:dict(file="models/" + design + ".json", training_ids=list(training_ids), target_mean=mean, target_scale=scale)}))
    return verify(root)


class Backend:
    def predict(self, model, dataset, output):
        root, output = Path(model), Path(output)
        if output.exists():
            raise FileExistsError(output)
        card = verify(root)
        validate_context(card, dataset)
        schema = json.loads((root / "FEATURE_SCHEMA.json").read_text())["columns"]
        frame, _ = features(dataset, schema)
        result = infer(root, frame)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix == ".parquet":
            result.to_parquet(output, index=False)
        elif output.suffix == ".csv":
            result.to_csv(output, index=False)
        else:
            raise ValueError("输出必须为CSV或Parquet")
        return dict(status="complete", rows=len(result), output=str(output), algorithm=ALGORITHM)

    def train(self, dataset, output, **options):
        y = pd.read_parquet(Path(dataset) / "labels.parquet")
        if y.design.nunique() != 1:
            raise ValueError("请每次选择一个design训练")
        design = y.design.iloc[0]
        p = Path(dataset) / "PARIS_TRAINING.json"
        card = fit(dataset, output, design, y.candidate_id.tolist(), json.loads(p.read_text()) if p.exists() else None)
        return dict(status="complete", algorithm=ALGORITHM, model=card["name"])
