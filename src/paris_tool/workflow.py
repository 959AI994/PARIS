from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid
import numpy as np
import pandas as pd
from . import config, data, models
from .collection import collect, assemble
from .jobs import label
from .screen import select
from .utils import fresh, save


def used(c):
    p = c["workspace"] / "budget.sqlite"
    if not p.exists():
        return 0
    with sqlite3.connect(p) as db:
        return db.execute("SELECT COUNT(*) FROM routes").fetchone()[0]


def run_workflow(c, budget, model=None, train=False):
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    rows = config.candidates(c)
    mapping = {(r["design"], r["candidate_id"]): r for r in rows}
    collect(c, "all" if any(r.get("dr_odb") for r in rows) else "gr")
    dataset = c["workspace"] / "dataset"
    f = data.features(dataset)
    if data.keys(f) != set(mapping):
        raise ValueError("Manifest and dataset candidates differ")
    if train:
        y = (
            data.labels(dataset)
            if (dataset / "labels.parquet").exists()
            else pd.DataFrame(columns=data.KEY + ["final_tns"])
        )
        needed = []
        for design, g in f.groupby("design"):
            existing = set(y[y.design.eq(design)].candidate_id)
            count = max(0, 24 - len(existing))
            pool = g[~g.candidate_id.isin(existing)].sort_values(
                ["grt_l2_tns", "candidate_id"]
            )
            if len(pool) < count:
                raise ValueError(
                    "Provide at least 24 calibration candidates for " + str(design)
                )
            if count:
                pos = np.linspace(0, len(pool) - 1, count).astype(int)
                needed.extend(
                    mapping[tuple(k)] for k in pool.iloc[pos][data.KEY].to_numpy()
                )
        need_new = sum(
            not r.get("dr_odb") or not Path(r["dr_odb"]).exists() for r in needed
        )
        if need_new + used(c) > budget:
            raise ValueError(
                "Insufficient DR budget for calibration; need "
                + str(need_new)
                + " more executions"
            )
        if needed:
            label(c, needed, budget)
            dataset = assemble(c)
    label_path = dataset / "labels.parquet"
    y = (
        data.labels(dataset)
        if label_path.exists()
        else pd.DataFrame(columns=data.KEY + ["final_tns"])
    )
    stamp = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )
    output = fresh(c["workspace"] / "runs" / stamp)
    if train:
        from .algorithms.base import backend

        model = output / "model"
        backend("paris-single-xgb").train(dataset, model, jobs=c["threads"])
    model = model or c.get("model")
    if not model:
        raise ValueError(
            "Provide --model / config.model, or --train with calibration data"
        )
    remaining = f[
        ~pd.MultiIndex.from_frame(f[data.KEY]).isin(
            pd.MultiIndex.from_frame(y[data.KEY])
        )
    ].copy()
    if len(remaining):
        if (remaining.groupby("design").size() < 2).any():
            raise ValueError(
                "A pending design has only one candidate; label it directly with paris label"
            )
        view = fresh(output / "inference_dataset")
        remaining.to_parquet(view / "features.parquet", index=False)
        for name in ["source_sink", "net_load"]:
            table = pd.read_parquet(dataset / (name + ".parquet")).merge(
                remaining[data.KEY], on=data.KEY, validate="many_to_one"
            )
            table.to_parquet(view / (name + ".parquet"), index=False)
        source_card = json.loads((dataset / "DATASET.json").read_text())
        data.manifest(view, {"feature_context": source_card.get("feature_context")})
        pred = output / "ranking.csv"
        models.predict(model, view, pred)
        ranked = pd.read_csv(pred, dtype={k: str for k in data.KEY})
        if "gr_tns" not in ranked:
            ranked = ranked.merge(
                remaining[data.KEY + ["grt_l2_tns"]].rename(
                    columns={"grt_l2_tns": "gr_tns"}
                ),
                on=data.KEY,
                validate="one_to_one",
            )
        slots = max(0, budget - used(c))
        if slots:
            picked = select(ranked, slots)
            picked.to_csv(output / "selected.csv", index=False)
            selected = [mapping[tuple(k)] for k in picked[data.KEY].to_numpy()]
            if any(not r.get("dr_odb") for r in selected) and not c.get("flow", {}).get(
                "dr_command"
            ):
                save(
                    output / "REPORT.json",
                    dict(
                        status="ranked_waiting_for_dr_hook",
                        model=str(model),
                        output=str(output),
                        remaining_dr_budget=slots,
                        note="Configure flow.dr_command or import DR checkpoints, then use paris label",
                    ),
                )
                return json.loads((output / "REPORT.json").read_text())
            label(c, selected, budget)
            y = data.labels(c["workspace"] / "dataset")
    best = []
    for design, g in y.groupby("design"):
        row = g.sort_values(
            ["final_tns", "candidate_id"], ascending=[False, True]
        ).iloc[0]
        best.append(
            dict(
                design=design,
                candidate_id=row.candidate_id,
                measured_dr_tns=float(row.final_tns),
            )
        )
    report = dict(
        status="complete",
        output=str(output),
        model=str(model),
        new_dr_attempts_total=used(c),
        dr_budget=budget,
        verified_best=best,
        quality_claim="best among actually labeled candidates; no global-optimum or signoff guarantee",
    )
    save(output / "REPORT.json", report)
    return report
