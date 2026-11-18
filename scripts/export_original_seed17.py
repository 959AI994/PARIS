"""Restore the COMPLETE historical PARIS4 seed-17 bank, never retune it.

Original candidate/wire weights are copied byte for byte. Missing endpoint and
ariane136 augmentation weights are recovered using original historical rows,
original selected parameters and original code. Target labels are never read.
The restored predictor must pass per-candidate reference parity separately.
"""

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import hashlib
import time


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def main(a):
    import numpy as np
    import pandas as pd
    import joblib
    import pyarrow.parquet as pq
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    run, out = a.run.resolve(), a.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    bundle = out / "bundle"
    bundle.mkdir()
    runtime = bundle / "runtime"
    runtime.mkdir()
    for name in ["v2", "v3"]:
        shutil.copytree(
            a.experiment / name,
            runtime / "experiment" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copytree(
        a.experiment.parents[1] / "src/paris",
        runtime / "paris",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        a.runtime_bundle / "runtime/timing_predict",
        runtime / "timing_predict",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(
        a.tool_source / "src/paris_tool/original_runtime.py", bundle / "predict.py"
    )
    sys.path[:0] = [str(runtime), str(runtime / "experiment")]
    from v2.models import baseline_fit
    from v3.models import (
        CorrectedPARIS,
        Correction,
        baseline_oof,
        base_frame,
        correction_features,
    )
    from paris.ranking.common import tns_cost

    key = ["design", "candidate_id"]
    U = run / "datasets/paris_1204_unified_20261012_v1"
    source = run / "datasets/seed17_all_1204_20261008_v1/merged"
    metadata = json.loads(
        (run / "five_split_wire_20261009/current/seed_17/manifest.json").read_text()
    )
    stages = {
        x["design"]: x
        for x in json.loads(
            (
                run
                / "advanced_wire_20261009/score_stack_blend_20261009/safe_score_stack_freeze.json"
            ).read_text()
        )["choices"]
        if x["seed"] == 17
    }
    guarded_path = (
        run / "paris4_history_guard_20261012/final/PARIS4_GUARDED_PREDICTIONS.parquet"
    )
    guard = json.loads(guarded_path.with_name("FREEZE.json").read_text())
    assert sha(guarded_path) == guard["prediction_sha256"]
    fusion = pd.read_parquet(guarded_path).query("seed==17")
    features = pd.read_parquet(U / "features.parquet")
    wiremeta = json.loads(
        (
            run / "models/seed17_wire_features_20261008_v1/wire_feature_manifest.json"
        ).read_text()
    )
    wiremodels = {r["design"]: r for r in wiremeta["models"]}
    inputs = {}

    def remember(p):
        p = Path(p)
        inputs[str(p)] = sha(p)
        return p

    for p in [
        guarded_path,
        U / "features.parquet",
        run / "five_split_wire_20261009/current/seed_17/manifest.json",
    ]:
        remember(p)
    target_parts = []
    counts = {}
    for j in metadata["jobs"]:
        design = j["design"]
        print("RESTORE", design, flush=True)
        folder = bundle / "models" / design
        folder.mkdir(parents=True)
        old = Path(j["history_features"])
        truth_path = Path(j["history_truth"])
        target_path = Path(j["target_features"])
        for p, k in [
            (old, "history_features_sha256"),
            (truth_path, "history_truth_sha256"),
            (target_path, "target_features_sha256"),
        ]:
            assert sha(p) == j[k], (design, k)
            remember(p)
        hist = pd.read_parquet(old)
        hy = pd.read_parquet(truth_path)[key + ["final_tns"]]
        target = pd.read_parquet(target_path)
        assert not set(hist.candidate_id) & set(target.candidate_id)
        core_path = (
            run
            / f"five_split_wire_20261009/current/seed_17/predictions/17/{design}/20260929_v3_model.joblib"
        )
        remember(core_path)
        core = joblib.load(core_path)
        assert set(core.history_keys.candidate_id) == set(hist.candidate_id)
        shutil.copy2(core_path, folder / "original_core.joblib")
        wirepath = Path(wiremodels[design]["path"])
        assert sha(wirepath) == wiremodels[design]["sha256"]
        remember(wirepath)
        shutil.copy2(wirepath, folder / "wire_delay_residual.json")
        augmented = None
        augmented_ids = []
        if design == "ariane136":
            prepared = run / "ariane136_augmented_history_20261009/prepared"
            extra = pd.read_parquet(remember(prepared / "augmented_features.parquet"))
            extra_y = pd.read_parquet(remember(prepared / "augmented_truth.parquet"))[
                key + ["final_tns"]
            ]
            assert len(extra) == 32 and not set(extra.candidate_id) & set(
                target.candidate_id
            )
            columns = list(dict.fromkeys([*hist.columns, *extra.columns]))
            ah = pd.concat(
                [hist.reindex(columns=columns), extra.reindex(columns=columns)],
                ignore_index=True,
            )
            ay = pd.concat([hy, extra_y], ignore_index=True)
            auditfile = remember(
                run / "ariane136_augmented_v3_20261009/final/augmented_v3_freeze.json"
            )
            audit = next(
                x for x in json.loads(auditfile.read_text())["jobs"] if x["seed"] == 17
            )["v3_audit"]
            chosen = audit["branches"]["legacy"]["chosen"]
            w = float(chosen["weight"])
            prior = baseline_fit(base_frame(ah), ay)
            correction = None
            if w:
                oof = baseline_oof(ah, ay, n=5)
                correction = Correction.fit(
                    correction_features(ah, "legacy"),
                    tns_cost(ay.final_tns),
                    oof,
                    tuple(chosen["spec"]),
                )
            augmented = CorrectedPARIS(
                prior, {"legacy": (correction, w)}, None, audit, ah[key].copy()
            )
            augmented_ids = extra.candidate_id.tolist()
            epath = remember(
                run / "endpoint_summary_20261011/candidate_features.parquet"
            )
            raw_ep = pd.read_parquet(epath)
            family = next(
                x
                for x in json.loads(
                    remember(run / "endpoint_summary_20261011/freeze.json").read_text()
                )["decisions"]
                if x["seed"] == 17
            )["selected_model"]
            ey = pd.concat(
                [
                    hy[["candidate_id", "final_tns"]],
                    extra_y[["candidate_id", "final_tns"]],
                ],
                ignore_index=True,
            )
        else:
            epath = remember(
                run / f"endpoint_summary_others_20261011/features/{design}.parquet"
            )
            raw_ep = pd.read_parquet(epath)
            decision = remember(
                run
                / f"endpoint_summary_others_20261011/jobs/seed_17/{design}/complete.json"
            )
            info = json.loads(decision.read_text())
            assert sha(epath) == info["feature_sha256"]
            family = info["selected_model"]
            ey = hy[["candidate_id", "final_tns"]]
        epcols = [c for c in raw_ep if c != "candidate_id"]
        assert len(epcols) == 104
        et = raw_ep.merge(ey, on="candidate_id", validate="one_to_one")
        assert len(et) == len(hy) + len(augmented_ids)
        if family.startswith("ridge_"):
            endpoint = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=float(family.split("_")[1])),
            )
        elif family == "xgb_shallow":
            endpoint = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                XGBRegressor(
                    n_estimators=200,
                    max_depth=2,
                    learning_rate=0.03,
                    min_child_weight=8,
                    reg_lambda=20,
                    subsample=1,
                    colsample_bytree=1,
                    tree_method="hist",
                    n_jobs=4,
                    random_state=17,
                ),
            )
        else:
            raise ValueError(family)
        endpoint.fit(
            et[epcols].to_numpy(float),
            np.log1p(-et.final_tns.to_numpy(float)) - et.gr_log_cost.to_numpy(float),
        )
        # Target inputs contain original GR descriptors and GR endpoint summaries.
        epnames = ["ep_" + c for c in epcols]
        f = target.drop(columns=[c for c in target if c.startswith("ep_")]).merge(
            features[key + epnames], on=key, validate="one_to_one"
        )
        assert len(f) == j["target_n"]
        columns = [c for c in target if c.startswith("phys_wire_")]
        assert len(columns) == 19
        weight = fusion.loc[fusion.design.eq(design), "paris4_weight"].unique()
        assert len(weight) == 1
        record = dict(
            core=core,
            augmented=augmented,
            endpoint=endpoint,
            ep_columns=epnames,
            endpoint_weight=float(weight[0]),
            stage=stages[design],
            wire_columns=columns,
            required_columns=list(f.columns),
            training_ids=hist.candidate_id.tolist() + augmented_ids,
            endpoint_family=family,
            original_core_sha256=sha(core_path),
            original_wire_sha256=sha(wirepath),
        )
        joblib.dump(record, folder / "ranker.joblib", compress=3)
        save(
            folder / "INPUT_SCHEMA.json",
            dict(
                candidate_columns=[c for c in f if not c.startswith("phys_wire_")],
                wire_features=wiremeta["features"],
                endpoint_columns=epnames,
                rank_direction="ascending",
                training_candidates=len(record["training_ids"]),
                target_candidates=j["target_n"],
            ),
        )
        save(
            folder / "RESTORE.json",
            dict(
                status="complete",
                candidate_model="original serialized weight",
                wire_model="original serialized weight",
                endpoint_model="original selected family; recovered using original historical labels only",
                augmentation=(
                    "original selected recipe; recovered using original 88+32 historical rows"
                    if augmented is not None
                    else None
                ),
                endpoint_weight=float(weight[0]),
                endpoint_family=family,
                stage_consensus=stages[design]["stage_consensus"],
            ),
        )
        target_parts.append(
            f.drop(columns=[c for c in f if c.startswith("phys_wire_")])
        )
        counts[design] = len(f)
        gc.collect()
    card = dict(
        name="PARIS4-original-seed17",
        algorithm="paris4.original.seed17.1",
        status="complete",
        designs=list(counts),
        seed=17,
        training_candidates=719,
        extra_ariane136_candidates=32,
        evaluation_target_candidates=485,
        scope="original same-design seed-17 bank, not all-data refit or simplified native recipe",
        input_contract="original candidate descriptors + 104 GR endpoint summaries + GR-only source_sink/net_load",
        output="batch-relative rank score, smaller is better; not physical DR TNS",
        fusion="original per-design stage/augmentation/endpoint decisions, including original small-history guard",
        limitations=[
            "Previously explored seed-17 target pool; not independent validation.",
            "Historical candidate banks retain the original historical label version; evaluation labels may be v2_flowlibs.",
            "Original wire sampling used GR/DR-connectivity eligibility; GR-only deployment distribution needs validation.",
        ],
        dependencies=dict(python=sys.version),
    )
    save(bundle / "MODEL_CARD.json", card)
    save(
        bundle / "INPUT_SCHEMA.json",
        dict(
            designs=counts,
            profile="original-full-paris4-seed17",
            per_design="models/<design>/INPUT_SCHEMA.json",
            prediction_target_labels=False,
        ),
    )
    # Original source needed by restored wire models; no labels/prediction caches.
    save(
        bundle / "SHA256.json",
        {
            str(p.relative_to(bundle)): sha(p)
            for p in bundle.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        },
    )
    save(out / "RESTORE_INPUTS.json", inputs)
    dataset = out / "target_dataset"
    dataset.mkdir()
    combined = pd.concat(target_parts, ignore_index=True)
    combined.to_parquet(dataset / "features.parquet", index=False)
    ids = combined.candidate_id.tolist()
    for name in ["source_sink", "net_load"]:
        p = source / (name + ".parquet")
        names = pq.read_schema(p).names
        columns = [
            c for c in names if not c.startswith(("dr_", "final_", "target_", "l3_"))
        ]
        print("PREPARE_GR_INPUT", name, flush=True)
        frame = pd.read_parquet(
            p,
            columns=columns,
            filters=[("candidate_id", "in", ids)],
            dtype_backend="pyarrow",
        )
        assert set(map(tuple, frame[key].to_numpy())) == set(
            map(tuple, combined[key].to_numpy())
        )
        frame.to_parquet(dataset / (name + ".parquet"), index=False)
        del frame
        gc.collect()
    save(
        dataset / "DATASET.json",
        dict(
            schema="paris.dataset.v1",
            feature_profile="original-full-paris4-seed17",
            hashes={p.name: sha(p) for p in dataset.glob("*.parquet")},
            target_labels_included=False,
        ),
    )
    save(
        out / "RESTORE_COMPLETE.json",
        dict(
            status="complete",
            designs=counts,
            total=485,
            no_target_labels_read=True,
            requires_separate_prediction_parity=True,
            model_sha256=sha(bundle / "SHA256.json"),
        ),
    )
    print("RESTORE_COMPLETE", out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for name in ["run", "experiment", "runtime-bundle", "tool-source", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
