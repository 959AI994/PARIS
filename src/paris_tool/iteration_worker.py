"""Full-architecture, design-specific refit using the original bundle's code.

A new design borrows a frozen *recipe*, never fitted weights or labels from the
reference design. Validation labels are not an argument of fit or predict.
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import time
import sys
from .utils import digest, save, seal, verify

ADAPTED = "paris4.original.design-refit.1"
ORIGINAL = "paris4.original.seed17.1"
KEY = ["design", "candidate_id"]


def activate(root):
    for p in [Path(root) / "runtime", Path(root) / "runtime/experiment"]:
        sys.path.insert(0, str(p))


def gr_tables(dataset, design, ids):
    import pandas as pd

    filters = [("design", "==", design), ("candidate_id", "in", ids)]
    return [
        pd.read_parquet(
            Path(dataset) / (n + ".parquet"), filters=filters, dtype_backend="pyarrow"
        )
        for n in ["features", "source_sink", "net_load"]
    ]


def validate_gr(source, columns):
    import numpy as np

    if not np.isfinite(source[list(columns) + ["gr_slack"]].to_numpy(float)).all():
        raise ValueError(
            "All selected GR arcs, including unmatched arcs, require finite features"
        )
    nonnegative = [
        "hpwl",
        "num_sinks",
        "driver_slew",
        "ss_length",
        "ss_r",
        "ss_c",
        "gr_wire_delay",
        "gr_sink_slew",
        "gr_pi_r",
        "gr_pi_c1",
        "gr_pi_c2",
    ]
    if (source[[c for c in nonnegative if c in source]].to_numpy(float) < 0).any():
        raise ValueError("Negative GR delay/slew magnitude/R/C/geometry")
    # gr_wire_slew is driver-minus-sink and may correctly be negative.


def fit(manifest, output):
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    import xgboost
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    m = json.loads(Path(manifest).read_text())
    for field in ["training_labels", "training_wire_labels"]:
        if digest(m[field]) != m[field + "_sha256"]:
            raise ValueError("Frozen training snapshot changed: " + field)
    out = Path(output)
    out.mkdir(exist_ok=False)

    def event(stage, **kw):
        save(
            out.parent / "TRAIN_STATUS.json", dict(stage=stage, time=time.time(), **kw)
        )
        print(json.dumps(dict(stage=stage, **kw)), flush=True)

    reference = Path(m["reference"])
    verify(reference)
    # Copy the full original implementations; no native/simplified backend.
    shutil.copytree(
        reference / "runtime",
        out / "runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    source = (Path(__file__).with_name("original_runtime.py")).read_text()
    needle = 'card.get("algorithm") != "paris4.original.seed17.1"'
    assert source.count(needle) == 1
    source = source.replace(
        needle,
        'card.get("algorithm") not in ["paris4.original.seed17.1", "' + ADAPTED + '"]',
    )
    source = source.replace(
        "    activate(a.bundle)\n",
        """    card = json.loads((a.bundle / "MODEL_CARD.json").read_text())
    expected = card.get("context", {})
    context_file = a.features.parent / "FEATURE_CONTEXT.json"
    if expected:
        if not context_file.is_file():
            raise ValueError("Design-adapted model requires FEATURE_CONTEXT.json")
        actual = json.loads(context_file.read_text())
        for key in ["pdk_id", "flow_id", "corner", "feature_version", "time_unit"]:
            if actual.get(key) != expected.get(key):
                raise ValueError("Prediction context mismatch: " + key)
    activate(a.bundle)
""",
    )
    (out / "predict.py").write_text(source)
    activate(out)
    from v2.models import baseline_fit
    from v3.models import (
        CorrectedPARIS,
        Correction,
        baseline_oof,
        base_frame,
        correction_features,
    )
    from paris.ranking.common import assert_deployable_features, tns_cost
    from timing_predict.model.history_split import (
        GR_RESIDUAL_FEATURES,
        _add_gr_net_features,
    )
    from timing_predict.model.wire_features import _fit_predict, _candidate_features

    template = joblib.load(reference / "models" / m["template"] / "ranker.joblib")
    event("load_training", training_candidates=len(m["training_ids"]))
    f, s, n = gr_tables(m["dataset"], m["design"], m["training_ids"])
    y = pd.read_parquet(m["training_labels"])[KEY + ["final_tns"]]
    f = f.sort_values("candidate_id").reset_index(drop=True)
    y = f[KEY].merge(y, on=KEY, validate="one_to_one")
    assert set(f.candidate_id) == set(m["training_ids"]) and y.final_tns.notna().all()
    assert_deployable_features(f)
    s = _add_gr_net_features(s, n)
    for c in [*GR_RESIDUAL_FEATURES, "gr_slack"]:
        s[c] = s[c].to_numpy(float)
    validate_gr(s, GR_RESIDUAL_FEATURES)
    pair = KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"]
    wire = pd.read_parquet(m["training_wire_labels"])
    train = s.merge(wire[pair + ["dr_wire_delay"]], on=pair, validate="one_to_one")
    if len(train) != len(wire):
        raise ValueError(
            "Orphan measured wire labels: GR inputs and measured arc keys differ"
        )
    train["dr_wire_delay"] = train.dr_wire_delay.to_numpy(float)
    if set(train.candidate_id) != set(m["training_ids"]):
        raise ValueError(
            "Each training candidate requires measured matched DR wire labels"
        )
    if not np.isfinite(
        train[list(GR_RESIDUAL_FEATURES) + ["dr_wire_delay"]].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite training wire values")
    ids = np.array(sorted(m["training_ids"]))
    oof = np.full(len(s), np.nan)
    fold_receipts = []
    for k, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=17).split(ids)):
        event("wire_crossfit", fold=k + 1, total=5)
        trids, vaids = ids[tr], ids[va]
        trmask = train.candidate_id.isin(trids)
        vamask = s.candidate_id.isin(vaids)
        model, pred = _fit_predict(
            train.loc[trmask], s.loc[vamask], device="cpu", seed=17 + k, n_jobs=2
        )
        oof[vamask] = pred
        fold_receipts.append(
            dict(fold=k, train=trids.tolist(), validation=vaids.tolist())
        )
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete wire OOF")
    event("wire_refit", wire_rows=len(train))
    model, _ = _fit_predict(train, s.iloc[:1], device="cpu", seed=17, n_jobs=2)
    folder = out / "models" / m["design"]
    folder.mkdir(parents=True)
    model.save_model(folder / "wire_delay_residual.json")
    s["predicted_dr_wire_delay"] = oof
    wf = _candidate_features(s)
    cols = list(template["wire_columns"])
    if set(cols) - set(wf):
        raise ValueError("Original wire summary schema unavailable")
    f = f.drop(columns=[c for c in f if c.startswith("phys_wire_")]).merge(
        wf[KEY + cols], on=KEY, validate="one_to_one"
    )

    def refit_core(old, branch):
        event("candidate_refit", branch=branch)
        prior = baseline_fit(base_frame(f), y)
        _, weight = old.branches[branch]
        correction = None
        audit = copy.deepcopy(old.audit)
        if weight:
            chosen = audit["branches"][branch]["chosen"]
            event("candidate_crossfit", branch=branch)
            prior_oof = baseline_oof(f, y, n=5)
            correction = Correction.fit(
                correction_features(f, branch),
                tns_cost(y.final_tns),
                prior_oof,
                tuple(chosen["spec"]),
            )
        audit["design_refit"] = dict(
            recipe_frozen_from_reference=True, validation_labels_used=False
        )
        return CorrectedPARIS(
            prior, {branch: (correction, weight)}, None, audit, f[KEY].copy()
        )

    core = refit_core(template["core"], "legacy_grt")
    augmented = (
        refit_core(template["augmented"], "legacy")
        if template["augmented"] is not None
        else None
    )
    event("endpoint_refit")
    epcols = list(template["ep_columns"])
    endpoint = clone(template["endpoint"])
    jobs = {
        k: 2
        for k in endpoint.get_params(deep=True)
        if k == "n_jobs" or k.endswith("__n_jobs")
    }
    if jobs:
        endpoint.set_params(**jobs)
    endpoint.fit(
        f[epcols].to_numpy(float),
        tns_cost(y.final_tns) - f.ep_gr_log_cost.to_numpy(float),
    )
    record = dict(
        core=core,
        augmented=augmented,
        endpoint=endpoint,
        ep_columns=epcols,
        endpoint_weight=template["endpoint_weight"],
        stage=copy.deepcopy(template["stage"]),
        wire_columns=cols,
        required_columns=list(f.columns),
        training_ids=ids.tolist(),
        endpoint_family=template.get("endpoint_family", "reference family"),
    )
    joblib.dump(record, folder / "ranker.joblib", compress=3)
    save(
        folder / "INPUT_SCHEMA.json",
        dict(
            candidate_columns=[c for c in f if not c.startswith("phys_wire_")],
            wire_features=list(GR_RESIDUAL_FEATURES),
            endpoint_columns=epcols,
            rank_direction="ascending",
        ),
    )
    save(
        out / "TRAINING_PROTOCOL.json",
        dict(
            algorithm=ADAPTED,
            reference_sha256=m["reference_sha256"],
            template=m["template"],
            training_ids=ids.tolist(),
            wire_folds=fold_receipts,
            wire_labeled_rows=len(train),
            wire_gr_rows=len(s),
            wire_label_coverage=len(train) / len(s),
            recipe="original baseline, candidate-error correction, stage consensus and endpoint fusion; reference decisions fixed; all fitted weights refitted",
            input_hashes=m["input_hashes"],
            context=m["context"],
            versions=dict(
                python=sys.version,
                numpy=np.__version__,
                pandas=pd.__version__,
                sklearn=sklearn.__version__,
                xgboost=xgboost.__version__,
                joblib=joblib.__version__,
            ),
            augmentation=(
                "legacy branch refit on cumulative local training pool"
                if augmented
                else None
            ),
            validation_labels_used=False,
            training_hyperparameters_selected_on_validation=False,
        ),
    )
    # The sealed bundle is complete before any validation labels are read.
    seal(
        out,
        dict(
            name="PARIS4-" + m["design"] + "-" + m["id"],
            status="complete",
            algorithm=ADAPTED,
            designs=[m["design"]],
            input_contract="full original CTS/GR candidate features + source_sink/net_load",
            training_candidates=len(f),
            purpose=m.get("purpose", "development"),
            training_ids=ids.tolist(),
            recipe="full original architecture; frozen reference recipe; design-specific cumulative refit",
            output="batch-relative ranking scores, not physical DR TNS",
            template=m["template"],
            line=m["line"],
            context=m["context"],
            validation_scope="repeated development holdout; not an independent test",
            validation_labels_used_for_training=False,
        ),
    )
    event("sealed", model_sha256=digest(out / "SHA256.json"))


def predict(manifest, bundle, output, baseline=False):
    import joblib
    import pandas as pd

    m = json.loads(Path(manifest).read_text())
    root = Path(bundle)
    verify(root)
    activate(root)
    if not (root / "models" / m["design"]).is_dir():
        save(
            Path(output).with_suffix(".skip.json"),
            dict(reason="reference model has no weights for this design"),
        )
        return
    record = joblib.load(root / "models" / m["design"] / "ranker.joblib")
    overlap = set(record["training_ids"]) & set(m["validation_ids"])
    if overlap:
        if not baseline:
            raise ValueError("Validation IDs leaked into candidate training")
        save(
            Path(output).with_suffix(".skip.json"),
            dict(
                reason="baseline trained on validation candidates",
                overlap_n=len(overlap),
            ),
        )
        return
    f, s, n = gr_tables(m["dataset"], m["design"], m["validation_ids"])
    spec = importlib.util.spec_from_file_location(
        "iteration_bundle_predict", root / "predict.py"
    )
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    result = api.predict(root, f, s, n)
    result.to_parquet(output, index=False)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("action", choices=["fit", "predict"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bundle")
    p.add_argument("--baseline", action="store_true")
    a = p.parse_args()
    if a.action == "fit":
        fit(a.manifest, a.output)
    else:
        predict(a.manifest, a.bundle, a.output, a.baseline)


if __name__ == "__main__":
    main()
