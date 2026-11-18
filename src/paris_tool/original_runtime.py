"""Full original PARIS4 seed-17 inference; no fitting or reference-score lookup.

Runs inside a trusted bundle containing the original PARIS classes and weights.
Candidate input descriptors, GR wire tables and GR endpoint summaries are the
only prediction inputs. Historical decisions are frozen in each model record.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata


KEY = ["design", "candidate_id"]


def rank(values):
    return rankdata(np.asarray(values, float), method="average") / len(values)


def activate(bundle):
    for name in ["runtime", "runtime/experiment"]:
        sys.path.insert(0, str(Path(bundle) / name))


def verify(bundle):
    bundle = Path(bundle).resolve()
    hashes = json.loads((bundle / "SHA256.json").read_text())
    for name, expected in hashes.items():
        p = (bundle / name).resolve()
        if not p.is_relative_to(bundle) or not p.is_file():
            raise ValueError("Invalid model member: " + name)
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(b)
        if h.hexdigest() != expected:
            raise ValueError("Model checksum changed: " + name)


def candidate_score(record, frame):
    # Original serialized class performs original preprocessing and correction.
    value = record["core"].predict(frame)["legacy_grt"]
    if record["stage"]["stage_consensus"]["accepted"]:
        from paris.ranking.common import tns_cost

        w = record["stage"]["stage"]["l1_weight"]
        early = w * tns_cost(frame.fast_l1_tns) + (1 - w) * tns_cost(frame.grt_l2_tns)
        value = 0.25 * value + 0.75 * early
    if record["augmented"] is not None:
        augmented = record["augmented"].predict(frame)["legacy"]
        value = 0.5 * rank(value) + 0.5 * rank(augmented)
    return value


def infer_record(record, frame):
    forbidden = set(frame.candidate_id) & set(record["training_ids"])
    if forbidden:
        raise ValueError("Training candidates cannot be used as new inference inputs")
    missing = set(record["required_columns"]) - set(frame)
    if missing:
        raise ValueError("Missing original input features: " + str(sorted(missing)))
    cost = candidate_score(record, frame)
    columns = record["ep_columns"]
    endpoint = frame.ep_gr_log_cost.to_numpy(float) + record["endpoint"].predict(
        frame[columns].to_numpy(float)
    )
    w = record["endpoint_weight"]
    out = frame[KEY].copy()
    out["candidate_branch_score"] = cost
    out["endpoint_branch_score"] = endpoint
    out["paris4_cost_rank"] = (1 - w) * rank(cost) + w * rank(endpoint)
    out["predicted_order"] = rankdata(out.paris4_cost_rank, method="min").astype(int)
    return out


def predict(bundle, features, source_sink, net_load):
    import xgboost as xgb
    from paris.ranking.common import assert_deployable_features
    from timing_predict.model.history_split import (
        GR_RESIDUAL_FEATURES,
        _add_gr_net_features,
    )
    from timing_predict.model.wire_features import _candidate_features

    bundle = Path(bundle)
    verify(bundle)
    card = json.loads((bundle / "MODEL_CARD.json").read_text())
    if (
        card.get("status") != "complete"
        or card.get("algorithm") != "paris4.original.seed17.1"
    ):
        raise ValueError("Not a complete original seed-17 model")
    assert_deployable_features(features)
    if features.empty or features.duplicated(KEY).any():
        raise ValueError("Empty/duplicate candidate inputs")
    if set(features.design) - set(card["designs"]):
        raise ValueError("Unsupported design for this original model bank")
    wanted = set(map(tuple, features[KEY].to_numpy()))
    for table in [features, source_sink, net_load]:
        if any(c.startswith(("dr_", "final_", "target_", "l3_")) for c in table):
            raise ValueError("Post-route label fields are forbidden at inference")
        if set(map(tuple, table[KEY].to_numpy())) != wanted:
            raise ValueError("Candidate inventory differs between input tables")
    pair = KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"]
    net = KEY + ["net_name", "driver_pin", "corner"]
    if source_sink.duplicated(pair).any() or net_load.duplicated(net).any():
        raise ValueError("Duplicate wire inputs")
    source = _add_gr_net_features(source_sink, net_load)
    frames = []
    for design, f in features.groupby("design", sort=True):
        if len(f) < 2:
            raise ValueError("Provide a batch of at least two same-design candidates")
        folder = bundle / "models" / str(design)
        record = joblib.load(folder / "ranker.joblib")
        s = source[source.design.eq(design)].copy()
        # Original training/extraction used NumPy float64 reductions. Normalize
        # Arrow-backed portable inputs before the original aggregation code.
        for column in [*GR_RESIDUAL_FEATURES, "gr_slack"]:
            s[column] = s[column].to_numpy(dtype=np.float64)
        if not np.isfinite(s[list(GR_RESIDUAL_FEATURES)].to_numpy(float)).all():
            raise ValueError("Nonfinite GR wire features")
        model = xgb.XGBRegressor(n_jobs=2, device="cpu")
        model.load_model(folder / "wire_delay_residual.json")
        model.set_params(n_jobs=2, device="cpu")
        s["predicted_dr_wire_delay"] = np.maximum(
            0.0,
            s.gr_wire_delay.to_numpy(float)
            + model.predict(s[list(GR_RESIDUAL_FEATURES)]).astype(float),
        )
        wire = _candidate_features(s)
        # Never consume precomputed target wire predictions, even if supplied.
        f = f.drop(columns=[c for c in f if c.startswith("phys_wire_")])
        f = f.merge(wire[KEY + record["wire_columns"]], on=KEY, validate="one_to_one")
        frames.append(infer_record(record, f))
    return pd.concat(frames, ignore_index=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ["bundle", "features", "source-sink", "net-load", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    activate(a.bundle)
    result = predict(
        a.bundle,
        pd.read_parquet(a.features),
        pd.read_parquet(a.source_sink),
        pd.read_parquet(a.net_load),
    )
    result.to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
