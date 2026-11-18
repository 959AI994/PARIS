"""Portable PARIS4 architecture, algorithm revision paris4.native.1.

Original fused CTS/GR Ridge baseline + cross-fitted wire-delay residual
summaries + candidate-error correction + stage consensus + endpoint-summary
rank fusion. Bounded choices are reselected on training candidates only.
The exact old frozen banks remain available via the separate bundle adapter.
"""

import itertools
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .. import data
from ..utils import fresh, save, seal, verify, digest
from .._vendor.ranking.model import fit_bundle
from .._vendor.ranking.common import tns_cost
from .correction import Correction, correction_features
from .wire import _candidate_features

KEY = data.KEY
GRID = [dict(correction=0.0, stage=False, endpoint=0.0, gr_only=True)] + [
    dict(correction=w, stage=st, endpoint=ep)
    for w, st, ep in itertools.product(
        [0.0, 0.1, 0.25, 0.5], [False, True], [0.0, 0.25, 0.5]
    )
]


def tau(a, b):
    v = float(kendalltau(a, b, variant="b").statistic)
    return v if np.isfinite(v) else 0.0


def base_frame(f):
    return f[[c for c in f if not c.startswith(("phys_", "ep_"))]]


def base_fit(f, y, seed):
    return fit_bundle(
        base_frame(f),
        y,
        config={"feature_counts": [16, 32, 64], "model": {"alpha": [1.0, 10.0, 100.0]}},
        seeds=[seed + i for i in [0, 24, 56, 84, 132]],
    )


def base_score(model, f):
    a = model.matrix(base_frame(f))
    return 0.5 * (tns_cost(f.fast_l1_tns) + tns_cost(f.grt_l2_tns)) + np.median(
        [m.predict(a) for m in model.members], axis=0
    )


def wire_fit(s, trees, jobs, seed):
    m = XGBRegressor(
        n_estimators=trees,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=jobs,
        random_state=seed,
    )
    size = s.groupby(KEY).candidate_id.transform("size").to_numpy(float)
    w = 1.0 / size
    w /= w.mean()
    m.fit(s[data.WIRE], s.dr_wire_delay - s.gr_wire_delay, sample_weight=w)
    return m


def wire_predict(m, s):
    p = s.drop(columns=[c for c in s if c.startswith("dr_")]).copy()
    p["predicted_dr_wire_delay"] = np.maximum(
        0.0, p.gr_wire_delay + m.predict(p[data.WIRE])
    )
    return _candidate_features(p)


def subset(s, f):
    return s.merge(f[KEY], on=KEY, validate="many_to_one")


def add_wire(f, w):
    f = f.drop(columns=[c for c in f if c.startswith("phys_wire_")])
    out = f.merge(w, on=KEY, validate="one_to_one")
    if len(out) != len(f):
        raise ValueError("Wire features dropped candidates")
    return out


def components(f, y, s, seed, jobs, trees):
    if len(f) < 12:
        raise ValueError("At least 12 training candidates needed for this fit")
    poof = np.full(len(f), np.nan)
    wf = []
    for k, (tr, va) in enumerate(KFold(3, shuffle=True, random_state=seed).split(f)):
        # A held-out candidate is absent from every source/sink row of its wire model.
        xt, xv = f.iloc[tr], f.iloc[va]
        yt = y.iloc[tr]
        bm = base_fit(xt, yt, seed + k)
        poof[va] = base_score(bm, xv)
        wm = wire_fit(subset(s, xt), trees, jobs, seed + k)
        wf.append(wire_predict(wm, subset(s, xv)))
    x = add_wire(f, pd.concat(wf, ignore_index=True))
    core = base_fit(f, y, seed)
    corr = Correction.fit(
        correction_features(x, "legacy_grt"),
        tns_cost(y.final_tns),
        poof,
        ("ridge", 16, 100.0),
    )
    cols = [c for c in f if c.startswith("ep_")]
    if not cols:
        raise ValueError("PARIS4 requires GR endpoint summaries")
    ep = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=100.0)
    )
    ep.fit(f[cols], tns_cost(y.final_tns) - f.ep_gr_log_cost.to_numpy(float))
    return dict(
        core=core,
        correction=corr,
        endpoint=ep,
        endpoint_columns=cols,
        wire=wire_fit(s, trees, jobs, seed),
        feature_columns=list(f.columns),
    )


def signals(m, f, s):
    missing = set(m["feature_columns"]) - set(f)
    if missing:
        raise ValueError("Missing model input columns: " + str(sorted(missing)))
    w = wire_predict(m["wire"], s)
    x = add_wire(f, w)
    return dict(
        gr=tns_cost(f.grt_l2_tns),
        base=base_score(m["core"], f),
        correction=m["correction"].predict(correction_features(x, "legacy_grt")),
        stage=0.5 * (tns_cost(f.fast_l1_tns) + tns_cost(f.grt_l2_tns)),
        endpoint=f.ep_gr_log_cost.to_numpy(float)
        + m["endpoint"].predict(f[m["endpoint_columns"]]),
    )


def score(z, cfg):
    if cfg.get("gr_only", False):
        return rankdata(z["gr"], method="average") / len(z["gr"])
    v = z["base"] + cfg["correction"] * z["correction"]
    if cfg["stage"]:
        v = 0.25 * v + 0.75 * z["stage"]
    # Raw rank numerators preserve exact ties before division.
    e = cfg["endpoint"]
    return (
        (1 - e) * rankdata(v, method="average")
        + e * rankdata(z["endpoint"], method="average")
    ) / len(v)


def fit_history(f, y, s, seed, jobs, trees):
    if len(f) < 18:
        raise ValueError(
            "PARIS4 self-training needs at least 18 history candidates after holdout; provide at least 24 labeled candidates per design"
        )
    rows = []
    for fold, (tr, va) in enumerate(KFold(3, shuffle=True, random_state=seed).split(f)):
        xt, xv = f.iloc[tr].reset_index(drop=True), f.iloc[va].reset_index(drop=True)
        yt, yv = y.iloc[tr].reset_index(drop=True), y.iloc[va].reset_index(drop=True)
        m = components(xt, yt, subset(s, xt), seed + fold, jobs, trees)
        z = signals(m, xv, subset(s, xv))
        for i, cfg in enumerate(GRID):
            rows.append(
                dict(fold=fold, choice=i, tau_b=tau(score(z, cfg), -yv.final_tns))
            )
    cv = pd.DataFrame(rows)
    best = int(
        cv.groupby("choice")
        .tau_b.mean()
        .sort_values(ascending=False, kind="stable")
        .index[0]
    )
    cfg = GRID[best]
    m = components(f, y, s, seed, jobs, trees)
    m["fusion"] = cfg
    m["training_keys"] = sorted(data.keys(f))
    m["algorithm"] = "paris4.native.1"
    return m, cv


class Backend:
    def train(
        self, dataset, output, seed=17, test_fraction=0.2, jobs=2, trees=900, **unused
    ):
        data.check(dataset)
        original_manifest = digest(Path(dataset) / "DATASET.json")
        dataset_card = json.loads((Path(dataset) / "DATASET.json").read_text())
        f = data.features(dataset)
        y = data.labels(dataset)
        f = (
            f.merge(y[KEY], on=KEY, validate="one_to_one")
            .sort_values(KEY)
            .reset_index(drop=True)
        )
        y = f[KEY].merge(y, on=KEY, validate="one_to_one")
        s = (
            data.source(dataset)
            .merge(data.wire_labels(dataset), on=data.PAIR, validate="one_to_one")
            .merge(f[KEY], on=KEY, validate="many_to_one")
        )
        if data.keys(s) != data.keys(f):
            raise ValueError(
                "Every labeled candidate needs matched training wire labels"
            )
        out = fresh(output)
        banks = {}
        metrics = []
        splits = []
        if not 0 < test_fraction < 0.5:
            raise ValueError("test_fraction must be in (0, .5)")
        for design, group in f.groupby("design", sort=True):
            ids = group.candidate_id.to_numpy(copy=True)
            np.random.default_rng(seed).shuffle(ids)
            n = max(2, int(np.ceil(len(ids) * test_fraction)))
            hold = set(ids[:n])
            train = group[~group.candidate_id.isin(hold)].reset_index(drop=True)
            test = group[group.candidate_id.isin(hold)].reset_index(drop=True)
            yt = train[KEY].merge(y, on=KEY, validate="one_to_one")
            yv = test[KEY].merge(y, on=KEY, validate="one_to_one")
            m, cv = fit_history(train, yt, subset(s, train), seed, jobs, trees)
            z = signals(m, test, subset(s, test))
            cost = score(z, m["fusion"])
            prediction = test[KEY].copy()
            prediction["rank_score"] = cost
            folder = out / "evaluation" / design
            folder.mkdir(parents=True)
            prediction.to_parquet(folder / "frozen_predictions.parquet", index=False)
            # Held-out truth joins only after the prediction file exists.
            metrics.append(
                dict(
                    design=design,
                    n=len(test),
                    gr_tau_b=tau(-test.grt_l2_tns, -yv.final_tns),
                    paris4_tau_b=tau(cost, -yv.final_tns),
                )
            )
            cv.to_csv(folder / "HISTORY_CV.csv", index=False)
            splits.append(
                dict(
                    design=design, history=sorted(train.candidate_id), test=sorted(hold)
                )
            )
            allf = group.reset_index(drop=True)
            ally = allf[KEY].merge(y, on=KEY, validate="one_to_one")
            # Refit using the frozen history-selected recipe, without selecting
            # again using held-out scores. Final weights absorb all labels.
            full = components(allf, ally, subset(s, allf), seed, jobs, trees)
            full.update(
                fusion=m["fusion"],
                training_keys=sorted(data.keys(allf)),
                algorithm="paris4.native.1",
            )
            banks[design] = full
        data.check(dataset)
        if digest(Path(dataset) / "DATASET.json") != original_manifest:
            raise ValueError(
                "Dataset changed during training; partial model retained without a release receipt"
            )
        joblib.dump(banks, out / "models.joblib", compress=3)
        pd.DataFrame(metrics).to_csv(out / "METRICS.csv", index=False)
        save(out / "SPLITS.json", splits)
        save(out / "RECIPE_GRID.json", GRID)
        versions = {}
        import importlib.metadata

        for k in [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "xgboost",
            "joblib",
            "pyarrow",
        ]:
            versions[k] = importlib.metadata.version(k)
        save(
            out / "INPUT_SCHEMA.json",
            {
                d: dict(candidate_columns=m["feature_columns"], wire_columns=data.WIRE)
                for d, m in banks.items()
            },
        )
        card = dict(
            status="complete",
            backend="paris4",
            algorithm="paris4.native.1",
            feature_profile="dataset-specific; never fabricated to match legacy weights",
            supported_designs=list(banks),
            training_candidates=len(f),
            seed=seed,
            wire_trees=trees,
            output="batch-relative cost rank; smaller is better; not physical DR TNS",
            full_refit_uses_all_labels=True,
            evaluation="candidate-disjoint heldout predictions from history-selected models; old explored corpus is not external validation",
            python=platform.python_version(),
            dependencies=versions,
            feature_context=dataset_card.get("feature_context"),
            selected_recipes={d: m["fusion"] for d, m in banks.items()},
            recipe="original PARIS fused Ridge baseline, OOF wire residuals, original candidate-error correction, stage consensus, GR endpoint Ridge and history-only fusion selection",
        )
        from paris_tool import __version__

        card["paris_package_version"] = __version__
        (out / "requirements.txt").write_text(
            "".join(k + "==" + v + "\n" for k, v in versions.items())
        )
        seal(out, card)
        return card

    def predict(self, model, dataset, output):
        card = verify(model)
        if card.get("backend") != "paris4":
            raise ValueError("Not a native PARIS4 model")
        if card.get("algorithm") != "paris4.native.1":
            raise ValueError(
                "Unsupported PARIS4 algorithm revision; install its recorded backend"
            )
        data.check(dataset)
        if card.get("feature_context") is not None:
            current = json.loads((Path(dataset) / "DATASET.json").read_text()).get(
                "feature_context"
            )
            if current != card["feature_context"]:
                raise ValueError(
                    "Model/feature flow context mismatch (Liberty, GR tool, RC settings, corner or extraction policy)"
                )
        f = data.features(dataset)
        s = data.source(dataset)
        banks = joblib.load(Path(model) / "models.joblib")
        parts = []
        for design, g in f.groupby("design", sort=True):
            if design not in banks:
                raise ValueError(
                    "No same-design bank; train this design or choose a separately validated shared backend: "
                    + str(design)
                )
            if len(g) < 2:
                raise ValueError("At least two same-design candidates required")
            m = banks[design]
            if data.keys(g) & set(map(tuple, m["training_keys"])):
                raise ValueError(
                    "Training IDs cannot be reported as new prediction candidates"
                )
            z = signals(m, g, subset(s, g))
            p = g[KEY].copy()
            p["rank_score"] = score(z, m["fusion"])
            p["predicted_order"] = p.rank_score.rank(method="min").astype(int)
            p["gr_tns"] = g.grt_l2_tns.to_numpy()
            parts.append(p)
        result = pd.concat(parts, ignore_index=True)
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        result.to_csv(output, index=False)
        return dict(status="complete", rows=len(result), output=str(output))
