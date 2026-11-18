"""Deployment-safe fused-residual rank model used by PARIS.

The model learns how post-route timing changes a fused L1/L2 cost prior. It
never receives L3 labels, runtimes, or post-route-only fields at prediction time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    IDENTITY_COLUMNS,
    assert_deployable_features,
    assert_unique_candidates,
    deployable_numeric_columns,
    enrich_target_free,
    kendall,
    select_features_train_only,
    sha256_file,
    tns_cost,
    truth_relevance,
)

MODEL_FAMILY = "ridge_fused_residual"


def _base_cost(frame: pd.DataFrame) -> np.ndarray:
    """Return the equal-weight L1/L2 timing-cost prior."""

    return 0.5 * (
        tns_cost(frame["fast_l1_tns"]) + tns_cost(frame["grt_l2_tns"])
    )


def _ranking_relevance(frame: pd.DataFrame) -> np.ndarray:
    """Return design-normalized L3 relevance for label-safe feature screening."""

    relevance = np.zeros(len(frame), dtype=float)
    for _, positions in frame.groupby("design", sort=False).indices.items():
        index = np.asarray(positions, dtype=int)
        relevance[index] = truth_relevance(frame.iloc[index]["final_tns"])
    return relevance


def _candidate_hash(candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(candidate_id.encode()).digest()[:8], "big")


def _validation_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Return design-held-out folds, or a stable split for one-design history."""

    designs = sorted(frame["design"].astype(str).unique())
    if len(designs) > 1:
        values = frame["design"].astype(str).to_numpy()
        return [
            (np.flatnonzero(values != design), np.flatnonzero(values == design), design)
            for design in designs
        ]
    if len(frame) < 5:
        return []
    order = np.argsort(
        np.asarray(
            [_candidate_hash(str(value)) for value in frame["candidate_id"]],
            dtype=np.uint64,
        ),
        kind="stable",
    )
    test_count = max(1, int(round(0.2 * len(frame))))
    test = np.sort(order[:test_count])
    train = np.sort(order[test_count:])
    return [(train, test, f"{designs[0]}_history_holdout")]


def _fit_ridge(matrix: np.ndarray, target: np.ndarray, alpha: float) -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=float(alpha))).fit(matrix, target)


@dataclass
class RankingBundle:
    """Serialized PARIS ranker, preprocessing state, and uncertainty ensemble."""

    feature_columns: list[str]
    source_columns: list[str]
    fill_values: dict[str, float]
    ood_median: dict[str, float]
    ood_iqr: dict[str, float]
    members: list[Any]
    selected_config: dict[str, Any]
    training_designs: list[str]
    training_candidate_keys: set[str]
    validation_scores: dict[str, float] = field(default_factory=dict)
    runtime_feature_columns: list[str] = field(default_factory=list)
    runtime_fill_values: dict[str, float] = field(default_factory=dict)
    runtime_members: list[Any] = field(default_factory=list)
    schema_version: int = 2
    family: str = MODEL_FAMILY

    def _enriched(self, features: pd.DataFrame) -> pd.DataFrame:
        assert_deployable_features(features)
        requested = {
            f"{row.design}\0{row.candidate_id}"
            for row in features[list(IDENTITY_COLUMNS)].itertuples(index=False)
        }
        overlap = requested.intersection(self.training_candidate_keys)
        if overlap:
            raise ValueError(
                "prediction candidates overlap offline training labels; "
                "use a disjoint target candidate table"
            )
        return enrich_target_free(features.reset_index(drop=True), self.source_columns)

    def matrix(self, features: pd.DataFrame) -> np.ndarray:
        enriched = self._enriched(features)
        numeric = enriched.reindex(columns=self.feature_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        return numeric.fillna(self.fill_values).to_numpy(dtype=np.float32)

    def predict_members(self, features: pd.DataFrame) -> np.ndarray:
        matrix = self.matrix(features)
        base_cost = _base_cost(features)
        member_scores: list[np.ndarray] = []
        for member in self.members:
            residual = np.asarray(member.predict(matrix), dtype=float)
            raw_score = -(base_cost + residual)
            normalized = np.zeros(len(features), dtype=float)
            for _, positions in features.groupby("design", sort=False).indices.items():
                index = np.asarray(positions, dtype=int)
                normalized[index] = pd.Series(raw_score[index]).rank(pct=True).to_numpy(float)
            member_scores.append(normalized)
        if not member_scores:
            raise ValueError("ranking bundle has no ensemble members")
        return np.stack(member_scores, axis=1)

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        features = features.reset_index(drop=True)
        members = self.predict_members(features)
        enriched = self._enriched(features)
        numeric = enriched.reindex(columns=self.feature_columns).apply(
            pd.to_numeric, errors="coerce"
        )
        filled = numeric.fillna(self.fill_values)
        median = pd.Series(self.ood_median)
        iqr = pd.Series(self.ood_iqr).clip(lower=1.0e-9)
        outside = numeric.isna() | filled.lt(median - 6.0 * iqr) | filled.gt(median + 6.0 * iqr)
        output = features[list(IDENTITY_COLUMNS)].copy()
        output["predicted_score"] = np.median(members, axis=1)
        output["rank_percentile"] = output.groupby("design")["predicted_score"].rank(pct=True)
        output["uncertainty"] = np.std(members, axis=1)
        output["ood_score"] = outside.mean(axis=1).to_numpy(float)
        if self.runtime_members:
            runtime_numeric = (
                enriched.reindex(columns=self.runtime_feature_columns)
                .apply(pd.to_numeric, errors="coerce")
                .fillna(self.runtime_fill_values)
            )
            runtime_matrix = runtime_numeric.to_numpy(dtype=np.float32)
            runtime_log = np.stack(
                [
                    np.asarray(member.predict(runtime_matrix), dtype=float)
                    for member in self.runtime_members
                ],
                axis=1,
            )
            output["predicted_runtime_sec"] = np.maximum(
                1.0, np.expm1(np.median(runtime_log, axis=1))
            )
        output["model_family"] = self.family
        return output

    def save(self, directory: Path, extra: dict[str, Any] | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "model.joblib"
        joblib.dump(self, model_path)
        card: dict[str, Any] = {
            "schema_version": self.schema_version,
            "family": self.family,
            "training_designs": self.training_designs,
            "training_candidate_count": len(self.training_candidate_keys),
            "source_feature_count": len(self.source_columns),
            "selected_feature_count": len(self.feature_columns),
            "selected_features": self.feature_columns,
            "selected_config": self.selected_config,
            "validation_tau_b": self.validation_scores,
            "ensemble_members": len(self.members),
            "runtime_feature_count": len(self.runtime_feature_columns),
            "runtime_ensemble_members": len(self.runtime_members),
            "model_sha256": sha256_file(model_path),
        }
        card.update(extra or {})
        (directory / "model_card.json").write_text(
            json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return model_path


def load_bundle(path: Path | str) -> RankingBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, RankingBundle):
        raise TypeError(f"not a PARIS RankingBundle: {path}")
    return bundle


def fit_bundle(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    config: dict[str, Any],
    seeds: Sequence[int],
) -> RankingBundle:
    """Fit the residual rank ensemble using only explicitly supplied labels."""

    assert_deployable_features(features)
    assert_unique_candidates(truth, "truth table")
    label_columns = ["final_tns"]
    if "runtime_sec" in truth:
        label_columns.append("runtime_sec")
    labels = truth[[*IDENTITY_COLUMNS, *label_columns]].copy()
    frame = features.merge(labels, on=list(IDENTITY_COLUMNS), how="inner", validate="one_to_one")
    if len(frame) < 5:
        raise ValueError("at least five labeled candidates are required")
    if not pd.to_numeric(frame["final_tns"], errors="coerce").notna().all():
        raise ValueError("every training candidate requires a finite final_tns")

    feature_frame = frame[features.columns].reset_index(drop=True)
    source_columns = deployable_numeric_columns(feature_frame)
    if not source_columns:
        raise ValueError("no deployable numeric feature columns")
    enriched = enrich_target_free(feature_frame, source_columns)
    target = tns_cost(frame["final_tns"]) - _base_cost(frame)
    selection_target = _ranking_relevance(frame)
    feature_counts = [int(value) for value in config.get("feature_counts", [16, 32, 64])]
    alpha_values = [
        float(value)
        for value in config.get("model", {}).get(
            "alpha",
            config.get("models", {})
            .get("ridge_fused_residual", {})
            .get("alpha", [1.0, 10.0, 100.0]),
        )
    ]
    folds = _validation_folds(frame)
    candidates: list[tuple[float, str, dict[str, Any], dict[str, float]]] = []
    for count in feature_counts:
        for alpha in alpha_values:
            fold_scores: dict[str, float] = {}
            for train, test, name in folds:
                if not len(train) or not len(test):
                    continue
                chosen = select_features_train_only(
                    enriched, selection_target, train, min(count, enriched.shape[1])
                )
                fill = enriched.iloc[train][chosen].median().fillna(0.0)
                matrix = enriched[chosen].fillna(fill).to_numpy(np.float32)
                member = _fit_ridge(matrix[train], target[train], alpha)
                score = -(
                    _base_cost(frame.iloc[test])
                    + np.asarray(member.predict(matrix[test]), dtype=float)
                )
                fold_scores[name] = kendall(score, frame.iloc[test]["final_tns"])
            mean_score = float(np.mean(list(fold_scores.values()))) if fold_scores else 0.0
            setting = {"alpha": alpha, "feature_count": count}
            candidates.append(
                (mean_score, json.dumps(setting, sort_keys=True), setting, fold_scores)
            )
    _, _, selected_config, selected_fold_scores = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    chosen_columns = select_features_train_only(
        enriched,
        selection_target,
        np.arange(len(frame)),
        min(int(selected_config["feature_count"]), enriched.shape[1]),
    )
    numeric = enriched[chosen_columns].apply(pd.to_numeric, errors="coerce")
    fill = numeric.median().fillna(0.0)
    filled = numeric.fillna(fill)
    member_seeds = [int(value) for value in seeds] or [17]
    # Ridge is deterministic.  Fit each recorded seed on the same complete
    # training fold so the released model reproduces the frozen Ranking V2
    # protocol exactly; target-design uncertainty is handled by OOD and
    # anchor-disagreement signals in the online policy, not bootstrap noise.
    members = [
        _fit_ridge(
            filled.to_numpy(np.float32),
            target,
            float(selected_config["alpha"]),
        )
        for _ in member_seeds
    ]

    runtime_feature_columns: list[str] = []
    runtime_fill_values: dict[str, float] = {}
    runtime_members: list[Any] = []
    if "runtime_sec" in frame:
        runtime = pd.to_numeric(frame["runtime_sec"], errors="coerce").to_numpy(float)
        valid = np.flatnonzero(np.isfinite(runtime) & (runtime >= 0.0))
        if len(valid) >= 5:
            runtime_target = np.log1p(runtime)
            runtime_feature_columns = select_features_train_only(
                enriched,
                runtime_target,
                valid,
                min(32, enriched.shape[1]),
            )
            runtime_numeric = enriched[runtime_feature_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            runtime_fill = runtime_numeric.iloc[valid].median().fillna(0.0)
            runtime_fill_values = {str(key): float(value) for key, value in runtime_fill.items()}
            runtime_matrix = runtime_numeric.fillna(runtime_fill).to_numpy(np.float32)
            for seed in member_seeds:
                rng = np.random.default_rng(seed)
                sampled = rng.choice(valid, size=len(valid), replace=True)
                runtime_members.append(
                    _fit_ridge(runtime_matrix[sampled], runtime_target[sampled], 10.0)
                )

    return RankingBundle(
        feature_columns=chosen_columns,
        source_columns=source_columns,
        fill_values={str(key): float(value) for key, value in fill.items()},
        ood_median={str(key): float(value) for key, value in filled.median().items()},
        ood_iqr={
            str(key): float(max(value, 1.0e-9))
            for key, value in (filled.quantile(0.75) - filled.quantile(0.25)).items()
        },
        members=members,
        selected_config=selected_config,
        training_designs=sorted(frame["design"].astype(str).unique()),
        training_candidate_keys={
            f"{row.design}\0{row.candidate_id}"
            for row in frame[list(IDENTITY_COLUMNS)].itertuples(index=False)
        },
        validation_scores={
            **selected_fold_scores,
            "macro": float(np.mean(list(selected_fold_scores.values())))
            if selected_fold_scores
            else 0.0,
        },
        runtime_feature_columns=runtime_feature_columns,
        runtime_fill_values=runtime_fill_values,
        runtime_members=runtime_members,
    )
