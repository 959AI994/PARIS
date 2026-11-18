"""Shared, leakage-safe helpers for the PARIS ranking model."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

IDENTITY_COLUMNS = ("design", "candidate_id")
REQUIRED_CHEAP_COLUMNS = ("fast_l1_tns", "grt_l2_tns")
TRUTH_COLUMNS = (
    "final_tns",
    "final_wns",
    "target_cost",
    "runtime_sec",
    "drc_count",
    "congestion",
    "routed_wirelength",
    "power",
)
TRUTH_PROVENANCE_COLUMNS = ("l3_label_path", "l3_label_sha256")
FORBIDDEN_FEATURE_PREFIXES = (
    "final_",
    "target_",
    "skip_rcx_",
    "routed_",
    "runtime",
    "teach_",
    "l3_",
    "current_l3_",
    "historical_overlay_",
    "metadata",
)
FORBIDDEN_FEATURE_SUBSTRINGS = ("runtime", "wall_sec", "cpu_sec", "peak_mem")
PROVENANCE_PREFIXES = ("l1_report_", "l2_report_", "cts_odb_", "grt_odb_")
PROVENANCE_COLUMNS = {
    "flow_git_commit",
    "openroad_commit",
    "feature_schema_version",
    "l1_source_stage",
    "l2_source_stage",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def stable_seed(name: str, base: int = 17) -> int:
    value = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big")
    return int((value + base) % (2**31 - 1))


def canonical_design(value: str) -> str:
    value = str(value)
    return "swerv_wrapper" if value == "swerv" else value


def assert_unique_candidates(frame: pd.DataFrame, label: str = "table") -> None:
    missing = set(IDENTITY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} misses identity columns: {sorted(missing)}")
    duplicated = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, list(IDENTITY_COLUMNS)].head(5).to_dict("records")
        raise ValueError(f"{label} has duplicate candidate keys: {sample}")


def forbidden_feature_columns(columns: Iterable[str]) -> list[str]:
    bad = []
    for column in columns:
        lowered = str(column).lower()
        if (
            column in TRUTH_COLUMNS
            or lowered.startswith(FORBIDDEN_FEATURE_PREFIXES)
            or any(token in lowered for token in FORBIDDEN_FEATURE_SUBSTRINGS)
        ):
            bad.append(str(column))
    return sorted(set(bad))


def assert_deployable_features(frame: pd.DataFrame) -> None:
    assert_unique_candidates(frame, "feature table")
    missing = set(REQUIRED_CHEAP_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"feature table misses cheap timing columns: {sorted(missing)}")
    bad = forbidden_feature_columns(frame.columns)
    if bad:
        raise ValueError(f"feature table contains forbidden columns: {bad}")
    for column in REQUIRED_CHEAP_COLUMNS:
        if not pd.to_numeric(frame[column], errors="coerce").notna().all():
            raise ValueError(f"{column} must be present for every candidate; no silent fallback")


def tns_cost(values: Sequence[float]) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    return np.log1p(-np.minimum(raw, 0.0))


def rank01(values: Sequence[float], *, higher_is_better: bool) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    if len(series) <= 1:
        return np.ones(len(series), dtype=float)
    ranks = series.rank(method="average", ascending=not higher_is_better)
    return (1.0 - (ranks.to_numpy(float) - 1.0) / (len(series) - 1.0)).astype(float)


def truth_relevance(final_tns: Sequence[float]) -> np.ndarray:
    """Best L3 candidate maps to one, worst to zero."""

    return rank01(final_tns, higher_is_better=True)


def kendall(scores: Sequence[float], final_tns: Sequence[float]) -> float:
    value = kendalltau(scores, final_tns, variant="b").statistic
    return float(value) if value is not None and np.isfinite(value) else 0.0


def topk_metrics(scores: Sequence[float], final_tns: Sequence[float], ks=(1, 5, 10)) -> dict:
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(final_tns, dtype=float)
    pred_order = np.argsort(-scores, kind="stable")
    true_order = np.argsort(-truth, kind="stable")
    true_position = np.empty(len(truth), dtype=int)
    true_position[true_order] = np.arange(len(truth))
    relevance = len(truth) - true_position
    output: dict[str, float | int | bool] = {
        "top1_hit": bool(pred_order[0] == true_order[0]) if len(truth) else False,
    }
    for requested in ks:
        k = min(int(requested), len(truth))
        if k == 0:
            continue
        output[f"recall_at_{requested}"] = float(
            len(set(pred_order[:k]).intersection(true_order[:k])) / k
        )
        gains = relevance[pred_order[:k]].astype(float)
        ideal = relevance[true_order[:k]].astype(float)
        discounts = np.log2(np.arange(2, k + 2))
        output[f"ndcg_at_{requested}"] = float(
            np.sum((2.0**gains - 1.0) / discounts)
            / max(np.sum((2.0**ideal - 1.0) / discounts), 1.0e-12)
        )
    return output


def macro_tau(frame: pd.DataFrame, score_column: str, truth_column: str = "final_tns") -> float:
    values = [
        kendall(group[score_column], group[truth_column])
        for _, group in frame.groupby("design", sort=True)
        if len(group) >= 2
    ]
    return float(np.mean(values)) if values else 0.0


def deployable_numeric_columns(frame: pd.DataFrame) -> list[str]:
    assert_deployable_features(frame)
    columns = []
    for column in frame.columns:
        if (
            column in IDENTITY_COLUMNS
            or column in PROVENANCE_COLUMNS
            or column.startswith(PROVENANCE_PREFIXES)
        ):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() < max(4, int(math.ceil(0.8 * len(frame)))):
            continue
        if float(numeric.std(skipna=True) or 0.0) <= 1.0e-12:
            continue
        columns.append(column)
    return sorted(columns)


def enrich_target_free(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add target-free design-relative transforms.

    Computing within-design ranks over the full cheap candidate set is allowed:
    this is transductive only in features and never observes an L3 value.
    """

    additions: dict[str, pd.Series | np.ndarray] = {}
    groups = frame["design"].astype(str)
    for column in columns:
        values = (
            pd.to_numeric(frame[column], errors="coerce")
            if column in frame
            else pd.Series(np.nan, index=frame.index, dtype=float)
        )
        additions[column] = values
        grouped = values.groupby(groups, sort=False)
        median = grouped.transform("median")
        q25 = grouped.transform(lambda item: item.quantile(0.25))
        q75 = grouped.transform(lambda item: item.quantile(0.75))
        scale = (q75 - q25).replace(0.0, 1.0)
        additions[f"{column}__rank"] = grouped.rank(method="average", pct=True)
        additions[f"{column}__robust_z"] = (values - median) / scale
    l1_cost = tns_cost(frame["fast_l1_tns"])
    l2_cost = tns_cost(frame["grt_l2_tns"])
    additions["timing__l2_minus_l1"] = l2_cost - l1_cost
    additions["timing__rank_disagreement"] = pd.Series(l2_cost, index=frame.index).groupby(
        groups
    ).rank(pct=True) - pd.Series(l1_cost, index=frame.index).groupby(groups).rank(pct=True)
    return pd.DataFrame(additions, index=frame.index).replace([np.inf, -np.inf], np.nan)


def select_features_train_only(
    matrix: pd.DataFrame,
    target: Sequence[float],
    train_indices: Sequence[int],
    count: int,
) -> list[str]:
    train_indices = np.asarray(train_indices, dtype=int)
    target = np.asarray(target, dtype=float)
    subset = matrix.iloc[train_indices].apply(pd.to_numeric, errors="coerce")
    target_series = pd.Series(target[train_indices], index=subset.index)
    # Constant columns are expected at this training-fold-only screen.  Avoid
    # one numpy warning per column while computing Spearman correlation.
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        correlations = subset.rank(method="average").corrwith(target_series.rank(method="average"))
    coverage = subset.notna().sum()
    spread = subset.std(skipna=True)
    correlations[(coverage < 4) | (spread <= 1.0e-12)] = 0.0
    scores = correlations.abs().fillna(0.0)
    ranked = sorted(
        ((float(scores[column]), column) for column in subset), key=lambda item: (-item[0], item[1])
    )
    return [column for _, column in ranked[:count]]


def schedule_lpt(durations: Iterable[float], workers: int) -> float:
    loads = [0.0] * max(1, int(workers))
    for duration in sorted((float(value) for value in durations), reverse=True):
        index = min(range(len(loads)), key=loads.__getitem__)
        loads[index] += duration
    return float(max(loads, default=0.0))
