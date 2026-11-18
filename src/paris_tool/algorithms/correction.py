"""PARIS4 candidate-error correction; retained research numerics."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import Ridge
from paris_tool._vendor.ranking.common import deployable_numeric_columns

IDS = ["design", "candidate_id"]


def tns_cost(x):
    return np.log1p(-np.asarray(x, float))


def correction_features(x, branch):
    if branch == "legacy":
        columns = [
            c
            for c in x
            if not c.startswith(("phys_", "ep_")) and c != "near_neighbor_group"
        ]
    elif branch == "legacy_grt":
        # True input-union ablation: preserve every legacy input and append
        # saved GRT features. This is distinct from endpoint+physical `joint`.
        columns = [
            c for c in x if not c.startswith("ep_") and c != "near_neighbor_group"
        ]
    elif branch == "grt":
        columns = (
            IDS
            + [c for c in x if c.startswith("phys_")]
            + ["fast_l1_tns", "grt_l2_tns"]
        )
    elif branch == "endpoint":
        columns = (
            IDS + [c for c in x if c.startswith("ep_")] + ["fast_l1_tns", "grt_l2_tns"]
        )
    elif branch == "joint":
        columns = (
            IDS
            + [c for c in x if c.startswith(("phys_", "ep_"))]
            + ["fast_l1_tns", "grt_l2_tns"]
        )
    else:
        raise ValueError(branch)
    z = x[list(dict.fromkeys(columns))].copy()
    z["timing_stage_delta"] = tns_cost(x.grt_l2_tns) - tns_cost(x.fast_l1_tns)
    if branch in ("grt", "joint", "legacy_grt") and not any(
        c.startswith("phys_") for c in z
    ):
        raise ValueError("physical branch requested without physical features")
    if branch in ("endpoint", "joint") and not any(c.startswith("ep_") for c in z):
        raise ValueError(
            "endpoint branch requested without cross-fitted endpoint signal"
        )
    return z


@dataclass
class Correction:
    names: list
    fill: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    selected: np.ndarray
    model: object
    family: str

    @classmethod
    def fit(cls, x, truth, prior_oof, spec):
        family, count, alpha = spec
        names = deployable_numeric_columns(x)
        if not names:
            raise ValueError("no valid correction features")
        a = x[names].to_numpy(float).copy()
        a[~np.isfinite(a)] = np.nan
        fill = np.nanmedian(a, axis=0)
        a = np.where(np.isfinite(a), a, fill)
        center = np.median(a, axis=0)
        scale = np.quantile(a, 0.75, axis=0) - np.quantile(a, 0.25, axis=0)
        scale = np.where(scale > 1e-9, scale, np.maximum(np.std(a, axis=0), 1.0))
        a = np.clip((a - center) / scale, -8.0, 8.0)
        residual = np.asarray(truth) - prior_oof
        # Features are screened against held-out PARIS errors, not total L3 cost.
        ranks = pd.DataFrame(a).rank().to_numpy()
        ranks -= ranks.mean(axis=0)
        target = pd.Series(residual).rank().to_numpy()
        target -= target.mean()
        corr = np.abs(ranks.T @ target) / np.maximum(
            np.sqrt((ranks * ranks).sum(axis=0) * (target * target).sum()), 1e-12
        )
        order = np.argsort(-corr, kind="stable")
        chosen = []
        # Remove redundant columns (including raw/relative copies) before fitting.
        for i in order:
            if chosen:
                denominator = np.sqrt(
                    (ranks[:, chosen] ** 2).sum(axis=0) * (ranks[:, i] ** 2).sum()
                )
                similarity = np.abs(ranks[:, chosen].T @ ranks[:, i]) / np.maximum(
                    denominator, 1e-12
                )
                if np.any(similarity > 0.98):
                    continue
            chosen.append(i)
            if len(chosen) >= min(count, max(2, len(x) // 4)):
                break
        chosen = np.asarray(chosen, dtype=int)
        a = a[:, chosen]
        if family == "ridge":
            model = Ridge(alpha=alpha).fit(a, residual)
        else:
            # Scalar score differences, not independent pair decisions/vote counts.
            i, j = np.triu_indices(len(x), 1)
            signs = np.sign(np.asarray(truth)[i] - np.asarray(truth)[j])
            keep = signs != 0  # no arbitrary order imposed on true ties
            d = a[i[keep]] - a[j[keep]]
            b = prior_oof[i[keep]] - prior_oof[j[keep]]
            signs = signs[keep]

            def loss(w):
                margin = signs * (b + d @ w)
                value = np.logaddexp(0.0, -margin).mean() + 0.5 * alpha * (w @ w)
                grad = d.T @ (-signs * expit(-margin)) / max(1, len(signs)) + alpha * w
                return value, grad

            model = (
                minimize(loss, np.zeros(a.shape[1]), jac=True, method="L-BFGS-B").x
                if len(signs)
                else np.zeros(a.shape[1])
            )
        return cls(names, fill, center, scale, chosen, model, family)

    def predict(self, x):
        a = x.reindex(columns=self.names).to_numpy(float)
        a = np.where(np.isfinite(a), a, self.fill)
        a = np.clip((a - self.center) / self.scale, -8.0, 8.0)[:, self.selected]
        return self.model.predict(a) if self.family == "ridge" else a @ self.model
