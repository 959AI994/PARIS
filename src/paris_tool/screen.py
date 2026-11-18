"""Budget-aware screening and offline comparison; no confidence claims."""

from pathlib import Path
import numpy as np
import pandas as pd
from . import data
from .utils import save


def select(frame, budget):
    if budget < 1:
        raise ValueError("budget must be positive")
    if frame.duplicated(data.KEY).any():
        raise ValueError("Duplicate ranked candidates")
    score = "rank_score" if "rank_score" in frame else "gr_cost"
    if not np.isfinite(frame[score]).all():
        raise ValueError("Nonfinite screening score")
    # Deterministic exploitation plus one GR-disagreement exploration candidate
    # when the budget allows. This is a heuristic, not calibrated uncertainty.
    f = frame.sort_values([score, *data.KEY], kind="stable")
    n = min(budget, len(f))
    explore = 1 if n >= 5 and "gr_tns" in f else 0
    chosen = f.iloc[: n - explore].copy()
    chosen["selection_reason"] = "predicted_best"
    if explore:
        rest = f[~f.index.isin(chosen.index)].copy()
        disagreement = (f[score].rank(pct=True) - (-f.gr_tns).rank(pct=True)).abs()
        rest["disagreement"] = disagreement.loc[rest.index]
        p = (
            rest.sort_values(
                ["disagreement", *data.KEY],
                ascending=[False, True, True],
                kind="stable",
            )
            .iloc[:1]
            .copy()
        )
        p["selection_reason"] = "GR_model_disagreement"
        chosen = pd.concat([chosen, p], ignore_index=True)
    return chosen


def benchmark(predictions, truth, output, budgets=(5, 10, 20)):
    p = data.predictions(predictions)
    y = pd.read_parquet(truth)
    if y.duplicated(data.KEY).any():
        raise ValueError("Duplicate truth keys")
    if not set(["final_tns", *data.KEY]) <= set(y):
        raise ValueError("Missing evaluation labels")
    f = p.merge(y[data.KEY + ["final_tns"]], on=data.KEY, validate="one_to_one")
    if len(f) != len(p):
        raise ValueError("Prediction/label inventory differs")
    rows = []
    for d, g in f.groupby("design"):
        for k in budgets:
            for method in ["paris4", "gr"]:
                # Select before consulting any joined final_tns value.
                x = g[data.KEY + ["rank_score", "gr_tns"]].copy()
                if method == "gr":
                    x["rank_score"] = -x.gr_tns
                picked = (
                    select(x, k)
                    if method == "paris4"
                    else x.sort_values(["rank_score", *data.KEY], kind="stable").iloc[
                        : min(k, len(x))
                    ]
                )
                selected = g.merge(picked[data.KEY], on=data.KEY, validate="one_to_one")
                best = float(selected.final_tns.max())
                oracle = float(g.final_tns.max())
                rows.append(
                    dict(
                        design=d,
                        budget=min(k, len(g)),
                        method=method,
                        best_measured_tns=best,
                        regret_ns=oracle - best,
                        reference_pool_n=len(g),
                    )
                )
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    pd.DataFrame(rows).to_csv(out, index=False)
    return dict(
        status="complete",
        rows=len(rows),
        note="Offline replay on frozen predictions; not measured wall-clock speedup",
    )
