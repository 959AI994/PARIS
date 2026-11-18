"""Deterministic candidate-level GR summaries; same calculation as the frozen experiment."""
KEY = ["design", "candidate_id"]
ARC = [
    "gr_wire_delay",
    "gr_sink_slew",
    "gr_wire_slew",
    "driver_slew",
    "hpwl",
    "num_sinks",
    "ss_length",
    "ss_r",
    "ss_c",
    "congestion_mean",
    "congestion_std",
    "macro_blockage",
    "macro_blockage_expanded",
    "gr_slack",
]
NET = ["gr_pi_r", "gr_pi_c1", "gr_pi_c2"]
GRID = [dict(max_depth=d, reg_lambda=l) for d in [2, 3] for l in [10.0, 100.0]]
PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    min_child_weight=3,
    reg_alpha=0.1,
    subsample=1.0,
    colsample_bytree=0.8,
    tree_method="hist",
    max_bin=128,
    n_jobs=2,
    random_state=17,
    objective="reg:squarederror",
)


def summaries(table, columns, prefix):
    import numpy as np

    grouped = table.groupby(KEY, sort=False)[columns]
    result = grouped.agg(["mean", "std", "min", "max", "median"])
    result.columns = [f"{prefix}{a}_{b}" for a, b in result.columns]
    for q, name in [(0.9, "p90"), (0.99, "p99")]:
        z = grouped.quantile(q)
        z.columns = [prefix + c + "_" + name for c in z]
        result = result.join(z)
    # A singleton set has population std zero; no test-fitted imputation.
    std = [c for c in result if c.endswith("_std")]
    result[std] = result[std].fillna(0.0)
    assert np.isfinite(result.to_numpy(float)).all()
    return result


def aggregate(source, net):
    import numpy as np
    import pandas as pd

    assert not any(
        c.startswith(("dr_", "final_", "target_", "phys_wire_")) for c in source
    )
    assert not source.duplicated(
        KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"]
    ).any()
    assert not net.duplicated(KEY + ["net_name", "driver_pin", "corner"]).any()
    for t, cols in [(source, ARC), (net, NET)]:
        for c in cols:
            t[c] = t[c].to_numpy(float)
        if not np.isfinite(t[cols].to_numpy()).all():
            raise ValueError("Nonfinite GR feature input")
    r = summaries(source, ARC, "gr_arc_")
    r["gr_arc_count"] = source.groupby(KEY, sort=False).size()
    critical = source[source.gr_slack < 0]
    ccols = [
        "gr_wire_delay",
        "gr_sink_slew",
        "driver_slew",
        "ss_r",
        "ss_c",
        "congestion_mean",
    ]
    if not critical.empty:
        z = critical.groupby(KEY, sort=False)[ccols].agg(["mean", "max"])
        z.columns = ["gr_negative_arc_" + a + "_" + b for a, b in z.columns]
        r = r.join(z)
    else:
        for col in ccols:
            for stat in ["mean", "max"]:
                r["gr_negative_arc_" + col + "_" + stat] = 0.0
    r["gr_negative_arc_count"] = critical.groupby(KEY, sort=False).size()
    # Zero is defined for the empty critical subset and flagged by count.
    neg = [c for c in r if c.startswith("gr_negative_arc_")]
    r[neg] = r[neg].fillna(0.0)
    r["gr_negative_arc_fraction"] = r.gr_negative_arc_count / r.gr_arc_count
    for transition in ["rise", "fall"]:
        subset = source[source.transition.eq(transition)]
        z = subset.groupby(KEY, sort=False)[["gr_wire_delay", "gr_sink_slew"]].mean()
        z.columns = ["gr_arc_" + transition + "_" + c + "_mean" for c in z.columns]
        r = r.join(z)
        r["gr_arc_" + transition + "_count"] = subset.groupby(KEY, sort=False).size()
    r = r.fillna(0.0)
    r = r.join(summaries(net, NET, "gr_net_"))
    r["gr_net_count"] = net.groupby(KEY, sort=False).size()
    assert np.isfinite(r.to_numpy(float)).all()
    return r.reset_index()

