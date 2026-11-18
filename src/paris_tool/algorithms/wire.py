"""PARIS4 GR-to-DR wire-summary features, preserved from the research implementation."""

import numpy as np
import pandas as pd


def _proxy_tns(frame: pd.DataFrame, slack: str, output: str) -> pd.DataFrame:
    endpoint = frame.groupby(
        ["design", "candidate_id", "sink_pin", "corner"],
        as_index=False,
        sort=False,
    )[slack].min()
    endpoint[output] = endpoint[slack].clip(upper=0.0)
    return endpoint.groupby(["design", "candidate_id"], as_index=False, sort=False)[
        output
    ].sum()


def _candidate_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["wire_prediction_residual"] = (
        data["predicted_dr_wire_delay"] - data["gr_wire_delay"]
    )
    data["predicted_dr_slack_proxy"] = (
        data["gr_slack"] - data["wire_prediction_residual"]
    )
    grouped = data.groupby(["design", "candidate_id"], sort=False)
    result = grouped.agg(
        phys_wire_path_count=("predicted_dr_wire_delay", "size"),
        phys_wire_gr_delay_sum=("gr_wire_delay", "sum"),
        phys_wire_pred_delay_sum=("predicted_dr_wire_delay", "sum"),
        phys_wire_pred_delay_mean=("predicted_dr_wire_delay", "mean"),
        phys_wire_pred_delay_std=("predicted_dr_wire_delay", "std"),
        phys_wire_pred_delay_max=("predicted_dr_wire_delay", "max"),
        phys_wire_residual_sum=("wire_prediction_residual", "sum"),
        phys_wire_residual_mean=("wire_prediction_residual", "mean"),
        phys_wire_residual_std=("wire_prediction_residual", "std"),
        phys_wire_residual_min=("wire_prediction_residual", "min"),
        phys_wire_residual_max=("wire_prediction_residual", "max"),
        phys_wire_gr_negative_path_count=("gr_slack", lambda x: int((x < 0).sum())),
    ).reset_index()
    quantiles = (
        grouped["wire_prediction_residual"].quantile([0.1, 0.5, 0.9, 0.99]).unstack()
    )
    quantiles.columns = [
        "phys_wire_residual_p10",
        "phys_wire_residual_p50",
        "phys_wire_residual_p90",
        "phys_wire_residual_p99",
    ]
    result = result.merge(
        quantiles.reset_index(), on=["design", "candidate_id"], validate="one_to_one"
    )
    critical = data[data["gr_slack"] < 0]
    if not critical.empty:
        critical_stats = (
            critical.groupby(["design", "candidate_id"], sort=False)[
                "wire_prediction_residual"
            ]
            .agg(["sum", "mean", "std", "max"])
            .rename(
                columns={
                    "sum": "phys_wire_critical_residual_sum",
                    "mean": "phys_wire_critical_residual_mean",
                    "std": "phys_wire_critical_residual_std",
                    "max": "phys_wire_critical_residual_max",
                }
            )
            .reset_index()
        )
        result = result.merge(
            critical_stats,
            on=["design", "candidate_id"],
            how="left",
            validate="one_to_one",
        )
    for column in (
        "phys_wire_critical_residual_sum",
        "phys_wire_critical_residual_mean",
        "phys_wire_critical_residual_std",
        "phys_wire_critical_residual_max",
    ):
        if column not in result:
            result[column] = 0.0
        result[column] = result[column].fillna(0.0)
    result = result.merge(
        _proxy_tns(data, "gr_slack", "phys_wire_gr_tns_proxy"),
        on=["design", "candidate_id"],
        validate="one_to_one",
    ).merge(
        _proxy_tns(data, "predicted_dr_slack_proxy", "phys_wire_pred_tns_proxy"),
        on=["design", "candidate_id"],
        validate="one_to_one",
    )
    result["phys_wire_tns_proxy_delta"] = (
        result["phys_wire_pred_tns_proxy"] - result["phys_wire_gr_tns_proxy"]
    )
    return result.fillna(0.0)
