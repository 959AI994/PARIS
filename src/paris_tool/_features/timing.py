"""Timing criticality providers and name mapping diagnostics."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .errors import ValidationError


def criticality_from_slack(slack: float, period: float) -> float:
    if period <= 0 or not math.isfinite(period):
        raise ValidationError("clock period must be positive and finite")
    if not math.isfinite(slack):
        return 0.0
    return min(1.0, max(0.0, -slack / period))


def canonical_name(name: str) -> str:
    return str(name).strip().replace("\\", "").replace("[", "_").replace("]", "")


def load_timing_table(path: str | Path, default_period: float | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise ValidationError(f"timing data not found: {source}")
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("records", payload) if isinstance(payload, Mapping) else payload
        if not isinstance(records, list):
            raise ValidationError("timing JSON must be an array or contain a records array")
        frame = pd.DataFrame(records)
    else:
        frame = pd.read_csv(source)
    if "name" not in frame.columns:
        for alternative in ("net", "net_name", "object", "object_name"):
            if alternative in frame.columns:
                frame = frame.rename(columns={alternative: "name"})
                break
    if "name" not in frame.columns:
        raise ValidationError("timing table requires a name/net_name column")
    frame["name"] = frame["name"].astype(str)
    frame["canonical_name"] = frame["name"].map(canonical_name)
    if "criticality" not in frame.columns:
        if "slack" not in frame.columns:
            raise ValidationError("timing table requires criticality or slack")
        if "period" not in frame.columns:
            if default_period is None:
                raise ValidationError("slack timing data requires period or --clock-period")
            frame["period"] = default_period
        frame["criticality"] = [
            criticality_from_slack(float(slack), float(period))
            for slack, period in zip(frame["slack"], frame["period"], strict=True)
        ]
    frame["criticality"] = frame["criticality"].astype(float).clip(0.0, 1.0)
    if "object_type" not in frame.columns:
        frame["object_type"] = "net"
    return frame


def apply_net_timing(
    nets: pd.DataFrame, timing: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = nets.copy()
    output["canonical_name"] = output["name"].astype(str).map(canonical_name)
    net_timing = timing.copy()
    if "canonical_name" not in net_timing.columns:
        net_timing["canonical_name"] = net_timing["name"].astype(str).map(canonical_name)
    net_timing = net_timing[net_timing["object_type"].astype(str).str.lower().isin(["net", "signal", "clock"])]
    if net_timing.empty:
        output = output.drop(columns=["canonical_name"])
        return output, {
            "net_count": len(nets),
            "timing_record_count": len(timing),
            "matched_nets": 0,
            "match_ratio": 0.0,
            "unmatched_nets": len(nets),
        }
    if net_timing["canonical_name"].duplicated().any():
        net_timing = net_timing.groupby("canonical_name", as_index=False)["criticality"].max()
    else:
        net_timing = net_timing[["canonical_name", "criticality"]]
    merged = output.merge(
        net_timing.rename(columns={"criticality": "timing_criticality"}),
        on="canonical_name",
        how="left",
    )
    matched = int(merged["timing_criticality"].notna().sum())
    existing = merged.get("criticality", pd.Series([0.0] * len(merged)))
    merged["criticality"] = merged["timing_criticality"].fillna(existing).fillna(0.0)
    merged = merged.drop(columns=["canonical_name", "timing_criticality"])
    report = {
        "net_count": len(nets),
        "timing_record_count": len(timing),
        "matched_nets": matched,
        "match_ratio": matched / len(nets) if len(nets) else 0.0,
        "unmatched_nets": len(nets) - matched,
    }
    return merged, report


def apply_node_timing(
    nodes: pd.DataFrame, timing: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = nodes.copy()
    output["canonical_name"] = output["name"].astype(str).map(canonical_name)
    node_timing = timing.copy()
    if "canonical_name" not in node_timing.columns:
        node_timing["canonical_name"] = node_timing["name"].astype(str).map(canonical_name)
    allowed = {"node", "macro", "instance", "object"}
    node_timing = node_timing[node_timing["object_type"].astype(str).str.lower().isin(allowed)]
    if node_timing.empty:
        output = output.drop(columns=["canonical_name"])
        return output, {
            "node_count": len(nodes),
            "timing_record_count": len(timing),
            "matched_nodes": 0,
            "match_ratio": 0.0,
            "unmatched_nodes": len(nodes),
        }
    if node_timing["canonical_name"].duplicated().any():
        node_timing = node_timing.groupby("canonical_name", as_index=False)["criticality"].max()
    else:
        node_timing = node_timing[["canonical_name", "criticality"]]
    merged = output.merge(
        node_timing.rename(columns={"criticality": "timing_criticality"}),
        on="canonical_name",
        how="left",
    )
    matched = int(merged["timing_criticality"].notna().sum())
    existing = merged.get("criticality", pd.Series([0.0] * len(merged)))
    merged["criticality"] = merged["timing_criticality"].fillna(existing).fillna(0.0)
    merged = merged.drop(columns=["canonical_name", "timing_criticality"])
    report = {
        "node_count": len(nodes),
        "timing_record_count": len(timing),
        "matched_nodes": matched,
        "match_ratio": matched / len(nodes) if len(nodes) else 0.0,
        "unmatched_nodes": len(nodes) - matched,
    }
    return merged, report


def load_path_crossings(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("paths", payload) if isinstance(payload, Mapping) else payload
    else:
        records = []
        with source.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                nodes = [item.strip() for item in row.get("nodes", "").split(";") if item.strip()]
                records.append({**row, "nodes": nodes})
    if not isinstance(records, list):
        raise ValidationError("path timing data must be an array")
    normalized = []
    for record in records:
        nodes = record.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) < 2:
            continue
        normalized.append(
            {
                "path_id": str(record.get("path_id", len(normalized))),
                "nodes": [str(node) for node in nodes],
                "slack": float(record.get("slack", 0.0)),
                "period": float(record.get("period", 1.0)),
                "criticality": float(
                    record.get(
                        "criticality",
                        criticality_from_slack(
                            float(record.get("slack", 0.0)),
                            float(record.get("period", 1.0)),
                        ),
                    )
                ),
            }
        )
    return normalized
