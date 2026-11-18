"""Deterministic candidate-level features from saved post-CTS reports."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


ENDPOINT_RE = re.compile(
    r"^(\S.*?) \(([^()]*)\)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+"
    r"([-+0-9.eE]+)(?: \(VIOLATED\))?$"
)
QUANTILES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
KEYWORDS = ("dcache", "icache", "cce", "network", "mem", "core", "fe", "be")
HASH_BUCKETS = 64


def _semantic_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\[[0-9]+\]", "[#]", value)
    value = re.sub(r"_[0-9]+_", "_#_", value)
    return re.sub(r"[0-9]+", "#", value)


def _bucket(value: str, namespace: bytes) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=namespace).digest()
    return int.from_bytes(digest, "little") % HASH_BUCKETS


def _summaries(prefix: str, values: list[float], output: dict[str, float]) -> None:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return
    output[f"{prefix}_mean"] = float(array.mean())
    output[f"{prefix}_std"] = float(array.std())
    for quantile, value in zip(QUANTILES, np.quantile(array, QUANTILES)):
        output[f"{prefix}_q{int(round(quantile * 100)):02d}"] = float(value)


def features(endpoint_path: Path, detail_path: Path) -> dict[str, float]:
    """Parse the endpoint table and path-detail JSON without using L3 data."""

    required: list[float] = []
    actual: list[float] = []
    slack: list[float] = []
    masters: list[str] = []
    endpoints: list[str] = []
    for line in endpoint_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ENDPOINT_RE.match(line.strip())
        if not match:
            continue
        endpoints.append(match.group(1))
        masters.append(match.group(2))
        required.append(float(match.group(3)))
        actual.append(float(match.group(4)))
        slack.append(float(match.group(5)))

    output: dict[str, float] = {"endpoint_path_count": float(len(slack))}
    for prefix, values in (("required", required), ("actual", actual), ("slack", slack)):
        _summaries(prefix, values, output)
    if slack:
        output["reported_negative_slack_sum"] = float(-np.minimum(slack, 0).sum())

    count = max(len(endpoints), 1)
    for keyword in KEYWORDS:
        output[f"endpoint_{keyword}_ratio"] = (
            sum(keyword in endpoint.lower() for endpoint in endpoints) / count
        )
    output["endpoint_upper_ratio"] = sum("_upper" in master for master in masters) / count
    output["endpoint_bottom_ratio"] = sum("_bottom" in master for master in masters) / count
    output["endpoint_macro_ratio"] = sum("fakeram" in master.lower() for master in masters) / count
    output["endpoint_hierarchy_depth_mean"] = float(
        np.mean([endpoint.count("/") for endpoint in endpoints]) if endpoints else 0.0
    )

    endpoint_counts = np.zeros(HASH_BUCKETS, dtype=float)
    endpoint_severity = np.zeros(HASH_BUCKETS, dtype=float)
    for endpoint, value in zip(endpoints, slack):
        index = _bucket(_semantic_name(endpoint), b"paris-endpoint")
        endpoint_counts[index] += 1.0
        endpoint_severity[index] += max(0.0, -value)
    endpoint_counts /= max(endpoint_counts.sum(), 1.0)
    endpoint_severity /= max(endpoint_severity.sum(), 1.0)
    for index in range(HASH_BUCKETS):
        output[f"endpoint_hash_count_{index:02d}"] = float(endpoint_counts[index])
        output[f"endpoint_hash_severity_{index:02d}"] = float(endpoint_severity[index])

    text = detail_path.read_text(encoding="utf-8")
    # OpenSTA may redirect deprecation warnings ahead of its JSON payload.
    start = text.find('{')
    if start < 0 or any(not re.fullmatch(r"\[WARNING STA-[0-9]+\].*", line) for line in text[:start].splitlines() if line.strip()):
        raise ValueError("Unexpected text before CTS JSON")
    checks = json.loads(text[start:]).get("checks", [])
    vectors: dict[str, list[float]] = {
        "detail_slack": [],
        "detail_arrival": [],
        "detail_required": [],
        "detail_points": [],
        "detail_span_x": [],
        "detail_span_y": [],
        "detail_manhattan": [],
        "detail_slew_max": [],
        "detail_slew_mean": [],
    }
    endpoint_x: list[float] = []
    endpoint_y: list[float] = []
    pair_severity = np.zeros(HASH_BUCKETS, dtype=float)
    for check in checks:
        slack_value = float(check["slack"])
        vectors["detail_slack"].append(slack_value)
        vectors["detail_arrival"].append(float(check["data_arrival_time"]))
        vectors["detail_required"].append(float(check["required_time"]))
        pair = _semantic_name(f"{check.get('startpoint', '')}->{check.get('endpoint', '')}")
        pair_severity[_bucket(pair, b"paris-path-pair")] += max(0.0, -slack_value)
        points = check.get("source_path", []) + check.get("target_clock_path", [])
        vectors["detail_points"].append(float(len(points)))
        xs = [float(point["x"]) for point in points if "x" in point]
        ys = [float(point["y"]) for point in points if "y" in point]
        slews = [float(point["slew"]) for point in points if "slew" in point]
        if xs and ys:
            vectors["detail_span_x"].append(max(xs) - min(xs))
            vectors["detail_span_y"].append(max(ys) - min(ys))
            vectors["detail_manhattan"].append(
                sum(abs(right - left) for left, right in zip(xs, xs[1:]))
                + sum(abs(right - left) for left, right in zip(ys, ys[1:]))
            )
            endpoint_x.append(xs[-1])
            endpoint_y.append(ys[-1])
        if slews:
            vectors["detail_slew_max"].append(max(slews))
            vectors["detail_slew_mean"].append(float(np.mean(slews)))

    for prefix, values in vectors.items():
        _summaries(prefix, values, output)
    _summaries("endpoint_x", endpoint_x, output)
    _summaries("endpoint_y", endpoint_y, output)
    pair_severity /= max(pair_severity.sum(), 1.0)
    for index in range(HASH_BUCKETS):
        output[f"path_pair_hash_severity_{index:02d}"] = float(pair_severity[index])
    return output


def grt_artifacts(path: Path) -> dict[str, float]:
    """Read measured per-layer routing summaries from the final GRT report."""
    text = path.read_text(encoding="utf-8", errors="strict")
    marker = "[INFO GRT-0096] Final congestion report:"
    if marker not in text:
        raise ValueError("Missing final GRT congestion table")
    table = text.rsplit(marker, 1)[1]
    result, usage = {}, []
    pattern = r"(?m)^\s*(metal\d+|Total)\s+(\d+)\s+(\d+)\s+([0-9.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s*$"
    for row in re.findall(pattern, table):
        name = row[0].replace("metal", "layer")
        values = list(map(float, row[1:]))
        for key,value in zip(("resource", "demand", "usage_pct", "max_h", "max_v", "overflow"), values):
            result[f"phys_artifact_grt_{name}_{key}"] = value
        if name != "Total":
            usage.append(values[2])
    if len(usage) != 20 or "phys_artifact_grt_Total_usage_pct" not in result:
        raise ValueError("Expected the complete Nangate45 3D 20-layer GRT table")
    result["phys_artifact_grt_peak_layer_usage_pct"] = float(max(usage))
    result["phys_artifact_grt_layer_usage_std"] = float(np.std(usage))
    for key, pattern in {
        "pin_vias": r"Via related to pin nodes:\s*(\d+)",
        "steiner_vias": r"Via related Steiner nodes:\s*(\d+)",
        "via_count": r"Final number of vias:\s*(\d+)",
        "usage_3d": r"Final usage 3D:\s*(\d+)",
        "wirelength_um": r"Total wirelength:\s*([0-9.]+)\s+um",
    }.items():
        values = re.findall(pattern,text)
        if not values:
            raise ValueError("Missing GRT measurement: " + key)
        result["phys_artifact_grt_"+key] = float(values[-1])
    result["phys_artifact_grt_vias_per_um"] = result["phys_artifact_grt_via_count"] / result["phys_artifact_grt_wirelength_um"]
    result["grt_congestion"] = result["phys_artifact_grt_Total_usage_pct"] / 100.0
    return result
