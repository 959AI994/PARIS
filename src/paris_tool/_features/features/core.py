"""Portable PARIS feature extraction.

The initial implementation operates on a compact JSON graph so feature logic and
CLI behavior can be developed on macOS. Linux integrations can export the same
representation from DREAMPlace/OpenROAD databases.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean, pstdev

from ..design import Design, Net
from ..io import Candidate


def _summary(prefix: str, values: Iterable[float]) -> dict[str, float]:
    data = sorted(float(value) for value in values)
    if not data:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p90": 0.0,
        }

    def percentile(q: float) -> float:
        if len(data) == 1:
            return data[0]
        position = q * (len(data) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return data[lower]
        fraction = position - lower
        return data[lower] * (1.0 - fraction) + data[upper] * fraction

    return {
        f"{prefix}_min": data[0],
        f"{prefix}_max": data[-1],
        f"{prefix}_mean": fmean(data),
        f"{prefix}_std": pstdev(data) if len(data) > 1 else 0.0,
        f"{prefix}_p50": percentile(0.5),
        f"{prefix}_p90": percentile(0.9),
    }


def _cut_net_stats(design: Design, candidate: Candidate) -> tuple[list[Net], list[int]]:
    cut_nets = list(design.iter_cut_nets(candidate))
    return cut_nets, [len(net.pins) for net in cut_nets]


def _area_features(design: Design, candidate: Candidate) -> dict[str, float]:
    macro_area = [0.0] * design.die_count
    macro_count = [0] * design.die_count
    for name, die in candidate.assignment.items():
        macro_area[die] += design.nodes[name].area
        macro_count[die] += 1
    total_area = [
        macro_area[die] + design.cell_area_by_die[die]
        for die in range(design.die_count)
    ]
    area_sum = sum(total_area)
    count_sum = sum(macro_count)
    features: dict[str, float] = {
        "area_imbalance": (
            (max(total_area) - min(total_area)) / area_sum if area_sum else 0.0
        ),
        "object_count_imbalance": (
            (max(macro_count) - min(macro_count)) / count_sum if count_sum else 0.0
        ),
    }
    for die in range(design.die_count):
        features[f"die_{die}_object_area"] = macro_area[die]
        features[f"die_{die}_total_area"] = total_area[die]
        features[f"die_{die}_object_count"] = float(macro_count[die])
        if design.die_width and design.die_height:
            die_area = design.die_width * design.die_height
            features[f"die_{die}_utilization"] = total_area[die] / die_area
    return features


def _hierarchy_features(design: Design, candidate: Candidate) -> dict[str, float]:
    per_hierarchy: dict[str, list[str]] = defaultdict(list)
    for name, node in design.nodes.items():
        if node.hierarchy:
            per_hierarchy[node.hierarchy].append(name)
    cohesion: list[float] = []
    area_weighted: list[tuple[float, float]] = []
    for hierarchy in design.hierarchy_names():
        names = per_hierarchy.get(hierarchy, [])
        memory_count = len(names)
        if memory_count == 0:
            continue
        bottom_count = sum(candidate.assignment[name] == 0 for name in names)
        logic_count = design.hierarchy_cell_counts.get(hierarchy, 0)
        denominator = logic_count * memory_count + memory_count * (memory_count - 1) / 2
        numerator = logic_count * bottom_count + bottom_count * (bottom_count - 1) / 2
        value = numerator / denominator if denominator else bottom_count / memory_count
        cohesion.append(value)
        area_weighted.append((value, sum(design.nodes[name].area for name in names)))
    result = _summary("hierarchy_cohesion", cohesion)
    total_weight = sum(weight for _, weight in area_weighted)
    result["hierarchy_cohesion_area_weighted"] = (
        sum(value * weight for value, weight in area_weighted) / total_weight
        if total_weight
        else 0.0
    )
    return result


def _timing_features(design: Design, candidate: Candidate, cut_nets: list[Net]) -> dict[str, float]:
    total_criticality = sum(max(net.criticality, 0.0) for net in design.nets)
    cut_criticality = sum(max(net.criticality, 0.0) for net in cut_nets)
    total_weighted = sum(max(net.criticality, 0.0) * max(net.weight, 0.0) for net in design.nets)
    cut_weighted = sum(max(net.criticality, 0.0) * max(net.weight, 0.0) for net in cut_nets)
    clock_total = sum(net.net_type == "clock" for net in design.nets)
    clock_cut = sum(net.net_type == "clock" for net in cut_nets)
    reset_total = sum(net.net_type == "reset" for net in design.nets)
    reset_cut = sum(net.net_type == "reset" for net in cut_nets)
    return {
        "critical_cut_ratio": cut_criticality / total_criticality if total_criticality else 0.0,
        "critical_weighted_cut_ratio": cut_weighted / total_weighted if total_weighted else 0.0,
        "clock_cut_ratio": clock_cut / clock_total if clock_total else 0.0,
        "reset_cut_ratio": reset_cut / reset_total if reset_total else 0.0,
        "cut_criticality_mean": fmean([net.criticality for net in cut_nets]) if cut_nets else 0.0,
        "cut_criticality_max": max((net.criticality for net in cut_nets), default=0.0),
    }


def _physical_features(design: Design, candidate: Candidate, cut_nets: list[Net]) -> dict[str, float]:
    hpwl = 0.0
    critical_hpwl = 0.0
    known_net_count = 0
    cut_span = []
    for net in design.nets:
        nodes = [design.nodes[name] for name in net.pins]
        if any(node.x is None or node.y is None for node in nodes):
            continue
        span = (max(node.x for node in nodes) - min(node.x for node in nodes)) + (
            max(node.y for node in nodes) - min(node.y for node in nodes)
        )
        hpwl += span * net.weight
        critical_hpwl += span * net.weight * net.criticality
        known_net_count += 1
        if len({candidate.assignment[name] for name in net.pins}) > 1:
            cut_span.append(span)
    result = {
        "prototype_hpwl": hpwl,
        "prototype_timing_weighted_hpwl": critical_hpwl,
        "prototype_position_coverage": known_net_count / len(design.nets) if design.nets else 0.0,
    }
    result.update(_summary("cut_planar_span", cut_span))
    if design.die_width and design.die_height:
        diagonal = math.hypot(design.die_width, design.die_height)
        result["cut_span_normalized_mean"] = (
            result["cut_planar_span_mean"] / diagonal if diagonal else 0.0
        )
    return result


def extract_features(design: Design, candidate: Candidate) -> dict[str, float]:
    design.validate_candidate(candidate)
    cut_nets, cut_degrees = _cut_net_stats(design, candidate)
    total_weight = sum(max(net.weight, 0.0) for net in design.nets)
    cut_weight = sum(max(net.weight, 0.0) for net in cut_nets)
    features: dict[str, float] = {
        "cut_count": float(len(cut_nets)),
        "normalized_cut_count": len(cut_nets) / len(design.nets) if design.nets else 0.0,
        "weighted_cut_ratio": cut_weight / total_weight if total_weight else 0.0,
        "cut_pin_count": float(sum(cut_degrees)),
        "high_fanout_cut_ratio": (
            sum(degree >= 8 for degree in cut_degrees) / len(cut_degrees)
            if cut_degrees
            else 0.0
        ),
    }
    features.update(_summary("cut_degree", cut_degrees))
    features.update(_area_features(design, candidate))
    features.update(_hierarchy_features(design, candidate))
    features.update(_timing_features(design, candidate, cut_nets))
    features.update(_physical_features(design, candidate, cut_nets))
    return features


def add_relative_features(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if not rows:
        return []
    keys = sorted(set.intersection(*(set(row) for row in rows)))
    columns = {key: sorted(float(row[key]) for row in rows) for key in keys}
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    for key, values in columns.items():
        middle = len(values) // 2
        medians[key] = (
            values[middle]
            if len(values) % 2
            else (values[middle - 1] + values[middle]) / 2
        )
        q1 = values[int(0.25 * (len(values) - 1))]
        q3 = values[int(0.75 * (len(values) - 1))]
        iqrs[key] = q3 - q1
    output: list[dict[str, float]] = []
    for row in rows:
        enriched = dict(row)
        for key in keys:
            value = float(row[key])
            scale = iqrs[key]
            sorted_values = columns[key]
            tolerance = max(1e-12, 1e-9 * max(abs(other) for other in sorted_values))
            robust = 0.0 if abs(scale) <= tolerance else (value - medians[key]) / scale
            enriched[f"rel_{key}_robust_z"] = max(-10.0, min(10.0, robust))
            below = sum(other < value for other in sorted_values)
            equal = sum(other == value for other in sorted_values)
            enriched[f"rel_{key}_percentile"] = (below + 0.5 * equal) / len(sorted_values)
        output.append(enriched)
    return output
