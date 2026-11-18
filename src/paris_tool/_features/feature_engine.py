"""Production batch feature extraction from a PARIS DesignCache."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cache import DesignCache
from .errors import ValidationError
from .features.core import extract_features
from .io import Candidate, load_candidates

FEATURE_SCHEMA_VERSION = "2.0"


@dataclass
class FeatureResult:
    table: pd.DataFrame
    metadata: dict[str, Any]

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".csv":
            self.table.to_csv(target, index=False)
        else:
            self.table.to_parquet(target, index=False, compression="zstd")
        metadata_path = target.with_suffix(target.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _candidate_matrix(cache: DesignCache, candidates: list[Candidate]) -> np.ndarray:
    names = cache.object_names
    expected = set(names)
    matrix = np.empty((len(candidates), len(names)), dtype=np.int16)
    for row, candidate in enumerate(candidates):
        if candidate.design != cache.name:
            raise ValidationError(
                f"candidate {candidate.candidate_id} belongs to {candidate.design}, expected {cache.name}"
            )
        if set(candidate.assignment) != expected:
            raise ValidationError(f"candidate {candidate.candidate_id} object set differs from cache")
        matrix[row] = [candidate.assignment[name] for name in names]
    return matrix


def _net_attributes(cache: DesignCache) -> tuple[np.ndarray, np.ndarray]:
    full_degree = np.empty(len(cache.nets), dtype=float)
    connected_cells = np.zeros(len(cache.nets), dtype=float)
    macro_degree = cache.net_offsets[1:] - cache.net_offsets[:-1]
    for index, value in enumerate(cache.nets.get("attributes", pd.Series([{}] * len(cache.nets)))):
        attributes = json.loads(value) if isinstance(value, str) and value else (value or {})
        full_degree[index] = float(attributes.get("degree", macro_degree[index]))
        connected_cells[index] = float(attributes.get("connected_cell_count", 0.0))
    return full_degree, connected_cells


def _net_arrays(cache: DesignCache) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    criticality = cache.nets.get("criticality", pd.Series(np.zeros(len(cache.nets)))).to_numpy(dtype=float)
    degree, connected_cells = _net_attributes(cache)
    # Existing caches used unit weights. Derive a portable structural weight from
    # the full net connectivity, then amplify timing-critical nets.
    weight = (1.0 + np.log1p(np.maximum(connected_cells, 0.0))) * (
        1.0 + np.clip(criticality, 0.0, 1.0)
    )
    return weight, criticality, degree, connected_cells


def _batch_cut_features(cache: DesignCache, assignment: np.ndarray) -> pd.DataFrame:
    candidate_count = assignment.shape[0]
    net_count = len(cache.nets)
    weights, criticality, degree, connected_cells = _net_arrays(cache)
    cut = np.zeros((candidate_count, net_count), dtype=bool)
    for net_id in range(net_count):
        start, end = int(cache.net_offsets[net_id]), int(cache.net_offsets[net_id + 1])
        pins = cache.net_pins[start:end]
        if not len(pins):
            continue
        values = assignment[:, pins]
        macro_cut = values.min(axis=1) != values.max(axis=1)
        # Open3DBench is memory-on-logic: logic cells are fixed on die 0.
        # Nets touching logic therefore cross tiers whenever any attached macro
        # is placed on an upper die, even if the cache stores only one macro pin.
        logic_cut = (connected_cells[net_id] > 0) & (values.max(axis=1) > 0)
        cut[:, net_id] = macro_cut | logic_cut
    cut_count = cut.sum(axis=1)
    cut_weight = cut @ weights
    total_weight = weights.sum()
    critical_weight = weights * criticality
    cut_critical = cut @ criticality
    cut_critical_weight = cut @ critical_weight
    clock_mask = cache.nets["net_type"].astype(str).str.lower().eq("clock").to_numpy()
    reset_mask = cache.nets["net_type"].astype(str).str.lower().eq("reset").to_numpy()
    rows = {
        "cut_count": cut_count.astype(float),
        "normalized_cut_count": cut_count / max(net_count, 1),
        "weighted_cut_ratio": cut_weight / total_weight if total_weight else np.zeros(candidate_count),
        "critical_cut_ratio": cut_critical / criticality.sum() if criticality.sum() else np.zeros(candidate_count),
        "critical_weighted_cut_ratio": (
            cut_critical_weight / critical_weight.sum()
            if critical_weight.sum()
            else np.zeros(candidate_count)
        ),
        "cut_pin_count": cut @ degree.astype(float),
        "cut_connected_cell_count": cut @ connected_cells,
        "logic_endpoint_cut_ratio": np.divide(
            (cut & (connected_cells[None, :] > 0)).sum(axis=1),
            np.maximum((connected_cells > 0).sum(), 1),
        ),
        "clock_cut_ratio": (
            cut[:, clock_mask].sum(axis=1) / clock_mask.sum()
            if clock_mask.sum()
            else np.zeros(candidate_count)
        ),
        "reset_cut_ratio": (
            cut[:, reset_mask].sum(axis=1) / reset_mask.sum()
            if reset_mask.sum()
            else np.zeros(candidate_count)
        ),
        "high_fanout_cut_ratio": np.divide(
            (cut & (degree[None, :] >= 8)).sum(axis=1),
            cut_count,
            out=np.zeros(candidate_count, dtype=float),
            where=cut_count > 0,
        ),
    }
    for label, percentile in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99)):
        values = []
        for row in range(candidate_count):
            selected = degree[cut[row]]
            values.append(float(np.percentile(selected, percentile)) if len(selected) else 0.0)
        rows[f"cut_degree_{label}"] = values
    rows["cut_degree_min"] = [float(degree[cut[row]].min()) if cut[row].any() else 0.0 for row in range(candidate_count)]
    rows["cut_degree_max"] = [float(degree[cut[row]].max()) if cut[row].any() else 0.0 for row in range(candidate_count)]
    rows["cut_degree_mean"] = [float(degree[cut[row]].mean()) if cut[row].any() else 0.0 for row in range(candidate_count)]
    rows["cut_degree_std"] = [float(degree[cut[row]].std()) if cut[row].any() else 0.0 for row in range(candidate_count)]
    return pd.DataFrame(rows)


def _hierarchy_pca_features(
    cache: DesignCache,
    assignment: np.ndarray,
    components: int,
) -> pd.DataFrame:
    """Compress per-cluster DOPP cohesion without losing cluster identity."""

    components = max(int(components), 0)
    output = np.zeros((len(assignment), components), dtype=float)
    if components == 0 or "hierarchy" not in cache.nodes:
        return pd.DataFrame(output, columns=[f"hierarchy_pca_{i:02d}" for i in range(components)])
    hierarchies = cache.nodes["hierarchy"].fillna("").astype(str).to_numpy()
    columns = []
    for hierarchy in sorted(set(hierarchies) | set(cache.hierarchy_cell_counts)):
        macro_ids = np.flatnonzero(hierarchies == hierarchy)
        if not len(macro_ids):
            continue
        memory_count = len(macro_ids)
        bottom_count = (assignment[:, macro_ids] == 0).sum(axis=1).astype(float)
        logic_count = float(cache.hierarchy_cell_counts.get(hierarchy, 0))
        denominator = logic_count * memory_count + memory_count * (memory_count - 1) / 2
        numerator = logic_count * bottom_count + bottom_count * (bottom_count - 1) / 2
        columns.append(numerator / denominator if denominator else bottom_count / memory_count)
    if not columns:
        return pd.DataFrame(output, columns=[f"hierarchy_pca_{i:02d}" for i in range(components)])
    matrix = np.column_stack(columns)
    varying = np.nanstd(matrix, axis=0) > 1e-12
    matrix = matrix[:, varying]
    if matrix.shape[1]:
        matrix = (matrix - matrix.mean(axis=0)) / np.where(matrix.std(axis=0) > 1e-12, matrix.std(axis=0), 1.0)
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        usable = min(components, len(singular))
        scores = left[:, :usable] * singular[:usable]
        # SVD signs are arbitrary. Fix them so feature extraction is reproducible.
        for index in range(usable):
            loading = right[index]
            pivot = int(np.argmax(np.abs(loading)))
            if loading[pivot] < 0:
                scores[:, index] *= -1
        output[:, :usable] = scores
    return pd.DataFrame(output, columns=[f"hierarchy_pca_{i:02d}" for i in range(components)])


def _area_features(cache: DesignCache, assignment: np.ndarray) -> pd.DataFrame:
    areas = cache.nodes["area"].to_numpy(dtype=float)
    widths = cache.nodes["width"].to_numpy(dtype=float)
    heights = cache.nodes["height"].to_numpy(dtype=float)
    die_count = cache.die_count
    candidate_count = assignment.shape[0]
    cell_area = np.asarray(cache.cell_area_by_die, dtype=float)
    rows: dict[str, Any] = {}
    total_area = np.zeros((candidate_count, die_count), dtype=float)
    object_count = np.zeros((candidate_count, die_count), dtype=float)
    for die in range(die_count):
        mask = assignment == die
        object_area = mask @ areas
        total_area[:, die] = object_area + cell_area[die]
        object_count[:, die] = mask.sum(axis=1)
        rows[f"die_{die}_object_area"] = object_area
        rows[f"die_{die}_total_area"] = total_area[:, die]
        rows[f"die_{die}_object_count"] = object_count[:, die]
        if cache.die_width and cache.die_height:
            rows[f"die_{die}_utilization"] = total_area[:, die] / (
                cache.die_width * cache.die_height
            )
    rows["area_imbalance"] = np.divide(
        total_area.max(axis=1) - total_area.min(axis=1),
        total_area.sum(axis=1),
        out=np.zeros(candidate_count),
        where=total_area.sum(axis=1) > 0,
    )
    rows["object_count_imbalance"] = np.divide(
        object_count.max(axis=1) - object_count.min(axis=1),
        object_count.sum(axis=1),
        out=np.zeros(candidate_count),
        where=object_count.sum(axis=1) > 0,
    )
    aspect = np.divide(widths, heights, out=np.zeros_like(widths), where=heights > 0)
    threshold = np.percentile(areas, 90) if len(areas) else 0.0
    for die in range(die_count):
        rows[f"die_{die}_large_macro_area_ratio"] = [
            float(areas[(assignment[row] == die) & (areas >= threshold)].sum() / max(areas.sum(), 1e-12))
            for row in range(candidate_count)
        ]
        rows[f"die_{die}_aspect_std"] = [
            float(aspect[assignment[row] == die].std()) if np.any(assignment[row] == die) else 0.0
            for row in range(candidate_count)
        ]
    return pd.DataFrame(rows)


def _path_features(
    assignment: np.ndarray, names: list[str], paths: list[Mapping[str, Any]] | None
) -> pd.DataFrame:
    count = assignment.shape[0]
    if not paths:
        return pd.DataFrame(
            {
                "timing_path_count": np.zeros(count),
                "path_crossing_mean": np.zeros(count),
                "path_crossing_max": np.zeros(count),
                "path_crossing_critical_weighted": np.zeros(count),
            }
        )
    name_to_index = {name: index for index, name in enumerate(names)}
    crossings = np.zeros((count, len(paths)), dtype=float)
    weights = np.zeros(len(paths), dtype=float)
    valid = 0
    for path_id, path in enumerate(paths):
        indices = [name_to_index[name] for name in path.get("nodes", []) if name in name_to_index]
        if len(indices) < 2:
            continue
        values = assignment[:, indices]
        crossings[:, path_id] = np.sum(values[:, 1:] != values[:, :-1], axis=1)
        weights[path_id] = float(path.get("criticality", 1.0))
        valid += 1
    weighted_denominator = weights.sum()
    return pd.DataFrame(
        {
            "timing_path_count": np.full(count, valid, dtype=float),
            "path_crossing_mean": crossings.mean(axis=1) if len(paths) else np.zeros(count),
            "path_crossing_max": crossings.max(axis=1) if len(paths) else np.zeros(count),
            "path_crossing_critical_weighted": (
                crossings @ weights / weighted_denominator
                if weighted_denominator
                else np.zeros(count)
            ),
        }
    )


def _rudy_features(cache: DesignCache, assignment: np.ndarray, grid_size: int) -> pd.DataFrame:
    candidate_count = assignment.shape[0]
    if not cache.die_width or not cache.die_height or "x" not in cache.nodes or "y" not in cache.nodes:
        return pd.DataFrame({"rudy_position_coverage": np.zeros(candidate_count)})
    x = cache.nodes["x"].to_numpy(dtype=float)
    y = cache.nodes["y"].to_numpy(dtype=float)
    valid_nodes = np.isfinite(x) & np.isfinite(y)
    rows = []
    for candidate_id in range(candidate_count):
        per_die = []
        for die in range(cache.die_count):
            demand = np.zeros((grid_size, grid_size), dtype=np.float32)
            for net_id in range(len(cache.nets)):
                start, end = int(cache.net_offsets[net_id]), int(cache.net_offsets[net_id + 1])
                pins = cache.net_pins[start:end]
                pins = pins[(assignment[candidate_id, pins] == die) & valid_nodes[pins]]
                if len(pins) < 2:
                    continue
                xl, xh = float(x[pins].min()), float(x[pins].max())
                yl, yh = float(y[pins].min()), float(y[pins].max())
                bx0 = max(0, min(grid_size - 1, int(xl / cache.die_width * grid_size)))
                bx1 = max(0, min(grid_size - 1, int(xh / cache.die_width * grid_size)))
                by0 = max(0, min(grid_size - 1, int(yl / cache.die_height * grid_size)))
                by1 = max(0, min(grid_size - 1, int(yh / cache.die_height * grid_size)))
                cells = max((bx1 - bx0 + 1) * (by1 - by0 + 1), 1)
                weight = float(cache.nets.iloc[net_id].get("weight", 1.0))
                demand[by0 : by1 + 1, bx0 : bx1 + 1] += weight / cells
            per_die.append(demand)
        flattened = np.concatenate([array.ravel() for array in per_die])
        positive = flattened[flattened > 0]
        total = flattened.sum()
        probabilities = positive / total if total > 0 else positive
        entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum()) if len(probabilities) else 0.0
        die_max = [float(array.max()) for array in per_die]
        rows.append(
            {
                "rudy_position_coverage": float(valid_nodes.mean()),
                "rudy_mean": float(flattened.mean()),
                "rudy_max": float(flattened.max()),
                "rudy_p90": float(np.percentile(flattened, 90)),
                "rudy_p95": float(np.percentile(flattened, 95)),
                "rudy_p99": float(np.percentile(flattened, 99)),
                "rudy_nonzero_ratio": float(np.mean(flattened > 0)),
                "rudy_spatial_entropy": entropy,
                "rudy_die_max_imbalance": max(die_max) - min(die_max),
            }
        )
    return pd.DataFrame(rows)


def _relative_frame(frame: pd.DataFrame, identity: set[str]) -> pd.DataFrame:
    relative: dict[str, pd.Series] = {}
    for column in frame.columns:
        if column in identity or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        values = frame[column].astype(float)
        median = values.median()
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        scale = float(q3 - q1)
        tolerance = max(1e-12, 1e-9 * max(float(values.abs().max()), 1.0))
        if not math.isfinite(scale) or abs(scale) <= tolerance:
            robust = pd.Series(np.zeros(len(values), dtype=float), index=values.index)
        else:
            robust = ((values - median) / scale).clip(-10.0, 10.0)
        relative[f"rel_{column}_robust_z"] = robust
        relative[f"rel_{column}_percentile"] = values.rank(method="average", pct=True)
    return pd.concat([frame, pd.DataFrame(relative, index=frame.index)], axis=1)


def extract_feature_table(
    cache: DesignCache,
    candidates: list[Candidate],
    config: Mapping[str, Any],
    *,
    timing_paths: list[Mapping[str, Any]] | None = None,
) -> FeatureResult:
    start = time.perf_counter()
    assignment = _candidate_matrix(cache, candidates)
    groups = set(config.get("feature_groups", []))
    criticality = pd.to_numeric(cache.nets.get("criticality", 0.0), errors="coerce").fillna(0.0)
    critical_coverage = float((criticality > 0).mean()) if len(criticality) else 0.0
    criticality_config = config.get("criticality", {})
    if (
        "timing" in groups
        and criticality_config.get("require_coverage", False)
        and critical_coverage < float(criticality_config.get("min_net_coverage", 0.0))
    ):
        raise ValidationError(
            "timing features requested but cache net criticality coverage is "
            f"{critical_coverage:.6f}; attach a design-level STA source first"
        )
    frames = [
        pd.DataFrame(
            {
                "design": [candidate.design for candidate in candidates],
                "candidate_id": [candidate.candidate_id for candidate in candidates],
                "candidate_hash": [candidate.assignment_hash for candidate in candidates],
            }
        )
    ]
    if groups & {"dopp_proxy", "cut_topology", "timing"}:
        frames.append(_batch_cut_features(cache, assignment))
    if groups & {"legality_area", "prototype_physical"}:
        frames.append(_area_features(cache, assignment))
    if "hierarchy" in groups or "prototype_physical" in groups:
        portable = cache.to_portable_design()
        portable_rows = [extract_features(portable, candidate) for candidate in candidates]
        keep_prefixes = (
            "hierarchy_",
            "prototype_",
            "cut_planar_",
            "cut_span_normalized",
        )
        frames.append(
            pd.DataFrame(
                [
                    {key: value for key, value in row.items() if key.startswith(keep_prefixes)}
                    for row in portable_rows
                ]
            )
        )
        frames.append(
            _hierarchy_pca_features(
                cache,
                assignment,
                int(config.get("hierarchy", {}).get("pca_components", 12)),
            )
        )
    if "timing" in groups:
        frames.append(_path_features(assignment, cache.object_names, timing_paths))
    if "rudy" in groups and config.get("rudy", {}).get("enabled", True):
        frames.append(_rudy_features(cache, assignment, int(config.get("rudy", {}).get("grid_size", 64))))
    table = pd.concat(frames, axis=1)
    table = table.loc[:, ~table.columns.duplicated()]
    coarse_path = config.get("coarse_features_path")
    if coarse_path:
        coarse = pd.read_csv(coarse_path) if str(coarse_path).endswith(".csv") else pd.read_parquet(coarse_path)
        required = {"design", "candidate_id"}
        if not required.issubset(coarse.columns):
            raise ValidationError("coarse feature table requires design and candidate_id")
        table = table.merge(coarse, on=["design", "candidate_id"], how="left", validate="one_to_one")
        coarse_columns = [column for column in coarse.columns if column not in required]
        for column in coarse_columns:
            table[f"{column}_missing"] = table[column].isna().astype(float)
    if config.get("relative_features", True):
        table = _relative_frame(table, {"design", "candidate_id", "candidate_hash"})
    table = table.replace([np.inf, -np.inf], np.nan).fillna(float(config.get("missing_value", 0.0)))
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "design": cache.name,
        "design_fingerprint": cache.fingerprint,
        "candidate_count": len(candidates),
        "feature_count": len(table.columns) - 3,
        "feature_groups": sorted(groups),
        "critical_net_coverage": critical_coverage,
        "runtime_sec": time.perf_counter() - start,
        "columns": table.columns.tolist(),
    }
    return FeatureResult(table, metadata)


def extract_from_paths(
    cache_path: str | Path,
    candidates_path: str | Path,
    config: Mapping[str, Any],
    output: str | Path,
    *,
    timing_paths: list[Mapping[str, Any]] | None = None,
) -> FeatureResult:
    cache = DesignCache.load(cache_path)
    candidates = load_candidates(candidates_path)
    result = extract_feature_table(cache, candidates, config, timing_paths=timing_paths)
    result.write(output)
    return result
