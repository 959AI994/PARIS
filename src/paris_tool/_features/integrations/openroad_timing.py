"""Backfill DesignCache timing from existing OpenROAD STA path reports.

OpenROAD candidate labels only store chip-level TNS/WNS. The same artifact
directories already contain OpenSTA ``violated_paths`` reports. Those paths are
mapped onto cache macros/nets so ``critical_cut_ratio`` and path-crossing
features become non-zero without re-running Place-MoL export.

Timing is treated as a *design-level* annotation: one or a few reference STA
reports are aggregated with max criticality. Candidate TNS labels are not copied
into the feature table.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import DesignCache, write_design_cache
from ..errors import ValidationError
from ..timing import (
    apply_net_timing,
    apply_node_timing,
    canonical_name,
    criticality_from_slack,
)

_INSTANCE_RE = re.compile(r"^\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[v^]\s+(\S+)\s+\(", re.MULTILINE)
_SLACK_RE = re.compile(r"(-?\d+\.\d+)\s+slack \((?:VIOLATED|MET)\)")
_CLOCK_EDGE_RE = re.compile(
    r"^\s+(\d+\.\d+)\s+(\d+\.\d+)\s+clock \S+ \(rise edge\)",
    re.MULTILINE,
)
_STARTPOINT_RE = re.compile(r"^Startpoint:\s+(\S+)", re.MULTILINE)
_ENDPOINT_RE = re.compile(r"^Endpoint:\s+(\S+)", re.MULTILINE)
_CLK_PERIOD_SET_RE = re.compile(r"set\s+clk_period\s+(\d+(?:\.\d+)?)")
_CREATE_CLOCK_RE = re.compile(r"create_clock\b.*?-period\s+(\d+(?:\.\d+)?)")


def parse_clock_period(text: str, default: float = 1.0) -> float:
    match = _CLK_PERIOD_SET_RE.search(text)
    if match:
        return float(match.group(1))
    match = _CREATE_CLOCK_RE.search(text)
    if match:
        return float(match.group(1))
    return default


def parse_violated_paths(text: str, default_period: float = 1.0) -> list[dict[str, Any]]:
    blocks = re.split(r"\n(?=Startpoint:)", text)
    paths: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if "Startpoint:" not in block:
            continue
        slack_match = _SLACK_RE.findall(block)
        if not slack_match:
            continue
        slack = float(slack_match[-1])
        period = default_period
        edges = _CLOCK_EDGE_RE.findall(block)
        if len(edges) >= 2:
            required = float(edges[-1][0])
            if required > 0:
                period = required
        instances = []
        for raw in _INSTANCE_RE.findall(block):
            cell = raw.rsplit("/", 1)[0] if "/" in raw else raw
            instances.append(cell)
        start = _STARTPOINT_RE.search(block)
        end = _ENDPOINT_RE.search(block)
        if start:
            instances.insert(0, start.group(1))
        if end:
            instances.append(end.group(1))
        paths.append(
            {
                "path_id": f"p{index}",
                "slack": slack,
                "period": period,
                "criticality": criticality_from_slack(slack, period),
                "instances": instances,
            }
        )
    return paths


def find_violated_paths(artifact_dir: str | Path) -> Path | None:
    root = Path(artifact_dir)
    exact = list(root.glob("work/results/*/*/*/violated_paths"))
    if exact:
        return exact[0]
    nested = root / "violated_paths"
    if nested.is_file():
        return nested
    hits = list(root.glob("**/violated_paths"))
    return hits[0] if hits else None


def find_sdc(artifact_dir: str | Path) -> Path | None:
    root = Path(artifact_dir)
    for pattern in ("work/results/*/*/*/6_final.sdc", "work/results/*/*/*/1_synth.sdc"):
        hits = list(root.glob(pattern))
        if hits:
            return hits[0]
    hits = list(root.glob("**/*.sdc"))
    return hits[0] if hits else None


def _region_for_instance(
    instance: str,
    name_to_index: Mapping[str, int],
    macros_by_length: Sequence[str],
    prefix_index: dict[str, list[int]],
) -> tuple[str, list[int]] | None:
    inst = canonical_name(instance)
    slash_inst = inst.replace(".", "/")
    for macro in macros_by_length:
        canon = canonical_name(macro)
        slash_macro = canon.replace(".", "/")
        if (
            inst == canon
            or slash_inst == slash_macro
            or inst.startswith(canon + "/")
            or inst.startswith(canon + ".")
            or slash_inst.startswith(slash_macro + "/")
        ):
            return macro, [name_to_index[macro]]
    parts = [part for part in slash_inst.split("/") if part]
    for depth in range(len(parts) - 1, 0, -1):
        prefix = "/".join(parts[:depth])
        hits = prefix_index.get(prefix)
        if hits:
            return prefix, list(hits)
    return None


def _prefix_index(
    macros: Sequence[str],
    hierarchies: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}

    def add(macro_id: int, name: str) -> None:
        slash = canonical_name(name).replace(".", "/")
        parts = [part for part in slash.split("/") if part]
        for depth in range(1, len(parts) + 1):
            prefix = "/".join(parts[:depth])
            bucket = index.setdefault(prefix, [])
            if macro_id not in bucket:
                bucket.append(macro_id)

    for macro_id, name in enumerate(macros):
        add(macro_id, name)
    if hierarchies is not None:
        for macro_id, hierarchy in enumerate(hierarchies):
            if hierarchy:
                add(macro_id, str(hierarchy))
    return index


def map_paths_to_cache(
    paths: Sequence[Mapping[str, Any]],
    cache: DesignCache,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    macros = cache.object_names
    if not macros:
        raise ValidationError("design cache has no macros")
    areas = cache.nodes["area"].to_numpy(dtype=float) if "area" in cache.nodes else np.ones(len(macros))
    macros_by_length = sorted(macros, key=len, reverse=True)
    name_to_index = {name: index for index, name in enumerate(macros)}
    hierarchies = (
        cache.nodes["hierarchy"].astype(str).tolist()
        if "hierarchy" in cache.nodes.columns
        else None
    )
    prefix_index = _prefix_index(macros, hierarchies)
    pin_sets = []
    for net_id in range(len(cache.nets)):
        start = int(cache.net_offsets[net_id])
        end = int(cache.net_offsets[net_id + 1])
        pin_sets.append(set(int(index) for index in cache.net_pins[start:end]))
    net_names = cache.nets["name"].astype(str).tolist()
    net_canon = [canonical_name(name) for name in net_names]
    net_canon_index = {name: net_id for net_id, name in enumerate(net_canon)}

    net_crit = np.zeros(len(net_names), dtype=float)
    node_crit = np.zeros(len(macros), dtype=float)
    mapped_paths: list[dict[str, Any]] = []
    instance_mapped = 0
    instance_total = 0
    multi_region = 0

    for path in paths:
        criticality = float(path.get("criticality", 0.0))
        slack = float(path.get("slack", 0.0))
        period = float(path.get("period", 1.0))
        regions: list[tuple[str, tuple[int, ...]]] = []
        for instance in path.get("instances", []):
            instance_total += 1
            mapped = _region_for_instance(str(instance), name_to_index, macros_by_length, prefix_index)
            if mapped is None:
                continue
            instance_mapped += 1
            key, macro_ids = mapped
            packed = (key, tuple(macro_ids))
            if not regions or regions[-1] != packed:
                regions.append(packed)
            for macro_id in macro_ids:
                node_crit[macro_id] = max(node_crit[macro_id], criticality)
            canon_inst = canonical_name(str(instance))
            net_id = net_canon_index.get(canon_inst)
            if net_id is not None:
                net_crit[net_id] = max(net_crit[net_id], criticality)
        if len(regions) >= 2:
            multi_region += 1
            representatives: list[str] = []
            for _key, macro_ids in regions:
                best = max(macro_ids, key=lambda index: float(areas[index]))
                name = macros[best]
                if not representatives or representatives[-1] != name:
                    representatives.append(name)
            if len(representatives) >= 2:
                mapped_paths.append(
                    {
                        "path_id": str(path.get("path_id", len(mapped_paths))),
                        "nodes": representatives,
                        "slack": slack,
                        "period": period,
                        "criticality": criticality,
                    }
                )
            for left, right in zip(regions, regions[1:], strict=False):
                left_ids = set(left[1])
                right_ids = set(right[1])
                if not left_ids.isdisjoint(right_ids):
                    continue
                for net_id, pins in enumerate(pin_sets):
                    if pins & left_ids and pins & right_ids:
                        net_crit[net_id] = max(net_crit[net_id], criticality)

    records: list[dict[str, Any]] = []
    for net_id, value in enumerate(net_crit):
        if value > 0:
            records.append(
                {
                    "name": net_names[net_id],
                    "criticality": float(value),
                    "object_type": "net",
                }
            )
    for node_id, value in enumerate(node_crit):
        if value > 0:
            records.append(
                {
                    "name": macros[node_id],
                    "criticality": float(value),
                    "object_type": "macro",
                }
            )
    frame = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["name", "criticality", "object_type"]
    )
    unique_paths: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in mapped_paths:
        key = tuple(record["nodes"])
        current = unique_paths.get(key)
        if current is None or record["criticality"] > current["criticality"]:
            unique_paths[key] = record
    collapsed = sorted(unique_paths.values(), key=lambda item: item["criticality"], reverse=True)
    report = {
        "sta_path_count": len(paths),
        "instance_total": instance_total,
        "instance_mapped": instance_mapped,
        "instance_match_ratio": instance_mapped / instance_total if instance_total else 0.0,
        "multi_region_paths": multi_region,
        "unique_macro_paths": len(collapsed),
        "critical_net_count": int((net_crit > 0).sum()),
        "critical_node_count": int((node_crit > 0).sum()),
    }
    return frame, collapsed, report


def _successful_label_dirs(labels_dir: Path) -> list[Path]:
    found: list[Path] = []
    for label_path in sorted(labels_dir.glob("*/label.json")):
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "success":
            found.append(label_path.parent)
    return found


def _select_reports(label_dirs: Sequence[Path], max_reports: int) -> list[Path]:
    if max_reports <= 0:
        return list(label_dirs)
    return list(label_dirs[:max_reports])


def attach_openroad_timing(
    cache_path: str | Path,
    labels_dir: str | Path,
    *,
    max_reports: int = 1,
    max_paths: int = 512,
    clock_period: float | None = None,
) -> dict[str, Any]:
    cache = DesignCache.load(cache_path)
    label_root = Path(labels_dir)
    if not label_root.is_dir():
        raise ValidationError(f"labels directory not found: {label_root}")
    selected = _select_reports(_successful_label_dirs(label_root), max_reports)
    if not selected:
        raise ValidationError(f"no successful OpenROAD labels under {label_root}")

    period = clock_period
    parsed_paths: list[dict[str, Any]] = []
    used_reports: list[str] = []
    for artifact_dir in selected:
        report_path = find_violated_paths(artifact_dir)
        if report_path is None:
            continue
        if period is None:
            sdc = find_sdc(artifact_dir)
            if sdc is not None:
                period = parse_clock_period(sdc.read_text(encoding="utf-8", errors="replace"))
        text = report_path.read_text(encoding="utf-8", errors="replace")
        parsed_paths.extend(parse_violated_paths(text, default_period=period or 1.0))
        used_reports.append(str(report_path))
    if not parsed_paths:
        raise ValidationError(f"no violated_paths reports under {label_root}")

    timing, paths, map_report = map_paths_to_cache(parsed_paths, cache)
    paths = paths[: max(max_paths, 0)] if max_paths else paths
    nets = cache.nets
    nodes = cache.nodes
    net_report = {"matched_nets": 0, "match_ratio": 0.0}
    node_report = {"matched_nodes": 0, "match_ratio": 0.0}
    if not timing.empty:
        nets, net_report = apply_net_timing(nets, timing)
        nodes, node_report = apply_node_timing(nodes, timing)
    nets = nets.copy()
    structural = []
    for value in nets.get("attributes", pd.Series([{}] * len(nets))):
        attributes = json.loads(value) if isinstance(value, str) and value else (value or {})
        cells = float(attributes.get("connected_cell_count", 0.0))
        structural.append(float(attributes.get("structural_weight", 1.0 + np.log1p(cells))))
    nets["weight"] = np.asarray(structural) * (1.0 + nets["criticality"].astype(float).clip(0.0, 1.0))

    cache_root = cache.root
    timing_dir = cache_root
    nets_file = timing_dir / "timing_nets.csv"
    paths_file = timing_dir / "timing_paths.json"
    if not timing.empty:
        timing.to_csv(nets_file, index=False)
    else:
        pd.DataFrame(columns=["name", "criticality", "object_type"]).to_csv(nets_file, index=False)
    paths_file.write_text(json.dumps({"paths": paths}, indent=2) + "\n", encoding="utf-8")

    metadata = dict(cache.manifest.get("metadata") or {})
    timing_report = {
        "source": "openroad_violated_paths",
        "clock_period": period or 1.0,
        "reports": used_reports,
        "max_reports": max_reports,
        "map": map_report,
        "nets": net_report,
        "nodes": node_report,
        "timing_nets": str(nets_file),
        "timing_paths": str(paths_file),
        "path_count": len(paths),
        "candidate_label_derived": True,
    }
    metadata["timing_report"] = timing_report
    write_design_cache(
        cache_root,
        cache.name,
        nodes,
        nets,
        np.asarray(cache.net_offsets, dtype=np.int64).copy(),
        np.asarray(cache.net_pins, dtype=np.int64).copy(),
        die_count=cache.die_count,
        cell_area_by_die=cache.cell_area_by_die,
        die_width=cache.die_width,
        die_height=cache.die_height,
        hierarchy_cell_counts=cache.hierarchy_cell_counts,
        metadata=metadata,
    )
    return timing_report


def attach_openroad_timing_runs(
    runs_root: str | Path,
    *,
    designs: Iterable[str] | None = None,
    max_reports: int = 1,
    max_paths: int = 512,
) -> dict[str, Any]:
    root = Path(runs_root)
    cache_root = root / "cache"
    labels_root = root / "labels"
    names = [path.name for path in sorted(cache_root.iterdir()) if path.is_dir()]
    if designs:
        wanted = set(designs)
        names = [name for name in names if name in wanted]
    reports: dict[str, Any] = {}
    for name in names:
        label_dir = labels_root / name
        if not label_dir.is_dir():
            continue
        if not any(label_dir.glob("*/label.json")):
            continue
        try:
            reports[name] = attach_openroad_timing(
                cache_root / name,
                label_dir,
                max_reports=max_reports,
                max_paths=max_paths,
            )
        except ValidationError as exc:
            reports[name] = {"error": str(exc)}
    return reports
