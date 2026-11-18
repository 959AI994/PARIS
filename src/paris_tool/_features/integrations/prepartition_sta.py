"""Pre-partition OpenSTA timing for DesignCache.

Uses the same 2D floorplan DEF that Place-MoL reads, plus the OpenROAD-3D
evaluation SDC and 2D Nangate/fakeram liberty. Net slacks are matched to cache
nets by canonical name (not by 3D post-route path heuristics).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..cache import DesignCache, write_design_cache
from ..errors import ValidationError
from ..timing import apply_net_timing, load_timing_table
from .openroad_timing import map_paths_to_cache, parse_violated_paths

TCL_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "dump_prepartition_sta.tcl"

DESIGN_SDC = {
    "ariane133": "ariane133/ariane.sdc",
    "ariane136": "ariane136/ariane.sdc",
    "bp": "black_parrot/black_parrot.sdc",
    "bp_be": "bp_be_top/bp_be_top.sdc",
    "bp_fe": "bp_fe_top/bp_fe_top.sdc",
    "bp_multi": "bp_multi_top/bp_multi_top.sdc",
    "bp_quad": "bp_quad/bsg_chip.sdc",
    "swerv_wrapper": "swerv_wrapper/swerv_wrapper.sdc",
}

_TRACK_RE = re.compile(r"LAYER metal(1[1-9]|20)\s*;")


def _unique_existing(paths: Iterable[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in paths:
        if path.is_file():
            seen[str(path.resolve())] = path.resolve()
    return list(seen.values())


def clean_floorplan_def(source: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace") as inf, output.open(
        "w", encoding="utf-8"
    ) as ouf:
        for line in inf:
            if _TRACK_RE.search(line):
                continue
            ouf.write(line)
    return output


def _platform_files(benchmarks: Path) -> tuple[list[Path], list[Path]]:
    lef_root = benchmarks / "nangate45" / "lef"
    lib_root = benchmarks / "nangate45" / "lib"
    lefs = [
        lef_root / "NangateOpenCellLibrary.tech.lef",
        lef_root / "NangateOpenCellLibrary.macro.mod.lef",
        *sorted(lef_root.glob("fakeram45_*.lef")),
    ]
    libs = [
        lib_root / "NangateOpenCellLibrary_typical.lib",
        *sorted(lib_root.glob("fakeram45_*.lib")),
    ]
    lefs = _unique_existing(lefs)
    libs = _unique_existing(libs)
    if len(lefs) < 3 or len(libs) < 2:
        raise ValidationError(f"incomplete 2D Nangate LEF/LIB under {benchmarks}")
    return lefs, libs


def dump_prepartition_sta(
    design: str,
    *,
    benchmarks: Path,
    flow_root: Path,
    output: Path,
    openroad: str = "openroad",
    path_count: int = 2000,
    threads: int = 4,
) -> dict[str, Any]:
    if design not in DESIGN_SDC:
        raise ValidationError(f"unsupported design for pre-partition STA: {design}")
    def_path = benchmarks / "or_3D" / design / "2_2_floorplan_io.def"
    sdc_path = flow_root / "designs" / "nangate45_3D" / DESIGN_SDC[design]
    if not def_path.is_file():
        raise ValidationError(f"floorplan DEF not found: {def_path}")
    if not sdc_path.is_file():
        raise ValidationError(f"SDC not found: {sdc_path}")
    if not TCL_SCRIPT.is_file():
        raise ValidationError(f"STA dump script not found: {TCL_SCRIPT}")
    output.mkdir(parents=True, exist_ok=True)
    cleaned = clean_floorplan_def(def_path, output / "cleaned.def")
    lefs, libs = _platform_files(benchmarks)
    env = os.environ.copy()
    env.update(
        {
            "LEF_FILES": " ".join(str(path) for path in lefs),
            "LIB_FILES": " ".join(str(path) for path in libs),
            "DEF_FILE": str(cleaned),
            "SDC_FILE": str(sdc_path),
            "OUT_DIR": str(output),
            "PATH_COUNT": str(path_count),
            "THREADS": str(threads),
        }
    )
    log_path = output / "openroad.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [openroad, "-exit", str(TCL_SCRIPT)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise ValidationError(f"OpenROAD STA failed for {design}; see {log_path}")
    nets = output / "net_slack.csv"
    paths = output / "paths.rpt"
    if not nets.is_file() or not paths.is_file():
        raise ValidationError(f"STA dump incomplete for {design}; see {log_path}")
    return {
        "design": design,
        "def": str(def_path),
        "sdc": str(sdc_path),
        "net_slack": str(nets),
        "paths": str(paths),
        "log": str(log_path),
    }


def _reset_timing(cache: DesignCache) -> tuple[pd.DataFrame, pd.DataFrame]:
    nets = cache.nets.copy()
    nodes = cache.nodes.copy()
    nets["criticality"] = 0.0
    nets["weight"] = _structural_weights(nets)
    nodes["criticality"] = 0.0
    return nodes, nets


def _structural_weights(nets: pd.DataFrame) -> np.ndarray:
    weights = []
    attributes = nets.get("attributes", pd.Series([{}] * len(nets), index=nets.index))
    for value in attributes:
        payload = json.loads(value) if isinstance(value, str) and value else (value or {})
        connected_cells = max(float(payload.get("connected_cell_count", 0.0)), 0.0)
        weights.append(float(payload.get("structural_weight", 1.0 + np.log1p(connected_cells))))
    return np.asarray(weights, dtype=float)


def _node_criticality_from_nets(cache: DesignCache, nets: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    values = np.zeros(len(nodes), dtype=float)
    crit = nets["criticality"].to_numpy(dtype=float)
    for net_id, value in enumerate(crit):
        if value <= 0:
            continue
        start = int(cache.net_offsets[net_id])
        end = int(cache.net_offsets[net_id + 1])
        pins = cache.net_pins[start:end]
        values[pins] = np.maximum(values[pins], value)
    out = nodes.copy()
    out["criticality"] = values
    return out


def attach_prepartition_timing(
    cache_path: str | Path,
    sta_dir: str | Path,
    *,
    max_paths: int = 512,
) -> dict[str, Any]:
    cache = DesignCache.load(cache_path)
    sta_root = Path(sta_dir)
    timing_csv = sta_root / "net_slack.csv"
    path_rpt = sta_root / "paths.rpt"
    if not timing_csv.is_file():
        raise ValidationError(f"STA net table not found: {timing_csv}")
    period = 1.0
    sta_meta: dict[str, Any] = {}
    if (sta_root / "sta_meta.json").is_file():
        sta_meta = json.loads((sta_root / "sta_meta.json").read_text(encoding="utf-8"))
        period = float(sta_meta.get("clock_period", period) or period)
    timing = load_timing_table(timing_csv, period)
    if "slack" in timing.columns:
        timing["slack"] = pd.to_numeric(timing["slack"], errors="coerce")
        timing = timing[timing["slack"].isna() | np.isfinite(timing["slack"])]
    nodes, nets = _reset_timing(cache)
    nets, net_report = apply_net_timing(nets, timing)
    nets["weight"] = _structural_weights(nets) * (
        1.0 + nets["criticality"].to_numpy(dtype=float)
    )
    nodes = _node_criticality_from_nets(cache, nets, nodes)
    paths: list[dict[str, Any]] = []
    map_report: dict[str, Any] = {}
    if path_rpt.is_file():
        parsed = parse_violated_paths(path_rpt.read_text(encoding="utf-8", errors="replace"), period)
        tmp_cache = DesignCache(cache.root, cache.manifest, nodes, nets, cache.net_offsets, cache.net_pins)
        _frame, paths, map_report = map_paths_to_cache(parsed, tmp_cache)
        paths = paths[: max(max_paths, 0)] if max_paths else paths
    nets_file = cache.root / "timing_nets.csv"
    paths_file = cache.root / "timing_paths.json"
    timing.to_csv(nets_file, index=False)
    paths_file.write_text(json.dumps({"paths": paths}, indent=2) + "\n", encoding="utf-8")
    metadata = dict(cache.manifest.get("metadata") or {})
    report = {
        "source": "prepartition_opensta",
        "clock_period": period,
        "clock_count": int(sta_meta.get("clock_count", 0)),
        "period_normalization": str(
            sta_meta.get("period_normalization", "minimum_clock_period")
        ),
        "sta_dir": str(sta_root),
        "nets": net_report,
        "map": map_report,
        "critical_net_count": int((nets["criticality"] > 0).sum()),
        "critical_node_count": int((nodes["criticality"] > 0).sum()),
        "path_count": len(paths),
        "timing_nets": str(nets_file),
        "timing_paths": str(paths_file),
    }
    metadata["timing_report"] = report
    write_design_cache(
        cache.root,
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
    return report


def attach_prepartition_timing_runs(
    runs_root: str | Path,
    *,
    benchmarks: str | Path,
    flow_root: str | Path,
    designs: Iterable[str] | None = None,
    openroad: str = "openroad",
    path_count: int = 2000,
    max_paths: int = 512,
    threads: int = 4,
) -> dict[str, Any]:
    root = Path(runs_root)
    bench = Path(benchmarks)
    flow = Path(flow_root)
    names = [path.name for path in sorted((root / "cache").iterdir()) if path.is_dir()]
    if designs:
        wanted = set(designs)
        names = [name for name in names if name in wanted]
    reports: dict[str, Any] = {}
    for name in names:
        sta_dir = root / "sta_prepartition" / name
        try:
            dump = dump_prepartition_sta(
                name,
                benchmarks=bench,
                flow_root=flow,
                output=sta_dir,
                openroad=openroad,
                path_count=path_count,
                threads=threads,
            )
            attached = attach_prepartition_timing(root / "cache" / name, sta_dir, max_paths=max_paths)
            reports[name] = {**dump, **attached}
        except ValidationError as exc:
            reports[name] = {"error": str(exc)}
    return reports
