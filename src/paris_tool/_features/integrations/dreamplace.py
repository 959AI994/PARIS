"""Export a production PARIS design cache from Place-MoL/DREAMPlace."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ..cache import DesignCache, write_design_cache
from ..errors import ValidationError
from ..timing import apply_net_timing, load_timing_table


def _decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hierarchy(name: str) -> str:
    for delimiter in ("/", "."):
        if delimiter in name:
            return name.rsplit(delimiter, 1)[0]
    return ""


def _net_type(name: str) -> str:
    lowered = name.lower()
    if "clk" in lowered or "clock" in lowered:
        return "clock"
    if "reset" in lowered or "rst" in lowered:
        return "reset"
    if "scan" in lowered or "test" in lowered:
        return "test"
    return "signal"


def _place_mol_args(config_path: Path, benchmark: str, seed: int, use_cuda: bool) -> SimpleNamespace:
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    config.setdefault("gpu", 0)
    config["use_cuda"] = bool(use_cuda)
    config.setdefault("use_timer_for_evaluation", False)
    config.setdefault("grid", 224)
    config["benchmark"] = benchmark
    config["seed"] = seed
    config["partition_params"] = config.get("partition_params", {})
    config["macro_refine_params"] = config.get("macro_refine_params", {})
    config["macro_place_params"] = config.get("macro_place_params", {})
    config["macro_legalize_params"] = config.get("macro_legalize_params", {})
    return SimpleNamespace(**config)


def _install_place_mol_imports(place_mol_root: Path) -> None:
    roots = [
        place_mol_root,
        place_mol_root / "DREAMPlace" / "install",
        place_mol_root / "DREAMPlace" / "install" / "dreamplace",
    ]
    for root in reversed(roots):
        value = str(root)
        if value not in sys.path:
            sys.path.insert(0, value)


def export_open3dbench_design(
    place_mol_root: str | Path,
    benchmark: str,
    output: str | Path,
    *,
    config_file: str = "or_3D.json",
    seed: int = 3,
    use_cuda: bool = False,
    timing_file: str | Path | None = None,
    clock_period: float | None = None,
) -> DesignCache:
    place_root = Path(place_mol_root).resolve()
    if not (place_root / "src" / "run_dmp.py").is_file():
        raise ValidationError(f"invalid Place-MoL root: {place_root}")
    _install_place_mol_imports(place_root)
    try:
        from src.run_dmp import ProblemInstance
    except Exception as exc:
        raise ValidationError(
            "cannot import Place-MoL/DREAMPlace. Build DREAMPlace inside the place container first: "
            f"{exc}"
        ) from exc
    config_path = Path(config_file)
    if not config_path.is_absolute():
        config_path = place_root / "config" / config_file
    args = _place_mol_args(config_path, benchmark, seed, use_cuda)
    instance = ProblemInstance(args, benchmark, rand_init=False)
    instance.determine_macro()
    db = instance.dmp_placedb
    macro_ids = sorted(int(value) for value in instance.macros)
    macro_id_set = set(macro_ids)
    physical_count = int(db.num_physical_nodes - db.num_terminal_NIs)
    cell_ids = [node_id for node_id in range(physical_count) if node_id not in macro_id_set]
    cell_id_set = set(cell_ids)
    all_names = [_decode(value) for value in db.node_names]

    node_rows = []
    macro_id_to_index: dict[int, int] = {}
    for index, node_id in enumerate(macro_ids):
        macro_id_to_index[node_id] = index
        name = all_names[node_id]
        width = float(db.node_size_x[node_id])
        height = float(db.node_size_y[node_id])
        node_rows.append(
            {
                "name": name,
                "area": width * height,
                "width": width,
                "height": height,
                "hierarchy": _hierarchy(name),
                "criticality": 0.0,
                "x": float(db.node_x[node_id]),
                "y": float(db.node_y[node_id]),
                "fixed_die": None,
                "attributes": {
                    "placedb_id": node_id,
                    "pin_count": len(db.node2pin_map[node_id]),
                },
            }
        )

    hierarchy_cell_counts: Counter[str] = Counter()
    hierarchy_cell_area: Counter[str] = Counter()
    total_cell_area = 0.0
    for node_id in cell_ids:
        name = all_names[node_id]
        hierarchy = _hierarchy(name)
        area = float(db.node_size_x[node_id]) * float(db.node_size_y[node_id])
        hierarchy_cell_counts[hierarchy] += 1
        hierarchy_cell_area[hierarchy] += area
        total_cell_area += area

    net_rows = []
    offsets = [0]
    incidence: list[int] = []
    net_names = getattr(db, "net_names", None)
    for net_id, pin_ids in enumerate(db.net2pin_map):
        macro_indices = sorted(
            {
                macro_id_to_index[int(db.pin2node_map[int(pin_id)])]
                for pin_id in pin_ids
                if int(db.pin2node_map[int(pin_id)]) in macro_id_set
            }
        )
        if not macro_indices:
            continue
        connected_cell_count = len(
            {
                int(db.pin2node_map[int(pin_id)])
                for pin_id in pin_ids
                if int(db.pin2node_map[int(pin_id)]) in cell_id_set
            }
        )
        name = _decode(net_names[net_id]) if net_names is not None else f"net_{net_id}"
        net_rows.append(
            {
                "name": name,
                "weight": 1.0 + math.log1p(connected_cell_count),
                "criticality": 0.0,
                "net_type": _net_type(name),
                "attributes": {
                    "placedb_id": net_id,
                    "degree": len(pin_ids),
                    "connected_cell_count": connected_cell_count,
                    "structural_weight": 1.0 + math.log1p(connected_cell_count),
                },
            }
        )
        incidence.extend(macro_indices)
        offsets.append(len(incidence))

    nodes = pd.DataFrame(node_rows)
    nets = pd.DataFrame(net_rows)
    timing_report = None
    if timing_file:
        timing = load_timing_table(timing_file, clock_period)
        nets, timing_report = apply_net_timing(nets, timing)

    metadata = {
        "source": "Place-MoL/DREAMPlace",
        "place_mol_root": str(place_root),
        "benchmark": benchmark,
        "config_file": str(config_path),
        "seed": seed,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(place_root.parent),
        "physical_node_count": physical_count,
        "standard_cell_count": len(cell_ids),
        "hierarchy_cell_area": dict(hierarchy_cell_area),
        "timing_report": timing_report,
        "bounds": {
            "xl": float(db.xl),
            "yl": float(db.yl),
            "xh": float(db.xh),
            "yh": float(db.yh),
        },
    }
    return write_design_cache(
        output,
        benchmark,
        nodes,
        nets,
        np.asarray(offsets, dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        die_count=2,
        cell_area_by_die=[total_cell_area, 0.0],
        die_width=float(db.xh) - float(db.xl),
        die_height=float(db.yh) - float(db.yl),
        hierarchy_cell_counts=dict(hierarchy_cell_counts),
        metadata=metadata,
    )
