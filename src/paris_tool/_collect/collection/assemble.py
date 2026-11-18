import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from paris_tool._collect.common.io import read_tsv

from .geometry import (
    BlockageCell,
    GCell,
    LayerRC,
    RoutePathIndex,
    ScalarGridIndex,
    bounding_box,
    expand_rectangle,
    hpwl_um,
    indexed_blockage_mean,
    indexed_congestion_statistics,
    path_features,
)
from .identity import unique_driver_and_sinks
from .models import NetRecord, PinRecord, Rectangle, Segment


def read_layer_rc(path: str | Path) -> dict[str, LayerRC]:
    result: dict[str, LayerRC] = {}
    for row in read_tsv(path):
        result[row["layer"]] = LayerRC(
            resistance_kohm_per_um=float(row["resistance_kohm_per_um"]),
            capacitance_ff_per_um=float(row["capacitance_ff_per_um"]),
        )
    if not result:
        raise ValueError(f"layer RC 导出为空: {path}")
    return result


def _timing_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (row["net_name"], row["sink_pin"], row["transition"])
        if key in result:
            raise ValueError(f"timing 导出主键重复: {key}")
        result[key] = row
    return result


def assemble_gr_tables(
    design: str,
    candidate_id: str,
    corner: str,
    selected_nets: Sequence[str],
    nets: Mapping[str, NetRecord],
    pins: Sequence[PinRecord],
    segments_by_net: Mapping[str, Sequence[Segment]],
    gr_timing: Sequence[dict[str, Any]],
    layer_rc: Mapping[str, LayerRC],
    g_cells: Sequence[GCell],
    obstructions: Sequence[Rectangle],
    blockage_cells: Sequence[BlockageCell],
    die: Rectangle,
    dbu_per_micron: int,
    expanded_alpha: float,
    wire_slew_sign_convention: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    grouped: dict[str, list[PinRecord]] = defaultdict(list)
    for pin in pins:
        grouped[pin.net_name].append(pin)
    gr_index = _timing_index(gr_timing)
    congestion_index = ScalarGridIndex.from_congestion(g_cells)
    blockage_index = ScalarGridIndex.from_blockage(blockage_cells)
    source_sink_rows: list[dict[str, Any]] = []
    net_load_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for net_name in selected_nets:
        try:
            net_pins = grouped[net_name]
            driver, sinks = unique_driver_and_sinks(net_pins)
            net_box = bounding_box(net_pins)
            congestion_mean, congestion_std = indexed_congestion_statistics(
                net_box, congestion_index
            )
            net_expanded = expand_rectangle(net_box, expanded_alpha, die)
            net_blockage = indexed_blockage_mean(net_box, blockage_index)
            net_blockage_expanded = indexed_blockage_mean(net_expanded, blockage_index)
            hpwl = hpwl_um(net_pins, dbu_per_micron)
            route_index = RoutePathIndex(
                segments_by_net[net_name], (driver.x_dbu, driver.y_dbu)
            )
            first_key = (net_name, sinks[0].pin_name, "rise")
            gr_pi = gr_index[first_key]
            _require_numeric(gr_pi, ("pi_r", "pi_c1", "pi_c2"), first_key)
            net_load_rows.append(
                {
                    "design": design,
                    "candidate_id": candidate_id,
                    "net_name": net_name,
                    "driver_pin": driver.pin_name,
                    "corner": corner,
                    "hpwl": hpwl,
                    "num_sinks": len(sinks),
                    "congestion_mean": congestion_mean,
                    "congestion_std": congestion_std,
                    "macro_blockage": net_blockage,
                    "macro_blockage_expanded": net_blockage_expanded,
                    "gr_pi_r": gr_pi["pi_r"],
                    "gr_pi_c1": gr_pi["pi_c1"],
                    "gr_pi_c2": gr_pi["pi_c2"],
                }
            )
            for sink in sinks:
                pair_box = bounding_box((driver, sink))
                pair_expanded = expand_rectangle(pair_box, expanded_alpha, die)
                path = route_index.path_to((sink.x_dbu, sink.y_dbu))
                ss_length, ss_r, ss_c = path_features(path, dbu_per_micron, layer_rc)
                for transition, transition_code in (("rise", 0), ("fall", 1)):
                    key = (net_name, sink.pin_name, transition)
                    gr = gr_index[key]
                    _require_numeric(gr, ("driver_slew",), key)
                    _require_numeric(gr, ("sink_slew", "slack", "wire_delay"), key)
                    if wire_slew_sign_convention == "driver_minus_sink":
                        gr_wire_slew = gr["driver_slew"] - gr["sink_slew"]
                    else:
                        gr_wire_slew = gr["sink_slew"] - gr["driver_slew"]
                    source_sink_rows.append(
                        {
                            "design": design,
                            "candidate_id": candidate_id,
                            "net_name": net_name,
                            "driver_pin": driver.pin_name,
                            "sink_pin": sink.pin_name,
                            "transition": transition,
                            "corner": corner,
                            "hpwl": hpwl,
                            "num_sinks": len(sinks),
                            "driver_slew": gr["driver_slew"],
                            "congestion_mean": congestion_mean,
                            "congestion_std": congestion_std,
                            "transition_code": transition_code,
                            "ss_length": ss_length,
                            "ss_r": ss_r,
                            "ss_c": ss_c,
                            "macro_blockage": indexed_blockage_mean(
                                pair_box, blockage_index
                            ),
                            "macro_blockage_expanded": indexed_blockage_mean(
                                pair_expanded, blockage_index
                            ),
                            "gr_wire_delay": gr["wire_delay"],
                            "gr_sink_slew": gr["sink_slew"],
                            "gr_slack": gr["slack"],
                            "gr_wire_slew": gr_wire_slew,
                            "wire_slew_sign_convention": wire_slew_sign_convention,
                        }
                    )
        except (KeyError, ValueError) as error:
            rejected.append({"net_name": net_name, "status": "assembly_failure", "detail": str(error)})
            if net_load_rows and net_load_rows[-1].get("net_name") == net_name:
                net_load_rows.pop()
            source_sink_rows = [row for row in source_sink_rows if row["net_name"] != net_name]
    return source_sink_rows, net_load_rows, rejected


def _require_numeric(row: Mapping[str, Any], names: Sequence[str], key: object) -> None:
    for name in names:
        value = row.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{key} 缺少有限数值字段 {name}")
        if name in {"pi_r", "pi_c1", "pi_c2", "driver_slew", "sink_slew", "wire_delay"} and float(value) < 0:
            raise ValueError(f"{key} invalid negative physical field {name}")
