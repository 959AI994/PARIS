from collections import defaultdict
from typing import Iterable

from .models import NetRecord, PinRecord


def connectivity_signature(
    nets: dict[str, NetRecord], pins: Iterable[PinRecord]
) -> dict[str, tuple[tuple[str, str, str], ...]]:
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for pin in pins:
        if pin.net_name in nets and nets[pin.net_name].signal_type in {"SIGNAL", "CLOCK"}:
            grouped[pin.net_name].append((pin.pin_name, pin.pin_kind, pin.direction))
    return {name: tuple(sorted(records)) for name, records in grouped.items()}


def compare_identity(
    gr_nets: dict[str, NetRecord],
    gr_pins: list[PinRecord],
    dr_nets: dict[str, NetRecord],
    dr_pins: list[PinRecord],
) -> dict[str, object]:
    gr = connectivity_signature(gr_nets, gr_pins)
    dr = connectivity_signature(dr_nets, dr_pins)
    missing_in_dr = sorted(set(gr) - set(dr))
    missing_in_gr = sorted(set(dr) - set(gr))
    changed = sorted(name for name in set(gr) & set(dr) if gr[name] != dr[name])
    matching = set(gr) & set(dr) - set(changed)
    report: dict[str, object] = {
        "match": not missing_in_dr and not missing_in_gr and not changed,
        "gr_signal_net_count": len(gr),
        "dr_signal_net_count": len(dr),
        "matching_connectivity_net_count": len(matching),
        "collection_policy": "unchanged_gr_dr_connectivity_only",
        "missing_in_dr": missing_in_dr,
        "missing_in_gr": missing_in_gr,
        "changed_connectivity": changed,
    }
    if not report["match"]:
        details = {
            name: {"gr": gr.get(name), "dr": dr.get(name)}
            for name in changed[:20]
        }
        report["changed_details"] = details
    return report


def matching_connectivity_names(
    gr_nets: dict[str, NetRecord],
    gr_pins: list[PinRecord],
    dr_nets: dict[str, NetRecord],
    dr_pins: list[PinRecord],
) -> set[str]:
    """Return nets whose named pins and directions are unchanged across stages."""
    gr = connectivity_signature(gr_nets, gr_pins)
    dr = connectivity_signature(dr_nets, dr_pins)
    return {name for name in set(gr) & set(dr) if gr[name] == dr[name]}


def unique_driver_and_sinks(pins: Iterable[PinRecord]) -> tuple[PinRecord, list[PinRecord]]:
    pin_list = list(pins)
    drivers = [pin for pin in pin_list if _is_driver(pin)]
    sinks = [pin for pin in pin_list if _is_sink(pin)]
    if len(drivers) != 1:
        raise ValueError(f"线网 driver 数量不是 1: {len(drivers)}")
    if not sinks:
        raise ValueError("线网没有 sink")
    return drivers[0], sorted(sinks, key=lambda item: item.pin_name)


def _is_driver(pin: PinRecord) -> bool:
    if pin.pin_kind == "iterm":
        return pin.direction in {"OUTPUT", "INOUT"}
    return pin.direction in {"INPUT", "INOUT"}


def _is_sink(pin: PinRecord) -> bool:
    if pin.pin_kind == "iterm":
        return pin.direction in {"INPUT", "INOUT"}
    return pin.direction in {"OUTPUT", "INOUT"}
