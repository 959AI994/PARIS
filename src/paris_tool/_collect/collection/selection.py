import fnmatch
from collections import defaultdict
from typing import Iterable

from paris_tool._collect.common.config import SelectionConfig

from .geometry import hpwl_um
from .identity import unique_driver_and_sinks
from .models import NetRecord, PinRecord


def select_nets(
    nets: dict[str, NetRecord],
    pins: Iterable[PinRecord],
    config: SelectionConfig,
    dbu_per_micron: int,
    routable_nets: set[str] | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    grouped: dict[str, list[PinRecord]] = defaultdict(list)
    for pin in pins:
        grouped[pin.net_name].append(pin)
    valid: list[tuple[float, str]] = []
    excluded: list[dict[str, str]] = []
    for name in sorted(nets):
        net = nets[name]
        if net.signal_type not in {"SIGNAL", "CLOCK"}:
            excluded.append({"net_name": name, "status": "excluded_power_ground"})
            continue
        if net.signal_type == "CLOCK" and not config.include_clock_nets:
            excluded.append({"net_name": name, "status": "excluded_clock"})
            continue
        if config.net_patterns and not any(fnmatch.fnmatchcase(name, pattern) for pattern in config.net_patterns):
            excluded.append({"net_name": name, "status": "excluded_pattern"})
            continue
        if routable_nets is not None and name not in routable_nets:
            excluded.append({"net_name": name, "status": "missing_route"})
            continue
        try:
            _, sinks = unique_driver_and_sinks(grouped[name])
        except ValueError as error:
            status = "unsupported_multi_driver" if "driver" in str(error) else "unsupported_no_sink"
            excluded.append({"net_name": name, "status": status})
            continue
        score = hpwl_um(grouped[name], dbu_per_micron) * max(1, len(sinks))
        valid.append((score, name))
    valid.sort(key=lambda item: (-item[0], item[1]))
    if config.mode == "limited":
        selected = [name for _, name in valid[: config.max_nets]]
        excluded.extend(
            {"net_name": name, "status": "excluded_limit"}
            for _, name in valid[config.max_nets :]
        )
    else:
        selected = [name for _, name in valid]
    return sorted(selected), sorted(excluded, key=lambda row: row["net_name"])
