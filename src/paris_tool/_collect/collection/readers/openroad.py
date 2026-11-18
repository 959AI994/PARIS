from pathlib import Path
from typing import Any

from paris_tool._collect.common.io import read_tsv

from ..models import NetRecord, PinRecord, Rectangle


def read_design_export(
    directory: str | Path,
) -> tuple[dict[str, NetRecord], list[PinRecord], list[Rectangle], dict[str, Any]]:
    root = Path(directory)
    metadata_rows = read_tsv(root / "metadata.tsv")
    if len(metadata_rows) != 1:
        raise ValueError("metadata.tsv 必须恰好包含一条记录")
    metadata: dict[str, Any] = dict(metadata_rows[0])
    metadata["dbu_per_micron"] = int(metadata["dbu_per_micron"])
    nets = {
        row["net_name"]: NetRecord(row["net_name"], row["signal_type"])
        for row in read_tsv(root / "nets.tsv")
    }
    pins = [
        PinRecord(
            net_name=row["net_name"],
            pin_name=row["pin_name"],
            pin_kind=row["pin_kind"],
            direction=row["direction"],
            x_dbu=int(row["x_dbu"]),
            y_dbu=int(row["y_dbu"]),
        )
        for row in read_tsv(root / "pins.tsv")
    ]
    obstructions = [
        Rectangle(int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"]))
        for row in read_tsv(root / "macro_obstructions.tsv")
    ]
    return nets, pins, obstructions, metadata


def read_timing_export(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric = {
        "driver_slew",
        "sink_slew",
        "slack",
        "wire_delay",
        "pi_r",
        "pi_c1",
        "pi_c2",
    }
    for row in read_tsv(path):
        converted: dict[str, Any] = dict(row)
        for name in numeric:
            value = row.get(name, "")
            converted[name] = float(value) if value != "" else None
        rows.append(converted)
    return rows
