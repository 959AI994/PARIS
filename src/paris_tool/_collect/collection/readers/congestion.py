import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from ..geometry import GCell


def read_congestion_cells(
    path: str | Path,
    x_boundaries: Sequence[int],
    y_boundaries: Sequence[int],
) -> list[GCell]:
    source = Path(path)
    totals: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"x", "y", "capacity", "usage"}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError(f"congestion 文件表头无效: {source}")
        for row in reader:
            x_value = row.get("x", "")
            if not x_value or not x_value.lstrip("-").isdigit():
                continue
            x_index = int(x_value)
            y_index = int(row["y"])
            if not 0 <= x_index < len(x_boundaries) - 1:
                raise ValueError(f"congestion x 索引越界: {x_index}")
            if not 0 <= y_index < len(y_boundaries) - 1:
                raise ValueError(f"congestion y 索引越界: {y_index}")
            capacity = max(0.0, float(row["capacity"]))
            usage = max(0.0, float(row["usage"]))
            totals[(x_index, y_index)][0] += capacity
            totals[(x_index, y_index)][1] += usage
    if not totals:
        raise ValueError(f"congestion 文件没有逐 GCell 数据: {source}")
    cells: list[GCell] = []
    for x_index in range(len(x_boundaries) - 1):
        for y_index in range(len(y_boundaries) - 1):
            capacity, usage = totals.get((x_index, y_index), [0, 0])
            congestion = min(1.0, usage / capacity) if capacity > 0 else 0.0
            cells.append(
                GCell(
                    x_boundaries[x_index],
                    y_boundaries[y_index],
                    x_boundaries[x_index + 1],
                    y_boundaries[y_index + 1],
                    congestion,
                )
            )
    return cells
