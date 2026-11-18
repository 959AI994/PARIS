import heapq
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from paris_tool._collect.common.units import dbu_to_um

from .models import PinRecord, Rectangle, Segment


@dataclass(frozen=True)
class LayerRC:
    resistance_kohm_per_um: float
    capacitance_ff_per_um: float


@dataclass(frozen=True)
class GCell:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    congestion: float


@dataclass(frozen=True)
class BlockageCell:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    blockage: float


@dataclass(frozen=True)
class ScalarGridIndex:
    x_boundaries: tuple[int, ...]
    y_boundaries: tuple[int, ...]
    values: Mapping[tuple[int, int], float]
    prefix_sum: np.ndarray
    prefix_square_sum: np.ndarray

    @classmethod
    def from_congestion(cls, cells: Sequence[GCell]) -> "ScalarGridIndex":
        return cls._build(cells, "congestion")

    @classmethod
    def from_blockage(cls, cells: Sequence[BlockageCell]) -> "ScalarGridIndex":
        return cls._build(cells, "blockage")

    @classmethod
    def _build(
        cls, cells: Sequence[GCell] | Sequence[BlockageCell], attribute: str
    ) -> "ScalarGridIndex":
        if not cells:
            raise ValueError("GCell 标量网格为空")
        x_boundaries = tuple(
            sorted({coordinate for cell in cells for coordinate in (cell.x_min, cell.x_max)})
        )
        y_boundaries = tuple(
            sorted({coordinate for cell in cells for coordinate in (cell.y_min, cell.y_max)})
        )
        x_indices = {value: index for index, value in enumerate(x_boundaries[:-1])}
        y_indices = {value: index for index, value in enumerate(y_boundaries[:-1])}
        values = {
            (x_indices[cell.x_min], y_indices[cell.y_min]): float(getattr(cell, attribute))
            for cell in cells
        }
        expected = (len(x_boundaries) - 1) * (len(y_boundaries) - 1)
        if len(values) != expected:
            raise ValueError("GCell 标量网格不完整或包含重复单元")
        matrix = np.empty((len(x_boundaries) - 1, len(y_boundaries) - 1), dtype=float)
        for (x_index, y_index), value in values.items():
            matrix[x_index, y_index] = value
        prefix_sum = np.pad(matrix, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        prefix_square_sum = (
            np.pad(matrix * matrix, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
        )
        return cls(x_boundaries, y_boundaries, values, prefix_sum, prefix_square_sum)

    def overlapping_values(self, region: Rectangle) -> list[float]:
        x_range = self._index_range(self.x_boundaries, region.x_min, region.x_max)
        y_range = self._index_range(self.y_boundaries, region.y_min, region.y_max)
        return [self.values[(x, y)] for x in x_range for y in y_range]

    def statistics(self, region: Rectangle) -> tuple[float, float]:
        x_range = self._index_range(self.x_boundaries, region.x_min, region.x_max)
        y_range = self._index_range(self.y_boundaries, region.y_min, region.y_max)
        count = len(x_range) * len(y_range)
        if count == 0:
            raise ValueError("bounding box 没有覆盖任何 GCell")
        total = self._rectangle_sum(self.prefix_sum, x_range, y_range)
        square_total = self._rectangle_sum(
            self.prefix_square_sum, x_range, y_range
        )
        # Congestion and blockage grids contain ratios in [0, 1].  Subtracting
        # large prefix sums for a small box can produce a value such as
        # -3e-13 for an exact zero, so restore the physical range here rather
        # than allowing harmless cancellation noise to fail validation.
        mean = min(1.0, max(0.0, total / count))
        variance = max(0.0, square_total / count - mean * mean)
        return mean, math.sqrt(variance)

    def mean(self, region: Rectangle) -> float:
        return self.statistics(region)[0]

    @staticmethod
    def _rectangle_sum(
        prefix: np.ndarray, x_range: range, y_range: range
    ) -> float:
        x1, x2 = x_range.start, x_range.stop
        y1, y2 = y_range.start, y_range.stop
        return float(
            prefix[x2, y2]
            - prefix[x1, y2]
            - prefix[x2, y1]
            + prefix[x1, y1]
        )

    @staticmethod
    def _index_range(boundaries: Sequence[int], lower: int, upper: int) -> range:
        cell_count = len(boundaries) - 1
        if lower == upper:
            index = bisect_right(boundaries, lower) - 1
            index = min(max(index, 0), cell_count - 1)
            return range(index, index + 1)
        start = min(max(bisect_right(boundaries, lower) - 1, 0), cell_count)
        stop = min(max(bisect_left(boundaries, upper), 0), cell_count)
        return range(start, stop)


def bounding_box(pins: Sequence[PinRecord]) -> Rectangle:
    if not pins:
        raise ValueError("计算 bounding box 至少需要一个 pin")
    return Rectangle(
        min(pin.x_dbu for pin in pins),
        min(pin.y_dbu for pin in pins),
        max(pin.x_dbu for pin in pins),
        max(pin.y_dbu for pin in pins),
    )


def hpwl_um(pins: Sequence[PinRecord], dbu_per_micron: int) -> float:
    box = bounding_box(pins)
    return dbu_to_um((box.x_max - box.x_min) + (box.y_max - box.y_min), dbu_per_micron)


def expand_rectangle(rectangle: Rectangle, alpha: float, clip: Rectangle) -> Rectangle:
    if alpha < 1.0:
        raise ValueError("alpha 必须大于或等于 1")
    center_x = (rectangle.x_min + rectangle.x_max) / 2
    center_y = (rectangle.y_min + rectangle.y_max) / 2
    half_width = max(1.0, (rectangle.x_max - rectangle.x_min) * alpha / 2)
    half_height = max(1.0, (rectangle.y_max - rectangle.y_min) * alpha / 2)
    return Rectangle(
        max(clip.x_min, math.floor(center_x - half_width)),
        max(clip.y_min, math.floor(center_y - half_height)),
        min(clip.x_max, math.ceil(center_x + half_width)),
        min(clip.y_max, math.ceil(center_y + half_height)),
    )


def intersection_area(left: Rectangle, right: Rectangle) -> int:
    return max(0, min(left.x_max, right.x_max) - max(left.x_min, right.x_min)) * max(
        0, min(left.y_max, right.y_max) - max(left.y_min, right.y_min)
    )


def blockage_ratio(region: Rectangle, obstructions: Sequence[Rectangle]) -> float:
    if region.area <= 0:
        return 0.0
    x_points = {region.x_min, region.x_max}
    y_points = {region.y_min, region.y_max}
    clipped: list[Rectangle] = []
    for obstruction in obstructions:
        if intersection_area(region, obstruction) == 0:
            continue
        item = Rectangle(
            max(region.x_min, obstruction.x_min),
            max(region.y_min, obstruction.y_min),
            min(region.x_max, obstruction.x_max),
            min(region.y_max, obstruction.y_max),
        )
        clipped.append(item)
        x_points.update((item.x_min, item.x_max))
        y_points.update((item.y_min, item.y_max))
    if not clipped:
        return 0.0
    xs = sorted(x_points)
    ys = sorted(y_points)
    covered = 0
    for x1, x2 in zip(xs, xs[1:]):
        for y1, y2 in zip(ys, ys[1:]):
            if any(
                item.x_min <= x1 and x2 <= item.x_max and item.y_min <= y1 and y2 <= item.y_max
                for item in clipped
            ):
                covered += (x2 - x1) * (y2 - y1)
    return min(1.0, covered / region.area)


def congestion_statistics(region: Rectangle, cells: Sequence[GCell]) -> tuple[float, float]:
    values = [
        cell.congestion
        for cell in cells
        if intersection_area(
            region, Rectangle(cell.x_min, cell.y_min, cell.x_max, cell.y_max)
        )
        > 0
    ]
    if not values:
        raise ValueError("bounding box 没有覆盖任何 GCell")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def indexed_congestion_statistics(
    region: Rectangle, index: ScalarGridIndex
) -> tuple[float, float]:
    return index.statistics(region)


def blockage_mean(region: Rectangle, cells: Sequence[BlockageCell]) -> float:
    values = [
        cell.blockage
        for cell in cells
        if intersection_area(
            region, Rectangle(cell.x_min, cell.y_min, cell.x_max, cell.y_max)
        )
        > 0
    ]
    if not values:
        raise ValueError("bounding box 没有覆盖任何 GCell")
    return sum(values) / len(values)


def indexed_blockage_mean(region: Rectangle, index: ScalarGridIndex) -> float:
    return index.mean(region)


def grid_boundaries(coordinates: Sequence[int], upper_bound: int) -> list[int]:
    points = sorted(set(coordinates))
    if not points:
        raise ValueError("GCell grid 坐标为空")
    if points[-1] < upper_bound:
        points.append(upper_bound)
    if len(points) < 2:
        raise ValueError("GCell grid 至少需要两个边界坐标")
    return points


def build_blockage_cells(
    x_boundaries: Sequence[int],
    y_boundaries: Sequence[int],
    obstructions: Sequence[Rectangle],
) -> list[BlockageCell]:
    if len(x_boundaries) < 2 or len(y_boundaries) < 2:
        raise ValueError("GCell grid 至少需要两个边界坐标")
    # A design can contain hundreds of thousands of GCells.  Scanning every
    # obstruction for every cell makes multi-candidate collection effectively
    # quadratic.  Bin each obstruction only into the GCells it overlaps, then
    # retain the exact union-area calculation inside each affected cell.
    bins: dict[tuple[int, int], list[Rectangle]] = defaultdict(list)
    x_count = len(x_boundaries) - 1
    y_count = len(y_boundaries) - 1
    for obstruction in obstructions:
        x_start = min(max(bisect_right(x_boundaries, obstruction.x_min) - 1, 0), x_count)
        x_stop = min(max(bisect_left(x_boundaries, obstruction.x_max), 0), x_count)
        y_start = min(max(bisect_right(y_boundaries, obstruction.y_min) - 1, 0), y_count)
        y_stop = min(max(bisect_left(y_boundaries, obstruction.y_max), 0), y_count)
        for x_index in range(x_start, x_stop):
            for y_index in range(y_start, y_stop):
                bins[(x_index, y_index)].append(obstruction)
    return [
        BlockageCell(
            x1,
            y1,
            x2,
            y2,
            blockage_ratio(
                Rectangle(x1, y1, x2, y2), bins.get((x_index, y_index), ())
            ),
        )
        for x_index, (x1, x2) in enumerate(zip(x_boundaries, x_boundaries[1:]))
        for y_index, (y1, y2) in enumerate(zip(y_boundaries, y_boundaries[1:]))
    ]


class RoutePathIndex:
    """One source-rooted shortest-path tree reused by every sink on a net."""

    def __init__(self, segments: Sequence[Segment], source: tuple[int, int]) -> None:
        if not segments:
            raise ValueError("线网没有 GR segments")
        self.segments = tuple(segments)
        self.source = source
        self._x1 = np.asarray([segment.x1 for segment in self.segments], dtype=np.int64)
        self._y1 = np.asarray([segment.y1 for segment in self.segments], dtype=np.int64)
        self._x2 = np.asarray([segment.x2 for segment in self.segments], dtype=np.int64)
        self._y2 = np.asarray([segment.y2 for segment in self.segments], dtype=np.int64)
        self._wire = np.asarray([segment.is_wire for segment in self.segments], dtype=bool)
        self._vertical = self._wire & (self._x1 == self._x2)
        self._horizontal = self._wire & (self._y1 == self._y2)
        if np.any(self._wire & ~self._vertical & ~self._horizontal):
            raise ValueError("GR wire segment 不是 Manhattan 线段")
        self.source_node, self.source_segment_index = self._attach(source)
        adjacency: dict[
            tuple[int, int, str],
            list[tuple[float, tuple[int, int, str], Segment]],
        ] = defaultdict(list)
        for segment in self.segments:
            start = (segment.x1, segment.y1, segment.layer1)
            end = (segment.x2, segment.y2, segment.layer2)
            self._add_edge(adjacency, start, end, segment)
        source_segment = self.segments[self.source_segment_index]
        for endpoint in self._endpoints(source_segment):
            connector = self._connector(source_segment.net_name, self.source_node, endpoint)
            self._add_edge(adjacency, self.source_node, endpoint, connector)
        self.distances: dict[tuple[int, int, str], float] = {self.source_node: 0.0}
        self.previous: dict[
            tuple[int, int, str], tuple[tuple[int, int, str], Segment]
        ] = {}
        queue: list[tuple[float, tuple[int, int, str]]] = [(0.0, self.source_node)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != self.distances.get(node):
                continue
            for weight, neighbor, segment in adjacency[node]:
                candidate = distance + weight
                if candidate < self.distances.get(neighbor, math.inf):
                    self.distances[neighbor] = candidate
                    self.previous[neighbor] = (node, segment)
                    heapq.heappush(queue, (candidate, neighbor))

    @staticmethod
    def _endpoints(segment: Segment) -> tuple[tuple[int, int, str], tuple[int, int, str]]:
        return (
            (segment.x1, segment.y1, segment.layer1),
            (segment.x2, segment.y2, segment.layer2),
        )

    @staticmethod
    def _connector(
        net_name: str,
        start: tuple[int, int, str],
        end: tuple[int, int, str],
    ) -> Segment:
        return Segment(net_name, start[0], start[1], start[2], end[0], end[1], end[2])

    @staticmethod
    def _add_edge(
        adjacency: dict[
            tuple[int, int, str],
            list[tuple[float, tuple[int, int, str], Segment]],
        ],
        start: tuple[int, int, str],
        end: tuple[int, int, str],
        segment: Segment,
    ) -> None:
        weight = float(segment.length_dbu) if segment.is_wire else 0.0
        adjacency[start].append((weight, end, segment))
        adjacency[end].append((weight, start, segment))

    def path_to(self, sink: tuple[int, int]) -> list[Segment]:
        sink_node, sink_segment_index = self._attach(sink)
        sink_segment = self.segments[sink_segment_index]
        candidates: list[
            tuple[float, tuple[int, int, str], Segment | None, bool]
        ] = []
        if sink_node in self.distances:
            candidates.append((self.distances[sink_node], sink_node, None, False))
        for endpoint in self._endpoints(sink_segment):
            if endpoint not in self.distances:
                continue
            connector = self._connector(sink_segment.net_name, endpoint, sink_node)
            weight = float(connector.length_dbu) if connector.is_wire else 0.0
            candidates.append(
                (self.distances[endpoint] + weight, endpoint, connector, False)
            )
        if self.source_segment_index == sink_segment_index:
            direct = self._connector(sink_segment.net_name, self.source_node, sink_node)
            candidates.append((float(direct.length_dbu), self.source_node, direct, True))
        if not candidates:
            raise ValueError("GR segment graph 中 source 与 sink 不连通")
        _, target, connector, direct = min(
            candidates,
            key=lambda item: (item[0], item[1], item[3]),
        )
        if direct:
            path = [] if connector is None or connector.length_dbu == 0 else [connector]
        else:
            path: list[Segment] = []
            while target != self.source_node:
                if target not in self.previous:
                    raise ValueError("GR segment graph 中 source 与 sink 不连通")
                target, segment = self.previous[target]
                path.append(segment)
            path.reverse()
            if connector is not None and (connector.length_dbu > 0 or not connector.is_wire):
                path.append(connector)
        source_access = self._connector(
            sink_segment.net_name,
            (self.source[0], self.source[1], self.source_node[2]),
            self.source_node,
        )
        sink_access = self._connector(
            sink_segment.net_name,
            sink_node,
            (sink[0], sink[1], sink_node[2]),
        )
        if source_access.length_dbu > 0:
            path.insert(0, source_access)
        if sink_access.length_dbu > 0:
            path.append(sink_access)
        return path

    def _attach(
        self, pin: tuple[int, int]
    ) -> tuple[tuple[int, int, str], int]:
        """Project a pin onto the nearest segment using vectorized exact distance."""
        pin_x, pin_y = pin
        projected_x = np.where(
            self._vertical,
            self._x1,
            np.where(
                self._horizontal,
                np.clip(pin_x, np.minimum(self._x1, self._x2), np.maximum(self._x1, self._x2)),
                self._x1,
            ),
        )
        projected_y = np.where(
            self._vertical,
            np.clip(pin_y, np.minimum(self._y1, self._y2), np.maximum(self._y1, self._y2)),
            self._y1,
        )
        distances = np.abs(pin_x - projected_x) + np.abs(pin_y - projected_y)
        index = int(np.argmin(distances))
        segment = self.segments[index]
        layer = segment.layer1 if segment.is_wire else min(segment.layer1, segment.layer2)
        return (int(projected_x[index]), int(projected_y[index]), layer), index


def shortest_route_path(
    segments: Sequence[Segment],
    source: tuple[int, int],
    sink: tuple[int, int],
) -> list[Segment]:
    return RoutePathIndex(segments, source).path_to(sink)


def _attach_pin(
    segments: Sequence[Segment], pin: tuple[int, int]
) -> tuple[tuple[int, int, str], int]:
    candidates: list[tuple[int, int, int, str, int]] = []
    pin_x, pin_y = pin
    for index, segment in enumerate(segments):
        if segment.is_wire:
            if segment.x1 == segment.x2:
                x = segment.x1
                y = min(max(pin_y, min(segment.y1, segment.y2)), max(segment.y1, segment.y2))
            elif segment.y1 == segment.y2:
                x = min(max(pin_x, min(segment.x1, segment.x2)), max(segment.x1, segment.x2))
                y = segment.y1
            else:
                raise ValueError("GR wire segment 不是 Manhattan 线段")
            layer = segment.layer1
        else:
            x = segment.x1
            y = segment.y1
            layer = min(segment.layer1, segment.layer2)
        distance = abs(pin_x - x) + abs(pin_y - y)
        candidates.append((distance, x, y, layer, index))
    _, x, y, layer, index = min(candidates)
    return (x, y, layer), index


def path_features(
    path: Sequence[Segment],
    dbu_per_micron: int,
    layer_rc: Mapping[str, LayerRC],
) -> tuple[float, float, float]:
    length = resistance = capacitance = 0.0
    for segment in path:
        if not segment.is_wire:
            continue
        if segment.layer1 not in layer_rc:
            raise KeyError(f"缺少 routing layer RC: {segment.layer1}")
        segment_um = dbu_to_um(segment.length_dbu, dbu_per_micron)
        rc = layer_rc[segment.layer1]
        length += segment_um
        resistance += segment_um * rc.resistance_kohm_per_um
        capacitance += segment_um * rc.capacitance_ff_per_um
    return length, resistance, capacitance


def build_congestion_cells(
    x_grid: Sequence[int],
    y_grid: Sequence[int],
    segments_by_net: Mapping[str, Sequence[Segment]],
) -> list[GCell]:
    if len(x_grid) < 2 or len(y_grid) < 2:
        raise ValueError("GCell grid 至少需要两个 X/Y 坐标")
    demand = [[0.0 for _ in range(len(y_grid) - 1)] for _ in range(len(x_grid) - 1)]
    for segments in segments_by_net.values():
        for segment in segments:
            if not segment.is_wire or segment.length_dbu == 0:
                continue
            for ix, (x1, x2) in enumerate(zip(x_grid, x_grid[1:])):
                for iy, (y1, y2) in enumerate(zip(y_grid, y_grid[1:])):
                    if segment.y1 == segment.y2 and y1 <= segment.y1 <= y2:
                        overlap = max(0, min(max(segment.x1, segment.x2), x2) - max(min(segment.x1, segment.x2), x1))
                    elif segment.x1 == segment.x2 and x1 <= segment.x1 <= x2:
                        overlap = max(0, min(max(segment.y1, segment.y2), y2) - max(min(segment.y1, segment.y2), y1))
                    else:
                        overlap = 0
                    demand[ix][iy] += overlap
    positive = [value for column in demand for value in column if value > 0]
    normalization = max(positive) if positive else 1.0
    return [
        GCell(x1, y1, x2, y2, min(1.0, demand[ix][iy] / normalization))
        for ix, (x1, x2) in enumerate(zip(x_grid, x_grid[1:]))
        for iy, (y1, y2) in enumerate(zip(y_grid, y_grid[1:]))
    ]
