"""A portable graph representation used by the macOS PARIS implementation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .io import Candidate


@dataclass(frozen=True)
class Node:
    name: str
    area: float = 1.0
    width: float = 1.0
    height: float = 1.0
    hierarchy: str = ""
    criticality: float = 0.0
    x: float | None = None
    y: float | None = None
    fixed_die: int | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Node:
        if "name" not in payload:
            raise ValidationError("node is missing name")
        values: dict[str, Any] = {}
        for key, default in (("area", 1.0), ("width", 1.0), ("height", 1.0), ("criticality", 0.0)):
            value = float(payload.get(key, default))
            if not math.isfinite(value) or value < 0:
                raise ValidationError(f"node {payload['name']} has invalid {key}")
            values[key] = value
        x = payload.get("x")
        y = payload.get("y")
        fixed_die = payload.get("fixed_die")
        attributes = payload.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise ValidationError("node attributes must be an object")
        return cls(
            name=str(payload["name"]),
            hierarchy=str(payload.get("hierarchy", "")),
            x=None if x is None else float(x),
            y=None if y is None else float(y),
            fixed_die=None if fixed_die is None else int(fixed_die),
            attributes=dict(attributes),
            **values,
        )


@dataclass(frozen=True)
class Net:
    name: str
    pins: tuple[str, ...]
    weight: float = 1.0
    criticality: float = 0.0
    net_type: str = "signal"
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Net:
        if "name" not in payload or "pins" not in payload:
            raise ValidationError("net requires name and pins")
        pins = payload["pins"]
        if not isinstance(pins, list) or len(pins) < 2:
            raise ValidationError(f"net {payload['name']} must contain at least two pins")
        if len(set(str(pin) for pin in pins)) < 2:
            raise ValidationError(f"net {payload['name']} must connect distinct nodes")
        attributes = payload.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise ValidationError("net attributes must be an object")
        return cls(
            name=str(payload["name"]),
            pins=tuple(str(pin) for pin in pins),
            weight=float(payload.get("weight", 1.0)),
            criticality=float(payload.get("criticality", 0.0)),
            net_type=str(payload.get("net_type", "signal")),
            attributes=dict(attributes),
        )


@dataclass
class Design:
    name: str
    nodes: dict[str, Node]
    nets: list[Net]
    die_count: int = 2
    cell_area_by_die: tuple[float, ...] = (0.0, 0.0)
    die_width: float | None = None
    die_height: float | None = None
    hierarchy_cell_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError("design name must be non-empty")
        if self.die_count < 2:
            raise ValidationError("die_count must be at least two")
        if len(self.cell_area_by_die) != self.die_count:
            raise ValidationError("cell_area_by_die length must equal die_count")
        for net in self.nets:
            missing = set(net.pins) - self.nodes.keys()
            if missing:
                raise ValidationError(
                    f"net {net.name} references unknown nodes: {sorted(missing)}"
                )
        for node in self.nodes.values():
            if node.fixed_die is not None and not 0 <= node.fixed_die < self.die_count:
                raise ValidationError(f"node {node.name} has invalid fixed_die")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Design:
        if "name" not in payload or "nodes" not in payload or "nets" not in payload:
            raise ValidationError("design requires name, nodes and nets")
        raw_nodes = payload["nodes"]
        raw_nets = payload["nets"]
        if not isinstance(raw_nodes, list) or not isinstance(raw_nets, list):
            raise ValidationError("nodes and nets must be arrays")
        node_items = [Node.from_dict(item) for item in raw_nodes]
        nodes = {node.name: node for node in node_items}
        if len(nodes) != len(node_items):
            raise ValidationError("node names must be unique")
        die_count = int(payload.get("die_count", 2))
        cell_area = payload.get("cell_area_by_die", [0.0] * die_count)
        if not isinstance(cell_area, list):
            raise ValidationError("cell_area_by_die must be an array")
        hierarchy_counts = payload.get("hierarchy_cell_counts", {})
        metadata = payload.get("metadata", {})
        if not isinstance(hierarchy_counts, Mapping) or not isinstance(metadata, Mapping):
            raise ValidationError("hierarchy_cell_counts and metadata must be objects")
        return cls(
            name=str(payload["name"]),
            nodes=nodes,
            nets=[Net.from_dict(item) for item in raw_nets],
            die_count=die_count,
            cell_area_by_die=tuple(float(value) for value in cell_area),
            die_width=None if payload.get("die_width") is None else float(payload["die_width"]),
            die_height=None if payload.get("die_height") is None else float(payload["die_height"]),
            hierarchy_cell_counts={str(k): int(v) for k, v in hierarchy_counts.items()},
            metadata=dict(metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> Design:
        source = Path(path)
        try:
            with source.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValidationError(f"cannot load design graph {source}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValidationError("design JSON must be an object")
        return cls.from_dict(payload)

    @property
    def object_names(self) -> set[str]:
        return set(self.nodes)

    def validate_candidate(self, candidate: Candidate) -> None:
        if candidate.design != self.name:
            raise ValidationError(
                f"candidate design {candidate.design} does not match {self.name}"
            )
        if candidate.die_count != self.die_count:
            raise ValidationError("candidate die_count does not match design")
        if set(candidate.assignment) != self.object_names:
            missing = sorted(self.object_names - set(candidate.assignment))
            extra = sorted(set(candidate.assignment) - self.object_names)
            raise ValidationError(
                f"candidate objects do not match design; missing={missing[:10]}, extra={extra[:10]}"
            )
        for name, node in self.nodes.items():
            if node.fixed_die is not None and candidate.assignment[name] != node.fixed_die:
                raise ValidationError(
                    f"candidate violates fixed die for {name}: "
                    f"expected {node.fixed_die}, got {candidate.assignment[name]}"
                )

    def hierarchy_names(self) -> list[str]:
        names = set(self.hierarchy_cell_counts)
        names.update(node.hierarchy for node in self.nodes.values() if node.hierarchy)
        return sorted(names)

    def iter_cut_nets(self, candidate: Candidate) -> Iterable[Net]:
        for net in self.nets:
            dies = {candidate.assignment[pin] for pin in net.pins}
            if len(dies) > 1:
                yield net
