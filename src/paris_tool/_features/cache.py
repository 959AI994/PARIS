"""Efficient on-disk design cache for production PARIS feature extraction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .design import Design, Net, Node
from .errors import ValidationError

CACHE_SCHEMA_VERSION = "1.0"


@dataclass
class DesignCache:
    root: Path
    manifest: dict[str, Any]
    nodes: pd.DataFrame
    nets: pd.DataFrame
    net_offsets: np.ndarray
    net_pins: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> DesignCache:
        root = Path(path).resolve()
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ValidationError(f"invalid design cache at {root}: {exc}") from exc
        if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported cache schema {manifest.get('schema_version')}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        nodes = pd.read_parquet(root / "nodes.parquet")
        nets = pd.read_parquet(root / "nets.parquet")
        incidence = np.load(root / "incidence.npz", mmap_mode="r")
        offsets = incidence["net_offsets"]
        pins = incidence["net_pins"]
        if len(offsets) != len(nets) + 1:
            raise ValidationError("cache incidence offsets do not match net table")
        if len(nodes) != int(manifest["node_count"]):
            raise ValidationError("cache node count does not match manifest")
        return cls(root, manifest, nodes, nets, offsets, pins)

    @property
    def name(self) -> str:
        return str(self.manifest["design"])

    @property
    def object_names(self) -> list[str]:
        return self.nodes["name"].astype(str).tolist()

    @property
    def die_count(self) -> int:
        return int(self.manifest.get("die_count", 2))

    @property
    def die_width(self) -> float | None:
        value = self.manifest.get("die_width")
        return None if value is None else float(value)

    @property
    def die_height(self) -> float | None:
        value = self.manifest.get("die_height")
        return None if value is None else float(value)

    @property
    def cell_area_by_die(self) -> list[float]:
        return [float(value) for value in self.manifest.get("cell_area_by_die", [0.0] * self.die_count)]

    @property
    def hierarchy_cell_counts(self) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in self.manifest.get("hierarchy_cell_counts", {}).items()
        }

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    def to_portable_design(self) -> Design:
        node_records: dict[str, Node] = {}
        for record in self.nodes.to_dict(orient="records"):
            name = str(record.pop("name"))
            attributes = record.pop("attributes", {})
            if isinstance(attributes, str):
                attributes = json.loads(attributes) if attributes else {}
            node_records[name] = Node(
                name=name,
                area=float(record.pop("area", 1.0)),
                width=float(record.pop("width", 1.0)),
                height=float(record.pop("height", 1.0)),
                hierarchy=str(record.pop("hierarchy", "") or ""),
                criticality=float(record.pop("criticality", 0.0)),
                x=_optional_float(record.pop("x", None)),
                y=_optional_float(record.pop("y", None)),
                fixed_die=_optional_int(record.pop("fixed_die", None)),
                attributes=dict(attributes or {}),
            )
        names = self.nodes["name"].astype(str).tolist()
        net_records: list[Net] = []
        for net_id, record in enumerate(self.nets.to_dict(orient="records")):
            start, end = int(self.net_offsets[net_id]), int(self.net_offsets[net_id + 1])
            pins = tuple(names[int(index)] for index in self.net_pins[start:end])
            attributes = record.get("attributes", {})
            if isinstance(attributes, str):
                attributes = json.loads(attributes) if attributes else {}
            net_records.append(
                Net(
                    name=str(record["name"]),
                    pins=pins,
                    weight=float(record.get("weight", 1.0)),
                    criticality=float(record.get("criticality", 0.0)),
                    net_type=str(record.get("net_type", "signal")),
                    attributes=dict(attributes or {}),
                )
            )
        return Design(
            name=self.name,
            nodes=node_records,
            nets=net_records,
            die_count=self.die_count,
            cell_area_by_die=tuple(self.cell_area_by_die),
            die_width=self.die_width,
            die_height=self.die_height,
            hierarchy_cell_counts=self.hierarchy_cell_counts,
            metadata=dict(self.manifest.get("metadata", {})),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _fingerprint(
    design: str,
    nodes: pd.DataFrame,
    nets: pd.DataFrame,
    net_offsets: np.ndarray,
    net_pins: np.ndarray,
    metadata: Mapping[str, Any],
) -> str:
    digest = sha256()
    digest.update(design.encode())
    digest.update(pd.util.hash_pandas_object(nodes, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(nets, index=True).values.tobytes())
    digest.update(np.asarray(net_offsets, dtype=np.int64).tobytes())
    digest.update(np.asarray(net_pins, dtype=np.int64).tobytes())
    digest.update(json.dumps(dict(metadata), sort_keys=True, default=str).encode())
    return digest.hexdigest()


def write_design_cache(
    path: str | Path,
    design: str,
    nodes: pd.DataFrame,
    nets: pd.DataFrame,
    net_offsets: np.ndarray,
    net_pins: np.ndarray,
    *,
    die_count: int = 2,
    cell_area_by_die: list[float] | tuple[float, ...] | None = None,
    die_width: float | None = None,
    die_height: float | None = None,
    hierarchy_cell_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DesignCache:
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    required_nodes = {"name", "area", "width", "height"}
    required_nets = {"name", "weight", "criticality", "net_type"}
    if not required_nodes.issubset(nodes.columns):
        raise ValidationError(f"nodes table missing columns: {sorted(required_nodes - set(nodes.columns))}")
    if not required_nets.issubset(nets.columns):
        raise ValidationError(f"nets table missing columns: {sorted(required_nets - set(nets.columns))}")
    if nodes["name"].duplicated().any():
        raise ValidationError("design cache node names must be unique")
    offsets = np.asarray(net_offsets, dtype=np.int64)
    pins = np.asarray(net_pins, dtype=np.int64)
    if len(offsets) != len(nets) + 1 or offsets[0] != 0 or offsets[-1] != len(pins):
        raise ValidationError("invalid CSR-style net incidence")
    if len(pins) and (pins.min() < 0 or pins.max() >= len(nodes)):
        raise ValidationError("net incidence references an invalid node index")
    nodes = nodes.copy()
    nets = nets.copy()
    for table in (nodes, nets):
        if "attributes" in table.columns:
            table["attributes"] = table["attributes"].map(
                lambda value: value if isinstance(value, str) else json.dumps(value or {}, sort_keys=True)
            )
    meta = dict(metadata or {})
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "design": design,
        "die_count": die_count,
        "node_count": len(nodes),
        "net_count": len(nets),
        "pin_count": len(pins),
        "die_width": die_width,
        "die_height": die_height,
        "cell_area_by_die": list(cell_area_by_die or [0.0] * die_count),
        "hierarchy_cell_counts": dict(hierarchy_cell_counts or {}),
        "metadata": meta,
    }
    manifest["fingerprint"] = _fingerprint(design, nodes, nets, offsets, pins, manifest)
    nodes.to_parquet(root / "nodes.parquet", index=False, compression="zstd")
    nets.to_parquet(root / "nets.parquet", index=False, compression="zstd")
    np.savez_compressed(root / "incidence.npz", net_offsets=offsets, net_pins=pins)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return DesignCache.load(root)


def portable_design_to_cache(design: Design, path: str | Path) -> DesignCache:
    names = sorted(design.nodes)
    name_to_index = {name: index for index, name in enumerate(names)}
    node_rows = []
    for name in names:
        node = design.nodes[name]
        node_rows.append(
            {
                "name": name,
                "area": node.area,
                "width": node.width,
                "height": node.height,
                "hierarchy": node.hierarchy,
                "criticality": node.criticality,
                "x": node.x,
                "y": node.y,
                "fixed_die": node.fixed_die,
                "attributes": node.attributes,
            }
        )
    net_rows = []
    offsets = [0]
    pins: list[int] = []
    for net in design.nets:
        net_rows.append(
            {
                "name": net.name,
                "weight": net.weight,
                "criticality": net.criticality,
                "net_type": net.net_type,
                "attributes": net.attributes,
            }
        )
        pins.extend(name_to_index[name] for name in net.pins)
        offsets.append(len(pins))
    return write_design_cache(
        path,
        design.name,
        pd.DataFrame(node_rows),
        pd.DataFrame(net_rows),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(pins, dtype=np.int64),
        die_count=design.die_count,
        cell_area_by_die=design.cell_area_by_die,
        die_width=design.die_width,
        die_height=design.die_height,
        hierarchy_cell_counts=design.hierarchy_cell_counts,
        metadata=design.metadata,
    )
