"""Input/output models and validation for PARIS artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from .errors import ValidationError

VALID_GRANULARITIES = {"macro", "cluster"}
VALID_STATUSES = {
    "success",
    "coarse_success",
    "global_route_failed",
    "detailed_route_failed",
    "placement_failed",
    "constraint_failed",
    "tool_error",
    "timeout",
}


def _read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise ValidationError(f"JSON file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {source}: {exc}") from exc


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(target)


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{field_name} must be finite")
    return number


@dataclass(frozen=True)
class Candidate:
    design: str
    candidate_id: str
    assignment: dict[str, int]
    granularity: str = "macro"
    die_count: int = 2
    schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.design.strip():
            raise ValidationError("design must be non-empty")
        if not self.candidate_id.strip():
            raise ValidationError("candidate_id must be non-empty")
        if self.granularity not in VALID_GRANULARITIES:
            raise ValidationError(
                f"granularity must be one of {sorted(VALID_GRANULARITIES)}"
            )
        if self.die_count < 2:
            raise ValidationError("die_count must be at least 2")
        if not self.assignment:
            raise ValidationError("assignment must not be empty")
        for name, die in self.assignment.items():
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("assignment keys must be non-empty strings")
            if isinstance(die, bool) or not isinstance(die, int):
                raise ValidationError(f"assignment for {name!r} must be an integer")
            if die < 0 or die >= self.die_count:
                raise ValidationError(
                    f"assignment for {name!r} is {die}; expected [0, {self.die_count - 1}]"
                )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Candidate:
        required = {"design", "candidate_id", "assignment"}
        missing = required - payload.keys()
        if missing:
            raise ValidationError(f"candidate is missing fields: {sorted(missing)}")
        assignment = payload["assignment"]
        if not isinstance(assignment, Mapping):
            raise ValidationError("assignment must be an object")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("metadata must be an object")
        return cls(
            schema_version=str(payload.get("schema_version", "1.0")),
            design=str(payload["design"]),
            candidate_id=str(payload["candidate_id"]),
            granularity=str(payload.get("granularity", "macro")),
            die_count=int(payload.get("die_count", 2)),
            assignment={str(key): value for key, value in assignment.items()},
            metadata=dict(metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> Candidate:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise ValidationError("candidate JSON must be an object")
        return cls.from_dict(payload)

    def canonical_assignment(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self.assignment.items()))

    @property
    def assignment_hash(self) -> str:
        encoded = json.dumps(
            self.canonical_assignment(), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    def names_on_die(self, die: int) -> list[str]:
        if die < 0 or die >= self.die_count:
            raise ValidationError(f"invalid die index: {die}")
        return sorted(name for name, assigned_die in self.assignment.items() if assigned_die == die)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assignment_hash"] = self.assignment_hash
        return payload

    def dump(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())


@dataclass(frozen=True)
class Label:
    design: str
    candidate_id: str
    status: str
    replicate: int = 0
    coarse_status: str | None = None
    coarse_tns: float | None = None
    coarse_wns: float | None = None
    coarse_congestion: float | None = None
    coarse_wirelength: float | None = None
    coarse_overflow: float | None = None
    coarse_power: float | None = None
    coarse_setup_violations: float | None = None
    coarse_runtime_sec: float | None = None
    coarse_peak_rss_mb: float | None = None
    final_tns: float | None = None
    final_wns: float | None = None
    congestion: float | None = None
    routed_wirelength: float | None = None
    power: float | None = None
    drc_count: int | None = None
    runtime_sec: float | None = None
    schema_version: str = "2.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.design.strip() or not self.candidate_id.strip():
            raise ValidationError("label design and candidate_id must be non-empty")
        if self.status not in VALID_STATUSES:
            raise ValidationError(f"unknown label status: {self.status}")
        if isinstance(self.replicate, bool) or self.replicate < 0:
            raise ValidationError("label replicate must be a non-negative integer")
        if self.coarse_status is not None and self.coarse_status not in VALID_STATUSES:
            raise ValidationError(f"unknown coarse label status: {self.coarse_status}")
        for name in (
            "coarse_tns",
            "coarse_wns",
            "coarse_congestion",
            "coarse_wirelength",
            "coarse_overflow",
            "coarse_power",
            "coarse_setup_violations",
            "coarse_runtime_sec",
            "coarse_peak_rss_mb",
            "final_tns",
            "final_wns",
            "congestion",
            "routed_wirelength",
            "power",
            "runtime_sec",
        ):
            value = getattr(self, name)
            if value is not None:
                _finite_number(value, name)
        if self.status == "success" and self.final_tns is None:
            raise ValidationError("a successful label requires final_tns")
        if self.status == "coarse_success" and self.coarse_tns is None:
            raise ValidationError("a coarse_success label requires coarse_tns")
        if self.drc_count is not None and (
            isinstance(self.drc_count, bool) or self.drc_count < 0
        ):
            raise ValidationError("drc_count must be a non-negative integer")

    @property
    def target_cost(self) -> float:
        if self.status != "success" or self.final_tns is None:
            return math.inf
        return -min(float(self.final_tns), 0.0)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Label:
        required = {"design", "candidate_id", "status"}
        missing = required - payload.keys()
        if missing:
            raise ValidationError(f"label is missing fields: {sorted(missing)}")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValidationError("metadata must be an object")
        values: dict[str, Any] = {}
        for name in (
            "coarse_tns",
            "coarse_wns",
            "coarse_congestion",
            "coarse_wirelength",
            "coarse_overflow",
            "coarse_power",
            "coarse_setup_violations",
            "coarse_runtime_sec",
            "coarse_peak_rss_mb",
            "final_tns",
            "final_wns",
            "congestion",
            "routed_wirelength",
            "power",
            "runtime_sec",
        ):
            value = payload.get(name)
            values[name] = None if value is None else float(value)
        drc_count = payload.get("drc_count")
        return cls(
            schema_version=str(payload.get("schema_version", "2.0")),
            design=str(payload["design"]),
            candidate_id=str(payload["candidate_id"]),
            status=str(payload["status"]),
            replicate=int(payload.get("replicate", 0)),
            coarse_status=(
                None if payload.get("coarse_status") is None else str(payload["coarse_status"])
            ),
            drc_count=None if drc_count is None else int(drc_count),
            metadata=dict(metadata),
            **values,
        )

    @classmethod
    def load(cls, path: str | Path) -> Label:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise ValidationError("label JSON must be an object")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_cost"] = None if math.isinf(self.target_cost) else self.target_cost
        return payload

    def dump(self, path: str | Path) -> None:
        _write_json(path, self.to_dict())


def load_candidates(path: str | Path) -> list[Candidate]:
    source = Path(path)
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    files = [
        file
        for file in files
        if file.name not in {"manifest.json", "stats.json"}
        and not file.name.endswith(".metadata.json")
    ]
    if not files:
        raise ValidationError(f"no candidate JSON files found in {source}")
    candidates = [Candidate.load(file) for file in files]
    validate_candidate_set(candidates)
    return candidates


def load_labels(path: str | Path) -> list[Label]:
    source = Path(path)
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    if not files:
        raise ValidationError(f"no label JSON files found in {source}")
    labels = [Label.load(file) for file in files]
    seen: set[tuple[str, str]] = set()
    for label in labels:
        key = (label.design, label.candidate_id)
        if key in seen:
            raise ValidationError(f"duplicate label: {key}")
        seen.add(key)
    return labels


def validate_candidate_set(
    candidates: Iterable[Candidate], expected_names: Iterable[str] | None = None
) -> None:
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValidationError("candidate set must not be empty")
    design = candidate_list[0].design
    granularity = candidate_list[0].granularity
    die_count = candidate_list[0].die_count
    names = set(candidate_list[0].assignment)
    ids: set[str] = set()
    hashes: dict[str, str] = {}
    for candidate in candidate_list:
        if candidate.design != design:
            raise ValidationError("all candidates in a set must have the same design")
        if candidate.granularity != granularity or candidate.die_count != die_count:
            raise ValidationError("all candidates must share granularity and die_count")
        if set(candidate.assignment) != names:
            missing = sorted(names - set(candidate.assignment))
            extra = sorted(set(candidate.assignment) - names)
            raise ValidationError(
                f"candidate {candidate.candidate_id} has inconsistent objects; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if candidate.candidate_id in ids:
            raise ValidationError(f"duplicate candidate_id: {candidate.candidate_id}")
        ids.add(candidate.candidate_id)
        previous = hashes.get(candidate.assignment_hash)
        if previous is not None:
            raise ValidationError(
                f"duplicate assignment: {previous} and {candidate.candidate_id}"
            )
        hashes[candidate.assignment_hash] = candidate.candidate_id
    if expected_names is not None:
        expected = set(expected_names)
        if names != expected:
            raise ValidationError(
                f"candidate objects differ from expected set; "
                f"missing={sorted(expected - names)[:10]}, extra={sorted(names - expected)[:10]}"
            )


def write_json(path: str | Path, payload: Any) -> None:
    _write_json(path, payload)
