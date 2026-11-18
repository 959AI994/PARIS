from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateInputs:
    design: str
    candidate_id: str
    candidate_root: Path
    results_dir: Path
    gr_input_odb: Path
    gr_odb: Path
    gr_sdc: Path
    gr_guide: Path
    placed_def: Path
    original_gr_log: Path
    original_gr_timing_log: Path | None
    dr_odb: Path
    dr_sdc: Path
    dr_def: Path | None
    rcx_rules: Path
    setrc: Path
    l3_label: Path | None
    gr_segments: Path | None
    gr_segments_target: Path
    gr_congestion: Path | None


@dataclass(frozen=True)
class PinRecord:
    net_name: str
    pin_name: str
    pin_kind: str
    direction: str
    x_dbu: int
    y_dbu: int


@dataclass(frozen=True)
class NetRecord:
    net_name: str
    signal_type: str


@dataclass(frozen=True)
class Segment:
    net_name: str
    x1: int
    y1: int
    layer1: str
    x2: int
    y2: int
    layer2: str

    @property
    def is_wire(self) -> bool:
        return self.layer1 == self.layer2

    @property
    def length_dbu(self) -> int:
        return abs(self.x2 - self.x1) + abs(self.y2 - self.y1)


@dataclass(frozen=True)
class Rectangle:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def area(self) -> int:
        return max(0, self.x_max - self.x_min) * max(0, self.y_max - self.y_min)
