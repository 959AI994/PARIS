from dataclasses import dataclass

@dataclass(frozen=True)
class SelectionConfig:
    mode: str = 'limited'
    max_nets: int | None = 1000
    include_clock_nets: bool = True
    net_patterns: tuple[str, ...] = ()

