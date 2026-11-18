#!/usr/bin/env python3
"""Transplant candidate coordinates into a cached timing-resized DEF."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PLACEMENT = re.compile(r"\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)")


def candidate_placement(path: Path) -> tuple[dict[str, tuple[str, str, str]], dict[str, str]]:
    coordinates: dict[str, tuple[str, str, str]] = {}
    macro_masters: dict[str, str] = {}
    inside = False
    current_name: str | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("COMPONENTS "):
                inside = True
                continue
            if stripped == "END COMPONENTS":
                break
            if not inside:
                continue
            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == "-":
                current_name = parts[1]
                if "fakeram" in parts[2].lower():
                    macro_masters[current_name] = parts[2]
            if current_name is not None:
                match = PLACEMENT.search(line)
                if match:
                    coordinates[current_name] = match.groups()
                    current_name = None
    return coordinates, macro_masters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prototype", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()

    coordinates, macro_masters = candidate_placement(args.candidate)
    common = moved = macro_swaps = 0
    inside = False
    with args.prototype.open("r", encoding="utf-8") as source, args.output.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            stripped = line.strip()
            if stripped.startswith("COMPONENTS "):
                inside = True
            elif stripped == "END COMPONENTS":
                inside = False
            if inside:
                parts = stripped.split()
                if len(parts) >= 3 and parts[0] == "-" and parts[1] in coordinates:
                    name = parts[1]
                    common += 1
                    x, y, orient = coordinates[name]
                    replacement = f"+ PLACED ( {x} {y} ) {orient}"
                    updated = PLACEMENT.sub(replacement, line, count=1)
                    if updated != line:
                        moved += 1
                    master = macro_masters.get(name)
                    if master is not None and parts[2] != master:
                        updated = updated.replace(f" {parts[2]} ", f" {master} ", 1)
                        macro_swaps += 1
                    line = updated
            target.write(line)

    stats = {
        "candidate_coordinate_count": len(coordinates),
        "common_instances": common,
        "moved_instances": moved,
        "macro_master_swaps": macro_swaps,
    }
    if args.stats:
        args.stats.write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
