#!/usr/bin/env python3
"""Run per-die detailed placement at the smallest successful displacement."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def schedule(design: str) -> list[int]:
    if design in {"bp_fe", "swerv_wrapper"}:
        return [120, 160, 240, 300]
    return [80, 120, 160, 240, 300]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in {"bottom", "upper"}:
        print("usage: adaptive_layer_legalize.py bottom|upper", file=sys.stderr)
        return 2
    die = sys.argv[1]
    results = Path(os.environ["RESULTS_DIR"])
    design = os.environ.get("DESIGN_NICKNAME", os.environ.get("DESIGN_NAME", ""))
    openroad = os.environ.get("OPENROAD_EXE", "openroad")
    script = Path(os.environ["FLOW_HOME"]) / "scripts_3D" / f"lg_{die}.tcl"
    output = results / f"{die}_legalized.def"

    for displacement in schedule(design):
        env = os.environ.copy()
        env["LAYER_DP_MAX_DISPLACEMENT"] = str(displacement)
        log = results / f"adaptive_legalize_{die}_{displacement}.log"
        output.unlink(missing_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(
                [openroad, "-exit", str(script)],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode == 0 and output.is_file():
            print(f"adaptive layer legalization: die={die} displacement={displacement}")
            (results / f"adaptive_legalize_{die}.txt").write_text(
                f"{displacement}\n", encoding="utf-8"
            )
            return 0
        print(
            f"adaptive layer legalization retry: die={die} "
            f"displacement={displacement} return_code={proc.returncode}",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
