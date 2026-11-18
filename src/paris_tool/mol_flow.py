"""Native Place-MoL / OpenROAD stage runner with isolated candidate outputs."""

import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from .utils import digest, save


def validate_placement(prepared, candidate, placed, output):
    """Check actual full-size macro boxes, tier assignment and containment."""
    prepared, placed = Path(prepared), Path(placed)
    c = json.loads(Path(candidate).read_text())
    config = (prepared / "config.mk").read_text()
    files = re.search(r"(?m)^export ADDITIONAL_LEFS = (.+)$", config)[1].split()
    sizes = {}
    for f in files:
        for match in re.finditer(
            r"\bMACRO\s+(\S+)(.*?)(?=\nEND\s+\1(?:\s|$))", Path(f).read_text(), re.S
        ):
            size = re.search(r"\bSIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;", match[2])
            sizes[match[1]] = tuple(map(float, size.groups()))
    text = placed.read_text()
    dbu = int(re.search(r"UNITS DISTANCE MICRONS\s+(\d+)", text)[1])
    die = tuple(
        map(
            int,
            re.search(
                r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)",
                text,
            ).groups(),
        )
    )
    section = re.search(r"\bCOMPONENTS\s+\d+\s*;(.*?)END COMPONENTS", text, re.S)[1]
    boxes, errors = {}, []
    for match in re.finditer(r"(?:^|\n)\s*-\s+(\S+)\s+(\S+)(.*?);", section, re.S):
        name, master = match[1], match[2]
        if name not in c["assignment"]:
            continue
        tier = c["assignment"][name]
        if not master.endswith("_upper" if tier else "_bottom") or master not in sizes:
            raise ValueError("Macro master/tier mismatch: " + name)
        location = re.search(
            r"\+\s+(FIXED|FIRM|PLACED)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\S+)", match[3]
        )
        if not location or location[4] != "N":
            raise ValueError("Expected placed N-oriented macro: " + name)
        x, y = int(location[2]), int(location[3])
        w, h = sizes[master]
        box = (x, y, x + round(w * dbu), y + round(h * dbu))
        boxes[name] = dict(tier=tier, master=master, box=box)
        if box[0] < die[0] or box[1] < die[1] or box[2] > die[2] or box[3] > die[3]:
            errors.append(dict(kind="outside_die", macro=name, box=box))
    if set(boxes) != set(c["assignment"]):
        raise ValueError("Placed macro inventory differs from partition candidate")
    items = list(boxes.items())
    for i, (name, a) in enumerate(items):
        for other, b in items[i + 1 :]:
            x, y = a["box"], b["box"]
            if (
                a["tier"] == b["tier"]
                and min(x[2], y[2]) > max(x[0], y[0])
                and min(x[3], y[3]) > max(x[1], y[1])
            ):
                errors.append(dict(kind="overlap", macros=[name, other]))
    result = dict(
        status="passed" if not errors else "failed",
        macro_count=len(boxes),
        die=die,
        dbu=dbu,
        errors=errors,
        macro_boxes=boxes,
        def_sha256=digest(placed),
    )
    save(output, result)
    if errors:
        raise ValueError("Macro geometry validation failed: " + str(errors[:3]))
    return result


def legalize_track_rounding(prepared, candidate, placed, output, audit):
    """Legalize local macro overlaps without changing sizes or tiers."""
    placed, output = Path(placed), Path(output)
    if output.exists():
        raise FileExistsError(output)
    text = placed.read_text()
    dbu = int(re.search(r"UNITS DISTANCE MICRONS\s+(\d+)", text)[1])
    sx, sy = round(0.19 * dbu), round(0.14 * dbu)
    offsets = sorted(
        ((a * sx, b * sy) for a in range(-5, 6) for b in range(-5, 6) if a or b),
        key=lambda v: (abs(v[0]) + abs(v[1]), v),
    )
    changes = []
    with tempfile.TemporaryDirectory(prefix="paris-track-") as td:
        trial, report = Path(td) / "trial.def", Path(td) / "audit.json"

        def check(value):
            trial.write_text(value)
            report.unlink(missing_ok=True)
            try:
                validate_placement(prepared, candidate, trial, report)
            except ValueError:
                if not report.exists():
                    raise
            return json.loads(report.read_text())

        current = check(text)
        for _ in range(36):
            if not current["errors"]:
                break
            error = current["errors"][0]
            names = list(reversed(error.get("macros", [error.get("macro")])))
            accepted = False
            actions = []

            def snapped(value, pitch):
                return (
                    int(math.copysign(math.ceil(abs(value) / pitch) * pitch, value))
                    if value
                    else 0
                )

            # Try exact separating edges before the small track neighborhood.
            # Upstream legalization can leave overlaps wider than five tracks.
            for name in names:
                box = current["macro_boxes"][name]["box"]
                if error["kind"] == "overlap":
                    other = next(n for n in error["macros"] if n != name)
                    boundary = current["macro_boxes"][other]["box"]
                    moves = [
                        (boundary[0] - box[2], 0),
                        (boundary[2] - box[0], 0),
                        (0, boundary[1] - box[3]),
                        (0, boundary[3] - box[1]),
                    ]
                else:
                    die = current["die"]
                    moves = [
                        (
                            max(die[0] - box[0], min(0, die[2] - box[2])),
                            max(die[1] - box[1], min(0, die[3] - box[3])),
                        )
                    ]
                for dx, dy in moves:
                    dx, dy = snapped(dx, sx), snapped(dy, sy)
                    if (dx or dy) and max(abs(dx), abs(dy)) <= 300 * dbu:
                        actions.append((name, dx, dy))
            actions.sort(key=lambda a: abs(a[1]) + abs(a[2]))
            actions += [(name, dx, dy) for dx, dy in offsets for name in names]
            tried = set()
            for name, dx, dy in actions:
                if (name, dx, dy) in tried:
                    continue
                tried.add((name, dx, dy))
                pattern = (
                    r"(?ms)(^\s*-\s+"
                    + re.escape(name)
                    + r"\s+\S+.*?\+\s+(?:FIXED|FIRM|PLACED)\s*\(\s*)(-?\d+)(\s+)(-?\d+)"
                )
                m = re.search(pattern, text)
                if not m:
                    raise ValueError("Cannot locate macro placement: " + name)
                value = (
                    text[: m.start(2)]
                    + str(int(m[2]) + dx)
                    + m[3]
                    + str(int(m[4]) + dy)
                    + text[m.end(4) :]
                )
                result = check(value)
                if len(result["errors"]) < len(current["errors"]):
                    changes.append(dict(macro=name, dx_dbu=dx, dy_dbu=dy))
                    text, current, accepted = value, result, True
                    break
            if not accepted:
                raise ValueError(
                    "Track rounding needs global macro legalization: " + str(error)
                )
        if current["errors"]:
            raise ValueError("Macro legalization did not converge")
    output.write_text(text)
    checked = validate_placement(prepared, candidate, output, audit)
    checked.update(source_sha256=digest(placed), changes=changes)
    save(audit, checked)
    return output


def run_candidate(prepared, candidate, output, python, openroad, *, placement_def=None):
    prepared = Path(prepared).resolve()
    candidate = Path(candidate).resolve()
    out = Path(output).resolve()
    meta = json.loads((prepared / "MOL_PROJECT.json").read_text())
    c = json.loads(candidate.read_text())
    design, cid = c["design"], c["candidate_id"]
    if design != meta["design"] or not all(
        re.fullmatch(r"[A-Za-z0-9_-]+", v) for v in (design, cid)
    ):
        raise ValueError("Invalid candidate identity")
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 60 * 1024**3:
        raise ValueError("Less than 60 GiB available")
    out.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="running",
        phase="placement",
        design=design,
        candidate_id=cid,
        started=time.time(),
        prepared=str(prepared),
        candidate=str(candidate),
        input_sha256=digest(candidate),
        openroad=str(openroad),
        python=str(python),
    )
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
        PYTHONUNBUFFERED="1",
    )
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")

    def run(argv, phase, cwd):
        state.update(phase=phase)
        save(out / "status.json", state)
        with (out / (phase + ".log")).open("x") as log:
            subprocess.run(
                list(map(str, argv)),
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        place = Path(meta["place_root"])
        if placement_def:
            placed = Path(placement_def).resolve()
            state["placement_input"] = dict(path=str(placed), sha256=digest(placed))
        else:
            run(
                [
                    python,
                    "-u",
                    place / "src/place_3d/main.py",
                    f"--benchmark={design}",
                    "--seed=17",
                    f"--config_file={place / 'config/project.json'}",
                    f"--partition_file={candidate}",
                    f"--run_id={cid}",
                    f"--output_root={out / 'placement'}",
                ],
                "placement",
                place,
            )
            placed = (
                out
                / "placement"
                / design
                / cid
                / "mol_final"
                / (design + "_suffixed.def")
            )
        if not placed.is_file():
            raise ValueError("Missing legalized 3D DEF")
        placed = legalize_track_rounding(
            prepared,
            candidate,
            placed,
            out / "placement.def",
            out / "PLACEMENT_AUDIT.json",
        )
        state["placement_def"] = str(placed)
        base = [
            "make",
            f"DEF_INPUT={placed}",
            f"DEF_VERSION={cid}",
            f"FLOW_VARIANT=paris_{cid}",
            f"WORK_HOME={out / 'work'}",
            "NPROC=2",
            "NUM_CORES=2",
            f"OPENROAD_EXE={Path(openroad).resolve()}",
        ]
        flow = Path(meta["flow_root"])
        run(
            base
            + [f"DESIGN_CONFIG={prepared / 'config_upper_shrink.mk'}", "do-autoflow"],
            "cts",
            flow,
        )
        run(
            base + [f"DESIGN_CONFIG={prepared / 'config.mk'}", "do-cts_coarse_eval"],
            "legalize",
            flow,
        )
        run(
            base
            + [
                f"DESIGN_CONFIG={prepared / 'config.mk'}",
                "do-5_1_grt",
                "do-report_grt_l2_eval",
                "do-extract_cts_surrogate",
            ],
            "global_route",
            flow,
        )
        results = out / "work/results/nangate45_3D" / design / ("paris_" + cid)
        logs = out / "work/logs/nangate45_3D" / design / ("paris_" + cid)
        outputs = {
            name: digest(results / name)
            for name in ("4_cts.odb", "4_cts.sdc", "5_1_grt.odb", "gr_segments.txt")
        }
        if digest(candidate) != state["input_sha256"]:
            raise ValueError("Candidate changed during flow")
        state.update(
            status="complete",
            phase="global_route",
            finished=time.time(),
            results=str(results),
            logs=str(logs),
            reports=str(out / "work/reports/nangate45_3D" / design / ("paris_" + cid)),
            outputs=outputs,
        )
        save(out / "complete.json", state)
        save(out / "status.json", state)
        return state
    except Exception as exc:
        state.update(status="failed", error=str(exc), finished=time.time())
        save(out / "status.json", state)
        raise


def finish_candidate(gr_run, output, *, route_iterations=1):
    if (
        isinstance(route_iterations, bool)
        or not isinstance(route_iterations, int)
        or not 1 <= route_iterations <= 64
    ):
        raise ValueError("route_iterations must be an integer from 1 to 64")
    gr = Path(gr_run).resolve()
    state = json.loads((gr / "complete.json").read_text())
    results = Path(state["results"])
    for name, expected in state["outputs"].items():
        if digest(results / name) != expected:
            raise ValueError("Changed GR artifact: " + name)
    out = Path(output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 60 * 1024**3:
        raise ValueError("Less than 60 GiB available")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(gr / "work", out / "work")
    prepared = Path(state["prepared"])
    meta = json.loads((prepared / "MOL_PROJECT.json").read_text())
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        CUDA_VISIBLE_DEVICES="",
    )
    env["PATH"] = str(Path(state["python"]).parent) + os.pathsep + env.get("PATH", "")
    cid, design = state["candidate_id"], state["design"]
    dr = dict(
        status="running",
        phase="detailed_route",
        gr_run=str(gr),
        route_iterations=route_iterations,
        started=time.time(),
    )
    save(out / "status.json", dr)
    try:
        argv = [
            "make",
            f"DESIGN_CONFIG={prepared / 'config.mk'}",
            f"DEF_INPUT={state['placement_def']}",
            f"DEF_VERSION={cid}",
            f"FLOW_VARIANT=paris_{cid}",
            f"WORK_HOME={out / 'work'}",
            "NPROC=2",
            "NUM_CORES=2",
            f"OPENROAD_EXE={state['openroad']}",
            f"DETAILED_ROUTE_ARGS=-droute_end_iter {route_iterations}",
            "do-5_2_route",
            "do-5_route",
            "do-5_route.sdc",
            "do-6_1_fill",
            "do-6_1_fill.sdc",
        ]
        with (out / "detailed_route.log").open("x") as log:
            subprocess.run(
                argv,
                cwd=meta["flow_root"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        results = out / "work/results/nangate45_3D" / design / ("paris_" + cid)
        dr.update(
            status="complete",
            finished=time.time(),
            results=str(results),
            outputs={n: digest(results / n) for n in ("6_1_fill.odb", "6_1_fill.sdc")},
        )
        save(out / "complete.json", dr)
        save(out / "status.json", dr)
        return dr
    except Exception as exc:
        dr.update(status="failed", error=str(exc), finished=time.time())
        save(out / "status.json", dr)
        raise
