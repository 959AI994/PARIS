"""Prepare an isolated two-tier Nangate45 memory-on-logic project."""

import json
from pathlib import Path
import re
import shutil
from importlib.resources import files

from .utils import digest, save


def placement_blockage(lef):
    """Reserve full macro rectangles in placement, retaining routing geometry.

    Fixed nonrectangular obstructions become pseudo nodes in DREAMPlace.
    The Place-MoL DEF exporter assumes one node per original fixed macro.
    Bounding rectangles keep that mapping one-to-one and conservatively
    reserve the entire macro footprint. Routing uses the original LEFs.
    """
    if not re.search(r"\bCLASS\s+BLOCK\s*;", lef):
        return lef
    names = re.findall(r"^\s*MACRO\s+(\S+)\s*$", lef, re.M)
    sizes = re.findall(r"\bSIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;", lef)
    if len(names) != 1 or len(sizes) != 1:
        raise ValueError("Placement profile requires one hard macro per LEF")
    w, h = sizes[0]
    obs = f"\n  OBS\n    LAYER metal1 ;\n      RECT 0 0 {w} {h} ;\n  END\n"
    result, count = re.subn(r"(?ms)^\s*OBS\b.*?^\s*END\s*$", obs, lef)
    if count > 1:
        raise ValueError("Multiple obstruction sections in macro LEF")
    if count == 0:
        result = re.sub(
            r"(?m)^END\s+" + re.escape(names[0]) + r"\s*$", obs + "END " + names[0], lef
        )
    return result


def macro_variant(lef, liberty, tier, *, shrink=False):
    """Use the Open3DBench macro layer reflection; preserve Liberty tables."""
    if tier not in {"bottom", "upper"}:
        raise ValueError("Unknown tier")
    names = re.findall(r"^\s*MACRO\s+(\S+)\s*$", lef, re.M)
    cells = re.findall(r'\bcell\s*\(\s*"?([^"\s)]+)"?\s*\)', liberty)
    if len(names) != 1 or cells != names:
        raise ValueError("Expected one matching LEF/Liberty macro")
    name = names[0]
    target = name + "_" + tier
    lef = re.sub(r"\b" + re.escape(name) + r"\b", target, lef)
    if tier == "upper":
        lef = re.sub(
            r"\bmetal([1-9]|10)\b", lambda m: "metal" + str(21 - int(m[1])), lef
        )
    if shrink:
        # Only the temporary mixed-tier placement abstraction is shrunk.
        # Physical legalization and routing load the full-size variant.
        lef, n = re.subn(
            r"\bSIZE\s+[0-9.]+\s+BY\s+[0-9.]+\s*;", "SIZE 0.19 BY 1.4 ;", lef
        )
        if n != 1:
            raise ValueError("Expected a single macro size")
    liberty = re.sub(
        r'(\bcell\s*\(\s*"?)' + re.escape(name) + r'("?\s*\))',
        lambda m: m[1] + target + m[2],
        liberty,
    )
    liberty = re.sub(
        r'(\blibrary\s*\(\s*"?)([^"\s)]+)("?\s*\))',
        lambda m: m[1] + m[2] + "_" + tier + m[3],
        liberty,
        count=1,
    )
    return lef, liberty


def prepare(project, preprocessed, output, place_root, flow_root):
    project, pre = Path(project).resolve(), Path(preprocessed).resolve()
    out, place_root, flow_root = map(
        lambda p: Path(p).resolve(), (output, place_root, flow_root)
    )
    p = json.loads((project / "PARIS_PROJECT.json").read_text())
    for key in ("lef", "liberty", "sdc"):
        if isinstance(p["inputs"][key], str):
            p["inputs"][key] = [p["inputs"][key]]
    complete = json.loads((pre / "complete.json").read_text())
    if complete["status"] != "complete":
        raise ValueError("2D preprocessing is incomplete")
    for name, expected in complete["outputs"].items():
        if digest(pre / name) != expected:
            raise ValueError("Changed preprocessing output: " + name)
    out.mkdir(parents=True, exist_ok=False)
    scripts = out / "flow_scripts"
    shutil.copytree(flow_root / "scripts", scripts)
    for script in scripts.glob("*.tcl"):
        text = script.read_text()
        text = re.sub(
            r"(?m)^(\s*)(save_image[^\n]*)$",
            r"\1if {[llength [info commands gui::DisplayControlMap]]} { \2 }",
            text,
        )
        text = re.sub(r"(?m)^\s*detailed_route_debug[^\n]*\n", "", text)
        script.write_text(text)
    # Some OpenROAD builds do not restore router state from ODB alone.
    # Reload the exact emitted segments for the independent L2 timing report.
    l2 = scripts / "report_grt_l2.tcl"
    if l2.exists():
        text = l2.read_text().replace(
            "estimate_parasitics -global_routing",
            "read_global_route_segments $::env(RESULTS_DIR)/gr_segments.txt\nestimate_parasitics -global_routing",
        )
        l2.write_text(text)
    with (scripts / "global_route.tcl").open("a") as f:
        f.write("\nwrite_global_route_segments $::env(RESULTS_DIR)/gr_segments.txt\n")
    place = out / "Place-MoL"
    place.mkdir()
    shutil.copytree(
        place_root / "src", place / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(place_root / "config", place / "config")
    dream = place / "DREAMPlace"
    dream.mkdir()
    for name in ("install", "thirdparty"):
        (dream / name).symlink_to(
            place_root / "DREAMPlace" / name, target_is_directory=True
        )
    test = dream / "test/or_3D"
    test.mkdir(parents=True)
    dp = json.loads(
        files("paris_tool").joinpath("resources/mol_dreamplace.json").read_text()
    )
    placement_lefs = []
    for index, value in enumerate(p["inputs"]["lef"]):
        original = project / value
        text = original.read_text()
        if re.search(r"\bCLASS\s+BLOCK\s*;", text):
            dest = out / "placement_lef" / f"{index}_{original.name}"
            dest.parent.mkdir(exist_ok=True)
            dest.write_text(placement_blockage(text))
            placement_lefs.append(str(dest))
        else:
            placement_lefs.append(str(original))
    dp.update(
        lef_input=placement_lefs,
        lib_input=[str(project / f) for f in p["inputs"]["liberty"]],
        def_input=str(pre / "floorplan.def"),
        # The linked DEF carries the authoritative connectivity.
        verilog_input=None,
        gpu=0,
        num_threads=2,
        result_dir=str(out / "dreamplace_results"),
        random_seed=17,
    )
    for suffix, density in (("", 0.7), ("_macroFixed", 0.4)):
        save(
            test / (p["design"] + "_3D" + suffix + ".json"),
            {**dp, "target_density": density},
        )
    config = json.loads((place / "config/or_3D.json").read_text())
    config.update(gpu=0, use_cuda=False, benchmark=p["design"], seed=17)
    save(place / "config/project.json", config)
    platform = flow_root / "platforms/nangate45_3D"
    variants = out / "technology"
    for tier in ("bottom", "upper", "upper_shrink"):
        (variants / tier).mkdir(parents=True)
    used_types = set(
        json.loads((pre / "STRUCTURE.json").read_text())["macros"].values()
    )
    macros = []
    liberty_files = [project / value for value in p["inputs"]["liberty"]]
    for value in p["inputs"]["lef"]:
        lf = project / value
        content = lf.read_text()
        if re.search(r"\bCLASS\s+BLOCK\s*;", content):
            names = re.findall(r"^\s*MACRO\s+(\S+)\s*$", content, re.M)
            if len(names) != 1:
                raise ValueError("This MoL profile requires one macro per LEF")
            if names[0] in used_types:
                macros.append((names[0], lf))
    if {name for name, _ in macros} != used_types:
        raise ValueError("Missing physical macro masters")
    standard = next(
        (f for f in liberty_files if f.name == "NangateOpenCellLibrary_typical.lib"),
        None,
    )
    if standard is None or digest(standard) != digest(
        platform / "lib/NangateOpenCellLibrary_typical.lib"
    ):
        raise ValueError(
            "The selected MoL technology does not match the source standard-cell Liberty"
        )
    inventory = []
    for name, lf in macros:
        matches = [
            lib
            for lib in liberty_files
            if re.search(
                r'\bcell\s*\(\s*"?' + re.escape(name) + r'"?\s*\)', lib.read_text()
            )
        ]
        if len(matches) != 1:
            raise ValueError("Expected one timing definition for macro: " + name)
        lib = matches[0]
        for variant in ("bottom", "upper", "upper_shrink"):
            tier = "bottom" if variant == "bottom" else "upper"
            lt, bt = macro_variant(
                lf.read_text(), lib.read_text(), tier, shrink=variant.endswith("shrink")
            )
            # OpenDB identifies LEF libraries by filename; upper and bottom
            # masters must not share a basename. ORFS also keys Liberty copies
            # by basename, so keep the tier in both filenames.
            dest = variants / variant / (name + "." + tier + ".lef")
            dest.write_text(lt)
            if variant != "upper_shrink":
                (dest.with_suffix(".lib")).write_text(bt)
            inventory.append(
                dict(
                    source=str(lf),
                    variant=variant,
                    output=str(dest),
                    sha256=digest(dest),
                )
            )
    # All standard cells stay on the bottom die. Full-size masters are restored
    # after the stock mixed-tier CTS placement abstraction, before GR/DR.
    common_lib = [
        platform / f"lib_{tier}/{name}.{tier}.lib"
        for tier in ("bottom", "upper")
        for name in ("NangateOpenCellLibrary_typical",)
    ]
    common_lib += list((variants / "bottom").glob("*.lib")) + list(
        (variants / "upper").glob("*.lib")
    )
    for shrink in (False, True):
        upper_dir = "lef_upper_shrink" if shrink else "lef_upper"
        upper_variant = "upper_shrink" if shrink else "upper"
        lefs = [
            platform / upper_dir / "NangateOpenCellLibrary.macro.mod.upper.lef",
        ]
        lefs += list((variants / "bottom").glob("*.lef")) + list(
            (variants / upper_variant).glob("*.lef")
        )
        for f in lefs + common_lib:
            if not f.is_file():
                raise FileNotFoundError(f)
        # Actual die/core coordinates come from the generated candidate DEF.
        text = "\n".join(
            [
                f"export DESIGN_NAME = {p['top']}",
                f"export DESIGN_NICKNAME = {p['design']}",
                "export PLATFORM = nangate45_3D",
                f"export SCRIPTS_DIR = {scripts}",
                f"export DESIGN_DIR = {out}",
                f"export VERILOG_FILES = {pre / 'netlist.v'}",
                f"export SDC_FILE = {project / p['inputs']['sdc'][0]}",
                "export ADDITIONAL_LEFS = " + " ".join(map(str, lefs)),
                "export ADDITIONAL_LIBS = " + " ".join(map(str, common_lib)),
                "export TNS_END_PERCENT = 100",
                "export PLACE_DENSITY = 0.7",
                "export NUM_CORES = 2",
                "export GLOBAL_ROUTE_ARGS = -allow_congestion -verbose -congestion_iterations 5",
                "export DETAILED_ROUTE_ARGS = -droute_end_iter 1",
                "export LAYER_DP_SAVE_IMAGES = 0",
                "export LAYER_DP_MAX_DISPLACEMENT = 300",
                "export SYNTH_HIERARCHICAL = 0",
                "",
            ]
        )
        (out / ("config_upper_shrink.mk" if shrink else "config.mk")).write_text(text)
    shutil.copyfile(project / p["inputs"]["sdc"][0], out / "constraint.sdc")
    save(
        out / "MOL_PROJECT.json",
        dict(
            status="complete",
            design=p["design"],
            place_root=str(place),
            flow_root=str(flow_root),
            preprocessing=str(pre),
            macro_variants=inventory,
            protocol=dict(
                tiers=2,
                logic_tier="bottom",
                macro_assignment="candidate",
                upper_macro_layers="metalN -> metal(21-N)",
                physical_masters="full size before legalization and routing",
            ),
        ),
    )
    return dict(status="complete", output=str(out), place_root=str(place))
