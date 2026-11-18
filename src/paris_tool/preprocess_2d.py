"""Native 2D netlist/RTL to linked LEF/DEF input for a 3D partitioner."""

import json
import re
from pathlib import Path
import subprocess
import time

from .utils import digest, save


def preprocess(project, output, yosys, openroad, threads=2):
    root = Path(project).resolve()
    manifest = root / "PARIS_PROJECT.json"
    p = json.loads(manifest.read_text())
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", p["top"]):
        raise ValueError("Invalid top module name")
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="running", started=time.time(), top=p["top"], inputs={})
    inputs = {}
    for name in ("netlist", "lef", "liberty", "sdc"):
        values = p["inputs"][name]
        values = [values] if isinstance(values, str) else values
        inputs[name] = []
        for value in values:
            path = (root / value).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Invalid project input: " + str(value))
            inputs[name].append(path)
            receipt["inputs"][value] = digest(path)
    save(out / "status.json", receipt)

    def run(argv, name):
        receipt.update(stage=name)
        save(out / "status.json", receipt)
        with (out / (name + ".log")).open("w") as log:
            subprocess.run(
                argv, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=out
            )

    def yquote(value):
        return json.dumps(str(value))

    def tq(value):
        # Tcl paths are data, never substituted commands.
        return (
            "{"
            + str(value).replace("\\", "\\\\").replace("}", "\\}").replace("{", "\\{")
            + "}"
        )

    try:
        standard = next(
            (
                f
                for f in inputs["liberty"]
                if f.name == "NangateOpenCellLibrary_typical.lib"
            ),
            None,
        )
        if standard is None:
            raise ValueError(
                "This flow profile requires Nangate45 standard-cell Liberty"
            )
        script = [f"read_liberty -lib {yquote(f)}" for f in inputs["liberty"]]
        script += [f"read_verilog {yquote(f)}" for f in inputs["netlist"]]
        script += [
            f"hierarchy -check -top {p['top']}",
            "setattr -mod -unset keep_hierarchy",
            f"synth -top {p['top']} -flatten",
            f"dfflibmap -liberty {yquote(standard)}",
            f"abc -liberty {yquote(standard)}",
            f"write_json {yquote(out / 'before_constants.json')}",
            "setundef -zero",
            "hilomap -hicell LOGIC1_X1 Z -locell LOGIC0_X1 Z",
            "clean",
            "check -assert",
            "stat",
            f"write_json {yquote(out / 'netlist.json')}",
            f"write_verilog -noattr -noexpr {yquote(out / 'netlist.v')}",
        ]
        ys = out / "synthesis.ys"
        ys.write_text("\n".join(script) + "\n")
        run([str(yosys), "-s", str(ys)], "synthesis")
        j = json.loads((out / "netlist.json").read_text())
        cells = j["modules"][p["top"]]["cells"]
        before = json.loads((out / "before_constants.json").read_text())["modules"][
            p["top"]
        ]

        def undefined_count(module):
            groups = [
                bits
                for cell in module["cells"].values()
                for bits in cell["connections"].values()
            ]
            groups += [port["bits"] for port in module["ports"].values()]
            return sum(
                isinstance(bit, str) and bit in {"x", "z"}
                for bits in groups
                for bit in bits
            )

        constant_policy = dict(
            policy="setundef -zero then hilomap, matching upstream OpenROAD-3D synth.tcl",
            undefined_pin_bits_before=undefined_count(before),
            undefined_pin_bits_after=undefined_count(j["modules"][p["top"]]),
        )
        save(out / "CONSTANT_POLICY.json", constant_policy)
        if constant_policy["undefined_pin_bits_after"]:
            raise ValueError(
                "Unresolved constant inputs remain after physical synthesis"
            )
        macro_types = set()

        for f in inputs["lef"]:
            text = f.read_text()
            for match in re.finditer(
                r"\bMACRO\s+(\S+)(.*?)(?=\nEND\s+\1(?:\s|$))", text, re.S
            ):
                if re.search(r"\bCLASS\s+BLOCK\s*;", match[2]):
                    macro_types.add(match[1])
        macros = {n: c["type"] for n, c in cells.items() if c["type"] in macro_types}
        if not macros:
            raise ValueError("No memory macros found for the MoL partition flow")
        save(
            out / "STRUCTURE.json",
            dict(
                design=p["design"],
                top=p["top"],
                instances=len(cells),
                macros=macros,
                standard_cells=len(cells) - len(macros),
            ),
        )
        tcl = [f"set_thread_count {int(threads)}"]
        tcl += [f"read_lef {tq(f)}" for f in inputs["lef"]]
        tcl += [f"read_liberty {tq(f)}" for f in inputs["liberty"]]
        tcl += [f"read_verilog {tq(out / 'netlist.v')}", f"link_design {tq(p['top'])}"]
        tcl += [
            "set empty_nets 0",
            "foreach net [[ord::get_db_block] getNets] {",
            "  if {[llength [$net getITerms]] == 0 && [llength [$net getBTerms]] == 0} {",
            "    odb::dbNet_destroy $net",
            "    incr empty_nets",
            "  }",
            "}",
            'puts "PARIS_REMOVED_EMPTY_NETS $empty_nets"',
            "foreach net [[ord::get_db_block] getNets] {",
            "  if {[$net getSigType] in {POWER GROUND} && ![$net isSpecial]} {",
            "    foreach it [$net getITerms] {",
            "      set mt [$it getMTerm]",
            '      if {[$mt getSigType] eq {SIGNAL}} { error "Unimplemented logic constant on non-routable PG net [$net getName]" }',
            "    }",
            "  }",
            "}",
        ]
        tcl += [f"read_sdc {tq(f)}" for f in inputs["sdc"]]
        tcl += [
            "initialize_floorplan -utilization 40 -aspect_ratio 1 -core_space 10 -site FreePDK45_38x28_10R_NP_162NW_34O",
            "make_tracks",
            "set_io_pin_constraint -group -region bottom:* -pin_names *",
            "place_pins -hor_layers metal3 -ver_layers metal2",
            f"write_def {tq(out / 'floorplan.def')}",
            f"write_db {tq(out / 'floorplan.odb')}",
            "report_design_area",
            "check_setup",
            "report_tns",
        ]
        (out / "floorplan.tcl").write_text("\n".join(tcl) + "\n")
        run([str(openroad), "-exit", str(out / "floorplan.tcl")], "floorplan")
        for value, expected in receipt["inputs"].items():
            if digest(root / value) != expected:
                raise ValueError("Input changed during preprocessing: " + value)
        receipt.update(
            status="complete",
            finished=time.time(),
            instances=len(cells),
            macros=len(macros),
            outputs={
                name: digest(out / name)
                for name in (
                    "netlist.v",
                    "netlist.json",
                    "floorplan.def",
                    "floorplan.odb",
                    "STRUCTURE.json",
                    "CONSTANT_POLICY.json",
                    "before_constants.json",
                )
            },
        )
        save(out / "complete.json", receipt)
        save(out / "status.json", receipt)
        return receipt
    except Exception as exc:
        receipt.update(status="failed", error=str(exc), finished=time.time())
        save(out / "status.json", receipt)
        raise
