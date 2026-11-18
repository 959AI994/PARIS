"""Independent GR feature extraction and measured-DR label ingestion.

No candidate feature or net choice consults a DR checkpoint. Historical wire
labels are joined only afterwards; unmatched arcs are reported, never invented.
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from . import config, data
from .utils import digest, fingerprint, save, run
from ._collect.collection.readers.openroad import read_design_export, read_timing_export
from ._collect.collection.readers.segments import read_route_segments
from ._collect.collection.readers.congestion import read_congestion_cells
from ._collect.collection.geometry import (
    Rectangle,
    grid_boundaries,
    build_blockage_cells,
)
from ._collect.collection.assemble import assemble_gr_tables, read_layer_rc
from ._collect.collection.selection import select_nets
from ._collect.common.config import SelectionConfig

RES = Path(__file__).parent / "resources"


def settings_hash(c, r, stage):
    o = c["openroad"]
    fields = {
        "gr": [
            "cts_odb",
            "cts_sdc",
            "gr_odb",
            "gr_sdc",
            "gr_segments",
            "gr_congestion",
        ],
        "dr": ["dr_odb", "dr_sdc"],
    }[stage]
    paths = (
        [r[k] for k in fields if r.get(k)]
        + o["liberty"]
        + [o["setrc"], o["gr_binary"] if stage == "gr" else o["dr_binary"]]
    )
    if stage == "dr":
        paths += [o["rcx_rules"]]
    paths += list(RES.glob("*.tcl"))
    paths += [
        Path(__file__),
        Path(config.__file__),
        *list((Path(__file__).parent / "_collect").rglob("*.py")),
    ]
    hashes = {str(p): fingerprint(p) for p in paths}
    hashes["settings"] = dict(
        corner=o.get("corner", "nominal"),
        selection=c.get("selection", {}),
        path_endpoint_limit=c.get("path_endpoint_limit", 1024),
        time_unit="ns",
    )
    return hashes


def tool(c, r, stage, folder, script, extra=None, name=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    o = c["openroad"]
    env = os.environ.copy()
    env.update(
        ODB_PATH=str(r[stage + "_odb"]),
        SDC_PATH=str(r[stage + "_sdc"]),
        LIB_FILES="|".join(map(str, o["liberty"])),
        SETRC_PATH=str(o["setrc"]),
        EXTRACTION_STAGE=stage,
        RCX_RULES=str(o.get("rcx_rules", "")),
        GR_SEGMENTS_PATH=str(r.get("gr_segments", "")),
        OUTPUT_DIR=str(folder),
        ENDPOINT_OUTPUT=str(folder / "endpoints.tsv"),
        PATH_OUTPUT=str(folder / "paths.tsv"),
        METRICS_OUTPUT=str(folder / "metrics.tsv"),
        DR_SPEF_PATH=str(folder / "extracted.spef"),
        OMP_NUM_THREADS=str(c["threads"]),
        PATH_ENDPOINT_LIMIT=str(c.get("path_endpoint_limit", 1024)),
    )
    env.update({k: str(v) for k, v in (extra or {}).items()})
    binary = o["dr_binary"] if stage == "dr" else o["gr_binary"]
    run(
        [binary, "-no_init", "-exit", RES / script],
        folder,
        env,
        c["timeout"],
        name or script.replace(".tcl", ""),
    )


def metrics(folder):
    f = pd.read_csv(Path(folder) / "metrics.tsv", sep="\t")
    m = dict(zip(f.metric, f.value))
    m = {k: float(v) for k, v in m.items()}
    if not all(np.isfinite(v) for v in m.values()) or abs(
        m["endpoint_sum"] - m["api_tns"]
    ) > max(0.002, abs(m["api_tns"]) * 2e-5):
        raise ValueError("Endpoint sum/STA TNS closure failed")
    if m["api_tns"] > 0 or m["finite_endpoints"] < 1:
        raise ValueError("Invalid STA metrics")
    return m


def stats(a, prefix, row):
    a = pd.to_numeric(a, errors="coerce").to_numpy(float)
    a = a[np.isfinite(a)]
    if not len(a):
        raise ValueError("No finite observations for endpoint statistic: " + prefix)
    row[prefix + "_mean"] = float(a.mean())
    row[prefix + "_std"] = float(a.std())
    for q in [0.1, 0.5, 0.9, 0.99]:
        row[prefix + "_q" + str(int(q * 100))] = float(np.quantile(a, q))


def endpoint_summary(folder, m):
    e = pd.read_csv(Path(folder) / "endpoints.tsv", sep="\t")
    p = pd.read_csv(Path(folder) / "paths.tsv", sep="\t")
    slack = e.slack.to_numpy(float)
    row = dict(
        gr_log_cost=float(np.log1p(-m["api_tns"])),
        gr_wns=m["api_wns"],
        negative_fraction=float((slack < 0).mean()),
        negative_count=int((slack < 0).sum()),
        slack_below_minus_1=float((slack < -1).mean()),
        slack_below_minus_2=float((slack < -2).mean()),
        slack_top100_mean=float(np.sort(slack)[:100].mean()),
        slack_top1000_mean=float(np.sort(slack)[:1000].mean()),
    )
    for k in [
        "slack",
        "slack_rise",
        "slack_fall",
        "arrival_rise",
        "arrival_fall",
        "slew_rise",
        "slew_fall",
    ]:
        stats(e[k], k, row)
    for k in [
        "path_slack",
        "path_span",
        "cell_delay",
        "wire_delay",
        "cell_arcs",
        "wire_arcs",
        "slew_mean",
        "slew_max",
        "point_count",
    ]:
        stats(p[k], "path_" + k, row)
    if len(row) != 104:
        raise ValueError("Endpoint schema mismatch")
    return {"ep_" + k: v for k, v in row.items()}


def collect_gr(c, r):
    folder = c["workspace"] / "jobs" / r["design"] / r["candidate_id"] / "gr"
    signature = settings_hash(c, r, "gr")
    receipt = folder / "complete.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["inputs"] != signature:
            raise ValueError("Inputs changed; use a new workspace")
        for n, h in old["outputs"].items():
            if digest(folder / n) != h:
                raise ValueError("Damaged cached GR output: " + n)
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(
            "Partial GR output retained; inspect or use a new workspace: " + str(folder)
        )
    folder.mkdir(parents=True, exist_ok=True)
    for stage in ["cts", "gr"]:
        tool(c, r, stage, folder / stage, "export_true_endpoints.tcl")
    a, b = metrics(folder / "cts"), metrics(folder / "gr")
    feature = {
        **{k: r[k] for k in data.KEY},
        "fast_l1_tns": a["api_tns"],
        "fast_l1_wns": a["api_wns"],
        "grt_l2_tns": b["api_tns"],
        "grt_l2_wns": b["api_wns"],
        **endpoint_summary(folder / "gr", b),
    }
    tool(c, r, "gr", folder / "geometry", "export_design.tcl")
    nets, pins, obs, meta = read_design_export(folder / "geometry")
    segments = read_route_segments(r["gr_segments"])
    selected, rejected = select_nets(
        nets,
        pins,
        SelectionConfig(
            max_nets=int(c.get("selection", {}).get("max_nets", 1000)),
            include_clock_nets=bool(c.get("selection", {}).get("include_clocks", True)),
        ),
        meta["dbu_per_micron"],
        routable_nets=set(segments),
    )
    if not selected:
        raise ValueError("No GR-only eligible nets")
    netlist = folder / "selected_nets.txt"
    netlist.write_text("".join(n + "\n" for n in selected))
    timing = folder / "wire"
    timing.mkdir()
    tool(
        c,
        r,
        "gr",
        timing,
        "export_timing.tcl",
        dict(
            NET_LIST_PATH=netlist,
            TIMING_OUTPUT=timing / "timing.tsv",
            LAYER_RC_OUTPUT=timing / "gr_layer_rc.tsv",
        ),
    )
    die = Rectangle(
        *(int(meta[k]) for k in ["die_x_min", "die_y_min", "die_x_max", "die_y_max"])
    )
    g = pd.read_csv(folder / "geometry/gcell_grid.tsv", sep="\t")
    xs = grid_boundaries(
        g[g.axis.eq("x")].coordinate_dbu.astype(int).tolist(), die.x_max
    )
    ys = grid_boundaries(
        g[g.axis.eq("y")].coordinate_dbu.astype(int).tolist(), die.y_max
    )
    src, load, bad = assemble_gr_tables(
        r["design"],
        r["candidate_id"],
        c["openroad"].get("corner", "nominal"),
        selected,
        nets,
        pins,
        segments,
        read_timing_export(timing / "timing.tsv"),
        read_layer_rc(timing / "gr_layer_rc.tsv"),
        read_congestion_cells(r["gr_congestion"], xs, ys),
        obs,
        build_blockage_cells(xs, ys, obs),
        die,
        meta["dbu_per_micron"],
        2.0,
        "driver_minus_sink",
    )
    if not src or not load:
        raise ValueError("No valid GR wire rows")
    if any(k.startswith("dr_") for row in src + load for k in row):
        raise ValueError("DR fields in GR path")
    pd.DataFrame([feature]).to_parquet(folder / "features.parquet", index=False)
    pd.DataFrame(src).to_parquet(folder / "source_sink.parquet", index=False)
    pd.DataFrame(load).to_parquet(folder / "net_load.parquet", index=False)
    save(folder / "rejected.json", rejected + bad)
    if settings_hash(c, r, "gr") != signature:
        raise ValueError("GR inputs changed during extraction")
    save(
        receipt,
        dict(
            status="complete",
            inputs=signature,
            metrics={"cts": a, "gr": b},
            selected_from="GR only; no DR files opened",
            outputs={p.name: digest(p) for p in [*folder.glob("*.parquet"), netlist]},
        ),
    )
    return folder


def collect_label(c, r):
    gr = c["workspace"] / "jobs" / r["design"] / r["candidate_id"] / "gr"
    if not (gr / "complete.json").exists():
        raise ValueError("Collect GR first")
    collect_gr(c, r)  # Recheck cached GR input/output signatures before labeling.
    folder = gr.parent / "dr"
    signature = settings_hash(c, r, "dr")
    receipt = folder / "complete.json"
    if receipt.exists():
        old = json.loads(receipt.read_text())
        if old["inputs"] != signature:
            raise ValueError("DR input signature changed")
        for n, h in old["outputs"].items():
            if digest(folder / n) != h:
                raise ValueError("DR label cache changed")
        return folder
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Partial DR output retained: " + str(folder))
    tool(c, r, "dr", folder / "endpoints", "export_true_endpoints.tcl")
    m = metrics(folder / "endpoints")
    t = folder / "wire"
    t.mkdir()
    tool(
        c,
        r,
        "dr",
        t,
        "export_timing.tcl",
        dict(
            NET_LIST_PATH=gr / "selected_nets.txt",
            TIMING_OUTPUT=t / "timing.tsv",
            LAYER_RC_OUTPUT=t / "dr_layer_rc.tsv",
            DR_SPEF_INPUT=folder / "endpoints/extracted.spef",
        ),
    )
    dr = pd.DataFrame(read_timing_export(t / "timing.tsv"))
    dr = dr[dr.status.eq("ok")].rename(columns={"wire_delay": "dr_wire_delay"})
    src = pd.read_parquet(gr / "source_sink.parquet")
    dr = dr.rename(columns={"sink_pin": "sink_pin", "driver_pin": "driver_pin"})
    names = ["net_name", "driver_pin", "sink_pin", "transition"]
    joined = src[data.PAIR].merge(
        dr[names + ["dr_wire_delay"]], on=names, validate="one_to_one"
    )
    if joined.empty:
        raise ValueError("No connectivity-matched historical wire labels")
    pd.DataFrame(
        [{**{k: r[k] for k in data.KEY}, "final_tns": m["api_tns"]}]
    ).to_parquet(folder / "labels.parquet", index=False)
    joined.to_parquet(folder / "wire_labels.parquet", index=False)
    if settings_hash(c, r, "dr") != signature:
        raise ValueError("DR inputs changed during extraction")
    save(
        receipt,
        dict(
            status="complete",
            inputs=signature,
            metrics=m,
            label_kind="measured_dr_rcx_sta_setup_tns",
            matched_wire_rows=len(joined),
            gr_wire_rows=len(src),
            outputs={p.name: digest(p) for p in folder.glob("*.parquet")},
        ),
    )
    return folder


def assemble(c):
    root = c["workspace"] / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    for stage, names in [
        ("gr", ["features", "source_sink", "net_load"]),
        ("dr", ["labels", "wire_labels"]),
    ]:
        for name in names:
            files = []
            for r in config.candidates(c):
                folder = (
                    c["workspace"] / "jobs" / r["design"] / r["candidate_id"] / stage
                )
                receipt = folder / "complete.json"
                p = folder / (name + ".parquet")
                if receipt.exists():
                    card = json.loads(receipt.read_text())
                    if (
                        name + ".parquet" not in card["outputs"]
                        or digest(p) != card["outputs"][name + ".parquet"]
                    ):
                        raise ValueError("Invalid candidate cache: " + str(p))
                    files.append(p)
            if files:
                pd.concat(
                    [pd.read_parquet(p) for p in files], ignore_index=True
                ).to_parquet(root / (name + ".parquet"), index=False)
    o = c["openroad"]
    context = dict(
        liberty_sha256=[fingerprint(p) for p in o["liberty"]],
        gr_binary_sha256=fingerprint(o["gr_binary"]),
        setrc_sha256=fingerprint(o["setrc"]),
        corner=o.get("corner", "nominal"),
        analysis="setup_max",
        units="ns/kOhm/fF/um",
        selection=c.get("selection", {}),
        path_endpoint_limit=c.get("path_endpoint_limit", 1024),
    )
    data.manifest(
        root,
        dict(
            corner=o.get("corner", "nominal"),
            analysis="setup_max",
            units="ns/kOhm/fF/um",
            feature_context=context,
        ),
    )
    return root


def collect(c, stage="gr"):
    config.doctor(c, need_rcx=stage in ["labels", "all"])
    for r in config.candidates(c):
        if stage in ["gr", "all"]:
            collect_gr(c, r)
        if stage in ["labels", "all"] and r.get("dr_odb") and r["dr_odb"].is_file():
            collect_label(c, r)
    root = assemble(c)
    validation = data.check(root)
    return dict(status="complete", dataset=str(root), validation=validation)
