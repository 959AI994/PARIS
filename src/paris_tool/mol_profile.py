"""Configured native Nangate45 memory-on-logic preparation recipe."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .utils import digest, save

RES = Path(__file__).parent / "resources"
KEYS = (
    "yosys",
    "openroad",
    "place_python",
    "feature_python",
    "place_mol_root",
    "flow_root",
)


def settings(path):
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text())
    for key in KEYS:
        item = Path(value[key]).expanduser()
        item = item if item.is_absolute() else source.parent / item
        if not item.exists():
            raise FileNotFoundError(item)
        # Preserve venv launcher paths; resolving a Python symlink loses its site-packages.
        value[key] = str(item.absolute() if key in {'place_python', 'feature_python'} else item.resolve(strict=True))
    return value


def write_profile(source, output):
    cfg = settings(source)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    tracked = [str(Path(source).resolve())]
    tracked += [
        str(p)
        for p in Path(__file__).parent.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".tcl", ".yaml", ".json"}
        and "__pycache__" not in p.parts
    ]
    tracked += [
        str(p)
        for p in (Path(cfg["flow_root"]) / "platforms/nangate45_3D").rglob("*")
        if p.is_file()
    ]
    tracked += [str(Path(cfg["flow_root"]) / "Makefile")]
    for folder in [Path(cfg["flow_root"]) / "scripts", Path(cfg["flow_root"]) / "scripts_3D",
                   Path(cfg["place_mol_root"]) / "src", Path(cfg["place_mol_root"]) / "config"]:
        tracked += [str(p) for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]

    artifacts = {
        "preprocess": [
            "preprocess/linked/complete.json",
            "preprocess/linked/STRUCTURE.json",
        ],
        "partition": [
            "partition/mol/MOL_PROJECT.json",
            "partition/partition_features.parquet",
            "partition/candidates/*.json",
        ],
        "physical": ["physical/GR_RUNS.json", "physical/runs/*/complete.json"],
        "features": [
            "dataset/*.parquet",
            "dataset/FEATURE_CONTEXT.json",
            "dataset/complete.json",
        ],
    }
    stages = {}
    for stage in artifacts:
        python = cfg["place_python"] if stage == "partition" else cfg["feature_python"]
        stages[stage] = dict(
            argv=[
                python,
                str(RES / "mol_stage.py"),
                stage,
                "--settings",
                str(Path(source).resolve()),
                "--project",
                "{project}",
                "--output",
                "{output}",
            ],
            artifacts=artifacts[stage],
            tracked_files=sorted(set(tracked)),
            timeout=86400,
        )
    profile = dict(
        version=1,
        name="Nangate45 · 双层 Memory-on-Logic",
        stops_at="global_route",
        stages=stages,
    )
    save(output, profile)
    return dict(profile=str(output), stages=list(stages))


def invoke(argv, output, env=None):
    with Path(output).open("x") as log:
        subprocess.run(
            list(map(str, argv)),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )


def row_config(prepared, gr_run, collection, metadata, binary, cid, design):
    meta = json.loads((prepared / "MOL_PROJECT.json").read_text())
    pdk = Path(meta["flow_root"]) / "platforms/nangate45_3D"
    txt = (prepared / "config.mk").read_text()
    libs = [
        Path(x)
        for x in re.search(r"^export ADDITIONAL_LIBS = (.+)$", txt, re.M)[1].split()
    ]
    rec = json.loads((Path(gr_run) / "complete.json").read_text())
    res = Path(rec["results"])
    c = dict(
        workspace=collection,
        threads=1,
        timeout=7200,
        openroad=dict(
            gr_binary=Path(binary),
            dr_binary=Path(binary),
            liberty=libs,
            setrc=pdk / "setRC.tcl",
            rcx_rules=pdk / "nangate45_3D.rules",
            corner="nominal",
        ),
        selection=dict(max_nets=1000, include_clocks=True),
        path_endpoint_limit=1024,
    )
    row = dict(
        design=design,
        candidate_id=cid,
        cts_odb=res / "4_cts.odb",
        cts_sdc=res / "4_cts.sdc",
        gr_odb=res / "5_1_grt.odb",
        gr_sdc=res / "4_cts.sdc",
        gr_segments=res / "gr_segments.txt",
        gr_congestion=metadata / "congestion.csv",
    )
    return c, row


def partition_area_policy(cache, base_tolerance=0.25):
    """Account for fixed logic before limiting movable macro imbalance.

    Normalize the avoidable range [unavoidable lower bound, 1]. The old
    absolute limit is unchanged whenever a balanced split is reachable.
    Geometric containment/overlap and physical legalization remain separate.
    """
    import math

    fixed = list(map(float, cache.cell_area_by_die))
    movable = float(cache.nodes["area"].sum())
    if (
        len(fixed) != 2
        or not 0 < base_tolerance < 1
        or movable <= 0
        or any(not math.isfinite(a) or a < 0 for a in fixed + [movable])
    ):
        raise ValueError("Invalid fixed/movable area budget")
    total = sum(fixed) + movable
    lower = max(0.0, abs(fixed[0] - fixed[1]) - movable) / total
    return dict(
        base_tolerance=base_tolerance,
        fixed_area_by_die=fixed,
        movable_macro_area=movable,
        unavoidable_imbalance=lower,
        effective_tolerance=lower + (1 - lower) * base_tolerance,
        selection_inputs="2D source geometry only; no DR or target scores",
    )


def stage(name, cfg, project, out):
    project, out = Path(project).resolve(), Path(out).resolve()
    part = out / "partition"
    manifest = json.loads((project / "PARIS_PROJECT.json").read_text())
    design = manifest["design"]
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    os.environ.update(
        {
            k: env[k]
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        }
    )
    if name == "preprocess":
        from .preprocess_2d import preprocess

        return preprocess(
            project, out / "preprocess/linked", cfg["yosys"], cfg["openroad"], 2
        )
    if name == "partition":
        from .mol_project import prepare
        from ._features.integrations.dreamplace import export_open3dbench_design
        from ._features.integrations.prepartition_sta import attach_prepartition_timing
        from ._features.candidates import generate_dopp_candidates
        from ._features.config import load_config
        from ._features.feature_engine import extract_from_paths
        from ._features.timing import load_path_crossings

        prepare(
            project,
            out / "preprocess/linked",
            part / "mol",
            cfg["place_mol_root"],
            cfg["flow_root"],
        )
        sta = part / "prepartition_sta"
        sta.mkdir()
        inputs = manifest["inputs"]

        def paths(key):
            values = inputs[key]
            return [
                project / v for v in ([values] if isinstance(values, str) else values)
            ]

        env.update(
            LEF_FILES=" ".join(map(str, paths("lef"))),
            LIB_FILES=" ".join(map(str, paths("liberty"))),
            DEF_FILE=str(out / "preprocess/linked/floorplan.def"),
            SDC_FILE=str(paths("sdc")[0]),
            OUT_DIR=str(sta),
            PATH_COUNT="2000",
            THREADS="2",
        )
        invoke(
            [cfg["openroad"], "-exit", RES / "dump_prepartition_sta.tcl"],
            sta / "openroad.log",
            env,
        )
        timing = json.loads((sta / "sta_meta.json").read_text())
        if timing["clock_count"] < 1 or timing["clock_period"] <= 0:
            raise ValueError("A constrained clock is required")
        seed = int(manifest.get("partition", {}).get("seed", 17))
        count = int(manifest.get("partition", {}).get("count", 12))
        cache = part / "cache"
        export_open3dbench_design(
            part / "mol/Place-MoL",
            design,
            cache,
            config_file="project.json",
            seed=seed,
            use_cuda=False,
            clock_period=timing["clock_period"],
        )
        report = attach_prepartition_timing(cache, sta, max_paths=512)
        save(part / "PREPARTITION_AUDIT.json", report)
        from ._features.cache import DesignCache

        design_cache = DesignCache.load(cache)
        area_policy = partition_area_policy(design_cache)
        save(part / "AREA_POLICY.json", area_policy)
        candidates = generate_dopp_candidates(
            design_cache,
            part / "candidates",
            seed=seed,
            iterations=20000,
            grid_size=0.01,
            area_tolerance=area_policy["effective_tolerance"],
            max_candidates=count,
        )
        if len(candidates) < 2:
            raise ValueError("At least two distinct feasible candidates are required")
        extract_from_paths(
            cache,
            part / "candidates",
            load_config(RES / "mol_features.yaml"),
            part / "partition_features.parquet",
            timing_paths=load_path_crossings(cache / "timing_paths.json"),
        )
        return dict(candidates=len(candidates))
    mapping_file = out / "physical/GR_RUNS.json"
    if name == "physical":
        from .mol_flow import run_candidate

        candidates = sorted((part / "candidates").glob("*.json"))
        # Candidate folder also contains an index/metadata file in some exporters.
        candidates = [
            p for p in candidates if "assignment" in json.loads(p.read_text())
        ]
        mapping = {}

        def run(path):
            cid = json.loads(path.read_text())["candidate_id"]
            dest = out / "physical/runs" / cid
            run_candidate(
                part / "mol", path, dest, cfg["place_python"], cfg["openroad"]
            )
            return cid, str(dest)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed([pool.submit(run, p) for p in candidates]):
                cid, dest = future.result()
                mapping[cid] = dest
                save(
                    out / "physical/PROGRESS.json",
                    dict(completed=len(mapping), total=len(candidates), unit="候选"),
                )
        save(mapping_file, mapping)
        return dict(candidates=len(mapping))
    if name == "features":
        from .collection import collect_gr
        from .mol_dataset import assemble

        mapping = json.loads(mapping_file.read_text())
        collection = out / "features/collection"
        done = 0

        def collect(cid):
            metadata = out / "features/metadata" / cid
            metadata.mkdir(parents=True, exist_ok=False)
            c, row = row_config(
                part / "mol",
                mapping[cid],
                collection,
                metadata,
                cfg["openroad"],
                cid,
                design,
            )
            export_env = dict(
                env,
                GR_ODB=str(row["gr_odb"]),
                ROUTING_MIN_LAYER="metal2",
                ROUTING_MAX_LAYER="metal20",
                CONGESTION_PATH=str(metadata / "congestion.csv"),
            )
            invoke(
                [cfg["openroad"], "-no_init", "-exit", RES / "export_congestion.tcl"],
                metadata / "export.log",
                export_env,
            )
            save(
                metadata / "complete.json",
                dict(
                    input_sha256=digest(row["gr_odb"]),
                    output_sha256=digest(metadata / "congestion.csv"),
                ),
            )
            collect_gr(c, row)

        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in as_completed(
                [pool.submit(collect, cid) for cid in sorted(mapping)]
            ):
                future.result()
                done += 1
                save(
                    out / "features/PROGRESS.json",
                    dict(completed=done, total=len(mapping), unit="候选"),
                )
        return assemble(
            part / "partition_features.parquet", mapping, collection, out / "dataset"
        )
    raise ValueError("Unknown native stage: " + name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["preprocess", "partition", "physical", "features"])
    p.add_argument("--settings", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            stage(a.stage, settings(a.settings), a.project, a.output), default=str
        )
    )


def run_project(project, configuration, output):
    """Run the same four stages as the GUI, in their configured runtimes."""
    cfg = settings(configuration)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", project=str(Path(project).resolve()), completed=[])
    save(out / "STATUS.json", state)
    try:
        for name in ("preprocess", "partition", "physical", "features"):
            if __import__("shutil").disk_usage(out).free < 60 * 1024**3:
                raise ValueError("Less than 60 GiB free; no new stage started")
            (out / name).mkdir()
            state["phase"] = name
            save(out / "STATUS.json", state)
            python = (
                cfg["place_python"] if name == "partition" else cfg["feature_python"]
            )
            invoke(
                [
                    python,
                    RES / "mol_stage.py",
                    name,
                    "--settings",
                    Path(configuration).resolve(),
                    "--project",
                    Path(project).resolve(),
                    "--output",
                    out,
                ],
                out / (name + ".log"),
            )
            state["completed"].append(name)
        state.update(status="complete", dataset=str(out / "dataset"))
        save(out / "STATUS.json", state)
        return state
    except Exception as exc:
        state.update(status="failed", error=str(exc))
        save(out / "STATUS.json", state)
        raise
