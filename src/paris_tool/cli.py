"""Small, composable PARIS command-line interface."""

import argparse
import json
import os
from pathlib import Path
import sys
from . import __version__


def parser():
    p = argparse.ArgumentParser(
        prog="paris", description="PARIS candidate ranking and dataset automation"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("design", help="Run raw-design preparation, optional calibration and prediction")
    x.add_argument("--project", type=Path, required=True)
    x.add_argument("--settings", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--candidates", type=int, default=36)
    x.add_argument("--seed", type=int, default=17)
    x.add_argument("--label-budget", type=int, default=0)
    x.add_argument("--model", type=Path)
    x.add_argument("--route-iterations", type=int, default=1)
    x = sub.add_parser("project", help="Import a reproducible raw 2D design project")
    x.add_argument(
        "action",
        choices=[
            "fetch",
            "preprocess",
            "prepare-3d",
            "run-candidate",
            "profile",
            "run",
            "route",
        ],
    )
    x.add_argument("--settings", type=Path)
    x.add_argument("--source", choices=["chipbench"], default="chipbench")
    x.add_argument("--design")
    x.add_argument("--project", type=Path)
    x.add_argument("--yosys", default="yosys")
    x.add_argument("--openroad", default="openroad")
    x.add_argument("--threads", type=int, default=2)
    x.add_argument("--preprocessed", type=Path)
    x.add_argument("--place-mol-root", type=Path)
    x.add_argument("--flow-root", type=Path)
    x.add_argument("--gr-run", type=Path)
    x.add_argument("--route-iterations", type=int, default=1)
    x.add_argument("--prepared", type=Path)
    x.add_argument("--partition-file", type=Path)
    x.add_argument("--python", default=sys.executable)
    x.add_argument("--revision", default="main")
    x.add_argument("--output", type=Path, required=True)
    x = sub.add_parser(
        "ui",
        help="Open the PARIS web workspace",
    )
    x.add_argument("--host", default="127.0.0.1")
    x.add_argument("--port", type=int, default=8765)
    x.add_argument("--workspace", type=Path, default=Path("./paris-ui-work"))
    x.add_argument(
        "--root",
        type=Path,
        action="append",
        help="Filesystem root visible in the interface; repeat for more than one",
    )
    x.add_argument("--model", type=Path)
    x.add_argument("--dataset", type=Path)
    x.add_argument(
        "--preparation-profile",
        type=Path,
        help="Trusted server-side raw design to GR feature recipe",
    )
    x.add_argument("--open", action="store_true", dest="open_browser")
    x = sub.add_parser("init", help="Write editable project configuration")
    x.add_argument("--directory", type=Path, required=True)
    x = sub.add_parser("doctor", help="Check configured tools/files")
    x.add_argument("--config", required=True)
    x.add_argument("--labels", action="store_true")
    x = sub.add_parser(
        "collect", help="Extract GR-only features or ingest existing measured DR labels"
    )
    x.add_argument("--config", required=True)
    x.add_argument("--stage", choices=["gr", "labels", "all"], default="gr")
    x = sub.add_parser(
        "label", help="Acquire measured labels, respecting the persistent new-DR budget"
    )
    x.add_argument("--config", required=True)
    x.add_argument("--budget", type=int, required=True)
    x.add_argument(
        "--selection", required=True, help="CSV containing design,candidate_id"
    )
    x = sub.add_parser("dataset", help="Validate/import a portable dataset")
    x.add_argument("action", choices=["validate", "import"])
    x.add_argument("--dataset", type=Path)
    for name in ["features", "source-sink", "net-load", "labels", "output"]:
        x.add_argument("--" + name, type=Path)
    x = sub.add_parser(
        "train",
        help="Train one design-specific PARIS model",
    )
    x.add_argument("--dataset", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--algorithm", default="paris-single-xgb")
    x.add_argument("--jobs", type=int, default=2)
    x.add_argument("--seed", type=int, default=17)
    x.add_argument("--test-fraction", type=float, default=0.2)
    x.add_argument("--wire-trees", type=int, default=900)
    x = sub.add_parser(
        "predict", help="Predict candidate TNS and ranking"
    )
    x.add_argument("--dataset", type=Path, required=True)
    x.add_argument("--model", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x = sub.add_parser("screen", help="Select candidates from frozen predictions")
    x.add_argument("--predictions", required=True)
    x.add_argument("--budget", type=int, required=True)
    x.add_argument("--output", type=Path, required=True)
    x = sub.add_parser(
        "benchmark", help="Offline budget replay on frozen holdout predictions"
    )
    x.add_argument("--predictions", required=True)
    x.add_argument("--truth", required=True)
    x.add_argument("--output", required=True)
    x.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    x = sub.add_parser("model", help="Import/export model bundles")
    x.add_argument("action", choices=["import", "export"])
    x.add_argument("--input", type=Path, required=True)
    x.add_argument("--output", type=Path, required=True)
    x.add_argument("--sha256")
    x = sub.add_parser("demo", help="Generate synthetic software-test datasets")
    x.add_argument("--output", type=Path, required=True)
    x = sub.add_parser(
        "run", help="Collect, optionally train, predict and validate within a DR budget"
    )
    x.add_argument("--config", required=True)
    x.add_argument("--budget", type=int, required=True)
    x.add_argument("--model", type=Path)
    x.add_argument("--train", action="store_true")
    return p


def execute(a):
    if a.command == "design":
        from .calibration import run
        return run(a.project, a.settings, a.output, a.candidates, a.seed,
                   a.label_budget, a.model, a.route_iterations)
    if a.command == "project":
        if a.action == "run":
            if not a.project or not a.settings:
                raise ValueError("run requires --project --settings")
            from .mol_profile import run_project

            return run_project(a.project, a.settings, a.output)
        if a.action == "route":
            if not a.gr_run:
                raise ValueError("route requires --gr-run")
            from .mol_flow import finish_candidate

            return finish_candidate(
                a.gr_run, a.output, route_iterations=a.route_iterations
            )
        if a.action == "profile":
            if not a.settings:
                raise ValueError("profile requires --settings")
            from .mol_profile import write_profile

            return write_profile(a.settings, a.output)
        if a.action == "run-candidate":
            if not a.prepared or not a.partition_file:
                raise ValueError("run-candidate requires --prepared --partition-file")
            from .mol_flow import run_candidate

            return run_candidate(
                a.prepared, a.partition_file, a.output, a.python, a.openroad
            )
        if a.action == "prepare-3d":
            if not all([a.project, a.preprocessed, a.place_mol_root, a.flow_root]):
                raise ValueError(
                    "prepare-3d requires --project --preprocessed --place-mol-root --flow-root"
                )
            from .mol_project import prepare

            return prepare(
                a.project, a.preprocessed, a.output, a.place_mol_root, a.flow_root
            )
        if a.action == "preprocess":
            if not a.project:
                raise ValueError("--project is required")
            from .preprocess_2d import preprocess

            return preprocess(a.project, a.output, a.yosys, a.openroad, a.threads)
        if not a.design:
            raise ValueError("--design is required")
        from .design_source import fetch_chipbench

        return fetch_chipbench(a.design, a.output, a.revision)
    if a.command == "ui":
        from .ui import serve

        return serve(
            a.workspace,
            a.root or [Path.cwd()],
            a.model,
            a.dataset,
            a.host,
            a.port,
            a.open_browser,
            a.preparation_profile,
        )
    if a.command == "init":
        from importlib.resources import files
        from .utils import fresh

        root = fresh(a.directory)
        for name in ["project.yaml", "candidates.csv"]:
            (root / name).write_text(
                files("paris_tool").joinpath("resources", name).read_text()
            )
        return dict(
            status="complete",
            project=str(root),
            next="Edit project.yaml/candidates.csv, then paris doctor --config project.yaml",
        )
    if a.command in ["doctor", "collect", "label", "run"]:
        from . import config

        c = config.load(a.config)
        if a.command != "doctor":
            config.prepare_workspace(c)
    if a.command == "doctor":
        return config.doctor(c, a.labels)
    if a.command == "collect":
        from .collection import collect

        return collect(c, a.stage)
    if a.command == "label":
        import pandas as pd
        from .jobs import label
        from . import data

        p = pd.read_csv(a.selection, dtype={k: str for k in data.KEY})
        wanted = data.keys(p)
        mapping = {tuple(r[k] for k in data.KEY): r for r in config.candidates(c)}
        if not wanted <= set(mapping):
            raise ValueError("Selection contains unknown candidates")
        return label(c, [mapping[k] for k in sorted(wanted)], a.budget)
    if a.command == "dataset":
        from .data import check, import_tables

        if a.action == "validate":
            if not a.dataset:
                raise ValueError("--dataset is required")
            return check(a.dataset)
        if not all([a.features, a.source_sink, a.net_load, a.output]):
            raise ValueError(
                "Import requires --features --source-sink --net-load --output"
            )
        return import_tables(a.features, a.source_sink, a.net_load, a.output, a.labels)
    if a.command == "train":
        if a.jobs < 1 or a.wire_trees < 1:
            raise ValueError("jobs/wire-trees must be positive")
        from .algorithms.base import backend

        return backend(a.algorithm).train(
            a.dataset,
            a.output,
            seed=a.seed,
            test_fraction=a.test_fraction,
            jobs=a.jobs,
            trees=a.wire_trees,
        )
    if a.command == "predict":
        from .models import predict

        return predict(a.model, a.dataset, a.output)
    if a.command == "screen":
        from .data import predictions
        from .screen import select

        if a.output.exists():
            raise FileExistsError(a.output)
        f = select(
            predictions(a.predictions),
            a.budget,
        )
        f.to_csv(a.output, index=False)
        return dict(status="complete", selected=len(f), output=str(a.output))
    if a.command == "benchmark":
        from .screen import benchmark

        return benchmark(a.predictions, a.truth, a.output, a.budgets)
    if a.command == "model":
        from .models import import_bundle, export

        return (
            dict(root=str(import_bundle(a.input, a.output, a.sha256)))
            if a.action == "import"
            else export(a.input, a.output)
        )
    if a.command == "demo":
        from .demo import demo

        return demo(a.output)
    if a.command == "run":
        from .workflow import run_workflow

        return run_workflow(c, a.budget, a.model, a.train)
    raise ValueError("Unknown command")


def main(argv=None):
    a = parser().parse_args(argv)
    for k in [
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[k] = str(getattr(a, "jobs", 2))
    try:
        if a.command in ["collect", "label", "run"]:
            from .config import load, prepare_workspace
            from .utils import project_lock

            c = load(a.config)
            prepare_workspace(c)
            with project_lock(c["workspace"]):
                result = execute(a)
        else:
            result = execute(a)
    except (Exception, KeyboardInterrupt) as e:
        print(
            json.dumps(dict(status="failed", error=str(e)), ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
