"""Native and original frozen PARIS4 bundles behind one prediction interface."""

import json
from pathlib import Path
import shutil
import sys
import tarfile
import os
from .utils import fresh, digest, save, verify, run


def import_bundle(archive, output, expected=None):
    archive = Path(archive)
    if expected and digest(archive) != expected:
        raise ValueError("Archive SHA256 mismatch")
    out = fresh(output)
    with tarfile.open(archive) as t:
        total = 0
        for m in t.getmembers():
            p = (out / m.name).resolve()
            total += m.size
            if (
                not p.is_relative_to(out.resolve())
                or not (m.isfile() or m.isdir())
                or total > 1024**3
            ):
                raise ValueError("Unsafe or oversized model archive member: " + m.name)
        if hasattr(tarfile, "data_filter"):
            t.extractall(out, filter="data")
        else:
            t.extractall(out)
    cards = list(out.rglob("MODEL_CARD.json"))
    if len(cards) != 1:
        raise ValueError("Expected one model card")
    root = cards[0].parent
    card = verify(root)
    if card.get("status") != "complete":
        raise ValueError("Incomplete model")
    if not card.get("backend") and not (root / "models.joblib").exists() and not (root / "predict.py").exists():
        raise ValueError("Unsupported bundle format")
    save(
        out / "IMPORT.json",
        dict(
            root=str(root.relative_to(out)),
            archive_sha256=digest(archive),
            note="Import trusted models only: joblib bundles contain executable Python objects.",
        ),
    )
    return root


def find_root(path):
    p = Path(path).resolve()
    if (p / "IMPORT.json").exists():
        p = p / json.loads((p / "IMPORT.json").read_text())["root"]
    verify(p)
    return p


def predict(model, dataset, output):
    root = find_root(model)
    card = verify(root)
    if card.get("backend"):
        from .algorithms.base import backend

        return backend(card["backend"]).predict(root, dataset, output)
    # Separate interpreter protects the host CLI from the legacy runtime's
    # top-level module names, and preserves the original model's calculations.
    dataset = Path(dataset).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.suffix not in [".csv", ".parquet"]:
        raise ValueError("Original model output must be .csv or .parquet")
    logs = output.parent / (output.stem + ".inference_logs")
    original_output = (
        logs / "original_scores.csv" if output.suffix == ".parquet" else output
    )
    command = [
        sys.executable,
        str(root / "predict.py"),
        "--bundle",
        str(root.resolve()),
        "--features",
        str(dataset / "features.parquet"),
        "--output",
        str(original_output),
    ]
    if (
        "source_sink/net_load" in card.get("input_contract", "")
        or (root / "models").is_dir()
    ):
        command += [
            "--source-sink",
            str(dataset / "source_sink.parquet"),
            "--net-load",
            str(dataset / "net_load.parquet"),
        ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run(command, logs, env=env, timeout=3600, name="predict")
    import pandas as pd

    f = pd.read_csv(
        original_output,
        dtype={"design": str, "candidate_id": str},
        float_precision="round_trip",
    )
    if "rank_score" not in f:
        old = next(
            (x for x in ["paris4_cost_rank", "paris4_zero_shot_cost_rank"] if x in f),
            None,
        )
        if not old:
            raise ValueError("Bundle did not produce an accepted rank column")
        f["rank_score"] = f[old]
    if output.suffix == ".parquet":
        f.to_parquet(output, index=False)
    else:
        f.to_csv(output, index=False)
    return dict(
        status="complete",
        rows=len(f),
        output=str(output),
        backend="original_frozen_bundle",
        algorithm=card.get("algorithm", card.get("name", "original_frozen_bundle")),
    )


def export(model, output):
    root = find_root(model)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(output, "w:gz") as t:
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                t.add(p, arcname="paris-model/" + str(p.relative_to(root)))
    return dict(archive=str(output), sha256=digest(output))
