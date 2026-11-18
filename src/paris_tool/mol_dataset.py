"""Assemble the original candidate, CTS, GR endpoint and wire feature tables."""

import json
from pathlib import Path

import pandas as pd

from .cts_features import features as cts_features, grt_artifacts
from .utils import digest, save


def assemble(partition_features, gr_runs, collection, output):
    base = pd.read_parquet(partition_features)
    mapping = (
        json.loads(Path(gr_runs).read_text())
        if not isinstance(gr_runs, dict)
        else gr_runs
    )
    collection = Path(collection)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows, sources, loads, receipts = [], [], [], []
    if (
        set(base.candidate_id) != set(mapping)
        or base.duplicated(["design", "candidate_id"]).any()
    ):
        raise ValueError("Candidate inventories differ")
    for record in base.to_dict("records"):
        cid, design = record["candidate_id"], record["design"]
        root = Path(mapping[cid])
        flow = json.loads((root / "complete.json").read_text())
        results, logs = Path(flow["results"]), Path(flow["logs"])
        for name, expected in flow["outputs"].items():
            if digest(results / name) != expected:
                raise ValueError("Changed GR checkpoint: " + cid + "/" + name)
        job = collection / "jobs" / design / cid / "gr"
        measured = json.loads((job / "complete.json").read_text())
        for name, expected in measured["outputs"].items():
            if digest(job / name) != expected:
                raise ValueError("Changed collected GR features: " + cid + "/" + name)
        electrical = pd.read_parquet(job / "features.parquet").iloc[0].to_dict()
        if (electrical["design"], electrical["candidate_id"]) != (design, cid):
            raise ValueError("GR feature identity mismatch")
        row = {
            **record,
            **electrical,
            **{
                "cts_" + k: v
                for k, v in cts_features(
                    logs / "4_cts_surrogate_endpoints.rpt",
                    logs / "4_cts_surrogate_detail.json",
                ).items()
            },
            **grt_artifacts(logs / "5_1_grt.log"),
        }
        row["feature_schema_version"] = 2.0
        for prefix, name in (("cts", "4_cts.odb"), ("grt", "5_1_grt.odb")):
            stat = (results / name).stat()
            row[prefix + "_odb_size_bytes"] = stat.st_size
            row[prefix + "_odb_mtime_ns"] = stat.st_mtime_ns
        rows.append(row)
        sources.append(pd.read_parquet(job / "source_sink.parquet"))
        loads.append(pd.read_parquet(job / "net_load.parquet"))
        receipts.append(
            dict(
                design=design,
                candidate_id=cid,
                gr_receipt_sha256=digest(root / "complete.json"),
                collection_receipt_sha256=digest(job / "complete.json"),
            )
        )
    f = (
        pd.DataFrame(rows)
        .sort_values(["design", "candidate_id"])
        .reset_index(drop=True)
    )
    if any(c.startswith(("dr_", "final_", "target_", "l3_")) for c in f):
        raise ValueError("A DR label entered the feature table")
    f.to_parquet(output / "features.parquet", index=False)
    pd.concat(sources, ignore_index=True).to_parquet(
        output / "source_sink.parquet", index=False
    )
    pd.concat(loads, ignore_index=True).to_parquet(
        output / "net_load.parquet", index=False
    )
    save(
        output / "FEATURE_CONTEXT.json",
        dict(
            pdk_id="nangate45_3D",
            flow_id="open3dbench.mol.native.c8e73fe.v1",
            corner="nominal",
            feature_version="paris4-original-candidate.2",
            time_unit="ns",
        ),
    )
    receipt = dict(
        status="complete",
        candidates=len(f),
        columns=len(f.columns),
        source="target GR/CTS only",
        inputs=receipts,
        outputs={p.name: digest(p) for p in output.glob("*.parquet")},
    )
    save(output / "complete.json", receipt)
    return receipt
