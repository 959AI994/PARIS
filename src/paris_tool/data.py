from pathlib import Path
import json
import numpy as np
import pandas as pd
from .utils import digest, save

KEY = ["design", "candidate_id"]
WIRE = [
    "hpwl",
    "num_sinks",
    "driver_slew",
    "congestion_mean",
    "congestion_std",
    "transition_code",
    "ss_length",
    "ss_r",
    "ss_c",
    "macro_blockage",
    "macro_blockage_expanded",
    "gr_wire_delay",
    "gr_sink_slew",
    "gr_wire_slew",
    "gr_pi_r",
    "gr_pi_c1",
    "gr_pi_c2",
]
PAIR = KEY + ["net_name", "driver_pin", "sink_pin", "transition", "corner"]
NET = KEY + ["net_name", "driver_pin", "corner"]


def predictions(path):
    """Read rank scores without changing one-ULP differences or exact ties."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={k: str for k in KEY}, float_precision="round_trip")


def keys(f):
    return set(map(tuple, f[KEY].astype(str).to_numpy()))


def features(root):
    f = pd.read_parquet(Path(root) / "features.parquet")
    if f.empty or f.duplicated(KEY).any() or f[KEY].isna().any().any():
        raise ValueError("Invalid candidate keys")
    if not all(isinstance(v, str) for v in f[KEY].to_numpy().ravel()):
        raise ValueError("design/candidate_id must be strings")
    for c in f:
        if c.startswith(("dr_", "final_", "target_", "l3_", "delta_")):
            raise ValueError("Label field in features: " + c)
    for c in ["fast_l1_tns", "grt_l2_tns", "ep_gr_log_cost"]:
        if c not in f or not np.isfinite(f[c]).all():
            raise ValueError("Missing/nonfinite feature: " + c)
    if (f[["fast_l1_tns", "grt_l2_tns"]] > 0).any().any():
        raise ValueError("TNS must be nonpositive ns")
    return f


def source(root):
    s = pd.read_parquet(Path(root) / "source_sink.parquet")
    n = pd.read_parquet(Path(root) / "net_load.parquet")
    for t, key in [(s, PAIR), (n, NET)]:
        if t.duplicated(key).any():
            raise ValueError("Duplicate wire keys")
        if any(c.startswith(("dr_", "final_", "target_")) for c in t):
            raise ValueError("Labels in GR wire inputs")
    s = s.merge(
        n[NET + ["gr_pi_r", "gr_pi_c1", "gr_pi_c2"]], on=NET, validate="many_to_one"
    )
    if not np.isfinite(s[WIRE + ["gr_slack"]]).all().all():
        raise ValueError("Incomplete GR wire features")
    if s.corner.nunique() != 1:
        raise ValueError("Version 0.1 requires one timing corner per dataset")
    nonnegative = [
        "hpwl",
        "num_sinks",
        "driver_slew",
        "ss_length",
        "ss_r",
        "ss_c",
        "gr_wire_delay",
        "gr_sink_slew",
        "gr_pi_r",
        "gr_pi_c1",
        "gr_pi_c2",
    ]
    if (s[nonnegative] < 0).any().any():
        raise ValueError("Negative physical GR delay/slew/R/C/geometry")
    return s


def labels(root):
    y = pd.read_parquet(Path(root) / "labels.parquet")[KEY + ["final_tns"]]
    if (
        y.empty
        or y.duplicated(KEY).any()
        or not np.isfinite(y.final_tns).all()
        or (y.final_tns > 0).any()
    ):
        raise ValueError("Invalid measured TNS labels")
    return y


def wire_labels(root):
    f = pd.read_parquet(Path(root) / "wire_labels.parquet")[PAIR + ["dr_wire_delay"]]
    if (
        f.duplicated(PAIR).any()
        or not np.isfinite(f.dr_wire_delay).all()
        or (f.dr_wire_delay < 0).any()
    ):
        raise ValueError("Invalid wire labels")
    return f


def manifest(root, extra=None):
    root = Path(root)
    save(
        root / "DATASET.json",
        dict(
            schema="paris.dataset.v1",
            feature_profile="paris4-native-v1",
            hashes={p.name: digest(p) for p in root.glob("*.parquet")},
            **(extra or {}),
        ),
    )


def check(root):
    root = Path(root)
    m = json.loads((root / "DATASET.json").read_text())
    for n, h in m["hashes"].items():
        if (root / n).resolve().parent != root.resolve() or digest(root / n) != h:
            raise ValueError("Dataset checksum mismatch: " + n)
    f = features(root)
    s = source(root)
    if keys(f) != keys(s):
        raise ValueError("Wire/candidate inventory mismatch")
    if (root / "labels.parquet").exists():
        y = labels(root)
        if not keys(y) <= keys(f):
            raise ValueError("Orphan labels")
    if (root / "wire_labels.parquet").exists():
        w = wire_labels(root)
        match = w[PAIR].merge(
            s[PAIR], on=PAIR, how="left", indicator=True, validate="one_to_one"
        )
        if not match["_merge"].eq("both").all():
            raise ValueError("Orphan wire labels")
    return dict(
        status="passed",
        candidates=len(f),
        wire_rows=len(s),
        labeled_candidates=len(y) if (root / "labels.parquet").exists() else 0,
    )


def import_tables(feature_file, source_file, load_file, output, label_file=None):
    from .utils import fresh

    out = fresh(output)
    f = pd.read_parquet(feature_file)
    s = pd.read_parquet(source_file)
    n = pd.read_parquet(load_file)
    forbidden = lambda c: c.startswith(("dr_", "final_", "target_", "l3_"))
    if any(forbidden(c) for c in f):
        raise ValueError("Candidate input table must contain GR/CTS features only")
    if "dr_wire_delay" in s:
        s[PAIR + ["dr_wire_delay"]].dropna().to_parquet(
            out / "wire_labels.parquet", index=False
        )
    for name, table in [("features", f), ("source_sink", s), ("net_load", n)]:
        if name != "features":
            table = table[[c for c in table if not forbidden(c)]]
        table.to_parquet(out / (name + ".parquet"), index=False)
    if label_file:
        pd.read_parquet(label_file)[KEY + ["final_tns"]].to_parquet(
            out / "labels.parquet", index=False
        )
    manifest(
        out,
        dict(
            imported_tables=True,
            source_hashes={
                k: digest(v)
                for k, v in [
                    ("features", feature_file),
                    ("source_sink", source_file),
                    ("net_load", load_file),
                ]
                + ([("labels", label_file)] if label_file else [])
            },
        ),
    )
    return check(out)
