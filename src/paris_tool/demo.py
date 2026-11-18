"""Small, explicitly synthetic fixtures for software acceptance only."""

from pathlib import Path
import numpy as np
import pandas as pd
from . import data
from .utils import fresh, save


def make(root, n, offset, with_labels):
    root = fresh(root)
    rng = np.random.default_rng(17 + offset)
    features = []
    src = []
    nets = []
    truth = []
    wl = []
    for i in range(n):
        cid = f"demo_{i+offset:04d}"
        q = float(rng.uniform(0.5, 2.0))
        gr = -100.0 * q
        final = gr * (0.7 + 0.15 * q)
        f = dict(
            design="demo_design",
            candidate_id=cid,
            fast_l1_tns=gr * 0.9,
            fast_l1_wns=-q * 0.8,
            grt_l2_tns=gr,
            grt_l2_wns=-q,
            ep_gr_log_cost=np.log1p(-gr),
            ep_gr_wns=-q,
            layout_density=float(rng.uniform(0.4, 0.8)),
        )
        for j in range(12):
            f[f"ep_summary_{j:02d}"] = q ** (1 + j % 3) + rng.normal(0, 0.04)
        features.append(f)
        truth.append(dict(design="demo_design", candidate_id=cid, final_tns=final))
        for j in range(3):
            base = dict(
                design="demo_design",
                candidate_id=cid,
                net_name=f"n{j}",
                driver_pin=f"u{j}/Z",
                corner="nominal",
            )
            nets.append(
                dict(**base, gr_pi_r=0.1 * q, gr_pi_c1=0.2 * q, gr_pi_c2=0.3 * q)
            )
            for tr, code in [("rise", 0), ("fall", 1)]:
                p = dict(
                    **base,
                    sink_pin=f"v{j}/A",
                    transition=tr,
                    hpwl=10 * q,
                    num_sinks=1,
                    driver_slew=0.05 * q,
                    congestion_mean=0.3,
                    congestion_std=0.05,
                    transition_code=code,
                    ss_length=11 * q,
                    ss_r=0.1 * q,
                    ss_c=0.2 * q,
                    macro_blockage=0.1,
                    macro_blockage_expanded=0.1,
                    gr_wire_delay=0.2 * q,
                    gr_sink_slew=0.06 * q,
                    gr_wire_slew=-0.01 * q,
                    gr_slack=-q * 0.1,
                )
                src.append(p)
                wl.append(
                    {
                        **{k: p[k] for k in data.PAIR},
                        "dr_wire_delay": 0.2 * q * (0.7 + 0.15 * q),
                    }
                )
    for name, rows in [
        ("features", features),
        ("source_sink", src),
        ("net_load", nets),
    ]:
        pd.DataFrame(rows).to_parquet(root / (name + ".parquet"), index=False)
    if with_labels:
        pd.DataFrame(truth).to_parquet(root / "labels.parquet", index=False)
        pd.DataFrame(wl).to_parquet(root / "wire_labels.parquet", index=False)
    data.manifest(
        root,
        dict(
            synthetic=True,
            purpose="software acceptance only; no physical accuracy claim",
        ),
    )
    return root


def demo(root):
    root = fresh(root)
    make(root / "history", 40, 0, True)
    make(root / "new_candidates", 12, 100, False)
    save(
        root / "README.json",
        dict(
            synthetic=True,
            training_dataset="history",
            prediction_dataset="new_candidates",
            note="No OpenROAD execution. Test software plumbing, not physical accuracy.",
        ),
    )
    return dict(status="complete", output=str(root), synthetic=True)
