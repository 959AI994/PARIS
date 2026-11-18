"""Check original seven-design seed-17 score and tie parity before evaluation.

Run `paris predict ... --output ranking.parquet` first. Reference is the original
PARIS4_GUARDED_PREDICTIONS.parquet. Target labels are opened only after parity.
"""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


COUNTS = dict(
    ariane133=104,
    ariane136=60,
    bp=100,
    bp_be=30,
    bp_fe=26,
    bp_multi=103,
    swerv_wrapper=62,
)
KEY = ["design", "candidate_id"]


def main(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    if a.predictions.suffix != ".parquet" or a.reference.suffix != ".parquet":
        raise ValueError("Use Parquet to retain the original float64 ties")
    prediction_hash = hashlib.sha256(a.predictions.read_bytes()).hexdigest()
    frame = pd.read_parquet(a.predictions)
    ref = pd.read_parquet(a.reference)
    ref = ref[ref.seed.eq(17)]
    for t in [frame, ref]:
        if t.duplicated(KEY).any() or t.groupby("design").size().to_dict() != COUNTS:
            raise ValueError(
                "Expected exactly seven designs and 485 unique seed-17 targets"
            )
    joined = frame.merge(ref[KEY + ["paris4_rank"]], on=KEY, validate="one_to_one")
    if len(joined) != 485:
        raise ValueError("Candidate IDs differ")
    rows = []
    for d, g in joined.groupby("design"):
        p = g.rank_score.to_numpy(dtype=np.float64)
        r = g.paris4_rank.to_numpy(dtype=np.float64)
        if not np.isfinite(p).all() or not np.isfinite(r).all():
            raise ValueError("Nonfinite scores")
        i, j = np.triu_indices(len(g), 1)
        rows.append(
            dict(
                design=d,
                n=len(g),
                score_bit_mismatches=int(
                    np.count_nonzero(p.view(np.uint64) != r.view(np.uint64))
                ),
                pair_relation_mismatches=int(
                    np.count_nonzero(np.sign(p[i] - p[j]) != np.sign(r[i] - r[j]))
                ),
            )
        )
    parity = pd.DataFrame(rows)
    parity.to_csv(a.output / "PARITY.csv", index=False)
    if parity[["score_bit_mismatches", "pair_relation_mismatches"]].to_numpy().any():
        raise ValueError("Original score parity failed; inspect PARITY.csv")
    # Only now may evaluation read target ground truth.
    truth = pd.read_parquet(a.labels)[KEY + ["final_tns"]]
    scored = joined.merge(truth, on=KEY, validate="one_to_one")
    if len(scored) != 485 or not np.isfinite(scored.final_tns).all():
        raise ValueError("Label inventory differs")
    metrics = []
    for d, g in scored.groupby("design"):
        metrics.append(
            dict(
                design=d,
                n=len(g),
                original_tau_b=float(kendalltau(g.paris4_rank, -g.final_tns).statistic),
                integrated_tau_b=float(
                    kendalltau(g.rank_score, -g.final_tns).statistic
                ),
            )
        )
    result = pd.DataFrame(metrics)
    result.to_csv(a.output / "PER_DESIGN.csv", index=False)
    card = dict(
        status="passed",
        n=485,
        designs=7,
        score_bitwise_equal=True,
        all_pair_relations_equal=True,
        prediction_sha256=prediction_hash,
        reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),
        macro_original_tau_b=float(result.original_tau_b.mean()),
        macro_integrated_tau_b=float(result.integrated_tau_b.mean()),
        target_labels_read_after_parity=True,
    )
    (a.output / "VALIDATION.json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    for name in ["predictions", "reference", "labels", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    main(p.parse_args())
