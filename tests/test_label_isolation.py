import json
import pandas as pd
from paris_tool import data
from paris_tool.demo import make
from paris_tool.algorithms.paris4 import Backend


def test_outer_candidate_and_wire_truth_cannot_change_frozen_predictions(tmp_path):
    root = make(tmp_path / "dataset", 24, 0, True)
    b = Backend()
    b.train(root, tmp_path / "first", jobs=2, trees=4)
    split = json.loads((tmp_path / "first/SPLITS.json").read_text())[0]
    ids = set(split["test"])
    y = pd.read_parquet(root / "labels.parquet")
    y.loc[y.candidate_id.isin(ids), "final_tns"] = -999999.0
    y.to_parquet(root / "labels.parquet", index=False)
    w = pd.read_parquet(root / "wire_labels.parquet")
    w.loc[w.candidate_id.isin(ids), "dr_wire_delay"] = 999.0
    w.to_parquet(root / "wire_labels.parquet", index=False)
    data.manifest(root)
    b.train(root, tmp_path / "second", jobs=2, trees=4)
    a = pd.read_parquet(
        tmp_path / "first/evaluation/demo_design/frozen_predictions.parquet"
    )
    z = pd.read_parquet(
        tmp_path / "second/evaluation/demo_design/frozen_predictions.parquet"
    )
    pd.testing.assert_frame_equal(a, z)
