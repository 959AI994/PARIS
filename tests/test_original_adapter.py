"""Original-score float/tie preservation through the installed CLI adapter."""

import numpy as np
import pandas as pd
import pytest
from paris_tool.models import predict
from paris_tool.data import predictions
from paris_tool.utils import seal


def test_original_scores_survive_csv_to_parquet_without_rounding(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    script = """import argparse
import pandas as pd
p=argparse.ArgumentParser()
for n in ['bundle','features','output']: p.add_argument('--'+n)
a=p.parse_args()
pd.DataFrame(dict(design=['chip']*4,candidate_id=['0001','0002','0003','0004'],
paris4_cost_rank=[0.5,0.5000000000000001,0.5,0.75],predicted_order=[1,3,1,4])).to_csv(a.output,index=False)
"""
    (model / "predict.py").write_text(script)
    seal(model, dict(status="complete", algorithm="test-original"))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    output = tmp_path / "ranking.parquet"
    result = predict(model, dataset, output)
    f = pd.read_parquet(output)
    expected = np.array([0.5, 0.5000000000000001, 0.5, 0.75])
    assert np.array_equal(
        f.paris4_cost_rank.to_numpy().view(np.uint64), expected.view(np.uint64)
    )
    assert np.array_equal(f.rank_score, expected)
    assert f.candidate_id.tolist() == ["0001", "0002", "0003", "0004"]
    assert f.predicted_order.tolist() == [1, 3, 1, 4]
    assert result["backend"] == "original_frozen_bundle"
    csv = tmp_path / "scores.csv"
    f.to_csv(csv, index=False)
    assert np.array_equal(
        predictions(csv).rank_score.to_numpy().view(np.uint64), expected.view(np.uint64)
    )
    assert np.array_equal(
        predictions(output).rank_score.to_numpy().view(np.uint64),
        expected.view(np.uint64),
    )
    with pytest.raises(FileExistsError):
        predict(model, dataset, output)
