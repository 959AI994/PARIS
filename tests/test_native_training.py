"""Small complete train/export/load test; synthetic, no accuracy assertion."""

import json
import pandas as pd
import pytest
from paris_tool.demo import make
from paris_tool.algorithms.paris4 import Backend
from paris_tool.models import export, import_bundle, predict


def test_train_export_import_predict(tmp_path):
    history = make(tmp_path / "history", 24, 0, True)
    target = make(tmp_path / "target", 6, 100, False)
    model = tmp_path / "model"
    b = Backend()
    card = b.train(history, model, trees=8, jobs=2)
    assert card["algorithm"] == "paris4.native.1"
    b.predict(model, target, tmp_path / "a.csv")
    with pytest.raises(ValueError):
        b.predict(model, history, tmp_path / "leaked.csv")
    archive = tmp_path / "model.tar.gz"
    export(model, archive)
    imported = import_bundle(archive, tmp_path / "imported")
    predict(imported, target, tmp_path / "b.csv")
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "a.csv"), pd.read_csv(tmp_path / "b.csv")
    )
    assert not set(json.loads((model / "SPLITS.json").read_text())[0]["history"]) & set(
        json.loads((model / "SPLITS.json").read_text())[0]["test"]
    )
