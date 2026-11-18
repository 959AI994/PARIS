import io
import json
import tarfile
from pathlib import Path
import pytest
import pandas as pd
from paris_tool import data
from paris_tool.demo import make
from paris_tool.jobs import reserve
from paris_tool.models import import_bundle
from paris_tool.screen import select, benchmark
from paris_tool.collection import endpoint_summary


def test_dataset_rejects_labels_in_features(tmp_path):
    root = make(tmp_path / "data", 8, 0, True)
    assert data.check(root)["candidates"] == 8
    f = pd.read_parquet(root / "features.parquet")
    f["final_tns"] = 0.0
    f.to_parquet(root / "features.parquet")
    with pytest.raises(ValueError):
        data.features(root)


def test_gr_wire_inputs_do_not_consult_labels(tmp_path):
    root = make(tmp_path / "data", 8, 0, True)
    before = data.source(root)
    y = pd.read_parquet(root / "wire_labels.parquet")
    y.dr_wire_delay = -12345.0
    y.to_parquet(root / "wire_labels.parquet")
    pd.testing.assert_frame_equal(before, data.source(root))


def test_invalid_physical_wire_labels_are_rejected(tmp_path):
    root = make(tmp_path / "data", 8, 0, True)
    y = pd.read_parquet(root / "wire_labels.parquet")
    y.dr_wire_delay = -1.0
    y.to_parquet(root / "wire_labels.parquet")
    with pytest.raises(ValueError):
        data.check(root)


def test_persistent_budget_does_not_refund_failures(tmp_path):
    db = tmp_path / "budget.sqlite"
    reserve(db, "a", 2)
    reserve(db, "b", 2)
    with pytest.raises(RuntimeError):
        reserve(db, "c", 2)
    with pytest.raises(RuntimeError):
        reserve(db, "a", 20)
    assert reserve(db, "c", 3) == 3


def test_budget_reservation_is_atomic(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    db = tmp_path / "budget.sqlite"

    def attempt(key):
        try:
            reserve(db, key, 1)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(2) as pool:
        result = list(pool.map(attempt, ["a", "b"]))
    assert sum(result) == 1


def test_workspace_lock_is_exclusive(tmp_path):
    from paris_tool.utils import project_lock

    with project_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with project_lock(tmp_path):
                pass


def test_imported_tables_separate_wire_labels(tmp_path):
    src = make(tmp_path / "source", 8, 0, True)
    s = pd.read_parquet(src / "source_sink.parquet").merge(
        pd.read_parquet(src / "wire_labels.parquet"), on=data.PAIR
    )
    combined = tmp_path / "combined.parquet"
    s.to_parquet(combined)
    result = data.import_tables(
        src / "features.parquet",
        combined,
        src / "net_load.parquet",
        tmp_path / "imported",
        src / "labels.parquet",
    )
    assert result["labeled_candidates"] == 8
    assert "dr_wire_delay" not in pd.read_parquet(
        tmp_path / "imported/source_sink.parquet"
    )


def test_old_nonproject_directory_is_not_overwritten(tmp_path):
    from paris_tool.config import prepare_workspace

    (tmp_path / "important").write_text("keep")
    with pytest.raises(ValueError):
        prepare_workspace(dict(workspace=tmp_path, _config=tmp_path / "config.yaml"))
    assert (tmp_path / "important").read_text() == "keep"


def test_archive_traversal_is_rejected(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        m = tarfile.TarInfo("../escape")
        m.size = 3
        t.addfile(m, io.BytesIO(b"bad"))
    with pytest.raises(ValueError):
        import_bundle(archive, tmp_path / "model")
    assert not (tmp_path / "escape").exists()


def test_selection_never_reads_true_tns():
    p = pd.DataFrame(
        dict(
            design=["x"] * 8,
            candidate_id=[str(i) for i in range(8)],
            rank_score=list(range(8)),
            gr_tns=list(range(-8, 0)),
        )
    )
    a = select(p, 5)[data.KEY]
    b = select(p.assign(final_tns=list(range(100, 108))), 5)[data.KEY]
    pd.testing.assert_frame_equal(a, b)


def test_budget_benchmark_uses_pure_gr_top_k(tmp_path):
    p = pd.DataFrame(
        dict(
            design=["x"] * 8,
            candidate_id=[str(i) for i in range(8)],
            rank_score=list(range(8)),
            gr_tns=list(range(-8, 0)),
        )
    )
    pred = tmp_path / "pred.csv"
    p.to_csv(pred, index=False)
    y = p[data.KEY].copy()
    y["final_tns"] = p.gr_tns
    y.to_parquet(tmp_path / "truth.parquet")
    benchmark(pred, tmp_path / "truth.parquet", tmp_path / "out.csv", [5])
    result = pd.read_csv(tmp_path / "out.csv")
    assert result[result.method.eq("gr")].regret_ns.iloc[0] == 0
