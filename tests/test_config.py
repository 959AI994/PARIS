import csv
from pathlib import Path
import pytest
import yaml
from paris_tool.config import load, candidates


def test_relative_paths_stage_binaries_and_identifier_zeroes(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    (root / "candidates.csv").write_text(
        "design,candidate_id,gr_odb\nchip,00001,./input with spaces.odb\n"
    )
    (root / "project.yaml").write_text(
        yaml.safe_dump(
            dict(
                version=1,
                candidates="./candidates.csv",
                workspace="./work",
                openroad=dict(binary="./bin/openroad", dr_binary="./bin/other"),
            )
        )
    )
    c = load(root / "project.yaml")
    r = candidates(c)[0]
    assert r["candidate_id"] == "00001"
    assert r["gr_odb"] == root / "input with spaces.odb"
    assert c["openroad"]["gr_binary"] == root / "bin/openroad"
    assert c["openroad"]["dr_binary"] == root / "bin/other"


def test_candidate_path_traversal_rejected(tmp_path):
    (tmp_path / "candidates.csv").write_text("design,candidate_id\nchip,../escape\n")
    with pytest.raises(ValueError):
        candidates(dict(manifest=tmp_path / "candidates.csv"))
