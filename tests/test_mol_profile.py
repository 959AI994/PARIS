"""Native flow profile contract and packaged feature engine integrity."""

import hashlib
import json
from pathlib import Path

import pytest

from paris_tool.mol_profile import write_profile


def test_profile_selects_eda_runtimes_and_preserves_existing(tmp_path):
    paths = {}
    for name in (
        "yosys",
        "openroad",
        "place_python",
        "feature_python",
        "place_mol_root",
        "flow_root",
    ):
        p = tmp_path / name
        p.mkdir() if name.endswith("root") else p.touch()
        paths[name] = str(p)
    (tmp_path / "flow_root/platforms/nangate45_3D").mkdir(parents=True)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(paths))
    output = tmp_path / "profile.json"
    write_profile(settings, output)
    profile = json.loads(output.read_text())
    assert profile["stops_at"] == "global_route"
    assert profile["stages"]["partition"]["argv"][0] == paths["place_python"]
    assert profile["stages"]["features"]["argv"][0] == paths["feature_python"]
    assert profile["stages"]["features"]["artifacts"][0] == "dataset/*.parquet"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_profile(settings, output)
    assert output.read_bytes() == before


def test_original_feature_engine_bytes_match_provenance():
    from paris_tool import _features

    root = Path(_features.__file__).parent
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for relative, expected in provenance["files"].items():
        assert (
            hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected
        ), relative


def test_partition_budget_accounts_for_unavoidable_fixed_logic():
    from types import SimpleNamespace
    import pandas as pd
    from paris_tool.mol_profile import partition_area_policy

    balanced = SimpleNamespace(
        cell_area_by_die=[1.0, 0.0], nodes=pd.DataFrame(dict(area=[1.0, 1.0]))
    )
    result = partition_area_policy(balanced)
    assert (
        result["unavoidable_imbalance"] == 0 and result["effective_tolerance"] == 0.25
    )
    logic_dominated = SimpleNamespace(
        cell_area_by_die=[7.0, 0.0], nodes=pd.DataFrame(dict(area=[1.0, 2.0]))
    )
    result = partition_area_policy(logic_dominated)
    assert result["unavoidable_imbalance"] == 0.4
    assert result["effective_tolerance"] == pytest.approx(0.55)
    assert result["movable_macro_area"] == 3
