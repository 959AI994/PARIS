import hashlib
import pytest

from paris_tool.design_source import verify_blob, fetch_chipbench
from paris_tool.mol_project import macro_variant, placement_blockage


def test_download_integrity():
    data = b"module example; endmodule\n"
    expected = hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data
    ).hexdigest()
    verify_blob(data, expected)
    with pytest.raises(ValueError, match="pinned"):
        verify_blob(data + b" ", expected)


def test_no_design_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="design name"):
        fetch_chipbench("../other", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_mol_layer_mapping_preserves_geometry_and_timing():
    lef = "MACRO ram\n CLASS BLOCK ;\n SIZE 10 BY 20 ;\n PIN D\n PORT\n LAYER metal3 ;\n RECT 1 2 3 4 ;\n END\n END D\nEND ram\n"
    lib = 'library (ram) { cell (ram) { pin (D) { capacitance : 3.4; } cell_rise(x) { values("1.25"); } } }'
    upper, timing = macro_variant(lef, lib, "upper")
    assert "MACRO ram_upper" in upper and "END ram_upper" in upper
    assert "LAYER metal18" in upper and "SIZE 10 BY 20" in upper
    assert "RECT 1 2 3 4" in upper
    assert 'values("1.25")' in timing and "capacitance : 3.4" in timing
    assert "cell (ram_upper)" in timing and "pin (D)" in timing
    bottom, _ = macro_variant(lef, lib, "bottom")
    assert "LAYER metal3" in bottom and "SIZE 10 BY 20" in bottom
    temporary, _ = macro_variant(lef, lib, "upper", shrink=True)
    assert "SIZE 0.19 BY 1.4" in temporary
    with pytest.raises(ValueError, match="matching"):
        macro_variant(lef, lib.replace("cell (ram)", "cell (other)"), "upper")


def test_placement_rectangle_preserves_macro_pins():
    lef = "MACRO ram\n CLASS BLOCK ;\n SIZE 10 BY 20 ;\n PIN D\n PORT\n LAYER metal3 ;\n RECT 1 2 3 4 ;\n END\n END D\n OBS\n LAYER metal1 ;\n RECT 1 2 8 9 ;\n END\nEND ram\n"
    placed = placement_blockage(lef)
    assert "RECT 1 2 3 4" in placed
    assert "RECT 0 0 10 20" in placed
    assert "RECT 1 2 8 9" not in placed
    assert "SIZE 10 BY 20" in placed
    assert placement_blockage(placed) == placed


def test_track_rounding_repairs_overlap_without_changing_tiers(tmp_path):
    import json
    from paris_tool.mol_flow import legalize_track_rounding

    for name, size in [("A", 2), ("B", 1)]:
        (tmp_path / (name + ".lef")).write_text(
            f"MACRO {name}_bottom\n SIZE {size} BY {size} ;\nEND {name}_bottom\n"
        )
    (tmp_path / "config.mk").write_text(
        "export ADDITIONAL_LEFS = "
        + str(tmp_path / "A.lef")
        + " "
        + str(tmp_path / "B.lef")
        + "\n"
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(dict(assignment=dict(a=0, b=0))))
    source = tmp_path / "source.def"
    source.write_text(
        "UNITS DISTANCE MICRONS 2000 ;\nDIEAREA ( 0 0 ) ( 20000 20000 ) ;\nCOMPONENTS 2 ;\n- a A_bottom\n + FIXED ( 260 210 ) N ;\n- b B_bottom\n + FIXED ( 4060 210 ) N ;\nEND COMPONENTS\n"
    )
    original = source.read_bytes()
    legalize_track_rounding(
        tmp_path, candidate, source, tmp_path / "fixed.def", tmp_path / "audit.json"
    )
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["status"] == "passed" and len(audit["changes"]) == 1
    assert audit["changes"][0]["macro"] == "b"
    assert source.read_bytes() == original
    legalize_track_rounding(
        tmp_path,
        candidate,
        tmp_path / "fixed.def",
        tmp_path / "again.def",
        tmp_path / "again.json",
    )
    assert json.loads((tmp_path / "again.json").read_text())["changes"] == []


def test_local_macro_overlap_wider_than_five_tracks(tmp_path):
    import json
    from paris_tool.mol_flow import legalize_track_rounding

    lef = tmp_path / "macro.lef"
    lef.write_text("MACRO ram_upper\n SIZE 12 BY 40 ;\nEND ram_upper\n")
    (tmp_path / "config.mk").write_text("export ADDITIONAL_LEFS = " + str(lef) + "\n")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(dict(assignment=dict(a=1, b=1))))
    source = tmp_path / "input.def"
    source.write_text(
        "UNITS DISTANCE MICRONS 2000 ;\nDIEAREA ( 0 0 ) ( 300000 300000 ) ;\nCOMPONENTS 2 ;\n- a ram_upper\n + FIXED ( 3800 2800 ) N ;\n- b ram_upper\n + FIXED ( 23800 2800 ) N ;\nEND COMPONENTS\n"
    )
    original = source.read_bytes()
    legalize_track_rounding(
        tmp_path, candidate, source, tmp_path / "fixed.def", tmp_path / "audit.json"
    )
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["status"] == "passed" and not audit["errors"]
    assert len(audit["changes"]) == 1 and audit["changes"][0]["macro"] == "b"
    assert audit["changes"][0]["dx_dbu"] >= 4000
    assert len(audit["macro_boxes"]) == 2 and source.read_bytes() == original
