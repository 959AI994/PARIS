"""Contracts of raw input preparation; these fixtures are NOT accuracy evidence."""

import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
import pandas as pd
from paris_tool.ui import App
from paris_tool.preparation import STAGES, catalog
from test_ui import fixture


def project(tmp):
    root = tmp / "raw"
    root.mkdir()
    for n in ["design.v", "tech.lef", "cells.lib", "design.sdc"]:
        (root / n).write_text("test fixture " + n)
    (root / "PARIS_PROJECT.json").write_text(
        json.dumps(
            dict(
                version=1,
                design="chip",
                top="chip",
                inputs=dict(
                    netlist="design.v",
                    lef=["tech.lef"],
                    liberty=["cells.lib"],
                    sdc="design.sdc",
                ),
                partition=dict(count=3, seed=17),
            )
        )
    )
    return root


def profile(tmp, dataset):
    # Real subprocess, tiny test data. No training or physical claims.
    script = tmp / "test_stage.py"
    script.write_text(
        """import pathlib,sys,shutil
out=pathlib.Path(sys.argv[1]);stage=sys.argv[2]
(out/stage).mkdir(exist_ok=True)
(out/stage/'receipt.txt').write_text(stage)
if stage=='features':
 (out/'dataset').mkdir()
 for p in pathlib.Path(sys.argv[3]).glob('*.parquet'):shutil.copyfile(p,out/'dataset'/p.name)
"""
    )
    stages = {
        k: dict(
            argv=[sys.executable, str(script), "{output}", k, str(dataset)],
            artifacts=[k + "/receipt.txt"],
        )
        for k in list(STAGES)[:-1]
    }
    p = tmp / "profile.json"
    p.write_text(
        json.dumps(
            dict(
                version=1,
                name="unit test recipe",
                stops_at="global_route",
                stages=stages,
            )
        )
    )
    return p


@pytest.fixture(autouse=True)
def disk(monkeypatch):
    monkeypatch.setattr(
        "paris_tool.preparation.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )


def test_no_recipe_is_explicit_blocker_and_does_not_start(tmp_path):
    raw = project(tmp_path)
    a = App(tmp_path / "ui", [tmp_path])
    info = a.preparation.inspect(raw)
    assert all(x["ok"] for x in info["checks"])
    assert not info["ready"] and "流程" in info["blockers"][0]
    with pytest.raises(ValueError, match="流程"):
        a.preparation.start(raw, tmp_path)
    assert not list((tmp_path / "ui/preparations").glob("*/PREPARATION.json"))


def test_safe_single_netlist_upload_requires_real_technology(tmp_path):
    a = App(tmp_path / "ui", [tmp_path])
    content = b"module chip(); endmodule\n"
    r = a.preparation.upload(io.BytesIO(content), len(content), "chip.v")
    assert Path(r["project"]).is_dir()
    assert [c["ok"] for c in r["checks"]] == [True, False, False, False]
    assert not r["ready"]


@pytest.mark.parametrize("name", ["../escape.v", "/escape.v", "x\\escape.v"])
def test_zip_traversal_is_rejected(tmp_path, name):
    a = App(tmp_path / "ui", [tmp_path])
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(name, "bad")
    data = b.getvalue()
    with pytest.raises(ValueError, match="不安全"):
        a.preparation.upload(io.BytesIO(data), len(data), "project.zip")
    assert not (tmp_path / "escape.v").exists()


def test_uploaded_commands_and_symlink_inputs_are_rejected(tmp_path):
    raw = project(tmp_path)
    outside = tmp_path / "secret"
    outside.write_text("secret")
    (raw / "design.v").unlink()
    (raw / "design.v").symlink_to(outside)
    a = App(tmp_path / "ui", [raw])
    assert not a.preparation.inspect(raw)["ready"]
    p = raw / "PARIS_PROJECT.json"
    j = json.loads(p.read_text())
    j["argv"] = ["bad"]
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError, match="服务器配置"):
        a.preparation.inspect(raw)


def test_stages_produce_artifacts_hashes_and_model_gate(tmp_path):
    model, dataset = fixture(tmp_path)
    raw = project(tmp_path)
    a = App(tmp_path / "ui", [tmp_path], preparation_profile=profile(tmp_path, dataset))
    r = a.preparation.start(raw, model)
    a.preparation.worker.join(30)
    r = a.preparation.state(r["id"])
    assert r["status"] == "complete", r.get("error")
    assert list(r["steps"].values()) == ["complete"] * 5
    assert len(r["artifacts"]) == 4
    assert r["model_inspection"]["candidates"] == 3
    assert r["catalog"]["columns"] == 20
    assert r["inspection"]["hashes"]["design.v"]
    assert a.preparation.active is None
    assert Path(r["dataset"]).is_relative_to(tmp_path / "ui/preparations")


def test_full_features_are_never_filled_or_replaced(tmp_path):
    model, dataset = fixture(tmp_path)
    frame = pd.read_parquet(dataset / "source_sink.parquet").drop(
        columns="gr_wire_delay"
    )
    frame.to_parquet(dataset / "source_sink.parquet", index=False)
    raw = project(tmp_path)
    a = App(tmp_path / "ui", [tmp_path], preparation_profile=profile(tmp_path, dataset))
    r = a.preparation.start(raw, model)
    a.preparation.worker.join(30)
    r = a.preparation.state(r["id"])
    assert r["status"] == "blocked" and "gr_wire_delay" in r["error"]
    assert r["steps"]["features"] == "complete" and r["steps"]["validate"] == "blocked"
    assert "gr_wire_delay" not in pd.read_parquet(
        Path(r["dataset"]) / "source_sink.parquet"
    )


def test_job_failure_retains_partial_outputs(tmp_path):
    model, dataset = fixture(tmp_path)
    raw = project(tmp_path)
    p = profile(tmp_path, dataset)
    j = json.loads(p.read_text())
    j["stages"]["partition"]["argv"] = [
        sys.executable,
        "-c",
        "raise RuntimeError('probe failure')",
    ]
    p.write_text(json.dumps(j))
    a = App(tmp_path / "ui", [tmp_path], preparation_profile=p)
    r = a.preparation.start(raw, model)
    a.preparation.worker.join(30)
    r = a.preparation.state(r["id"])
    assert r["status"] == "failed" and r["steps"]["partition"] == "failed"
    assert (Path(r["output"]) / "artifacts/preprocess/receipt.txt").exists()
    assert "probe failure" in a.preparation.log(r["id"])


def test_catalog_reads_actual_schema_and_values_even_if_partial(tmp_path):
    _, dataset = fixture(tmp_path)
    (dataset / "net_load.parquet").unlink()
    r = catalog(dataset)
    assert r["tables"][2]["status"] == "missing"
    f = next(
        f
        for t in r["tables"]
        for f in t.get("fields", [])
        if f["name"] == "gr_wire_delay"
    )
    assert f["sample"] == ["0.1"] * 3
    assert f["family"] == "GR 线网 delay / slew / slack"


def test_changed_recipe_and_low_disk_prevent_execution(tmp_path, monkeypatch):
    model, dataset = fixture(tmp_path)
    raw = project(tmp_path)
    p = profile(tmp_path, dataset)
    a = App(tmp_path / "ui", [tmp_path], preparation_profile=p)
    p.write_text(p.read_text() + "\n")
    assert not a.preparation.inspect(raw)["ready"]
    with pytest.raises(ValueError, match="配置已变化"):
        a.preparation.start(raw, model)
    monkeypatch.setattr(
        "paris_tool.preparation.shutil.disk_usage",
        lambda _: SimpleNamespace(free=59 * 1024**3),
    )
    with pytest.raises(ValueError, match="60 GiB"):
        a.preparation.upload(io.BytesIO(b"x"), 1, "x.v")


def test_concurrent_inference_prevents_preparation(tmp_path):
    model, dataset = fixture(tmp_path)
    raw = project(tmp_path)
    a = App(tmp_path / "ui", [tmp_path], preparation_profile=profile(tmp_path, dataset))
    a.active = "existing-inference"
    with pytest.raises(ValueError, match="正在运行"):
        a.preparation.start(raw, model)


def test_preparation_http_contract_and_upload_auth(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from urllib.parse import urlencode
    from paris_tool.ui import Handler

    model, dataset = fixture(tmp_path)
    a = App(tmp_path / "ui", [tmp_path])
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.app = a
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_address[1])
    headers = {"X-PARIS-Token": a.token}
    try:
        with pytest.raises(HTTPError) as e:
            urlopen(
                Request(
                    base + "/api/projects/upload?filename=chip.v",
                    data=b"module chip;endmodule",
                )
            )
        assert e.value.code == 403
        r = json.loads(
            urlopen(
                Request(
                    base + "/api/projects/upload?filename=chip.v",
                    data=b"module chip;endmodule",
                    headers=headers,
                )
            ).read()
        )
        assert not r["ready"] and r["checks"][0]["ok"]
        r = json.loads(
            urlopen(
                Request(
                    base + "/api/features?" + urlencode(dict(dataset=str(dataset))),
                    headers=headers,
                )
            ).read()
        )
        assert r["columns"] == 20 and r["tables"][0]["rows"] == 3
        with pytest.raises(HTTPError) as e:
            urlopen(
                Request(
                    base + "/api/projects/upload?filename=broken.zip",
                    data=b"corrupt",
                    headers=headers,
                )
            )
        assert e.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_open3dbench_def_connectivity_can_replace_verilog(tmp_path):
    raw = project(tmp_path)
    p = raw / "PARIS_PROJECT.json"
    j = json.loads(p.read_text())
    j["inputs"].pop("netlist")
    j["inputs"]["def"] = "layout.def"
    (raw / "layout.def").write_text("VERSION 5.8 ;\nDESIGN chip ;\nEND DESIGN")
    p.write_text(json.dumps(j))
    a = App(tmp_path / "ui", [tmp_path])
    info = a.preparation.inspect(raw)
    assert all(c["ok"] for c in info["checks"])
    assert "DEF" in info["checks"][0]["message"]
    assert len(info["blockers"]) == 1  # only the explicitly unconfigured flow
