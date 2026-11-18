"""UI orchestration, confinement and score preservation; model inference unchanged."""

import json
from pathlib import Path
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import pytest
from paris_tool.ui import App, Handler
from paris_tool.utils import seal


def fixture(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "models/chip").mkdir(parents=True)
    (model / "models/chip/INPUT_SCHEMA.json").write_text(
        json.dumps(
            dict(
                candidate_columns=["design", "candidate_id", "grt_l2_tns"],
                wire_features=["gr_wire_delay", "gr_pi_r", "gr_pi_c1", "gr_pi_c2"],
            )
        )
    )
    (model / "predict.py").write_text("")
    seal(
        model,
        dict(
            name="test-original",
            status="complete",
            algorithm="paris4.original.seed17.1",
            designs=["chip"],
        ),
    )
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    pd.DataFrame(
        dict(
            design=["chip"] * 3,
            candidate_id=["0001", "0002", "0003"],
            grt_l2_tns=[-1.0, -2.0, -3.0],
        )
    ).to_parquet(dataset / "features.parquet", index=False)
    pd.DataFrame(
        dict(
            design=["chip"] * 3,
            candidate_id=["0001", "0002", "0003"],
            net_name=["n"] * 3,
            driver_pin=["a"] * 3,
            sink_pin=["b"] * 3,
            transition=["rise"] * 3,
            corner=["nominal"] * 3,
            gr_wire_delay=[0.1] * 3,
            gr_slack=[-0.1] * 3,
        )
    ).to_parquet(dataset / "source_sink.parquet", index=False)
    pd.DataFrame(
        dict(
            design=["chip"] * 3,
            candidate_id=["0001", "0002", "0003"],
            net_name=["n"] * 3,
            driver_pin=["a"] * 3,
            corner=["nominal"] * 3,
            gr_pi_r=[0.1] * 3,
            gr_pi_c1=[0.1] * 3,
            gr_pi_c2=[0.1] * 3,
        )
    ).to_parquet(dataset / "net_load.parquet", index=False)
    return model, dataset


def test_ui_inspection_rejects_non_original_or_labels(tmp_path):
    model, dataset = fixture(tmp_path)
    a = App(tmp_path / "ui", [tmp_path])
    info = a.inspect(model, dataset)
    assert info["candidates"] == 3
    assert info["algorithm"] == "paris4.original.seed17.1"
    f = pd.read_parquet(dataset / "features.parquet")
    f["final_tns"] = -1.0
    f.to_parquet(dataset / "features.parquet", index=False)
    with pytest.raises(ValueError, match="DR"):
        a.inspect(model, dataset)
    f = f.drop(columns="final_tns")
    f.to_parquet(dataset / "features.parquet", index=False)
    seal(
        model,
        dict(
            name="native",
            status="complete",
            algorithm="paris4.native.1",
            designs=["chip"],
        ),
    )
    with pytest.raises(ValueError, match="完整原版"):
        a.inspect(model, dataset)


def test_ui_path_scope_and_symlinks(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "outside"
    secret.mkdir()
    (allowed / "escape").symlink_to(secret, target_is_directory=True)
    a = App(tmp_path / "ui", [allowed])
    with pytest.raises(ValueError):
        a.path(secret)
    with pytest.raises(ValueError):
        a.path(allowed / "escape")
    assert not a.browse(allowed)["entries"]


def test_explicit_shared_bundle_accepts_new_design_schema(tmp_path):
    model, dataset = fixture(tmp_path)
    for file in dataset.glob("*.parquet"):
        f = pd.read_parquet(file)
        f["design"] = "new_chip"
        f.to_parquet(file, index=False)
    (model / "INPUT_SCHEMA.json").write_text(json.dumps(dict(numeric_columns=["grt_l2_tns"])))
    seal(model, dict(name="test-shared", status="complete", algorithm="paris4.cross-design.1", unknown_design_names_allowed=True))
    app = App(tmp_path / "ui", [tmp_path])
    result = app.inspect(model, dataset)
    assert result["algorithm"] == "paris4.cross-design.1"
    assert result["designs"] == [dict(name="new_chip", n=3)]
    f = pd.read_parquet(dataset / "features.parquet").drop(columns="grt_l2_tns")
    f.to_parquet(dataset / "features.parquet", index=False)
    with pytest.raises(ValueError, match="GR TNS"):
        app.inspect(model, dataset)


def test_prediction_model_relative_path_survives_child_working_directory(tmp_path,monkeypatch):
    from paris_tool.models import predict

    model,dataset=fixture(tmp_path)
    (model/"predict.py").write_text('''import argparse,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--features');p.add_argument('--output');a,_=p.parse_known_args()
f=pd.read_parquet(a.features)[['design','candidate_id']];f['rank_score']=[0.,.5,1.];f['predicted_order']=[1,2,3];f.to_csv(a.output,index=False)
''')
    card=json.loads((model/"MODEL_CARD.json").read_text());seal(model,card)
    monkeypatch.chdir(tmp_path)
    result=predict(Path("model"),Path("dataset"),Path("ranking.csv"))
    assert result["rows"]==3
    assert pd.read_csv(tmp_path/"ranking.csv").rank_score.tolist()==[0.,.5,1.]


def test_ui_import_pointer_cannot_escape_roots(tmp_path):
    model, dataset = fixture(tmp_path)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    pointer = allowed / "imported"
    pointer.mkdir()
    (pointer / "IMPORT.json").write_text(json.dumps(dict(root=str(model))))
    a = App(tmp_path / "ui", [allowed])
    with pytest.raises(ValueError, match="可访问目录"):
        a.inspect(pointer, dataset)


def test_ui_job_uses_cli_and_preserves_scores(tmp_path, monkeypatch):
    model, dataset = fixture(tmp_path)
    a = App(tmp_path / "ui", [tmp_path])
    seen = []

    class Process:
        pid = 12345

        def __init__(self, cmd, **kw):
            seen.extend(cmd)
            out = Path(cmd[cmd.index("--output") + 1])
            scores = np.array([0.5, 0.5000000000000001, 0.5])
            pd.DataFrame(
                dict(
                    design=["chip"] * 3,
                    candidate_id=["0001", "0002", "0003"],
                    rank_score=scores,
                    paris4_cost_rank=scores,
                    predicted_order=[1, 3, 1],
                )
            ).to_parquet(out, index=False)

        def wait(self):
            return 0

    monkeypatch.setattr("paris_tool.ui.subprocess.Popen", Process)
    r = a.start(model, dataset)
    a.worker.join(timeout=5)
    final = a.state(r["id"])
    assert final["status"] == "complete"
    assert seen[1:4] == ["-m", "paris_tool.cli", "predict"]
    rows = a.results(r["id"])["rows"]
    assert rows[0]["candidate_id"] == "0001" and rows[1]["candidate_id"] == "0003"
    assert rows[0]["tie_count"] == 2
    f = pd.read_parquet(Path(final["output"]) / "ranking.parquet")
    assert f.rank_score.iloc[1] == 0.5000000000000001
    assert a.active is None
    assert a.results(r["id"], query="0002")["total"] == 1


def test_ui_http_auth_static_and_origin(tmp_path):
    fixture(tmp_path)
    a = App(tmp_path / "ui", [tmp_path])
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.app = a
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    base = "http://127.0.0.1:" + str(server.server_address[1])
    try:
        html = urlopen(base).read().decode()
        assert a.token in html and "PARIS · 候选质量评估" in html
        assert urlopen(base + "/app.js").status == 200
        picture = urlopen(base + "/method-diagram.png")
        assert picture.headers.get_content_type() == "image/png"
        assert picture.read().startswith(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(HTTPError) as e:
            urlopen(base + "/api/config")
        assert e.value.code == 403
        data = json.loads(
            urlopen(
                Request(base + "/api/config", headers={"X-PARIS-Token": a.token})
            ).read()
        )
        assert data["roots"] == [str(tmp_path)]
        with pytest.raises(HTTPError) as e:
            urlopen(
                Request(
                    base + "/api/config",
                    headers={
                        "X-PARIS-Token": a.token,
                        "Origin": "http://attacker.example",
                    },
                )
            )
        assert e.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        t.join()
