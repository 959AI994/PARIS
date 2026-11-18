"""Iteration protocol tests use fixtures, never model accuracy evidence."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from paris_tool.ui import App
from paris_tool.iteration import promotion_gate, metric
from paris_tool.utils import seal, digest


def setup(tmp_path, n=36):
    model = tmp_path / "reference"
    (model / "models/chip").mkdir(parents=True)
    (model / "runtime/experiment/v3").mkdir(parents=True)
    (model / "runtime/experiment/v3/models.py").write_text("# test-only placeholder")
    (model / "models/chip/INPUT_SCHEMA.json").write_text(
        json.dumps(
            dict(
                candidate_columns=["design", "candidate_id", "grt_l2_tns"],
                wire_features=["gr_wire_delay", "gr_pi_r", "gr_pi_c1", "gr_pi_c2"],
            )
        )
    )
    seal(
        model,
        dict(
            name="test reference",
            status="complete",
            algorithm="paris4.original.seed17.1",
            designs=["chip"],
        ),
    )
    root = tmp_path / "samples"
    root.mkdir()
    dataset(root, n)
    a = App(tmp_path / "ui", [tmp_path])
    return a, model, root


def dataset(root, n):
    ids = [f"c{i:03}" for i in range(n)]
    pd.DataFrame(
        dict(
            design=["chip"] * n,
            candidate_id=ids,
            grt_l2_tns=-np.arange(1, n + 1, dtype=float),
        )
    ).to_parquet(root / "features.parquet", index=False)
    source = pd.DataFrame(
        dict(
            design=["chip"] * n,
            candidate_id=ids,
            net_name=["n"] * n,
            driver_pin=["a"] * n,
            sink_pin=["b"] * n,
            transition=["rise"] * n,
            corner=["nominal"] * n,
            gr_wire_delay=[0.1] * n,
        )
    )
    source.to_parquet(root / "source_sink.parquet", index=False)
    source[["design", "candidate_id", "net_name", "driver_pin", "corner"]].assign(
        gr_pi_r=0.1, gr_pi_c1=0.2, gr_pi_c2=0.3
    ).to_parquet(root / "net_load.parquet", index=False)
    source[
        [
            "design",
            "candidate_id",
            "net_name",
            "driver_pin",
            "sink_pin",
            "transition",
            "corner",
        ]
    ].assign(dr_wire_delay=0.15).to_parquet(root / "wire_labels.parquet", index=False)
    pd.DataFrame(
        dict(
            design=["chip"] * n,
            candidate_id=ids,
            final_tns=-np.arange(1, n + 1, dtype=float),
        )
    ).to_parquet(root / "labels.parquet", index=False)
    (root / "PARIS_TRAINING.json").write_text(
        json.dumps(
            dict(
                version=1,
                pdk_id="test-pdk",
                flow_id="test-flow",
                corner="nominal",
                label_version="v1",
                feature_version="full-v1",
                time_unit="ns",
                label_source="measured_dr_sta",
            )
        )
    )


def fake_worker(monkeypatch):
    def execute(command, cwd, env, timeout, name):
        out = Path(cwd)
        m = json.loads((out / "TRAIN.json").read_text())
        if name == "fit":
            y = pd.read_parquet(m["training_labels"])
            w = pd.read_parquet(m["training_wire_labels"])
            assert not set(y.candidate_id) & set(m["validation_ids"])
            assert not set(w.candidate_id) & set(m["validation_ids"])
            model = out / "model"
            model.mkdir()
            (model / "predict.py").write_text("# fixture")
            seal(
                model,
                dict(
                    status="complete",
                    algorithm="paris4.original.design-refit.1",
                    designs=[m["design"]],
                ),
            )
        elif name == "predict":
            ids = m["validation_ids"]
            pd.DataFrame(
                dict(
                    design=[m["design"]] * len(ids),
                    candidate_id=ids,
                    paris4_cost_rank=[int(i[1:]) for i in ids],
                )
            ).to_parquet(out / "validation_predictions.parquet", index=False)
        elif name == "baseline":
            bundle = Path(command[command.index("--bundle") + 1])
            if (
                json.loads((bundle / "MODEL_CARD.json").read_text())["algorithm"]
                == "paris4.original.design-refit.1"
            ):
                ids = m["validation_ids"]
                pd.DataFrame(
                    dict(
                        design=[m["design"]] * len(ids),
                        candidate_id=ids,
                        paris4_cost_rank=[int(i[1:]) for i in ids],
                    )
                ).to_parquet(out / "baseline_predictions.parquet", index=False)
            else:
                (out / "baseline_predictions.skip.json").write_text(
                    json.dumps(dict(reason="baseline trained on validation candidates"))
                )

    monkeypatch.setattr("paris_tool.iteration.run", execute)
    monkeypatch.setattr(
        "paris_tool.iteration.shutil.disk_usage",
        lambda _: SimpleNamespace(free=100 * 1024**3),
    )


def train(a, model, root):
    r = a.iteration.start(root, model)
    a.iteration.worker.join(10)
    r = a.iteration.state(r["id"])
    assert r["status"] == "complete", r.get("error")
    return r


def test_fixed_holdout_excludes_future_training_and_keeps_previous_versions(
    tmp_path, monkeypatch
):
    a, model, root = setup(tmp_path)
    fake_worker(monkeypatch)
    first = train(a, model, root)
    assert first["eligible"] and first["inspection"]["training"] == 24
    old_hash = digest(Path(first["model"]) / "SHA256.json")
    one = a.iteration.activate(first["id"])
    dataset(root, 40)
    info = a.iteration.inspect(root, model)
    assert info["validation_ids"] == first["inspection"]["validation_ids"]
    assert info["new_training"] == 4 and info["training"] == 28
    second = train(a, model, root)
    assert (
        a.iteration.config()["lines"][0]["current"]["id"] == first["id"]
    )  # no auto-promotion
    a.iteration.activate(second["id"])
    a.iteration.activate(first["id"])  # explicit rollback
    assert a.iteration.config()["lines"][0]["current"]["id"] == first["id"]
    assert digest(Path(first["model"]) / "SHA256.json") == old_hash
    assert (
        len(
            list(
                (a.iteration.root / "lines" / info["line"] / "activations").glob(
                    "*.json"
                )
            )
        )
        == 3
    )


def test_labels_cannot_silently_change_inside_same_validation_protocol(
    tmp_path, monkeypatch
):
    a, model, root = setup(tmp_path)
    fake_worker(monkeypatch)
    r = train(a, model, root)
    y = pd.read_parquet(root / "labels.parquet")
    y.loc[0, "final_tns"] = -900.0
    y.to_parquet(root / "labels.parquet", index=False)
    info = a.iteration.inspect(root, model)
    assert not info["ready"] and any("label_version" in s for s in info["blockers"])
    c = json.loads((root / "PARIS_TRAINING.json").read_text())
    c["label_version"] = "v2"
    (root / "PARIS_TRAINING.json").write_text(json.dumps(c))
    assert a.iteration.inspect(root, model)["line"] != r["inspection"]["line"]


def test_validation_gr_cannot_drift_between_versions(tmp_path, monkeypatch):
    a, model, root = setup(tmp_path)
    fake_worker(monkeypatch)
    r = train(a, model, root)
    f = pd.read_parquet(root / "features.parquet")
    f.loc[f.candidate_id.eq(r["inspection"]["validation_ids"][0]), "grt_l2_tns"] = (
        -999.0
    )
    f.to_parquet(root / "features.parquet", index=False)
    r = a.iteration.start(root, model)
    a.iteration.worker.join(10)
    r = a.iteration.state(r["id"])
    assert r["status"] == "failed" and "GR输入发生变化" in r["error"]


def test_insufficient_samples_and_pseudo_labels_are_blocked(tmp_path):
    a, model, root = setup(tmp_path, 20)
    assert not a.iteration.inspect(root, model)["ready"]
    c = json.loads((root / "PARIS_TRAINING.json").read_text())
    c["label_source"] = "predicted"
    (root / "PARIS_TRAINING.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="伪标签"):
        a.iteration.inspect(root, model)


def test_gate_retains_worse_or_degenerate_models():
    baseline = dict(n=12, tau_b=0.8, top1_regret_ns=0.0)
    assert not promotion_gate(dict(n=12, tau_b=0.7, top1_regret_ns=0.0), [baseline])
    assert not promotion_gate(dict(n=12, tau_b=0.9, top1_regret_ns=1.0), [baseline])
    assert not promotion_gate(dict(n=12, tau_b=None, top1_regret_ns=0.0), [baseline])
    assert promotion_gate(baseline, [baseline])
    m = metric(np.array([0.0, 0.0, 1.0]), np.array([-10.0, -20.0, -30.0]))
    assert m["top1_ties"] == 2 and m["top1_regret_ns"] == 5.0


def test_smoke_models_are_not_deployable_even_with_perfect_fixture_scores(
    tmp_path, monkeypatch
):
    a, model, root = setup(tmp_path)
    fake_worker(monkeypatch)
    p = root / "PARIS_TRAINING.json"
    c = json.loads(p.read_text())
    c["purpose"] = "engineering_smoke"
    p.write_text(json.dumps(c))
    r = train(a, model, root)
    assert not r["eligible"] and r["metrics"]["candidate"]["tau_b"] == pytest.approx(
        1.0
    )
    with pytest.raises(ValueError, match="验证"):
        a.iteration.activate(r["id"])


def test_no_overlapping_tasks_and_missing_cumulative_samples(tmp_path, monkeypatch):
    a, model, root = setup(tmp_path)
    fake_worker(monkeypatch)
    a.active = "inference"
    with pytest.raises(ValueError, match="任务"):
        a.iteration.start(root, model)
    a.active = None
    r = train(a, model, root)
    dataset(root, 35)
    assert not a.iteration.inspect(root, model)["ready"]


def test_broken_model_receipts_cannot_be_promoted(tmp_path, monkeypatch):
    a, model, root = setup(tmp_path)
    fake_worker(monkeypatch)
    r = train(a, model, root)
    p = Path(r["output"]) / "VALIDATION.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="回执"):
        a.iteration.activate(r["id"])


def test_malformed_training_context_is_a_readable_error(tmp_path):
    a, model, root = setup(tmp_path)
    (root / "PARIS_TRAINING.json").write_text("[]")
    with pytest.raises(ValueError, match="JSON对象"):
        a.iteration.inspect(root, model)


def test_adapted_prediction_requires_matching_physical_context(tmp_path):
    from test_ui import fixture

    model, dataset = fixture(tmp_path)
    expected = dict(
        pdk_id="pdk",
        flow_id="flow",
        corner="nominal",
        feature_version="full-v1",
        time_unit="ns",
    )
    seal(
        model,
        dict(
            name="adapted-test",
            status="complete",
            algorithm="paris4.original.design-refit.1",
            designs=["chip"],
            context=expected,
        ),
    )
    a = App(tmp_path / "ui", [tmp_path])
    with pytest.raises(ValueError, match="FEATURE_CONTEXT"):
        a.inspect(model, dataset)
    p = dataset / "FEATURE_CONTEXT.json"
    p.write_text(json.dumps(dict(expected, flow_id="different-flow")))
    with pytest.raises(ValueError, match="flow_id"):
        a.inspect(model, dataset)
    p.write_text(json.dumps(expected))
    assert a.inspect(model, dataset)["algorithm"] == "paris4.original.design-refit.1"


def test_new_design_uses_reference_recipe_without_requiring_old_design_weights(
    tmp_path,
):
    a, model, root = setup(tmp_path)
    for name in ["features", "source_sink", "net_load", "labels", "wire_labels"]:
        p = root / (name + ".parquet")
        f = pd.read_parquet(p)
        f["design"] = "new_3d_design"
        f.to_parquet(p, index=False)
    info = a.iteration.inspect(root, model)
    assert info["design"] == "new_3d_design" and info["template"] == "chip"
    assert info["ready"] and info["training"] == 24 and info["validation"] == 12


def test_positive_gr_tns_is_not_silently_clipped_during_adaptation(tmp_path):
    a, model, root = setup(tmp_path)
    p = root / "features.parquet"
    f = pd.read_parquet(p)
    f.loc[0, "grt_l2_tns"] = 1.0
    f.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="非正"):
        a.iteration.inspect(root, model)


def test_all_gr_arcs_must_be_finite_even_when_dr_labels_are_missing():
    from paris_tool.iteration_worker import validate_gr

    f = pd.DataFrame(
        dict(
            gr_wire_delay=[0.1, np.nan],
            gr_wire_slew=[-0.4, -0.5],
            gr_slack=[-1.0, -2.0],
            gr_pi_r=[0.1, 0.2],
        )
    )
    with pytest.raises(ValueError, match="including unmatched"):
        validate_gr(f, ["gr_wire_delay", "gr_wire_slew", "gr_pi_r"])


def test_negative_physical_values_are_rejected_but_slew_difference_is_allowed():
    from paris_tool.iteration_worker import validate_gr

    f = pd.DataFrame(
        dict(gr_wire_delay=[0.1], gr_wire_slew=[-0.4], gr_slack=[-1.0], gr_pi_r=[-0.1])
    )
    with pytest.raises(ValueError, match="Negative GR"):
        validate_gr(f, ["gr_wire_delay", "gr_wire_slew", "gr_pi_r"])
    f.gr_pi_r = 0.1
    validate_gr(f, ["gr_wire_delay", "gr_wire_slew", "gr_pi_r"])
