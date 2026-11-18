"""Mock-flow contracts only: no EDA accuracy or real detailed routing."""

import sys
from pathlib import Path
import pytest
from paris_tool import jobs


def setup(tmp_path, monkeypatch, body):
    script = tmp_path / "flow.py"
    script.write_text(
        'import os,json\nfrom pathlib import Path\np=Path(os.environ["PARIS_JOB_DIR"])\n'
        + body
    )
    c = dict(
        workspace=tmp_path / "work",
        threads=1,
        timeout=10,
        openroad=dict(binary=Path(sys.executable), dr_binary=Path(sys.executable)),
        flow=dict(dr_command=[sys.executable, str(script)]),
    )
    row = dict(design="chip", candidate_id="0001", gr_odb=tmp_path / "gr.odb")
    monkeypatch.setattr(jobs.config, "doctor", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "collect_label", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "assemble", lambda c: c["workspace"] / "dataset")
    return c, row


def test_successful_flow_is_cached_without_another_route(tmp_path, monkeypatch):
    c, r = setup(
        tmp_path,
        monkeypatch,
        '(p/"calls.txt").write_text("one")\n'
        '(p/"dr.odb").write_text("mock")\n(p/"dr.sdc").write_text("mock")\n'
        'Path(os.environ["PARIS_DR_RESULT"]).write_text(json.dumps(dict(dr_odb="dr.odb",dr_sdc="dr.sdc")))\n',
    )
    jobs.label(c, [r], 1)
    jobs.label(c, [r], 1)
    job = c["workspace"] / "route_jobs/chip/0001"
    assert (job / "calls.txt").read_text() == "one"
    assert (job / "complete.json").exists()


def test_failed_flow_is_not_blindly_restarted(tmp_path, monkeypatch):
    c, r = setup(
        tmp_path,
        monkeypatch,
        '(p/"partial.txt").write_text("keep")\nraise SystemExit(1)\n',
    )
    with pytest.raises(RuntimeError):
        jobs.label(c, [r], 1)
    with pytest.raises(RuntimeError):
        jobs.label(c, [r], 10)
    job = c["workspace"] / "route_jobs/chip/0001"
    assert (job / "partial.txt").read_text() == "keep"
    assert not (job / "complete.json").exists()


def test_incomplete_result_is_not_marked_complete(tmp_path, monkeypatch):
    c, r = setup(
        tmp_path,
        monkeypatch,
        'Path(os.environ["PARIS_DR_RESULT"]).write_text(json.dumps(dict(dr_odb="missing.odb")))\n',
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        jobs.label(c, [r], 1)
    assert not (c["workspace"] / "route_jobs/chip/0001/complete.json").exists()
