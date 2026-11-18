import os
import subprocess
import sys
from pathlib import Path
from paris_tool.cli import main


def test_init_and_demo(tmp_path):
    assert main(["init", "--directory", str(tmp_path / "project")]) == 0
    assert (tmp_path / "project/project.yaml").is_file()
    assert main(["demo", "--output", str(tmp_path / "demo")]) == 0
    assert (
        main(["dataset", "validate", "--dataset", str(tmp_path / "demo/history")]) == 0
    )


def test_public_commands_help():
    p = subprocess.run(
        [sys.executable, "-m", "paris_tool.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0
    for cmd in ["collect", "train", "predict", "label", "screen", "model", "run"]:
        assert cmd in p.stdout
