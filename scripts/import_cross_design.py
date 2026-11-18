"""Wrap an existing frozen shared bank for the PARIS UI, preserving weights."""

import argparse
import json
from pathlib import Path
import shutil

from paris_tool.utils import digest, seal, verify


def main(source, output):
    source = Path(source).resolve()
    card = verify(source)
    if not card.get("unknown_design_names_allowed") or card.get("status") != "complete":
        raise ValueError("Expected a complete shared cross-design bundle")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output)
    card.update(algorithm="paris4.cross-design.1", input_contract="features.parquet", source_bundle_sha256=digest(source / "SHA256.json"))
    seal(output, card)
    assert digest(source / "shared_ranker.joblib") == digest(output / "shared_ranker.joblib")
    return dict(status="complete", output=str(output), algorithm=card["algorithm"], weights_sha256=digest(output / "shared_ranker.joblib"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(main(a.source,a.output)))
