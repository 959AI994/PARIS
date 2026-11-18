"""Create a source-only release; reject private paths, weights and design data."""

from pathlib import Path
import argparse
import ast
import hashlib
import json
import re
import tarfile
import zipfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(output):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/paris_tool/__init__.py").read_text())
    version = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    forbidden = [
        "/" + "data" + "/wjx/",
        "/" + "home" + "/wjx/",
        "/" + "Users" + "/alex/",
        "gpu40" + ".s.acelab.top",
    ]
    files = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if (
            not p.is_file()
            or any(
                x in rel.parts
                for x in [
                    "dist",
                    "build",
                    "__pycache__",
                    ".pytest_cache",
                    ".ruff_cache",
                    ".git",
                ]
            )
            or any(x.endswith(".egg-info") for x in rel.parts)
        ):
            continue
        if p.suffix in [
            ".pyc",
            ".joblib",
            ".parquet",
            ".odb",
            ".spef",
            ".lib",
            ".gz",
            ".zip",
            ".whl",
        ]:
            raise ValueError("Private/generated artifact in source tree: " + str(rel))
        text = p.read_text()
        if any(x in text for x in forbidden) or re.search(
            r"-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----", text
        ):
            raise ValueError("Private path/key found in " + str(rel))
        files.append(p)
    manifest = {str(p.relative_to(root)): sha(p) for p in sorted(files)}
    payload = (
        json.dumps(
            dict(
                version=version,
                source_files=manifest,
                pretrained_weights_included=False,
                private_design_data_included=False,
            ),
            indent=2,
        )
        + "\n"
    )
    for extension in ["zip", "tar.gz"]:
        target = output / (f"paris-eda-{version}-source." + extension)
        if target.exists():
            raise FileExistsError(target)
        if extension == "zip":
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(files):
                    z.write(p, "paris-eda/" + str(p.relative_to(root)))
                z.writestr("paris-eda/SOURCE_MANIFEST.json", payload)
        else:
            import io

            with tarfile.open(target, "w:gz") as t:
                for p in sorted(files):
                    t.add(p, arcname="paris-eda/" + str(p.relative_to(root)))
                b = payload.encode()
                m = tarfile.TarInfo("paris-eda/SOURCE_MANIFEST.json")
                m.size = len(b)
                t.addfile(m, io.BytesIO(b))
    result = dict(
        version=version,
        source_file_count=len(files),
        source_scan="passed",
        artifacts={
            p.name: sha(p) for p in output.glob(f"paris-eda-{version}-source.*")
        },
    )
    (output / "SOURCE_RELEASE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args().output)
