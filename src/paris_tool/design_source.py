"""Import reproducible, data-only 2D benchmark projects."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO = "MIRALab-USTC/ChiPBench"
PLATFORM = "flow/platforms/nangate45/"
PLATFORM_FILES = (
    "lef/NangateOpenCellLibrary.tech.lef",
    "lef/NangateOpenCellLibrary.macro.mod.lef",
    "lib/NangateOpenCellLibrary_typical.lib",
)


def _get(url):
    with urlopen(
        Request(url, headers={"User-Agent": "PARIS-design-import"}), timeout=90
    ) as r:
        return r.read()


def verify_blob(data, expected):
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != expected:
        raise ValueError("Downloaded file differs from the pinned Git blob")


def fetch_chipbench(design, output, revision="main"):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", design):
        raise ValueError("Invalid design name")
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    status = root / "SOURCE_STATUS.json"

    def save(state):
        temp = status.with_suffix(".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(status)

    try:
        commit = json.loads(
            _get(
                f"https://api.github.com/repos/{REPO}/commits/{quote(revision, safe='')}"
            )
        )["sha"]
        tree = json.loads(
            _get(f"https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1")
        )
        if tree.get("truncated"):
            raise ValueError("Incomplete repository inventory")
        prefix = f"flow/designs/nangate45/{design}/"
        config_item = next((x for x in tree["tree"] if x["path"]==prefix+"config.mk" and x["type"]=="blob"),None)
        if config_item is None:
            raise ValueError("Design not found: " + design)
        config_bytes = _get(f"https://raw.githubusercontent.com/{REPO}/{commit}/{config_item['path']}")
        verify_blob(config_bytes,config_item["sha"])
        platform_files = set(PLATFORM_FILES)
        platform_files.update(kind+"/"+name for kind,name in re.findall(r"\$\(PLATFORM_DIR\)/(lef|lib)/([A-Za-z0-9_.-]+)",config_bytes.decode()))
        entries = []
        for item in tree["tree"]:
            src = item["path"]
            dest = None
            if src.startswith(prefix) and PurePosixPath(src).suffix in {
                ".v",
                ".lef",
                ".lib",
                ".sdc",
                ".mk",
            }:
                dest = "design/" + src[len(prefix) :]
            elif src.startswith(PLATFORM) and src[len(PLATFORM) :] in platform_files:
                dest = "platform/" + src[len(PLATFORM) :]
            elif src in {"LICENSE", "LICENSE.md", "LICENSE.txt"}:
                dest = src
            if dest is None:
                continue
            if (
                item["type"] != "blob"
                or item.get("mode") != "100644"
                or ".." in PurePosixPath(dest).parts
            ):
                raise ValueError("Unsupported source entry: " + src)
            entries.append((item, dest))
        if not any(dest == "design/config.mk" for _, dest in entries):
            raise ValueError("Design not found: " + design)
        missing = platform_files - {
            d[len("platform/") :] for _, d in entries if d.startswith("platform/")
        }
        if missing:
            raise ValueError("Missing platform files: " + str(sorted(missing)))
        receipts = []
        for item, dest in entries:
            save(
                dict(
                    status="downloading",
                    commit=commit,
                    completed=len(receipts),
                    total=len(entries),
                    current=dest,
                )
            )
            data = _get(
                f"https://raw.githubusercontent.com/{REPO}/{commit}/{item['path']}"
            )
            verify_blob(data, item["sha"])
            path = root / dest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            receipts.append(
                dict(
                    source=item["path"],
                    path=dest,
                    git_blob=item["sha"],
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                )
            )
        config = (root / "design/config.mk").read_text()
        match = re.search(
            r"^\s*export\s+DESIGN_NAME\s*[:?+]?=\s*([A-Za-z_][A-Za-z0-9_$]*)\s*$",
            config,
            re.M,
        )
        if not match:
            match = re.search(
                r"^\s*DESIGN_NAME\s*[:?+]?=\s*([A-Za-z_][A-Za-z0-9_$]*)\s*$",
                config,
                re.M,
            )
        if not match:
            raise ValueError("Cannot determine top module from design data")
        paths = [r["path"] for r in receipts]
        inputs = {
            name: sorted(p for p in paths if p.endswith(ext))
            for name, ext in (
                ("netlist", ".v"),
                ("lef", ".lef"),
                ("liberty", ".lib"),
                ("sdc", ".sdc"),
            )
        }
        # Technology must be loaded before macro geometry.
        inputs["lef"].sort(key=lambda p: (not p.endswith(".tech.lef"), p))
        inputs["liberty"].sort(
            key=lambda p: (not p.endswith("NangateOpenCellLibrary_typical.lib"), p)
        )
        if any(not values for values in inputs.values()):
            raise ValueError("Source project is missing required inputs")
        project = dict(
            version=1,
            design=design,
            top=match[1],
            inputs=inputs,
            partition=dict(count=12, seed=17),
            description=f"ChiPBench {design}, commit {commit}",
        )
        (root / "PARIS_PROJECT.json").write_text(json.dumps(project, indent=2) + "\n")
        receipt = dict(
            version=1,
            repository=f"https://github.com/{REPO}",
            revision=commit,
            design=design,
            files=receipts,
        )
        (root / "SOURCE.json").write_text(json.dumps(receipt, indent=2) + "\n")
        save(
            dict(
                status="complete",
                commit=commit,
                completed=len(receipts),
                total=len(receipts),
            )
        )
        return dict(
            status="complete",
            project=str(root / "PARIS_PROJECT.json"),
            revision=commit,
            files=len(receipts),
        )
    except Exception as exc:
        save(dict(status="failed", error=str(exc)))
        raise
