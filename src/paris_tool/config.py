import csv
import os
from pathlib import Path
import re
import shutil
import yaml


def path(value, base):
    def env(m):
        if m[1] not in os.environ:
            raise ValueError("Undefined environment variable: " + m[1])
        return os.environ[m[1]]

    text = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", env, str(value))
    p = Path(text).expanduser()
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def load(file):
    file = Path(file).resolve()
    c = yaml.safe_load(file.read_text())
    if not isinstance(c, dict) or c.get("version") != 1:
        raise ValueError("Project config requires version: 1")
    c["_config"] = file
    c["_base"] = file.parent
    c["workspace"] = path(c.get("workspace", "./paris-work"), file.parent)
    c["manifest"] = path(c["candidates"], file.parent)
    c["threads"] = int(c.get("threads", 2))
    c["timeout"] = float(c.get("timeout", 3600))
    if c["threads"] < 1 or c["timeout"] <= 0:
        raise ValueError("threads/timeout must be positive")
    o = c.get("openroad", {})
    binary = str(o.get("binary", "openroad"))
    binary = os.path.expandvars(binary)
    found = shutil.which(binary)
    o["binary"] = Path(found).resolve() if found else path(binary, file.parent)
    for k in ["gr_binary", "dr_binary"]:
        value = str(o.get(k, o["binary"]))
        found = shutil.which(os.path.expandvars(value))
        o[k] = Path(found).resolve() if found else path(value, file.parent)
    for name in ["setrc", "rcx_rules"]:
        if o.get(name):
            o[name] = path(o[name], file.parent)
    o["liberty"] = [path(x, file.parent) for x in o.get("liberty", [])]
    o["time_unit"] = o.get("time_unit", "ns")
    if o["time_unit"] != "ns":
        raise ValueError("Version 0.1 requires ns timing inputs; convert explicitly")
    c["openroad"] = o
    if c.get("model"):
        c["model"] = path(c["model"], file.parent)
    return c


def candidates(c):
    with c["manifest"].open(newline="") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        for k in ["design", "candidate_id"]:
            if (
                not r.get(k)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", r[k])
                or r[k] in [".", ".."]
            ):
                raise ValueError(
                    "Identifiers must contain only letters, numbers, _, -, ."
                )
        key = (r["design"], r["candidate_id"])
        if key in seen:
            raise ValueError("Duplicate candidate: " + str(key))
        seen.add(key)
        for k, v in list(r.items()):
            if k not in ["design", "candidate_id"] and v:
                r[k] = path(v, c["manifest"].parent)
    if not rows:
        raise ValueError("Empty candidate manifest")
    return rows


def doctor(c, need_rcx=False):
    o = c["openroad"]
    required = [o["binary"], o["gr_binary"], o["dr_binary"], *o["liberty"]]
    if not o["liberty"]:
        raise ValueError("List the exact design Liberty files in openroad.liberty")
    for k in ["setrc"] + (["rcx_rules"] if need_rcx else []):
        if not o.get(k):
            raise ValueError("Missing openroad." + k)
        required.append(o[k])
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing configured inputs: " + str(missing))
    if not os.access(o["binary"], os.X_OK):
        raise ValueError("OpenROAD binary is not executable")
    rows = candidates(c)
    if len({r["design"] for r in rows}) != 1:
        raise ValueError(
            "Use one design per project so its exact Liberty/SDC/RCX context is unambiguous"
        )
    for r in rows:
        for k in [
            "cts_odb",
            "cts_sdc",
            "gr_odb",
            "gr_sdc",
            "gr_segments",
            "gr_congestion",
        ]:
            if not r.get(k) or not Path(r[k]).is_file():
                raise FileNotFoundError(f'{r["candidate_id"]}: missing {k}')
    return dict(
        status="passed",
        openroad=str(o["binary"]),
        candidates=len(rows),
        workspace=str(c["workspace"]),
    )


def prepare_workspace(c):
    from .utils import save

    root = c["workspace"]
    marker = root / "PARIS_WORKSPACE.json"
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise ValueError(
            "Existing non-PARIS directory cannot be used as workspace: " + str(root)
        )
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        save(marker, dict(format="paris.workspace.v1", config=str(c["_config"])))
