"""Persistent route budget: reserve before starting; failures consume budget."""

import json
import os
from pathlib import Path
import sqlite3
from . import config
from .collection import collect_label, assemble
from .utils import save, run, digest


def reserve(db, key, budget):
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db, timeout=30) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS routes (key TEXT PRIMARY KEY, state TEXT NOT NULL)"
        )
        con.execute("BEGIN IMMEDIATE")
        old = con.execute("SELECT state FROM routes WHERE key=?", (key,)).fetchone()
        if old:
            raise RuntimeError(
                "Existing route job retained; inspect before explicit recovery: " + key
            )
        used = con.execute("SELECT COUNT(*) FROM routes").fetchone()[0]
        if used >= budget:
            raise RuntimeError("Total DR budget exhausted")
        con.execute("INSERT INTO routes VALUES (?,?)", (key, "reserved"))
    return used + 1


def label(c, rows, budget):
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    config.doctor(c, need_rcx=True)
    completed = []
    for row in rows:
        r = dict(row)
        if not r.get("dr_odb") or not Path(r["dr_odb"]).exists():
            command = c.get("flow", {}).get("dr_command")
            if not isinstance(command, list) or not command:
                raise ValueError(
                    "No existing DR checkpoint and no flow.dr_command configured"
                )
            key = r["design"] + "/" + r["candidate_id"]
            job = c["workspace"] / "route_jobs" / r["design"] / r["candidate_id"]
            cached = job / "DR_RESULT.json"
            complete = job / "complete.json"
            if complete.exists():
                if digest(cached) != json.loads(complete.read_text())["result_sha256"]:
                    raise ValueError("DR result cache changed")
                result = json.loads(cached.read_text())
            else:
                reserve(c["workspace"] / "budget.sqlite", key, budget)
                job.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(
                    PARIS_DESIGN=r["design"],
                    PARIS_CANDIDATE_ID=r["candidate_id"],
                    PARIS_GR_ODB=str(r["gr_odb"]),
                    PARIS_JOB_DIR=str(job),
                    PARIS_DR_RESULT=str(cached),
                    PARIS_OPENROAD_BIN=str(c["openroad"]["dr_binary"]),
                    OMP_NUM_THREADS=str(c["threads"]),
                )
                values = {
                    **r,
                    "openroad": c["openroad"]["dr_binary"],
                    "job_dir": job,
                    "result_json": cached,
                }
                run(
                    [str(x).format_map(values) for x in command],
                    job,
                    env,
                    c["timeout"],
                    "detailed_route",
                )
                if not cached.exists():
                    raise ValueError(
                        "DR flow must write DR_RESULT.json with dr_odb/dr_sdc"
                    )
                result = json.loads(cached.read_text())
            for k in ["dr_odb", "dr_sdc"]:
                if k not in result:
                    raise ValueError("DR flow result missing " + k)
                p = Path(result[k])
                r[k] = (job / p).resolve() if not p.is_absolute() else p.resolve()
                if not r[k].is_file():
                    raise FileNotFoundError(r[k])
            if not complete.exists():
                save(complete, dict(status="complete", result_sha256=digest(cached)))
        collect_label(c, r)
        completed.append((r["design"], r["candidate_id"]))
    return dict(status="complete", completed=completed, dataset=str(assemble(c)))
