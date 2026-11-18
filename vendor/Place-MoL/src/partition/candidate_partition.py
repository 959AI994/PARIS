"""Load an externally supplied PARIS macro partition candidate.

This module intentionally uses only the Python standard library so Place-MoL's
Docker environment does not need the PARIS package installed. The accepted JSON
format is defined in ``PARIS/schemas/candidate.schema.json``.
"""

import hashlib
import json
import os


def _decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def load_candidate_partition(path, problem_instance):
    candidate_path = os.path.abspath(path)
    with open(candidate_path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)

    required = {"design", "candidate_id", "assignment"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Partition candidate is missing fields: {sorted(missing)}")
    if payload["design"] != problem_instance.benchmark:
        raise ValueError(
            f"Partition candidate design '{payload['design']}' does not match "
            f"benchmark '{problem_instance.benchmark}'"
        )
    if payload.get("granularity", "macro") != "macro":
        raise ValueError("Place-MoL currently accepts only macro-granularity candidates")
    if int(payload.get("die_count", 2)) != 2:
        raise ValueError("Place-MoL currently accepts only two-die candidates")
    assignment = payload["assignment"]
    if not isinstance(assignment, dict) or not assignment:
        raise ValueError("Partition candidate assignment must be a non-empty object")

    if not hasattr(problem_instance, "macros") or problem_instance.macros is None:
        problem_instance.determine_macro()
    db = problem_instance.dmp_placedb
    macro_names = {_decode_name(db.node_names[node_id]) for node_id in problem_instance.macros}
    assigned_names = set(assignment)
    if assigned_names != macro_names:
        raise ValueError(
            "Partition candidate macro set mismatch: "
            f"missing={sorted(macro_names - assigned_names)[:10]}, "
            f"extra={sorted(assigned_names - macro_names)[:10]}"
        )
    invalid = {name: die for name, die in assignment.items() if die not in (0, 1)}
    if invalid:
        raise ValueError(f"Partition candidate contains invalid die assignments: {invalid}")

    name_to_id = {
        _decode_name(db.node_names[node_id]): node_id for node_id in problem_instance.macros
    }
    bottom_names = sorted(name for name, die in assignment.items() if die == 0)
    upper_names = sorted(name for name, die in assignment.items() if die == 1)
    bottom_ids = [name_to_id[name] for name in bottom_names]
    upper_ids = [name_to_id[name] for name in upper_names]

    def node_area(node_id):
        return float(db.node_size_x[node_id]) * float(db.node_size_y[node_id])

    macro_ids = set(problem_instance.macros)
    physical_node_count = db.num_physical_nodes - db.num_terminal_NIs
    cell_ids = set(range(physical_node_count)) - macro_ids
    cells_area = sum(node_area(node_id) for node_id in cell_ids)
    scale_factor = problem_instance.args.partition_params.get("scale_factor", 1.0)
    if 0 < scale_factor <= 1.0:
        cells_area /= scale_factor

    canonical = json.dumps(sorted(assignment.items()), separators=(",", ":"))
    candidate_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "upper_die_macros": upper_ids,
        "upper_die_macro_names": upper_names,
        "bottom_die_macros": bottom_ids,
        "bottom_die_macro_names": bottom_names,
        "upper_die_area": sum(node_area(node_id) for node_id in upper_ids),
        "bottom_die_area": sum(node_area(node_id) for node_id in bottom_ids) + cells_area,
        "cells_area": cells_area,
        "moved_macros_count": len(bottom_ids),
        "candidate_id": str(payload["candidate_id"]),
        "candidate_hash": candidate_hash,
        "candidate_path": candidate_path,
    }
