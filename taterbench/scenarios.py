from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def load_suite(name_or_path: str = "core") -> dict[str, Any]:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        source = candidate
    else:
        source = Path(str(files("taterbench").joinpath("suites", f"{name_or_path}.json")))
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError(f"Invalid benchmark suite: {source}")
    ids: set[str] = set()
    for scenario in payload["scenarios"]:
        if not isinstance(scenario, dict) or not str(scenario.get("id") or ""):
            raise ValueError(f"Suite contains an invalid scenario: {source}")
        scenario_id = str(scenario["id"])
        if scenario_id in ids:
            raise ValueError(f"Duplicate scenario id {scenario_id}: {source}")
        ids.add(scenario_id)
    return payload
