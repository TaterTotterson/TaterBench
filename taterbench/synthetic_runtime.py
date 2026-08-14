"""Frozen Tater runtime context used by benchmark prompts.

Nothing in this module reads a live Tater installation. The fixture is versioned so
every model in a benchmark cohort receives exactly the same identity and catalog.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any, Iterable


CATALOG_FILE = "tater-shop-2026-08-14.json"


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


@lru_cache(maxsize=1)
def load_runtime_fixture() -> dict[str, Any]:
    source = files("taterbench").joinpath("fixtures", CATALOG_FILE)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def enabled_tool_rows(extra_tool_ids: Iterable[str] = ()) -> list[dict[str, str]]:
    fixture = load_runtime_fixture()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in (fixture.get("verbas") or [], fixture.get("kernel_tools") or []):
        for row in group:
            tool_id = str(row.get("id") or "").strip()
            if not tool_id or tool_id in seen:
                continue
            seen.add(tool_id)
            rows.append(
                {
                    "id": tool_id,
                    "description": _short(row.get("description") or "available Tater capability", 80),
                }
            )
    for value in extra_tool_ids:
        tool_id = str(value or "").strip()
        if tool_id and tool_id not in seen:
            seen.add(tool_id)
            rows.append({"id": tool_id, "description": "Synthetic scenario-specific Tater capability."})
    return sorted(rows, key=lambda row: row["id"])


def enabled_tool_ids(extra_tool_ids: Iterable[str] = ()) -> list[str]:
    return [row["id"] for row in enabled_tool_rows(extra_tool_ids)]


def render_tool_index(extra_tool_ids: Iterable[str] = ()) -> str:
    fixture = load_runtime_fixture()
    extras = {str(value or "").strip() for value in extra_tool_ids if str(value or "").strip()}
    known = {
        str(row.get("id") or "").strip()
        for group in (fixture.get("verbas") or [], fixture.get("kernel_tools") or [])
        for row in group
    }
    lines = ["Available kernel tools (id | description):"]
    lines.extend(
        f"- id: {row['id']} | description: {_short(row['description'], 80)}"
        for row in fixture.get("kernel_tools") or []
    )
    lines.append("Available enabled verba tools on this platform (id | description):")
    lines.extend(
        f"- id: {row['id']} | description: {_short(row['description'], 80)}"
        for row in fixture.get("verbas") or []
    )
    lines.extend(
        f"- id: {tool_id} | description: Synthetic scenario-specific Tater capability."
        for tool_id in sorted(extras - known)
    )
    return "\n".join(lines)


def render_system_status() -> str:
    fixture = load_runtime_fixture()
    lines = [
        "Tater System Status",
        "",
        "Verba (Capabilities)",
        "These are the Verba tools you currently have available for reference.",
    ]
    for row in fixture.get("verbas") or []:
        lines.append(f"- {row['id']}: {_short(row['description'], 120)} (enabled)")

    lines.extend(
        [
            "",
            "Kernel Tools (Built-ins)",
            "These are kernel tools currently available for direct execution.",
        ]
    )
    for row in fixture.get("kernel_tools") or []:
        lines.append(f"- {row['id']}: {_short(row['description'], 120)}")

    lines.extend(
        [
            "",
            "Portals (Platforms)",
            "These are the Portals you are currently running or connected through.",
        ]
    )
    for row in fixture.get("portals") or []:
        lines.append(f"- {row['name']}: {_short(row['description'], 120)} (connected, enabled)")

    lines.extend(
        [
            "",
            "Cores (Systems)",
            "These are the Cores currently active in your system.",
        ]
    )
    for row in fixture.get("cores") or []:
        lines.append(f"- {row['name']}: {_short(row['description'], 120)} (running, enabled)")

    lines.extend(
        [
            "",
            "Rules:",
            "- Use this information for awareness of current capability and system status.",
            "- You may reference these Verba tools, Kernel tools, Portals, and Cores when relevant.",
            "- Do NOT simulate calling Verba or Kernel tools in this response.",
            "- Do NOT pretend to execute actions in chat mode.",
            "- Do NOT mention internal modes, pipelines, or branches unless asked.",
            "- If the user asks to perform an action, respond naturally without claiming execution occurred.",
            "- Keep responses immersive and user-facing, not mechanical.",
        ]
    )
    return "\n".join(lines)


def render_core_context(role: str) -> str:
    normalized = str(role or "").strip().lower()
    parts = [
        str(row.get("content") or "").strip()
        for row in load_runtime_fixture().get("core_contexts") or []
        if normalized in {str(value or "").strip().lower() for value in row.get("roles") or []}
        and str(row.get("content") or "").strip()
    ]
    return "\n\n".join(parts)


def synthetic_memory_context() -> dict[str, str]:
    return dict(load_runtime_fixture().get("memory_context") or {})


def synthetic_identity() -> dict[str, str]:
    return dict(load_runtime_fixture().get("identity") or {})
