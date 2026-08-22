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

# Prefer the first tool when more than one equivalent capability is accidentally
# selected for the same routing turn. Provider-specific tools remain available
# when they are the only member explicitly requested by a scenario.
OVERLAPPING_TOOL_FAMILIES = (
    ("music_play", "music_assistant", "roon_music"),
)


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


def enabled_tool_rows(tool_ids: Iterable[str] | None = None) -> list[dict[str, str]]:
    requested = (
        None
        if tool_ids is None
        else {str(value or "").strip() for value in tool_ids if str(value or "").strip()}
    )
    if requested is not None:
        for family in OVERLAPPING_TOOL_FAMILIES:
            selected = [tool_id for tool_id in family if tool_id in requested]
            if len(selected) > 1:
                requested.difference_update(selected[1:])
    fixture = load_runtime_fixture()
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in (fixture.get("verbas") or [], fixture.get("kernel_tools") or []):
        for row in group:
            tool_id = str(row.get("id") or "").strip()
            if not tool_id or tool_id in seen or (requested is not None and tool_id not in requested):
                continue
            seen.add(tool_id)
            rows.append(
                {
                    "id": tool_id,
                    "description": _short(row.get("description") or "available Tater capability", 80),
                }
            )
    for value in requested or ():
        tool_id = str(value or "").strip()
        if tool_id and tool_id not in seen:
            seen.add(tool_id)
            rows.append({"id": tool_id, "description": "Synthetic scenario-specific Tater capability."})
    return sorted(rows, key=lambda row: row["id"])


def enabled_tool_ids(tool_ids: Iterable[str] | None = None) -> list[str]:
    return [row["id"] for row in enabled_tool_rows(tool_ids)]


def render_tool_index(tool_ids: Iterable[str] | None = None) -> str:
    fixture = load_runtime_fixture()
    requested = (
        None
        if tool_ids is None
        else {str(value or "").strip() for value in tool_ids if str(value or "").strip()}
    )
    selected = enabled_tool_rows(tool_ids)
    selected_by_id = {row["id"]: row for row in selected}
    kernel_ids = {str(row.get("id") or "").strip() for row in fixture.get("kernel_tools") or []}
    verba_ids = {str(row.get("id") or "").strip() for row in fixture.get("verbas") or []}
    lines = ["Available kernel tools (id | description):"]
    lines.extend(
        f"- id: {tool_id} | description: {selected_by_id[tool_id]['description']}"
        for tool_id in sorted(kernel_ids & selected_by_id.keys())
    )
    lines.append("Available enabled verba tools on this platform (id | description):")
    lines.extend(
        f"- id: {tool_id} | description: {selected_by_id[tool_id]['description']}"
        for tool_id in sorted(verba_ids & selected_by_id.keys())
    )
    lines.extend(
        f"- id: {tool_id} | description: {selected_by_id[tool_id]['description']}"
        for tool_id in sorted(selected_by_id.keys() - kernel_ids - verba_ids)
    )
    if requested is not None:
        lines.append("Only the tools listed above are available for this benchmark turn.")
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
