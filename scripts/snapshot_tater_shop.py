#!/usr/bin/env python3
"""Create an immutable, publish-safe Tater Shop catalog fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen


VERBA_URL = "https://raw.githubusercontent.com/TaterTotterson/Tater_Shop/main/manifest.json"
CORE_URL = "https://raw.githubusercontent.com/TaterTotterson/Tater_Shop/main/core_manifest.json"

KERNEL_TOOLS = [
    ("attach_file", "Attach an available artifact or local file."),
    ("delete_file", "Delete a local file."),
    ("download_file", "Download a file from a concrete file URL."),
    ("extract_archive", "Extract an archive into a target directory."),
    ("get_verba_help", "Show usage and guidance for a Verba."),
    ("image_describe", "Describe an explicit image artifact, URL, blob, or local path."),
    ("inspect_webpage", "Inspect and extract content from a specific webpage URL."),
    ("intercom", "Manage a room-to-room Voice Core intercom handoff."),
    ("list_archive", "Inspect archive entries."),
    ("list_directory", "List files and folders."),
    ("list_tools", "List kernel and enabled Verba tools."),
    ("read_file", "Read local file contents."),
    ("rewrite_text", "Rewrite supplied text for a downstream action."),
    ("run_terminal_task", "Run policy-controlled terminal work through Spudex."),
    ("search_files", "Search text across local files."),
    ("send_message", "Send an explicitly requested cross-portal message."),
    ("websearch", "Search the web, inspect sources, and synthesize an answer."),
    ("write_file", "Write content to a local file."),
]

PORTALS = [
    ("webui", "Tater Web UI"),
    ("macos", "macOS"),
    ("discord", "Discord"),
    ("homeassistant", "Home Assistant"),
    ("homekit", "HomeKit / Siri"),
    ("irc", "IRC"),
    ("matrix", "Matrix"),
    ("meshtastic", "Meshtastic"),
    ("telegram", "Telegram"),
    ("xbmc", "XBMC / Original Xbox"),
    ("little_spud", "Little Spud"),
    ("voice_core", "Voice Core"),
]


def _fetch(url: str) -> tuple[dict, str]:
    request = Request(url, headers={"User-Agent": "TaterBench catalog snapshot"})
    with urlopen(request, timeout=30) as response:
        raw = response.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def _catalog_row(item: dict, *, core: bool = False) -> dict:
    row = {
        "id": str(item.get("id") or ""),
        "name": str(item.get("name") or item.get("id") or ""),
        "version": str(item.get("version") or ""),
        "description": str(item.get("description") or ""),
        "enabled": True,
    }
    if core:
        row["module_key"] = str(item.get("module_key") or f"{row['id']}_core")
        row["running"] = True
    else:
        row["portals"] = [str(value) for value in item.get("portals") or []]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--snapshot-date", default=date.today().isoformat())
    args = parser.parse_args()
    output = args.output or Path(f"taterbench/fixtures/tater-shop-{args.snapshot_date}.json")

    verba_manifest, verba_sha = _fetch(VERBA_URL)
    core_manifest, core_sha = _fetch(CORE_URL)
    verbas = [_catalog_row(item) for item in verba_manifest.get("verbas") or []]
    cores = [_catalog_row(item, core=True) for item in core_manifest.get("cores") or []]
    payload = {
        "schema": 1,
        "profile": f"tater-full-synthetic-{args.snapshot_date}",
        "snapshot_date": args.snapshot_date,
        "sources": {
            "verbas": {"url": VERBA_URL, "sha256": verba_sha},
            "cores": {"url": CORE_URL, "sha256": core_sha},
        },
        "identity": {
            "first_name": "Tater",
            "last_name": "Bench",
            "personality": "Friendly, capable, concise, and direct.",
            "platform": "webui",
            "platform_label": "Tater Web UI",
            "now": "Friday, August 14, 2026 at 12:00 PM UTC",
        },
        "verbas": verbas,
        "kernel_tools": [
            {"id": tool_id, "name": tool_id, "description": description, "enabled": True}
            for tool_id, description in KERNEL_TOOLS
        ],
        "portals": [
            {
                "id": portal_id,
                "name": name,
                "description": f"Interact through {name}.",
                "enabled": True,
                "connected": True,
            }
            for portal_id, name in PORTALS
        ],
        "cores": cores,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(verbas)} Verbas and {len(cores)} Cores to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
