from __future__ import annotations

import json
import unittest

from taterbench.prompts import build_messages
from taterbench.scenarios import load_suite
from taterbench.synthetic_runtime import enabled_tool_ids, load_runtime_fixture, render_system_status
from taterbench.version import PROMPT_PROFILE_VERSION


class SyntheticRuntimeTests(unittest.TestCase):
    def test_frozen_shop_catalog_is_complete_and_all_enabled(self) -> None:
        fixture = load_runtime_fixture()
        verbas = fixture["verbas"]
        cores = fixture["cores"]
        self.assertEqual(fixture["profile"], PROMPT_PROFILE_VERSION)
        self.assertEqual(len(verbas), 63)
        self.assertEqual(len({row["id"] for row in verbas}), 63)
        self.assertTrue(all(row["enabled"] is True for row in verbas))
        self.assertEqual(len(cores), 10)
        self.assertEqual(len({row["id"] for row in cores}), 10)
        self.assertTrue(all(row["enabled"] is True and row["running"] is True for row in cores))

    def test_full_tool_catalog_reaches_astraeus_payload(self) -> None:
        messages = build_messages(
            {
                "kind": "astraeus",
                "user": "Turn on the test lamp.",
                "available_tools": ["synthetic_lamp_control"],
            }
        )
        prefix = "Astraeus stable execution catalog:\n"
        stable_payload = json.loads(messages[1]["content"].removeprefix(prefix))
        dynamic_payload = json.loads(messages[2]["content"])
        tool_ids = stable_payload["available_tool_ids"]
        self.assertEqual(tool_ids, enabled_tool_ids(["synthetic_lamp_control"]))
        self.assertIn("automatic_plugin", tool_ids)
        self.assertIn("run_terminal_task", tool_ids)
        self.assertIn("synthetic_lamp_control", tool_ids)
        self.assertTrue(dynamic_payload["synthetic_runtime"]["all_shop_verbas_enabled"])
        self.assertTrue(dynamic_payload["synthetic_runtime"]["all_shop_cores_running"])

    def test_core_suite_uses_cataloged_tool_ids(self) -> None:
        available = set(enabled_tool_ids())
        for scenario in load_suite("core")["scenarios"]:
            for tool_id in scenario.get("available_tools") or []:
                self.assertIn(tool_id, available, scenario["id"])
            tool_hint = str(scenario.get("tool_hint") or "")
            if tool_hint:
                self.assertIn(tool_hint, available, scenario["id"])

    def test_status_contains_each_catalog_group_without_live_state(self) -> None:
        status = render_system_status()
        self.assertIn("Verba (Capabilities)", status)
        self.assertIn("Kernel Tools (Built-ins)", status)
        self.assertIn("Portals (Platforms)", status)
        self.assertIn("Cores (Systems)", status)
        self.assertIn("AI Task Scheduler Core", status)
        self.assertIn("Tater Tube Core", status)
        self.assertNotIn("/Users/", status)
        self.assertNotIn(".taterassistant", status)

    def test_chat_receives_system_status_and_spudex_uses_fake_paths(self) -> None:
        chat = build_messages({"kind": "chat", "user": "What can you do?"})
        self.assertIn("Tater System Status", chat[0]["content"])
        spudex = build_messages({"kind": "spudex", "user": "Check memory."})
        payload = json.loads(spudex[1]["content"])
        self.assertEqual(payload["working_folder"], "/synthetic/tater/agent_lab/workspace")
        self.assertEqual(payload["system_info"]["memory_total"], "64 GiB")


if __name__ == "__main__":
    unittest.main()
