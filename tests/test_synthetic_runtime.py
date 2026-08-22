from __future__ import annotations

import json
import unittest

from taterbench.prompts import build_messages
from taterbench.scenarios import load_suite
from taterbench.synthetic_runtime import (
    enabled_tool_ids,
    load_runtime_fixture,
    render_core_context,
    render_system_status,
)
from taterbench.version import PROMPT_PROFILE_VERSION


class SyntheticRuntimeTests(unittest.TestCase):
    def test_frozen_shop_catalog_is_complete_and_all_enabled(self) -> None:
        fixture = load_runtime_fixture()
        verbas = fixture["verbas"]
        cores = fixture["cores"]
        self.assertEqual(fixture["profile"], "tater-full-synthetic-2026-08-14")
        self.assertEqual(PROMPT_PROFILE_VERSION, "tater-curated-synthetic-2026-08-22")
        self.assertEqual(len(verbas), 63)
        self.assertEqual(len({row["id"] for row in verbas}), 63)
        self.assertTrue(all(row["enabled"] is True for row in verbas))
        self.assertEqual(len(cores), 10)
        self.assertEqual(len({row["id"] for row in cores}), 10)
        self.assertTrue(all(row["enabled"] is True and row["running"] is True for row in cores))
        self.assertEqual(len(fixture["kernel_tools"]), 45)
        self.assertEqual(sum(row["source"] == "builtin" for row in fixture["kernel_tools"]), 18)
        self.assertEqual(sum(row["source"] == "core" for row in fixture["kernel_tools"]), 27)
        self.assertEqual(len(fixture["portals"]), 12)
        self.assertEqual(len(fixture["core_contexts"]), 5)

    def test_only_curated_tools_reach_astraeus_payload(self) -> None:
        messages = build_messages(
            {
                "kind": "astraeus",
                "user": "Turn on the test lamp.",
                "available_tools": ["synthetic_lamp_control"],
            }
        )
        prefix = "Astraeus stable execution catalog:\n"
        stable_payload = json.loads(messages[0]["content"].split("\n\n" + prefix, 1)[1])
        dynamic_payload = json.loads(messages[1]["content"])
        tool_ids = stable_payload["available_tool_ids"]
        self.assertEqual(tool_ids, ["synthetic_lamp_control"])
        self.assertNotIn("automatic_plugin", tool_ids)
        self.assertNotIn("music_play", tool_ids)
        self.assertNotIn("music_assistant", tool_ids)
        self.assertIn("Only the tools listed above are available", stable_payload["available_capabilities"])
        self.assertTrue(dynamic_payload["synthetic_runtime"]["all_shop_verbas_enabled"])
        self.assertTrue(dynamic_payload["synthetic_runtime"]["all_shop_cores_running"])
        self.assertEqual(
            dynamic_payload["synthetic_runtime"]["routing_catalog_scope"],
            "scenario_curated_non_overlapping",
        )

    def test_core_routing_prompts_omit_overlapping_music_tools(self) -> None:
        overlap = {"music_assistant", "music_control", "music_play", "roon_music"}
        for scenario in load_suite("core")["scenarios"]:
            if scenario.get("kind") != "astraeus":
                continue
            messages = build_messages(scenario)
            prefix = "Astraeus stable execution catalog:\n"
            stable_payload = json.loads(messages[0]["content"].split("\n\n" + prefix, 1)[1])
            prompted = set(stable_payload["available_tool_ids"])
            self.assertEqual(prompted, set(scenario.get("available_tools") or []), scenario["id"])
            self.assertLessEqual(len(prompted & overlap), 1, scenario["id"])

    def test_prompt_catalog_defensively_removes_equivalent_tools(self) -> None:
        self.assertEqual(
            enabled_tool_ids(["music_assistant", "music_play", "send_message"]),
            ["music_play", "send_message"],
        )

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

    def test_fake_core_context_is_available_to_chat_and_hermes(self) -> None:
        chat_context = render_core_context("chat")
        self.assertIn("Riley Example", chat_context)
        self.assertIn("Guardian Core network context", chat_context)
        self.assertIn("Private Tater Tube activity context", chat_context)
        self.assertEqual(chat_context, render_core_context("hermes"))

    def test_each_core_awareness_scenario_receives_its_frozen_fact(self) -> None:
        facts = {
            "chat-core-memory": "roasted potatoes",
            "chat-core-personal": "Neighborhood potluck",
            "chat-core-guardian": "Bench Printer",
            "chat-core-music": "upbeat electronic music",
            "chat-core-tater-tube": "Example Space Show",
        }
        scenarios = {scenario["id"]: scenario for scenario in load_suite("core")["scenarios"]}
        self.assertEqual(set(facts), set(scenarios) & set(facts))
        for scenario_id, fact in facts.items():
            messages = build_messages(scenarios[scenario_id])
            self.assertIn(fact, messages[0]["content"], scenario_id)
            self.assertNotIn(fact, messages[-1]["content"], scenario_id)

    def test_chat_receives_system_status_and_spudex_uses_fake_paths(self) -> None:
        chat = build_messages({"kind": "chat", "user": "What can you do?"})
        self.assertIn("Tater System Status", chat[0]["content"])
        spudex = build_messages({"kind": "spudex", "user": "Check memory."})
        payload = json.loads(spudex[1]["content"])
        self.assertEqual(payload["working_folder"], "/synthetic/tater/agent_lab/workspace")
        self.assertEqual(payload["system_info"]["memory_total"], "64 GiB")


if __name__ == "__main__":
    unittest.main()
