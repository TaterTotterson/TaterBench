from __future__ import annotations

import unittest

from taterbench.prompts import build_messages
from taterbench.scenarios import load_suite


class SuiteTests(unittest.TestCase):
    def test_core_suite_is_versioned_and_renderable(self) -> None:
        suite = load_suite("core")
        self.assertEqual(suite["version"], "tater-core-0.4")
        self.assertEqual(len(suite["scenarios"]), 27)
        for scenario in suite["scenarios"]:
            messages = build_messages(scenario)
            self.assertGreaterEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[-1]["role"], "user")
            self.assertEqual(sum(message["role"] == "system" for message in messages), 1)

    def test_execution_routes_require_non_empty_nl_without_grading_wording(self) -> None:
        suite = load_suite("core")
        execution_routes = [
            scenario
            for scenario in suite["scenarios"]
            if scenario["kind"] == "astraeus" and scenario["expected"]["mode"] == "execute"
        ]
        self.assertTrue(execution_routes)
        for scenario in execution_routes:
            self.assertEqual(
                scenario["expected"]["step_nl"],
                [{"non_empty": True}] * len(scenario["expected"]["tool_sequence"]),
            )

    def test_nl_tool_contracts_use_runtime_fields(self) -> None:
        suite = load_suite("core")
        scenarios = {scenario["id"]: scenario for scenario in suite["scenarios"]}
        expected_fields = {
            "tool-timer": "query",
            "tool-light": "query",
            "tool-weather": "request",
            "tool-spudex": "request",
        }
        for scenario_id, field in expected_fields.items():
            arguments = scenarios[scenario_id]["expected"]["arguments"]
            self.assertEqual(arguments, {field: {"non_empty": True}})
        self.assertEqual(scenarios["tool-spudex"]["category"], "spudex")

    def test_thanatos_combines_instructions_into_the_initial_system_message(self) -> None:
        messages = build_messages(
            {
                "kind": "thanatos",
                "user": "Set a timer",
                "instruction": "Set a timer for 15 minutes",
                "tool_hint": "set_timer",
            }
        )
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Turn focus", messages[0]["content"])
        self.assertIn("Current agent state JSON", messages[0]["content"])
        self.assertIn("Execution step lock", messages[0]["content"])
        self.assertIn("Execution tool contract", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
