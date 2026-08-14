from __future__ import annotations

import unittest

from taterbench.grading import grade_response


class GradingTests(unittest.TestCase):
    def test_astraeus_exact_plan(self) -> None:
        scenario = {
            "kind": "astraeus",
            "available_tools": ["one", "two"],
            "expected": {"mode": "execute", "tool_sequence": ["one", "two"]},
        }
        response = '{"mode":"execute","goal":"done","steps":[{"step_id":1,"intent":"first","nl":"first","tool_hint":"one"},{"step_id":2,"intent":"second","nl":"second","tool_hint":"two"}]}'
        grade = grade_response(scenario, response)
        self.assertEqual(grade["score"], 1.0)
        self.assertTrue(grade["passed"])

    def test_tool_arguments_are_graded(self) -> None:
        scenario = {
            "kind": "thanatos",
            "expected": {
                "function": "set_timer",
                "arguments": {"duration_seconds": 900, "label": {"contains": "pizza"}},
            },
        }
        grade = grade_response(
            scenario,
            '{"function":"set_timer","arguments":{"duration_seconds":900,"label":"pizza timer"}}',
        )
        self.assertEqual(grade["score"], 1.0)

    def test_fenced_json_loses_strict_format_points(self) -> None:
        scenario = {
            "kind": "thanatos",
            "expected": {"function": "weather", "arguments": {"location": "Chicago"}},
        }
        grade = grade_response(
            scenario,
            '```json\n{"function":"weather","arguments":{"location":"Chicago"}}\n```',
        )
        self.assertLess(grade["score"], 1.0)

    def test_blocker_rejects_fake_tool_call(self) -> None:
        scenario = {"kind": "thanatos", "expected": {"blocker": True, "contains": ["destination"]}}
        good = grade_response(scenario, "The destination is missing.")
        bad = grade_response(scenario, '{"function":"send_message","arguments":{}}')
        self.assertEqual(good["score"], 1.0)
        self.assertLess(bad["score"], good["score"])


if __name__ == "__main__":
    unittest.main()
