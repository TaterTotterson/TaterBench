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

    def test_astraeus_accepts_declared_equivalent_tool(self) -> None:
        scenario = {
            "kind": "astraeus",
            "available_tools": ["device_control", "music_play"],
            "expected": {
                "mode": "execute",
                "tool_sequence": [
                    "device_control",
                    {"one_of": ["music_play", "music_assistant"]},
                ],
            },
        }
        response = '{"mode":"execute","goal":"done","steps":[{"step_id":1,"intent":"lights","nl":"turn off lights","tool_hint":"device_control"},{"step_id":2,"intent":"music","nl":"play music","tool_hint":"music_assistant"}]}'
        grade = grade_response(scenario, response)
        self.assertEqual(grade["score"], 1.0)
        self.assertTrue(grade["passed"])
        self.assertEqual(
            grade["details"]["expected_tool_options"],
            [["device_control"], ["music_play", "music_assistant"]],
        )

    def test_astraeus_rejects_unrelated_tool_as_equivalent(self) -> None:
        scenario = {
            "kind": "astraeus",
            "available_tools": ["device_control", "music_play", "send_message"],
            "expected": {
                "mode": "execute",
                "tool_sequence": [
                    "device_control",
                    {"one_of": ["music_play", "music_assistant"]},
                ],
            },
        }
        response = '{"mode":"execute","goal":"done","steps":[{"step_id":1,"intent":"lights","nl":"turn off lights","tool_hint":"device_control"},{"step_id":2,"intent":"music","nl":"play music","tool_hint":"send_message"}]}'
        grade = grade_response(scenario, response)
        self.assertLess(grade["score"], 1.0)
        self.assertFalse(grade["passed"])

    def test_astraeus_gives_declared_less_preferred_tool_partial_credit(self) -> None:
        scenario = {
            "kind": "astraeus",
            "available_tools": ["device_control", "music_play"],
            "expected": {
                "mode": "execute",
                "tool_sequence": [
                    "device_control",
                    {"one_of": ["music_play"], "partial_credit": {"music_control": 0.5}},
                ],
            },
        }
        response = '{"mode":"execute","goal":"done","steps":[{"step_id":1,"intent":"lights","nl":"turn off lights","tool_hint":"device_control"},{"step_id":2,"intent":"music","nl":"play music","tool_hint":"music_control"}]}'
        grade = grade_response(scenario, response)
        self.assertEqual(grade["score"], 0.9)
        self.assertEqual(
            grade["details"]["expected_tool_partial_credit"],
            [{}, {"music_control": 0.5}],
        )

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

    def test_astraeus_requires_nl_but_does_not_grade_its_wording(self) -> None:
        scenario = {
            "kind": "astraeus",
            "available_tools": ["device_control"],
            "expected": {
                "mode": "execute",
                "tool_sequence": ["device_control"],
                "step_nl": [{"non_empty": True}],
            },
        }
        arbitrary = grade_response(
            scenario,
            '{"mode":"execute","goal":"done","steps":[{"step_id":1,"intent":"lights","nl":"any natural-language instruction","tool_hint":"device_control"}]}',
        )
        missing = grade_response(
            scenario,
            '{"mode":"execute","goal":"done","steps":[{"step_id":1,"intent":"lights","nl":"","tool_hint":"device_control"}]}',
        )
        self.assertEqual(arbitrary["score"], 1.0)
        self.assertTrue(arbitrary["passed"])
        self.assertEqual(missing["score"], 0.79)
        self.assertFalse(missing["passed"])

    def test_verba_nl_argument_only_needs_to_be_non_empty(self) -> None:
        scenario = {
            "kind": "thanatos",
            "expected": {
                "function": "device_control",
                "arguments": {"query": {"non_empty": True}},
            },
        }
        arbitrary = grade_response(
            scenario,
            '{"function":"device_control","arguments":{"query":"any wording is accepted"}}',
        )
        wrong_field = grade_response(
            scenario,
            '{"function":"device_control","arguments":{"nl":"some text"}}',
        )
        self.assertEqual(arbitrary["score"], 1.0)
        self.assertTrue(arbitrary["passed"])
        self.assertEqual(wrong_field["score"], 0.65)
        self.assertFalse(wrong_field["passed"])

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
        scenario = {"kind": "thanatos", "expected": {"blocker": True}}
        good = grade_response(scenario, "Who should receive the update?")
        empty = grade_response(scenario, "")
        bad = grade_response(scenario, '{"function":"send_message","arguments":{}}')
        self.assertEqual(good["score"], 1.0)
        self.assertFalse(empty["passed"])
        self.assertLess(bad["score"], good["score"])


if __name__ == "__main__":
    unittest.main()
