from __future__ import annotations

import unittest

from taterbench.prompts import build_messages
from taterbench.scenarios import load_suite


class SuiteTests(unittest.TestCase):
    def test_core_suite_is_versioned_and_renderable(self) -> None:
        suite = load_suite("core")
        self.assertEqual(suite["version"], "tater-core-0.1")
        self.assertGreaterEqual(len(suite["scenarios"]), 20)
        for scenario in suite["scenarios"]:
            messages = build_messages(scenario)
            self.assertGreaterEqual(len(messages), 2)
            self.assertEqual(messages[0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
