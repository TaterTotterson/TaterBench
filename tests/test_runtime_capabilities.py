from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from taterbench.paths import llama_supported_speculative_methods


class RuntimeCapabilityTests(unittest.TestCase):
    def test_speculative_probe_rejects_unknown_methods(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            server = home / "runtime/llama.cpp/build/bin/llama-server"
            server.parent.mkdir(parents=True)
            server.write_text("binary", encoding="utf-8")
            server.chmod(0o755)

            def fake_run(args, **_kwargs):
                method = args[args.index("--spec-type") + 1]
                return mock.Mock(
                    returncode=0 if method == "draft-mtp" else 1,
                    stdout="version" if method == "draft-mtp" else "",
                    stderr="" if method == "draft-mtp" else "unknown speculative type",
                )

            with mock.patch("taterbench.paths.subprocess.run", side_effect=fake_run):
                self.assertEqual(llama_supported_speculative_methods(home), {"draft-mtp"})


if __name__ == "__main__":
    unittest.main()
