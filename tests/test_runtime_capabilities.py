from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from taterbench.engines.llamacpp import LlamaCppSession, _gpu_layers_argument
from taterbench.paths import llama_server_candidates, llama_supported_speculative_methods, tater_home
from taterbench.types import ModelCandidate, RunVariant


class RuntimeCapabilityTests(unittest.TestCase):
    @staticmethod
    def _session(variant: RunVariant) -> LlamaCppSession:
        model = ModelCandidate(
            id="llama_cpp:test",
            label="test.gguf",
            provider="llama_cpp",
            model_path=Path("/tmp/test.gguf"),
            filename="test.gguf",
        )
        return LlamaCppSession(model, variant)

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

            with (
                mock.patch.dict("os.environ", {"TATER_BENCH_LLAMA_SERVER": str(server)}, clear=False),
                mock.patch("taterbench.paths.subprocess.run", side_effect=fake_run),
            ):
                self.assertEqual(llama_supported_speculative_methods(home), {"draft-mtp"})

    def test_tater_app_engine_precedes_downloaded_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "tater-home"
            resources = Path(temp) / "Tater.app" / "Contents" / "Resources"
            with mock.patch.dict("os.environ", {"TATER_APP_RESOURCES_DIR": str(resources)}, clear=False):
                candidates = llama_server_candidates(home)
            bundled = resources / "Native" / "llama.cpp" / "bin" / "llama-server"
            runtime = home / "runtime" / "llama.cpp" / "build" / "bin" / "llama-server"
            self.assertLess(candidates.index(bundled), candidates.index(runtime))

    def test_linux_source_checkout_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            user_home = Path(temp)
            source_home = user_home / "Tater"
            (source_home / "agent_lab/models").mkdir(parents=True)
            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch("taterbench.paths.Path.home", return_value=user_home),
            ):
                self.assertEqual(tater_home(), source_home.resolve())
                source_server = source_home / ".runtime/llama.cpp/build/bin/llama-server"
                self.assertIn(source_server, llama_server_candidates(source_home))

    def test_non_strix_defaults_to_automatic_target_offload(self) -> None:
        session = self._session(RunVariant("baseline"))
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("taterbench.engines.llamacpp._strix_halo_full_offload_default", return_value=False),
        ):
            command = session._command(Path("/tmp/llama-server"))
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "auto")
        self.assertNotIn("--spec-draft-ngl", command)

    def test_strix_fully_offloads_target_and_speculative_draft(self) -> None:
        variant = RunVariant(
            "mtp",
            speculative_method="draft-mtp",
            draft_path=Path("/tmp/draft.gguf"),
            draft_tokens=3,
        )
        session = self._session(variant)
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch("taterbench.engines.llamacpp._strix_halo_full_offload_default", return_value=True),
        ):
            command = session._command(Path("/tmp/llama-server"))
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "all")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "all")
        self.assertEqual(session.target_gpu_layers, "all")
        self.assertEqual(session.draft_gpu_layers, "all")

    def test_explicit_gpu_layer_overrides_win_on_strix(self) -> None:
        variant = RunVariant(
            "dflash",
            speculative_method="draft-dflash",
            draft_path=Path("/tmp/draft.gguf"),
            draft_tokens=15,
        )
        session = self._session(variant)
        with (
            mock.patch.dict(
                "os.environ",
                {
                    "TATER_BENCH_LLAMA_N_GPU_LAYERS": "auto",
                    "TATER_BENCH_LLAMA_DRAFT_N_GPU_LAYERS": "7",
                },
                clear=True,
            ),
            mock.patch("taterbench.engines.llamacpp._strix_halo_full_offload_default", return_value=True),
        ):
            command = session._command(Path("/tmp/llama-server"))
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "auto")
        self.assertEqual(command[command.index("--spec-draft-ngl") + 1], "7")

    def test_tater_gpu_layer_values_are_normalized(self) -> None:
        self.assertEqual(_gpu_layers_argument("-1", default="all"), "auto")
        self.assertEqual(_gpu_layers_argument("-2", default="auto"), "all")
        self.assertEqual(_gpu_layers_argument("cpu", default="all"), "0")
        self.assertEqual(_gpu_layers_argument("19", default="auto"), "19")


if __name__ == "__main__":
    unittest.main()
