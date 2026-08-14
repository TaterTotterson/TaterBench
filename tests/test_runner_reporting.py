from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from taterbench.engines.base import EngineSession
from taterbench.reporting import write_reports
from taterbench.runner import BenchmarkRunner, save_batch
from taterbench.types import GenerationResult, ModelCandidate, RunVariant


class FakeSession(EngineSession):
    def __init__(self) -> None:
        self.load_seconds = 1.25
        self.peak_rss_bytes = 0
        self.metadata = {"engine": "fake", "context_size": 4096, "server_path": "/private/path"}

    def start(self) -> None:
        return None

    def generate(self, messages, *, max_tokens, temperature, seed):
        user = messages[-1]["content"]
        text = "READY" if "Ready check" in user else "The answer is 102."
        return GenerationResult(
            text=text,
            elapsed_seconds=0.5,
            ttft_seconds=0.1,
            prompt_tokens=20,
            completion_tokens=5,
            completion_tokens_per_second=10.0,
        )

    def close(self) -> None:
        self.peak_rss_bytes = 1024 * 1024


class RunnerReportingTests(unittest.TestCase):
    def test_batch_is_privacy_safe_and_reports_render(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_path = root / "PrivateUser/Model-Q4_K_M.gguf"
            model_path.parent.mkdir()
            model_path.write_bytes(b"model")
            model = ModelCandidate(
                id="llama_cpp:test",
                label="org/model::Model-Q4_K_M.gguf",
                provider="llama_cpp",
                model_path=model_path,
                repo_id="org/model",
                filename=model_path.name,
                quantization="Q4_K_M",
            )
            suite = {
                "name": "Tiny",
                "version": "tiny-1",
                "max_tokens": 32,
                "scenarios": [
                    {
                        "id": "math",
                        "category": "chat",
                        "kind": "chat",
                        "user": "17 times 6?",
                        "expected": {"contains": ["102"]},
                    }
                ],
            }
            runner = BenchmarkRunner(
                context_size=4096,
                session_factory=lambda _model, _variant: FakeSession(),
            )
            batch = runner.run_batch([(model, RunVariant("baseline"))], suite)
            result_text = json.dumps(batch)
            self.assertNotIn("PrivateUser", result_text)
            self.assertNotIn("/private/path", result_text)
            self.assertEqual(batch["runs"][0]["accuracy"]["score"], 100.0)
            results_dir = root / "results"
            save_batch(batch, results_dir)
            outputs = write_reports(
                results_dir=results_dir,
                markdown_path=root / "RESULTS.md",
                docs_dir=root / "docs",
            )
            self.assertIn("org/model", outputs["markdown"].read_text(encoding="utf-8"))
            self.assertIn("Tater Bench", outputs["html"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
