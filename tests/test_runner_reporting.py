from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from taterbench.engines.base import EngineSession
from taterbench.reporting import aggregate_payload, render_html, write_reports
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
    def test_tater_score_weights_accuracy_speed_ttft_and_memory(self) -> None:
        def run(label: str, accuracy: float, speed: float, ttft: float, memory: int) -> dict:
            return {
                "status": "complete",
                "model": {"id": label, "repo_id": f"org/{label}", "provider": "llama_cpp"},
                "engine": {"engine": "llama.cpp"},
                "variant": {"name": "baseline"},
                "suite": {"version": "suite-1"},
                "configuration": {"context_size": 4096, "prompt_profile": "profile-1"},
                "accuracy": {"score": accuracy},
                "performance": {
                    "median_generation_tokens_per_second": speed,
                    "median_ttft_seconds": ttft,
                },
                "peak_rss_bytes": memory,
            }

        aggregate = aggregate_payload(
            [
                {
                    "created_at": "2026-08-14T00:00:00Z",
                    "hardware": {"hardware_id": "same-machine"},
                    "runs": [
                        run("fast", 100.0, 20.0, 0.1, 100),
                        run("slow", 50.0, 10.0, 0.2, 200),
                    ],
                }
            ]
        )
        scores = {item["model"]["id"]: item["tater_score"] for item in aggregate["runs"]}
        self.assertEqual(scores, {"fast": 100.0, "slow": 50.0})
        components = {item["model"]["id"]: item["tater_score_components"] for item in aggregate["runs"]}
        self.assertEqual(
            components["fast"],
            {"accuracy": 90.0, "generation_speed": 7.0, "ttft": 2.0, "memory": 1.0},
        )
        html_text = render_html(aggregate)
        self.assertLess(html_text.index("org/fast"), html_text.index("org/slow"))

    def test_tater_critical_accuracy_outranks_raw_performance(self) -> None:
        def run(label: str, categories: dict[str, float], speed: float, ttft: float, memory: int) -> dict:
            return {
                "status": "complete",
                "model": {"id": label, "repo_id": f"org/{label}", "provider": "llama_cpp"},
                "engine": {"engine": "llama.cpp"},
                "variant": {"name": "baseline"},
                "suite": {"version": "suite-1"},
                "configuration": {"context_size": 4096, "prompt_profile": "profile-1"},
                "accuracy": {"score": sum(categories.values()) / len(categories), "categories": categories},
                "performance": {
                    "median_generation_tokens_per_second": speed,
                    "median_ttft_seconds": ttft,
                },
                "peak_rss_bytes": memory,
            }

        aggregate = aggregate_payload(
            [
                {
                    "created_at": "2026-08-14T00:00:00Z",
                    "hardware": {"hardware_id": "same-machine"},
                    "runs": [
                        run(
                            "ultra-fast-unreliable",
                            {
                                "tool_accuracy": 60.05,
                                "routing": 35.12,
                                "spudex": 36.67,
                                "synthesis": 90.83,
                                "chat": 100.0,
                            },
                            100.0,
                            0.1,
                            100,
                        ),
                        run(
                            "slower-reliable",
                            {
                                "tool_accuracy": 87.43,
                                "routing": 85.29,
                                "spudex": 90.0,
                                "synthesis": 100.0,
                                "chat": 100.0,
                            },
                            20.0,
                            0.5,
                            500,
                        ),
                    ],
                }
            ]
        )
        scores = {item["model"]["id"]: item["tater_score"] for item in aggregate["runs"]}
        self.assertEqual(scores["ultra-fast-unreliable"], 59.38)
        self.assertEqual(scores["slower-reliable"], 82.42)
        self.assertGreater(scores["slower-reliable"], scores["ultra-fast-unreliable"])

    def test_cross_device_average_weights_each_device_equally(self) -> None:
        def run(run_id: str, accuracy: float) -> dict:
            return {
                "run_id": run_id,
                "status": "complete",
                "finished_at": f"2026-08-14T00:00:0{run_id[-1]}Z",
                "model": {
                    "id": "shared-model",
                    "repo_id": "org/shared-model",
                    "filename": "shared-Q4.gguf",
                    "provider": "llama_cpp",
                    "quantization": "Q4",
                },
                "engine": {"engine": "llama.cpp"},
                "variant": {"name": "baseline"},
                "suite": {"version": "suite-1"},
                "configuration": {"context_size": 4096, "prompt_profile": "profile-1"},
                "accuracy": {"score": accuracy, "categories": {"routing": accuracy}},
                "performance": {
                    "median_generation_tokens_per_second": 20.0,
                    "median_prompt_tokens_per_second": 100.0,
                    "median_ttft_seconds": 0.1,
                    "median_scenario_seconds": 0.2,
                },
                "peak_rss_bytes": 100,
                "load_seconds": 1.0,
            }

        aggregate = aggregate_payload(
            [
                {
                    "created_at": "2026-08-14T00:00:00Z",
                    "hardware": {"hardware_id": "device-a", "cpu": "Device A", "memory_bytes": 64},
                    "runs": [run("run-1", 100.0)],
                },
                {
                    "created_at": "2026-08-14T00:00:30Z",
                    "hardware": {"hardware_id": "device-a-2", "cpu": "Device A", "memory_bytes": 64},
                    "runs": [run("run-2", 50.0)],
                },
                {
                    "created_at": "2026-08-14T00:01:00Z",
                    "hardware": {"hardware_id": "device-b", "cpu": "Device B", "memory_bytes": 128},
                    "runs": [run("run-3", 0.0)],
                },
            ]
        )
        result = aggregate["leaderboard"][0]
        self.assertEqual(result["device_count"], 2)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["tater_score"], 43.75)
        self.assertEqual(len(aggregate["devices"]), 2)
        self.assertEqual(aggregate["devices"][0]["leaderboard"][0]["sample_count"], 2)
        self.assertEqual(aggregate["devices"][0]["hardware_profile_count"], 2)

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
            report_payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
            self.assertEqual(report_payload["runs"][0]["tater_score"], 100.0)
            self.assertEqual(report_payload["runs"][0]["tater_score_components"]["accuracy"], 90.0)
            html_text = outputs["html"].read_text(encoding="utf-8")
            self.assertIn("Tater Bench", html_text)
            self.assertIn("Best models for Tater", html_text)
            self.assertIn('class="score-entry"', html_text)
            self.assertIn(
                "90 category-weighted accuracy (35 tool · 25 routing · 15 Spudex · 10 synthesis · 5 chat) · 7 generation speed · 2 TTFT · 1 memory efficiency",
                html_text,
            )
            self.assertIn('id="score-bars"', html_text)
            self.assertIn("All Devices results — sorted by Tater Score", html_text)
            self.assertIn('class="detail-rank"', html_text)
            self.assertIn("scoreBars.style.setProperty('--bar-count'", html_text)
            self.assertIn('data-scope-button="overall"', html_text)
            self.assertNotIn('data-scope-button="runs"', html_text)
            self.assertIn('id="sort-results"', html_text)
            self.assertIn('<details class="run-details"', html_text)
            self.assertIn("See 1 individual run", html_text)
            self.assertIn("@media(max-width:560px)", html_text)
            self.assertIn(".score-bars{width:100%;min-width:0", html_text)
            self.assertNotIn('id="benchmark-chart"', html_text)


if __name__ == "__main__":
    unittest.main()
