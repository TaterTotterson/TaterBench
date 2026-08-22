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
            {
                "accuracy": 90.0,
                "generation_speed": 7.0,
                "ttft": 2.0,
                "memory": 1.0,
                "readiness_adjustment": 0.0,
            },
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
                        run(
                            "fast-limited",
                            {
                                "tool_accuracy": 100.0,
                                "routing": 100.0,
                                "spudex": 79.0,
                                "synthesis": 100.0,
                                "chat": 100.0,
                            },
                            100.0,
                            0.1,
                            100,
                        ),
                    ],
                }
            ]
        )
        scores = {item["model"]["id"]: item["tater_score"] for item in aggregate["runs"]}
        raw_scores = {item["model"]["id"]: item["raw_tater_score"] for item in aggregate["runs"]}
        self.assertEqual(raw_scores["ultra-fast-unreliable"], 56.89)
        self.assertEqual(scores["ultra-fast-unreliable"], 49.9)
        self.assertEqual(scores["slower-reliable"], 82.21)
        self.assertEqual(raw_scores["fast-limited"], 96.85)
        self.assertEqual(scores["fast-limited"], 79.9)
        accuracy_components = {
            item["model"]["id"]: item["tater_score_components"]["accuracy"]
            for item in aggregate["runs"]
        }
        self.assertEqual(accuracy_components["ultra-fast-unreliable"], 46.89)
        self.assertEqual(accuracy_components["slower-reliable"], 80.21)
        self.assertGreater(scores["slower-reliable"], scores["ultra-fast-unreliable"])
        fitness = {item["model"]["id"]: item["fitness"]["status"] for item in aggregate["runs"]}
        self.assertEqual(
            fitness,
            {
                "ultra-fast-unreliable": "not_fit",
                "slower-reliable": "ready",
                "fast-limited": "limited",
            },
        )
        self.assertEqual(
            [item["model"]["id"] for item in aggregate["leaderboard"]],
            ["slower-reliable", "fast-limited", "ultra-fast-unreliable"],
        )

    def test_overall_uses_best_device_result_and_preserves_device_verdicts(self) -> None:
        def run(run_id: str, categories: dict[str, float], accuracy: float) -> dict:
            return {
                "run_id": run_id,
                "status": "complete",
                "model": {
                    "id": "gemma",
                    "repo_id": "org/gemma",
                    "filename": "gemma.gguf",
                    "provider": "llama_cpp",
                },
                "engine": {"engine": "llama.cpp"},
                "variant": {"name": "baseline"},
                "suite": {"version": "suite-1"},
                "configuration": {"context_size": 4096, "prompt_profile": "profile-1"},
                "accuracy": {"score": accuracy, "categories": categories},
                "performance": {
                    "median_generation_tokens_per_second": 20.0,
                    "median_ttft_seconds": 0.1,
                },
                "peak_rss_bytes": 100,
            }

        ready = {
            "tool_accuracy": 93.14,
            "routing": 89.88,
            "spudex": 90.0,
            "synthesis": 90.0,
            "chat": 100.0,
        }
        not_fit = {
            "tool_accuracy": 90.0,
            "routing": 65.0,
            "spudex": 90.0,
            "synthesis": 90.0,
            "chat": 100.0,
        }
        aggregate = aggregate_payload(
            [
                {
                    "hardware": {"hardware_id": "apple", "cpu": "Apple M3 Ultra", "memory_bytes": 100},
                    "runs": [run("apple-run", ready, 91.78)],
                },
                {
                    "hardware": {"hardware_id": "amd", "cpu": "AMD Ryzen", "memory_bytes": 100},
                    "runs": [run("amd-run", not_fit, 30.1)],
                },
            ]
        )
        overall = aggregate["leaderboard"][0]
        self.assertEqual(overall["fitness"]["status"], "ready")
        self.assertEqual(overall["fitness"]["label"], "Tater Ready")
        self.assertTrue(overall["fitness"]["provisional"])
        self.assertEqual(overall["raw_tater_score"], 92.24)
        self.assertEqual(overall["tater_score"], 92.24)
        self.assertEqual(overall["best_hardware"]["cpu"], "Apple M3 Ultra")
        self.assertEqual(overall["tested_device_count"], 2)
        self.assertEqual(overall["device_count"], 1)
        device_fitness = {
            device["hardware"]["cpu"]: device["leaderboard"][0]["fitness"]["status"]
            for device in aggregate["devices"]
        }
        self.assertEqual(device_fitness, {"AMD Ryzen": "not_fit", "Apple M3 Ultra": "ready"})

    def test_overall_selects_best_device_after_device_averaging(self) -> None:
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
        self.assertEqual(result["device_count"], 1)
        self.assertEqual(result["tested_device_count"], 2)
        self.assertEqual(result["best_hardware"]["cpu"], "Device A")
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["observation_count"], 2)
        self.assertEqual(result["tater_score"], 77.5)
        self.assertEqual(len(aggregate["devices"]), 2)
        self.assertEqual(aggregate["devices"][0]["leaderboard"][0]["sample_count"], 2)
        self.assertEqual(aggregate["devices"][0]["hardware_profile_count"], 2)

    def test_duplicate_graded_outcomes_on_same_device_are_omitted(self) -> None:
        def run(run_id: str, finished_at: str, scenario_score: float, speed: float) -> dict:
            passed = scenario_score == 1.0
            return {
                "run_id": run_id,
                "status": "complete",
                "finished_at": finished_at,
                "model": {
                    "id": "shared-model",
                    "repo_id": "org/shared-model",
                    "filename": "shared.gguf",
                    "provider": "llama_cpp",
                },
                "engine": {"engine": "llama.cpp"},
                "variant": {"name": "baseline"},
                "suite": {"version": "suite-1"},
                "configuration": {"context_size": 4096, "prompt_profile": "profile-1"},
                "accuracy": {"score": scenario_score * 100.0, "categories": {"chat": scenario_score * 100.0}},
                "scenario_results": [
                    {
                        "id": "chat-one",
                        "category": "chat",
                        "kind": "chat",
                        "passed": passed,
                        "score": scenario_score,
                        "weight": 1.0,
                        "attempts": [
                            {
                                "repeat": 1,
                                "grade": {"passed": passed, "score": scenario_score},
                                "performance": {"elapsed_seconds": 1.0 / speed},
                                "response": f"response from {run_id}",
                            }
                        ],
                    }
                ],
                "performance": {
                    "median_generation_tokens_per_second": speed,
                    "median_ttft_seconds": 0.1,
                },
                "peak_rss_bytes": 100,
            }

        aggregate = aggregate_payload(
            [
                {
                    "hardware": {"hardware_id": "apple-old", "cpu": "Apple M3", "memory_bytes": 100},
                    "runs": [run("apple-old", "2026-08-14T00:00:00Z", 1.0, 20.0)],
                },
                {
                    "hardware": {"hardware_id": "apple-new", "cpu": "Apple M3", "memory_bytes": 100},
                    "runs": [run("apple-new", "2026-08-14T00:01:00Z", 1.0, 30.0)],
                },
                {
                    "hardware": {"hardware_id": "amd", "cpu": "AMD Ryzen", "memory_bytes": 100},
                    "runs": [run("amd-same-outcome", "2026-08-14T00:02:00Z", 1.0, 40.0)],
                },
                {
                    "hardware": {"hardware_id": "apple-different", "cpu": "Apple M3", "memory_bytes": 100},
                    "runs": [run("apple-different", "2026-08-14T00:03:00Z", 0.5, 25.0)],
                },
            ]
        )
        self.assertEqual(aggregate["raw_run_count"], 4)
        self.assertEqual(aggregate["run_count"], 3)
        self.assertEqual(aggregate["duplicate_run_count"], 1)
        self.assertEqual(
            {item["run_id"] for item in aggregate["runs"]},
            {"apple-new", "amd-same-outcome", "apple-different"},
        )
        newest = next(item for item in aggregate["runs"] if item["run_id"] == "apple-new")
        self.assertEqual(newest["performance"]["median_generation_tokens_per_second"], 30.0)
        self.assertEqual(newest["observation_count"], 2)
        device_counts = {device["hardware"]["cpu"]: device["run_count"] for device in aggregate["devices"]}
        self.assertEqual(device_counts, {"AMD Ryzen": 1, "Apple M3": 2})
        observation_counts = {
            device["hardware"]["cpu"]: device["observation_count"]
            for device in aggregate["devices"]
        }
        self.assertEqual(observation_counts, {"AMD Ryzen": 1, "Apple M3": 3})
        self.assertEqual(aggregate["leaderboard"][0]["best_hardware"]["cpu"], "AMD Ryzen")
        self.assertEqual(aggregate["leaderboard"][0]["tested_device_count"], 2)
        self.assertEqual(aggregate["leaderboard"][0]["sample_count"], 1)
        self.assertEqual(aggregate["leaderboard"][0]["observation_count"], 1)

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
            self.assertEqual(report_payload["runs"][0]["raw_tater_score"], 100.0)
            self.assertEqual(report_payload["raw_run_count"], 1)
            self.assertEqual(report_payload["run_count"], 1)
            self.assertEqual(report_payload["duplicate_run_count"], 0)
            self.assertEqual(report_payload["runs"][0]["observation_count"], 1)
            self.assertEqual(report_payload["runs"][0]["tater_score_components"]["accuracy"], 90.0)
            self.assertEqual(
                report_payload["runs"][0]["tater_score_components"]["readiness_adjustment"],
                0.0,
            )
            html_text = outputs["html"].read_text(encoding="utf-8")
            self.assertIn("Tater Bench", html_text)
            self.assertIn("Best models for Tater", html_text)
            self.assertIn('class="score-entry"', html_text)
            self.assertIn("Unrated", html_text)
            self.assertIn(
                "90 category-weighted accuracy (35 Astraeus routing/tool selection · 25 Thanatos execution · 15 Spudex · 10 synthesis · 5 chat) · 7 generation speed · 2 TTFT · 1 memory",
                html_text,
            )
            self.assertIn("Limited results cap at 79.9; Not Fit results cap at 49.9", html_text)
            self.assertIn('id="score-bars"', html_text)
            self.assertIn("Overall results — sorted by Final Tater Score", html_text)
            self.assertIn("Best device result", html_text)
            self.assertIn('class="detail-rank"', html_text)
            self.assertIn("scoreBars.style.setProperty('--bar-count'", html_text)
            self.assertIn('data-scope-button="overall"', html_text)
            self.assertNotIn('data-scope-button="runs"', html_text)
            self.assertIn('id="sort-results"', html_text)
            self.assertIn('<details class="run-details"', html_text)
            self.assertIn("See 1 unique run", html_text)
            self.assertIn("1 unique run · 0 duplicates omitted", html_text)
            self.assertIn("1 unique / 1 observed", html_text)
            self.assertIn("@media(max-width:560px)", html_text)
            self.assertIn(".score-bars{width:100%;min-width:0", html_text)
            self.assertNotIn('id="benchmark-chart"', html_text)


if __name__ == "__main__":
    unittest.main()
