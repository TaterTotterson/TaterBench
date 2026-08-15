from __future__ import annotations

import json
import statistics
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .engines import LlamaCppSession, MlxSession
from .engines.base import EngineSession
from .grading import grade_response
from .hardware import capture_hardware
from .prompts import build_messages
from .types import ModelCandidate, RunVariant
from .version import PROMPT_PROFILE_VERSION, SUITE_VERSION, __version__


ProgressCallback = Callable[[str], None]
SessionFactory = Callable[[ModelCandidate, RunVariant], EngineSession]


def _median(values: list[float]) -> float:
    usable = [float(value) for value in values if float(value) > 0]
    return statistics.median(usable) if usable else 0.0


def _public_model(model: ModelCandidate) -> dict[str, Any]:
    return {
        "id": model.id,
        "label": model.label,
        "provider": model.provider,
        "repo_id": model.repo_id,
        "filename": model.filename or model.model_path.name,
        "quantization": model.quantization,
        "supports_vision": model.supports_vision,
        "max_context_tokens": model.max_context_tokens,
        "model_bytes": _path_size(model.model_path),
    }


def _public_variant(variant: RunVariant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "speculative_method": variant.speculative_method,
        "draft_filename": variant.draft_path.name if variant.draft_path else "",
        "draft_tokens": variant.draft_tokens,
        "draft_bytes": _path_size(variant.draft_path) if variant.draft_path else 0,
    }


def _path_size(path: Path) -> int:
    try:
        if path.is_file():
            return int(path.stat().st_size)
        if path.is_dir():
            return sum(int(item.stat().st_size) for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0
    return 0


def _public_engine_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "engine",
        "server_version",
        "server_system_info",
        "target_gpu_layers",
        "draft_gpu_layers",
        "python_version",
        "mlx_version",
        "context_size",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


class BenchmarkRunner:
    def __init__(
        self,
        *,
        home: str | Path | None = None,
        context_size: int = 8192,
        repeats: int = 1,
        max_tokens: int | None = None,
        seed: int = 42,
        include_hostname: bool = False,
        progress: ProgressCallback | None = None,
        session_factory: SessionFactory | None = None,
    ):
        self.home = home
        self.context_size = max(2048, int(context_size))
        self.repeats = max(1, int(repeats))
        self.max_tokens = max_tokens
        self.seed = int(seed)
        self.hardware = capture_hardware(include_hostname=include_hostname)
        self.progress = progress or (lambda _message: None)
        self.session_factory = session_factory

    def _session(self, model: ModelCandidate, variant: RunVariant) -> EngineSession:
        if self.session_factory is not None:
            return self.session_factory(model, variant)
        session_type = LlamaCppSession if model.provider == "llama_cpp" else MlxSession
        return session_type(
            model,
            variant,
            home=self.home,
            context_size=self.context_size,
        )

    def run_one(
        self,
        model: ModelCandidate,
        variant: RunVariant,
        suite: dict[str, Any],
        *,
        limit: int = 0,
    ) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        wall_started = time.perf_counter()
        scenarios = list(suite.get("scenarios") or [])
        if limit > 0:
            scenarios = scenarios[:limit]
        run: dict[str, Any] = {
            "run_id": uuid.uuid4().hex,
            "status": "running",
            "started_at": started_at,
            "model": _public_model(model),
            "variant": _public_variant(variant),
            "suite": {
                "name": str(suite.get("name") or ""),
                "version": str(suite.get("version") or SUITE_VERSION),
                "scenario_count": len(scenarios),
                "repeats": self.repeats,
            },
            "configuration": {
                "context_size": self.context_size,
                "temperature": float(suite.get("temperature") or 0.0),
                "max_tokens": int(self.max_tokens or suite.get("max_tokens") or 320),
                "seed": self.seed,
                "prompt_profile": PROMPT_PROFILE_VERSION,
            },
            "scenario_results": [],
        }
        session = self._session(model, variant)
        try:
            self.progress(f"Loading {model.label} [{variant.name}]")
            session.start()
            run["engine"] = _public_engine_metadata(session.metadata)
            run["load_seconds"] = round(float(session.load_seconds), 6)
            self.progress("Warming up the model")
            session.generate(
                [
                    {"role": "system", "content": "Reply with exactly READY."},
                    {"role": "user", "content": "Ready check"},
                ],
                max_tokens=12,
                temperature=0.0,
                seed=self.seed,
            )
            for index, scenario in enumerate(scenarios, start=1):
                scenario_id = str(scenario.get("id") or f"scenario-{index}")
                self.progress(f"[{index}/{len(scenarios)}] {scenario_id}")
                attempts: list[dict[str, Any]] = []
                for repeat_index in range(self.repeats):
                    generation = session.generate(
                        build_messages(scenario),
                        max_tokens=int(self.max_tokens or suite.get("max_tokens") or 320),
                        temperature=float(suite.get("temperature") or 0.0),
                        seed=self.seed + repeat_index,
                    )
                    grade = grade_response(scenario, generation.text)
                    attempts.append(
                        {
                            "repeat": repeat_index + 1,
                            "response": generation.text,
                            "grade": grade,
                            "performance": generation.to_dict() | {"raw": generation.raw},
                        }
                    )
                score = sum(float(item["grade"]["score"]) for item in attempts) / len(attempts)
                run["scenario_results"].append(
                    {
                        "id": scenario_id,
                        "category": str(scenario.get("category") or "other"),
                        "kind": str(scenario.get("kind") or "chat"),
                        "weight": float(scenario.get("weight") or 1.0),
                        "score": round(score, 6),
                        "passed": score >= float(scenario.get("pass_threshold") or 0.8),
                        "attempts": attempts,
                    }
                )
            self._summarize_run(run)
            run["status"] = "complete"
        except Exception as exc:
            run["status"] = "failed"
            run["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            session.close()
            run["peak_rss_bytes"] = int(session.peak_rss_bytes)
            run["wall_seconds"] = round(time.perf_counter() - wall_started, 6)
            run["finished_at"] = datetime.now(timezone.utc).isoformat()
        return run

    @staticmethod
    def _summarize_run(run: dict[str, Any]) -> None:
        rows = run.get("scenario_results") or []
        total_weight = sum(float(row.get("weight") or 1.0) for row in rows)
        weighted_score = sum(float(row.get("score") or 0.0) * float(row.get("weight") or 1.0) for row in rows)
        category_weight: dict[str, float] = defaultdict(float)
        category_score: dict[str, float] = defaultdict(float)
        ttft: list[float] = []
        elapsed: list[float] = []
        generation_speed: list[float] = []
        prompt_speed: list[float] = []
        total_completion_tokens = 0
        for row in rows:
            category = str(row.get("category") or "other")
            weight = float(row.get("weight") or 1.0)
            category_weight[category] += weight
            category_score[category] += float(row.get("score") or 0.0) * weight
            for attempt in row.get("attempts") or []:
                perf = attempt.get("performance") or {}
                ttft.append(float(perf.get("ttft_seconds") or 0.0))
                elapsed.append(float(perf.get("elapsed_seconds") or 0.0))
                generation_speed.append(float(perf.get("completion_tokens_per_second") or 0.0))
                prompt_speed.append(float(perf.get("prompt_tokens_per_second") or 0.0))
                total_completion_tokens += int(perf.get("completion_tokens") or 0)
        run["accuracy"] = {
            "score": round((weighted_score / total_weight) * 100.0, 2) if total_weight else 0.0,
            "passed": sum(bool(row.get("passed")) for row in rows),
            "total": len(rows),
            "categories": {
                category: round((category_score[category] / weight) * 100.0, 2) if weight else 0.0
                for category, weight in sorted(category_weight.items())
            },
        }
        run["performance"] = {
            "median_ttft_seconds": round(_median(ttft), 6),
            "median_generation_tokens_per_second": round(_median(generation_speed), 4),
            "median_prompt_tokens_per_second": round(_median(prompt_speed), 4),
            "median_scenario_seconds": round(_median(elapsed), 6),
            "completion_tokens": total_completion_tokens,
        }

    def run_batch(
        self,
        jobs: list[tuple[ModelCandidate, RunVariant]],
        suite: dict[str, Any],
        *,
        limit: int = 0,
    ) -> dict[str, Any]:
        batch = {
            "schema_version": 1,
            "taterbench_version": __version__,
            "suite_version": str(suite.get("version") or SUITE_VERSION),
            "prompt_profile": PROMPT_PROFILE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hardware": self.hardware,
            "runs": [],
        }
        for index, (model, variant) in enumerate(jobs, start=1):
            self.progress(f"Run {index}/{len(jobs)}")
            batch["runs"].append(self.run_one(model, variant, suite, limit=limit))
        return batch


def save_batch(payload: dict[str, Any], output_dir: str | Path = "results") -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    hardware_id = str((payload.get("hardware") or {}).get("hardware_id") or "hardware")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"{stamp}-{hardware_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
