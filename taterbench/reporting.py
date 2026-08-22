from __future__ import annotations

import hashlib
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hardware import bytes_label


TATER_SCORE_WEIGHTS = {
    "accuracy": 90.0,
    "generation_speed": 7.0,
    "ttft": 2.0,
    "memory": 1.0,
}

TATER_ACCURACY_WEIGHTS = {
    "tool_accuracy": 25.0,
    "routing": 35.0,
    "spudex": 15.0,
    "synthesis": 10.0,
    "chat": 5.0,
}

FITNESS_LABELS = {
    "ready": "Tater Ready",
    "limited": "Limited — Not Ready",
    "mixed": "Mixed by Hardware",
    "not_fit": "Not Fit",
    "unrated": "Unrated",
}

FITNESS_RANKS = {
    "ready": 0,
    "limited": 1,
    "mixed": 2,
    "not_fit": 3,
    "unrated": 4,
}

FITNESS_SCORE_CAPS = {
    "limited": 79.9,
    "not_fit": 49.9,
}


def load_batches(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    batches: list[dict[str, Any]] = []
    if not root.is_dir():
        return batches
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            payload["_source"] = path.name
            batches.append(payload)
    return batches


def flatten_runs(batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in batches:
        hardware = batch.get("hardware") if isinstance(batch.get("hardware"), dict) else {}
        for run in batch.get("runs") or []:
            if not isinstance(run, dict):
                continue
            rows.append({**run, "hardware": hardware, "result_source": batch.get("_source", "")})
    rows.sort(key=lambda row: str(row.get("finished_at") or ""), reverse=True)
    return rows


def _speedup(run: dict[str, Any], rows: list[dict[str, Any]]) -> float:
    variant = str((run.get("variant") or {}).get("name") or "baseline")
    if variant == "baseline":
        return 0.0
    model_id = str((run.get("model") or {}).get("id") or "")
    hardware_id = str((run.get("hardware") or {}).get("hardware_id") or "")
    suite_version = str((run.get("suite") or {}).get("version") or "")
    current = float((run.get("performance") or {}).get("median_generation_tokens_per_second") or 0.0)
    for candidate in rows:
        if str((candidate.get("variant") or {}).get("name") or "") != "baseline":
            continue
        if str((candidate.get("model") or {}).get("id") or "") != model_id:
            continue
        if str((candidate.get("hardware") or {}).get("hardware_id") or "") != hardware_id:
            continue
        if str((candidate.get("suite") or {}).get("version") or "") != suite_version:
            continue
        baseline = float((candidate.get("performance") or {}).get("median_generation_tokens_per_second") or 0.0)
        return ((current / baseline) - 1.0) * 100.0 if current and baseline else 0.0
    return 0.0


def _comparison_key(run: dict[str, Any]) -> tuple[str, str, int, str]:
    hardware = run.get("hardware") or {}
    suite = run.get("suite") or {}
    configuration = run.get("configuration") or {}
    return (
        str(hardware.get("hardware_id") or ""),
        str(suite.get("version") or ""),
        int(configuration.get("context_size") or 0),
        str(configuration.get("prompt_profile") or ""),
    )


def _positive_metric(run: dict[str, Any], name: str) -> float:
    if name == "memory":
        value = run.get("peak_rss_bytes")
    else:
        value = (run.get("performance") or {}).get(name)
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def _bounded_percent(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, number))


def _accuracy_component(run: dict[str, Any]) -> float:
    accuracy = run.get("accuracy") if isinstance(run.get("accuracy"), dict) else {}
    categories = accuracy.get("categories")
    if isinstance(categories, dict) and all(name in categories for name in TATER_ACCURACY_WEIGHTS):
        return sum(
            (_bounded_percent(categories.get(name)) / 100.0) * weight
            for name, weight in TATER_ACCURACY_WEIGHTS.items()
        )
    return (_bounded_percent(accuracy.get("score")) / 100.0) * TATER_SCORE_WEIGHTS["accuracy"]


def _fitness_for_accuracy(accuracy: Any) -> dict[str, Any]:
    accuracy = accuracy if isinstance(accuracy, dict) else {}
    categories = accuracy.get("categories")
    if not isinstance(categories, dict) or not all(name in categories for name in TATER_ACCURACY_WEIGHTS):
        return {
            "status": "unrated",
            "label": FITNESS_LABELS["unrated"],
            "rank": FITNESS_RANKS["unrated"],
            "reasons": ["A complete five-category accuracy map is required."],
            "provisional": True,
        }

    overall = _bounded_percent(accuracy.get("score"))
    scores = {name: _bounded_percent(categories.get(name)) for name in TATER_ACCURACY_WEIGHTS}
    not_fit_reasons: list[str] = []
    if overall < 70.0:
        not_fit_reasons.append(f"Overall accuracy {overall:.1f}% is below 70%.")
    if scores["tool_accuracy"] < 70.0:
        not_fit_reasons.append(f"Tool accuracy {scores['tool_accuracy']:.1f}% is below 70%.")
    if scores["routing"] < 70.0:
        not_fit_reasons.append(f"Routing {scores['routing']:.1f}% is below 70%.")
    if not_fit_reasons:
        return {
            "status": "not_fit",
            "label": FITNESS_LABELS["not_fit"],
            "rank": FITNESS_RANKS["not_fit"],
            "reasons": not_fit_reasons,
            "provisional": True,
        }

    readiness_reasons: list[str] = []
    if overall < 85.0:
        readiness_reasons.append(f"Overall accuracy {overall:.1f}% is below 85%.")
    if scores["tool_accuracy"] < 85.0:
        readiness_reasons.append(f"Tool accuracy {scores['tool_accuracy']:.1f}% is below 85%.")
    if scores["routing"] < 80.0:
        readiness_reasons.append(f"Routing {scores['routing']:.1f}% is below 80%.")
    for name in ("spudex", "synthesis", "chat"):
        if scores[name] < 80.0:
            label = "Spudex" if name == "spudex" else name.title()
            readiness_reasons.append(f"{label} {scores[name]:.1f}% is below 80%.")
    if readiness_reasons:
        return {
            "status": "limited",
            "label": FITNESS_LABELS["limited"],
            "rank": FITNESS_RANKS["limited"],
            "reasons": readiness_reasons,
            "provisional": True,
        }
    return {
        "status": "ready",
        "label": FITNESS_LABELS["ready"],
        "rank": FITNESS_RANKS["ready"],
        "reasons": ["All Tater readiness gates pass."],
        "provisional": True,
    }


def _aggregate_fitness(rows: list[dict[str, Any]], *, device_weighted: bool) -> dict[str, Any]:
    child_fitness = [
        row.get("fitness") if isinstance(row.get("fitness"), dict) else _fitness_for_accuracy(row.get("accuracy"))
        for row in rows
    ]
    statuses = {str(fitness.get("status") or "unrated") for fitness in child_fitness}
    observation_count = sum(
        int(row.get("observation_count") or row.get("sample_count") or 1)
        for row in rows
    )
    if len(statuses) == 1:
        status = next(iter(statuses), "unrated")
        reasons = list(
            dict.fromkeys(
                str(reason)
                for fitness in child_fitness
                for reason in fitness.get("reasons") or []
                if str(reason)
            )
        )
        return {
            "status": status,
            "label": FITNESS_LABELS.get(status, FITNESS_LABELS["unrated"]),
            "rank": FITNESS_RANKS.get(status, FITNESS_RANKS["unrated"]),
            "reasons": reasons,
            "provisional": observation_count < 2,
        }

    reasons: list[str] = []
    if device_weighted:
        for row, fitness in zip(rows, child_fitness):
            device = str(row.get("hardware_label") or "Unknown device")
            reasons.append(f"{device}: {fitness.get('label') or FITNESS_LABELS['unrated']}.")
    else:
        counts = {
            status: sum(1 for fitness in child_fitness if fitness.get("status") == status)
            for status in FITNESS_RANKS
        }
        reasons.append(
            "Individual runs disagree: "
            + ", ".join(
                f"{count} {FITNESS_LABELS[status]}" for status, count in counts.items() if count
            )
            + "."
        )
    return {
        "status": "mixed",
        "label": FITNESS_LABELS["mixed"] if device_weighted else "Mixed Results",
        "rank": FITNESS_RANKS["mixed"],
        "reasons": reasons,
        "provisional": observation_count < 2,
    }


def _readiness_adjusted_score(raw_score: float, fitness: dict[str, Any]) -> float:
    cap = FITNESS_SCORE_CAPS.get(str(fitness.get("status") or ""))
    return min(raw_score, cap) if cap is not None else raw_score


def _tater_score(run: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    cohort = [
        candidate
        for candidate in rows
        if str(candidate.get("status") or "complete") == "complete"
        and _comparison_key(candidate) == _comparison_key(run)
    ]
    speed = _positive_metric(run, "median_generation_tokens_per_second")
    ttft = _positive_metric(run, "median_ttft_seconds")
    memory = _positive_metric(run, "memory")
    cohort_speeds = [_positive_metric(candidate, "median_generation_tokens_per_second") for candidate in cohort]
    cohort_ttfts = [_positive_metric(candidate, "median_ttft_seconds") for candidate in cohort]
    cohort_memory = [_positive_metric(candidate, "memory") for candidate in cohort]
    max_speed = max(cohort_speeds or [0.0])
    min_ttft = min((value for value in cohort_ttfts if value > 0), default=0.0)
    min_memory = min((value for value in cohort_memory if value > 0), default=0.0)
    components = {
        "accuracy": _accuracy_component(run),
        "generation_speed": (speed / max_speed) * TATER_SCORE_WEIGHTS["generation_speed"] if max_speed else 0.0,
        "ttft": (min_ttft / ttft) * TATER_SCORE_WEIGHTS["ttft"] if ttft and min_ttft else 0.0,
        "memory": (min_memory / memory) * TATER_SCORE_WEIGHTS["memory"] if memory and min_memory else 0.0,
    }
    rounded = {name: round(value, 2) for name, value in components.items()}
    rounded["total"] = round(sum(components.values()), 2)
    return rounded


def _mean(values: list[Any], *, positive: bool = False) -> float:
    usable: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if positive and number <= 0:
            continue
        usable.append(number)
    return statistics.fmean(usable) if usable else 0.0


def _result_key(run: dict[str, Any]) -> tuple[str, ...]:
    model = run.get("model") or {}
    variant = run.get("variant") or {}
    suite = run.get("suite") or {}
    configuration = run.get("configuration") or {}
    return (
        str(model.get("repo_id") or model.get("label") or model.get("id") or ""),
        str(model.get("filename") or ""),
        str(model.get("provider") or ""),
        str(model.get("quantization") or ""),
        str(variant.get("name") or "baseline"),
        str(suite.get("version") or ""),
        str(configuration.get("prompt_profile") or ""),
        str(configuration.get("context_size") or ""),
    )


def _hardware_label(hardware: dict[str, Any]) -> str:
    cpu = str(hardware.get("cpu") or hardware.get("architecture") or "Unknown device")
    memory = bytes_label(hardware.get("memory_bytes") or 0)
    return f"{cpu} · {memory}"


def _device_key(hardware: dict[str, Any]) -> str:
    """Group submissions by hardware class, not OS/Python patch-level fingerprints."""
    def integer(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    gpu_rows = []
    for gpu in hardware.get("gpus") or []:
        if not isinstance(gpu, dict):
            continue
        gpu_rows.append(
            {
                "name": str(gpu.get("name") or ""),
                "backend": str(gpu.get("backend") or ""),
                "vram": str(gpu.get("vram") or ""),
                "vram_mib": integer(gpu.get("vram_mib")),
            }
        )
    identity = {
        "os": str(hardware.get("os") or ""),
        "architecture": str(hardware.get("architecture") or ""),
        "cpu": str(hardware.get("cpu") or ""),
        "logical_cores": integer(hardware.get("logical_cores")),
        "physical_cores": integer(hardware.get("physical_cores")),
        "memory_bytes": integer(hardware.get("memory_bytes")),
        "gpus": gpu_rows,
    }
    if not any(value for key, value in identity.items() if key != "gpus") and not gpu_rows:
        return str(hardware.get("hardware_id") or "unknown")
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _graded_outcome_key(run: dict[str, Any]) -> str:
    scenarios: list[dict[str, Any]] = []
    for scenario in run.get("scenario_results") or []:
        if not isinstance(scenario, dict):
            continue
        attempts = [
            {
                "repeat": attempt.get("repeat"),
                "grade": attempt.get("grade"),
            }
            for attempt in scenario.get("attempts") or []
            if isinstance(attempt, dict)
        ]
        attempts.sort(key=lambda attempt: str(attempt.get("repeat") or ""))
        scenarios.append(
            {
                "id": scenario.get("id"),
                "category": scenario.get("category"),
                "kind": scenario.get("kind"),
                "passed": scenario.get("passed"),
                "score": scenario.get("score"),
                "weight": scenario.get("weight"),
                "attempts": attempts,
            }
        )
    scenarios.sort(key=lambda scenario: str(scenario.get("id") or ""))
    outcome = {
        "status": str(run.get("status") or "complete"),
        "accuracy": run.get("accuracy") or {},
        "scenarios": scenarios,
        "error": run.get("error"),
    }
    encoded = json.dumps(outcome, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def deduplicate_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest run for identical graded outcomes on the same hardware type."""
    seen: dict[tuple[str, tuple[str, ...], str], int] = {}
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = (_device_key(row.get("hardware") or {}), _result_key(row), _graded_outcome_key(row))
        if key in seen:
            index = seen[key]
            unique[index]["observation_count"] += int(row.get("observation_count") or 1)
            continue
        item = dict(row)
        item["observation_count"] = int(row.get("observation_count") or 1)
        seen[key] = len(unique)
        unique.append(item)
    return unique


def _average_categories(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = {
        str(name)
        for row in rows
        for name in ((row.get("accuracy") or {}).get("categories") or {})
    }
    return {
        name: round(
            _mean([((row.get("accuracy") or {}).get("categories") or {}).get(name) for row in rows]),
            2,
        )
        for name in sorted(names)
    }


def _average_result(rows: list[dict[str, Any]], *, device_weighted: bool) -> dict[str, Any]:
    first = rows[0]
    performance_rows = [row.get("performance") or {} for row in rows]
    component_names = (*TATER_SCORE_WEIGHTS, "readiness_adjustment")
    run_ids: list[str] = []
    result_sources: list[str] = []
    for row in rows:
        nested_ids = row.get("run_ids") if isinstance(row.get("run_ids"), list) else []
        candidates = nested_ids or [str(row.get("run_id") or "")]
        for run_id in candidates:
            token = str(run_id or "")
            if token and token not in run_ids:
                run_ids.append(token)
        nested_sources = row.get("result_sources") if isinstance(row.get("result_sources"), list) else []
        source_candidates = nested_sources or [str(row.get("result_source") or "")]
        for source in source_candidates:
            token = str(source or "")
            if token and token not in result_sources:
                result_sources.append(token)
    hardware_ids = sorted(
        {
            str(value)
            for row in rows
            for value in (
                row.get("hardware_ids")
                if isinstance(row.get("hardware_ids"), list)
                else [str((row.get("hardware") or {}).get("hardware_id") or "")]
            )
            if str(value)
        }
    )
    sample_count = sum(int(row.get("sample_count") or 1) for row in rows)
    observation_count = sum(
        int(row.get("observation_count") or row.get("sample_count") or 1)
        for row in rows
    )
    output = {
        "status": "complete",
        "aggregate": True,
        "device_weighted": bool(device_weighted),
        "model": dict(first.get("model") or {}),
        "variant": dict(first.get("variant") or {}),
        "engine": dict(first.get("engine") or {}),
        "suite": dict(first.get("suite") or {}),
        "configuration": dict(first.get("configuration") or {}),
        "tater_score": round(_mean([row.get("tater_score") for row in rows]), 2),
        "raw_tater_score": round(
            _mean([row.get("raw_tater_score", row.get("tater_score")) for row in rows]),
            2,
        ),
        "fitness": _aggregate_fitness(rows, device_weighted=device_weighted),
        "tater_score_components": {
            name: round(_mean([(row.get("tater_score_components") or {}).get(name) for row in rows]), 2)
            for name in component_names
        },
        "accuracy": {
            "score": round(_mean([(row.get("accuracy") or {}).get("score") for row in rows]), 2),
            "categories": _average_categories(rows),
        },
        "performance": {
            "median_generation_tokens_per_second": round(
                _mean([row.get("median_generation_tokens_per_second") for row in performance_rows], positive=True),
                4,
            ),
            "median_prompt_tokens_per_second": round(
                _mean([row.get("median_prompt_tokens_per_second") for row in performance_rows], positive=True),
                4,
            ),
            "median_ttft_seconds": round(
                _mean([row.get("median_ttft_seconds") for row in performance_rows], positive=True),
                6,
            ),
            "median_scenario_seconds": round(
                _mean([row.get("median_scenario_seconds") for row in performance_rows], positive=True),
                6,
            ),
        },
        "load_seconds": round(_mean([row.get("load_seconds") for row in rows], positive=True), 6),
        "peak_rss_bytes": round(_mean([row.get("peak_rss_bytes") for row in rows], positive=True)),
        "speedup_percent": round(_mean([row.get("speedup_percent") for row in rows]), 2),
        "sample_count": sample_count,
        "observation_count": observation_count,
        "device_count": len(rows) if device_weighted else 1,
        "hardware_ids": hardware_ids,
        "run_ids": run_ids,
        "result_sources": result_sources,
        "finished_at": max((str(row.get("finished_at") or "") for row in rows), default=""),
    }
    return output


def _leaderboards(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def ranking_key(row: dict[str, Any]) -> tuple[float, int, float, tuple[str, ...]]:
        fitness = row.get("fitness") or {}
        return (
            -float(row.get("tater_score") or 0.0),
            int(fitness.get("rank") or 0),
            -float((row.get("accuracy") or {}).get("score") or 0.0),
            _result_key(row),
        )

    completed = [row for row in rows if str(row.get("status") or "complete") == "complete"]
    by_device_group: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    hardware_by_device: dict[str, dict[str, Any]] = {}
    for row in completed:
        hardware = row.get("hardware") or {}
        device_id = _device_key(hardware)
        hardware_by_device.setdefault(device_id, dict(hardware))
        by_device_group[(device_id, _result_key(row))].append(row)

    device_averages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (device_id, _key), group_rows in by_device_group.items():
        averaged = _average_result(group_rows, device_weighted=False)
        averaged["device_id"] = device_id
        averaged["hardware"] = hardware_by_device[device_id]
        averaged["hardware_label"] = _hardware_label(hardware_by_device[device_id])
        device_averages[device_id].append(averaged)

    all_device_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for device_rows in device_averages.values():
        for row in device_rows:
            all_device_groups[_result_key(row)].append(row)
    overall = [_average_result(group_rows, device_weighted=True) for group_rows in all_device_groups.values()]
    overall.sort(key=ranking_key)

    devices: list[dict[str, Any]] = []
    for device_id, device_rows in sorted(
        device_averages.items(), key=lambda item: _hardware_label(hardware_by_device[item[0]]).lower()
    ):
        device_rows.sort(key=ranking_key)
        profile_ids = sorted(
            {
                str(profile_id)
                for row in device_rows
                for profile_id in row.get("hardware_ids") or []
                if str(profile_id)
            }
        )
        devices.append(
            {
                "device_id": device_id,
                "hardware_ids": profile_ids,
                "hardware_profile_count": len(profile_ids),
                "label": _hardware_label(hardware_by_device[device_id]),
                "hardware": hardware_by_device[device_id],
                "run_count": sum(int(row.get("sample_count") or 0) for row in device_rows),
                "observation_count": sum(
                    int(row.get("observation_count") or row.get("sample_count") or 0)
                    for row in device_rows
                ),
                "leaderboard": device_rows,
            }
        )
    return overall, devices


def aggregate_payload(batches: list[dict[str, Any]]) -> dict[str, Any]:
    raw_rows = flatten_runs(batches)
    rows = deduplicate_runs(raw_rows)
    public_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["speedup_percent"] = round(_speedup(row, rows), 2)
        score = _tater_score(row, rows)
        raw_score = score["total"]
        fitness = _fitness_for_accuracy(item.get("accuracy"))
        fitness["provisional"] = int(item.get("observation_count") or 1) < 2
        final_score = round(_readiness_adjusted_score(raw_score, fitness), 2)
        item["raw_tater_score"] = raw_score
        item["tater_score"] = final_score
        item["tater_score_components"] = {
            **{name: value for name, value in score.items() if name != "total"},
            "readiness_adjustment": round(final_score - raw_score, 2),
        }
        item["fitness"] = fitness
        public_rows.append(item)
    leaderboard, devices = _leaderboards(public_rows)
    timestamps = [str(batch.get("created_at") or "") for batch in batches if str(batch.get("created_at") or "")]
    return {
        "generated_at": max(timestamps) if timestamps else "",
        "run_count": len(public_rows),
        "raw_run_count": len(raw_rows),
        "duplicate_run_count": len(raw_rows) - len(public_rows),
        "model_result_count": len(leaderboard),
        "device_count": len(devices),
        "leaderboard": leaderboard,
        "devices": devices,
        "runs": public_rows,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fitness_display_label(fitness: Any) -> str:
    fitness = fitness if isinstance(fitness, dict) else {}
    label = str(fitness.get("label") or FITNESS_LABELS["unrated"])
    return f"{label} — Provisional" if fitness.get("provisional") else label


def render_markdown(aggregate: dict[str, Any]) -> str:
    rows = list(aggregate.get("leaderboard") or [])
    lines = [
        "# Tater Bench Results",
        "",
        "Cross-device average accuracy and real-world speed for models running through Tater's llama.cpp and MLX engines.",
        "",
        f"> {int(aggregate.get('duplicate_run_count') or 0)} duplicate runs with identical graded outcomes on the same hardware type are omitted, keeping the newest representative. Distinct outcomes are averaged per hardware type before hardware-type averages are combined.",
        "",
        "| Model | Engine | Mode | Fitness | Avg Tater Score | Avg Accuracy | Avg Gen tok/s | Avg TTFT | Devices | Unique Runs | Observations | Suite |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in rows:
        model = run.get("model") or {}
        variant = run.get("variant") or {}
        performance = run.get("performance") or {}
        engine = run.get("engine") or {}
        mode = str(variant.get("name") or "baseline").upper()
        speedup = float(run.get("speedup_percent") or 0.0)
        if mode != "BASELINE" and speedup:
            mode += f" ({speedup:+.1f}%)"
        label = str(model.get("repo_id") or model.get("label") or model.get("filename") or "Unknown").replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(engine.get("engine") or model.get("provider") or ""),
                    mode,
                    _fitness_display_label(run.get("fitness")),
                    _fmt(run.get("tater_score")),
                    _fmt((run.get("accuracy") or {}).get("score")) + "%",
                    _fmt(performance.get("median_generation_tokens_per_second")),
                    _fmt(performance.get("median_ttft_seconds")) + "s",
                    str(int(run.get("device_count") or 0)),
                    str(int(run.get("sample_count") or 0)),
                    str(int(run.get("observation_count") or run.get("sample_count") or 0)),
                    str((run.get("suite") or {}).get("version") or ""),
                ]
            )
            + " |"
        )
    if not rows:
        lines.extend(["| _No published benchmark runs yet_ | | | | | | | | | | | |", ""])
    lines.extend(
        [
            "",
            "## Devices",
            "",
            "| Device | Unique published runs | Observations |",
            "|---|---:|---:|",
        ]
    )
    for device in aggregate.get("devices") or []:
        device_label = str(device.get("label") or "Unknown").replace("|", "\\|")
        lines.append(
            f"| {device_label} | {int(device.get('run_count') or 0)} | "
            f"{int(device.get('observation_count') or device.get('run_count') or 0)} |"
        )
    if not aggregate.get("devices"):
        lines.append("| _No devices published yet_ | | |")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "Tater Score starts with a 100-point raw formula: 90 points for category-weighted accuracy (35 Astraeus routing and tool selection, 25 Thanatos tool execution, 15 Spudex, 10 synthesis, and 5 chat), 7 for generation speed, 2 for time to first token, and 1 for peak-memory efficiency. Limited results are capped at 79.9 and Not Fit results at 49.9 before repeated runs and hardware results are averaged. Performance and efficiency are normalized within matching hardware, suite, context, and prompt profile.",
            "",
            "Fitness is a separate reliability verdict. Tater Ready requires at least 85% overall accuracy, 85% tool accuracy, 80% routing, and 80% in every remaining category. Limited results miss a readiness gate; Not Fit results have overall, tool, or routing accuracy below 70%. Aggregate labels are unanimous: a model is Tater Ready or Not Fit only when every underlying result has that verdict, while device disagreements are labeled Mixed by Hardware. Results based on fewer than two observations are provisional.",
            "",
            "Tater Bench uses a versioned, frozen synthetic Tater runtime for routing, strict tool-call, synthesis, chat, and Spudex scenarios. Runs are considered duplicates only when the hardware type, model configuration, accuracy summary, and every graded scenario outcome match; timing variation is ignored. The newest duplicate is published while the original batch files remain untouched. Hidden duplicates still count as observations for reproducibility and provisional-status checks. Outcome-distinct runs are averaged by hardware type before every represented hardware type receives equal weight. Each result records the model, engine, speculative mode, suite version, prompt profile, hardware fingerprint, context, and raw per-scenario response.",
            "",
            "MTP, DFlash, and DSpark percentages compare generation speed against the matching baseline run on the same hardware and suite.",
            "",
        ]
    )
    return "\n".join(lines)


def _chart_label(repo_id: str, provider: str) -> str:
    identity = repo_id.lower()
    if "qwen3.8-27b" in identity:
        return "Qwen 3.8 27B"
    if "qwen3.6-35b" in identity:
        return "Qwen 3.6 35B A3B"
    if "nemotron-3.5" in identity:
        return "Nemotron 3.5 30B A3B"
    if "gemma-4-26b" in identity:
        return "Gemma 4 26B" + (" MLX" if provider == "mlx_lm" else "")
    return repo_id.rsplit("/", 1)[-1][:36]


def render_html(aggregate: dict[str, Any]) -> str:
    raw_runs = list(aggregate.get("runs") or [])
    run_lookup = {str(run.get("run_id") or ""): run for run in raw_runs if str(run.get("run_id") or "")}
    scopes: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("overall", "All Devices", list(aggregate.get("leaderboard") or []))
    ]
    scopes.extend(
        (
            f"device-{device.get('device_id')}",
            str(device.get("label") or "Unknown device"),
            list(device.get("leaderboard") or []),
        )
        for device in aggregate.get("devices") or []
    )
    present_modes: set[str] = set()
    present_engines: set[str] = set()
    cards: list[str] = []
    leaderboard_items: list[str] = []

    def run_details(result: dict[str, Any]) -> str:
        ids = result.get("run_ids") if isinstance(result.get("run_ids"), list) else []
        if not ids and str(result.get("run_id") or ""):
            ids = [str(result.get("run_id"))]
        detail_runs = [run_lookup[run_id] for run_id in ids if run_id in run_lookup]
        if not detail_runs:
            detail_runs = [result]
        rows_html: list[str] = []
        for detail in sorted(detail_runs, key=lambda row: str(row.get("finished_at") or ""), reverse=True):
            hardware = detail.get("hardware") or {}
            performance = detail.get("performance") or {}
            rows_html.append(
                "<tr>"
                f"<td>{html.escape(_hardware_label(hardware))}</td>"
                f"<td>{html.escape(str(detail.get('finished_at') or detail.get('started_at') or ''))}</td>"
                f"<td>{int(detail.get('observation_count') or 1)}</td>"
                f"<td>{html.escape(_fitness_display_label(detail.get('fitness')))}</td>"
                f"<td>{_fmt(detail.get('tater_score'), 1)}</td>"
                f"<td>{_fmt(detail.get('raw_tater_score', detail.get('tater_score')), 1)}</td>"
                f"<td>{_fmt((detail.get('accuracy') or {}).get('score'), 1)}%</td>"
                f"<td>{_fmt(performance.get('median_generation_tokens_per_second'))}</td>"
                f"<td>{_fmt(performance.get('median_ttft_seconds'))}s</td>"
                "</tr>"
            )
        count = len(detail_runs)
        return (
            f'<details class="run-details"><summary>See {count} unique run{"s" if count != 1 else ""}</summary>'
            '<div class="table-wrap"><table><thead><tr><th>Device</th><th>Finished</th><th>Observations</th><th>Fitness</th><th>Final score</th><th>Raw formula</th><th>Accuracy</th><th>tok/s</th><th>TTFT</th></tr></thead>'
            f'<tbody>{"".join(rows_html)}</tbody></table></div></details>'
        )

    for scope_index, (scope_id, _scope_label, rows) in enumerate(scopes):
        ordered = sorted(
            rows,
            key=lambda run: (
                -float(run.get("tater_score") or 0.0),
                int((run.get("fitness") or {}).get("rank") or 0),
                -float((run.get("accuracy") or {}).get("score") or 0.0),
                str((run.get("model") or {}).get("repo_id") or ""),
            ),
        )
        for index, run in enumerate(ordered, start=1):
            model = run.get("model") or {}
            variant = run.get("variant") or {}
            performance = run.get("performance") or {}
            engine = run.get("engine") or {}
            categories = (run.get("accuracy") or {}).get("categories") or {}
            provider = str(model.get("provider") or "")
            mode = str(variant.get("name") or "baseline").lower()
            repo_id = str(model.get("repo_id") or model.get("label") or "Unknown model")
            tater_score = float(run.get("tater_score") or 0.0)
            raw_tater_score = float(run.get("raw_tater_score", tater_score) or 0.0)
            accuracy = float((run.get("accuracy") or {}).get("score") or 0.0)
            speed = float(performance.get("median_generation_tokens_per_second") or 0.0)
            ttft = float(performance.get("median_ttft_seconds") or 0.0)
            memory = float(run.get("peak_rss_bytes") or 0.0)
            samples = int(run.get("sample_count") or 1)
            observations = int(run.get("observation_count") or samples)
            devices = int(run.get("device_count") or (1 if run.get("hardware") else 0))
            card_id = f"result-{scope_index}-{index}"
            label = _chart_label(repo_id, provider)
            score_components = run.get("tater_score_components") or {}
            fitness = run.get("fitness") or {}
            fitness_status = str(fitness.get("status") or "unrated")
            fitness_rank = int(fitness.get("rank", FITNESS_RANKS["unrated"]))
            fitness_label = _fitness_display_label(fitness)
            fitness_reason = " ".join(str(reason) for reason in fitness.get("reasons") or [])
            data = (
                f'data-scope="{html.escape(scope_id)}" data-engine="{html.escape(provider)}" '
                f'data-mode="{html.escape(mode)}" data-score="{tater_score}" data-accuracy="{accuracy}" '
                f'data-raw-score="{raw_tater_score}" data-speed="{speed}" data-ttft="{ttft}" data-memory="{memory}" '
                f'data-model="{html.escape(label.lower())}" data-samples="{observations}" '
                f'data-fitness="{html.escape(fitness_status)}" data-fitness-rank="{fitness_rank}"'
            )
            category_html = "".join(
                f'<span><b>{html.escape(str(name).replace("_", " ").title())}</b>{_fmt(score)}%</span>'
                for name, score in categories.items()
            )
            average_label = "Cross-device average" if scope_id == "overall" else "Device average"
            cards.append(
                f'''<article id="{card_id}" class="result-card" {data}>
                  <div class="card-head"><div><p class="eyebrow"><span class="detail-rank">#{index}</span> · {html.escape(average_label)} · {html.escape(str(engine.get("engine") or provider))} · {html.escape(mode.upper())}</p><h2>{html.escape(repo_id)}</h2><p>{html.escape(str(model.get("filename") or ""))}</p></div><div class="score">{_fmt(tater_score, 1)}<small>Tater score</small></div></div>
                  <div class="fitness-verdict fitness-{html.escape(fitness_status)}"><b>{html.escape(fitness_label)}</b><span>{html.escape(fitness_reason)}</span></div>
                  <div class="metrics"><span><b>{_fmt(raw_tater_score, 1)}</b> raw formula</span><span><b>{_fmt(accuracy, 1)}%</b> accuracy</span><span><b>{_fmt(speed)}</b> tok/s</span><span><b>{_fmt(ttft)}s</b> TTFT</span><span><b>{bytes_label(memory)}</b> peak RSS</span><span><b>{devices}</b> device{"s" if devices != 1 else ""}</span><span><b>{samples}</b> unique run{"s" if samples != 1 else ""}</span><span><b>{observations}</b> observation{"s" if observations != 1 else ""}</span></div>
                  <div class="categories">{category_html}</div>
                  {run_details(run)}
                  <footer>Suite {html.escape(str((run.get("suite") or {}).get("version") or ""))} · Prompt {html.escape(str((run.get("configuration") or {}).get("prompt_profile") or ""))} · Score mix: {_fmt(score_components.get("accuracy"), 1)} accuracy + {_fmt(score_components.get("generation_speed"), 1)} speed + {_fmt(score_components.get("ttft"), 1)} TTFT + {_fmt(score_components.get("memory"), 1)} memory + {_fmt(score_components.get("readiness_adjustment"), 1)} readiness gate</footer>
                </article>'''
            )
            if str(run.get("status") or "complete") == "complete" and accuracy > 0 and speed > 0:
                engine_label = str(engine.get("engine") or provider)
                leaderboard_items.append(
                    f'''<a class="score-entry" href="#{card_id}" {data} data-label="{html.escape(label)}, {tater_score:.1f} Tater Score" style="--bar-score:{tater_score}%;--bar-color:var(--mode-{html.escape(mode)})" aria-label="Rank {index}: {html.escape(label)}, {tater_score:.1f} Tater Score">
                      <span class="score-rank">#{index}</span><span class="score-bar-stage"><span class="score-bar"><b>{tater_score:.1f}</b></span></span><span class="score-copy"><strong>{html.escape(label)}</strong><small>{html.escape(fitness_label)} · {html.escape(engine_label)} · {html.escape(mode.upper())} · {samples} unique / {observations} observed</small></span>
                    </a>'''
                )
                present_modes.add(mode)
                present_engines.add(provider)

    cards_html = "\n".join(cards) or '<div class="empty">No benchmark results have been published yet.</div>'
    leaderboard_html = "\n".join(leaderboard_items) or '<div class="empty">No completed benchmark runs yet.</div>'
    mode_labels = {"baseline": "Base", "mtp": "MTP", "dflash": "DFlash", "dspark": "DSpark"}
    engine_labels = {"llama_cpp": "llama.cpp", "mlx_lm": "MLX"}
    scope_tabs = "".join(
        f'<button class="{"active" if index == 0 else ""}" type="button" aria-pressed="{"true" if index == 0 else "false"}" data-scope-button="{html.escape(scope_id)}">{html.escape(label)}</button>'
        for index, (scope_id, label, _rows) in enumerate(scopes)
    )
    filter_parts = ['<button class="active" type="button" aria-pressed="true" data-filter="all">All modes & engines</button>']
    filter_parts.extend(
        f'<button type="button" aria-pressed="false" data-filter="{html.escape(mode)}">{label}</button>'
        for mode, label in mode_labels.items()
        if mode in present_modes
    )
    filter_parts.extend(
        f'<button type="button" aria-pressed="false" data-filter="{html.escape(engine)}">{label}</button>'
        for engine, label in engine_labels.items()
        if engine in present_engines
    )
    filters_html = "".join(filter_parts)
    template = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tater Bench</title>
<meta name="description" content="Cross-device ranked accuracy and speed results for local models running through Tater.">
<style>
:root{--ink:#f7f7f5;--muted:#aaa9a2;--line:#31302c;--orange:#ff7a18;--orange2:#ffb13b;--blue:#42b8ff;--purple:#b38cff;--green:#56d68b;--gold:#ffd166;--grid:#2b2a27;--mode-baseline:var(--blue);--mode-mtp:var(--orange);--mode-dflash:var(--purple);--mode-dspark:var(--green)}
*{box-sizing:border-box}[hidden]{display:none!important}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 75% 0,#2b1608 0,transparent 34%),#090909;color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}.shell{width:min(1240px,calc(100% - 32px));margin:auto;padding:34px 0 70px}header{display:grid;grid-template-columns:1fr auto;align-items:center;gap:28px;border-bottom:1px solid var(--line);padding-bottom:28px}.brand{display:flex;align-items:center;gap:18px}.brand img{width:88px;height:88px;border-radius:50%;border:2px solid var(--orange);box-shadow:0 0 34px #ff7a1840}.eyebrow{margin:0 0 5px;color:var(--orange2);font-size:12px;font-weight:800;letter-spacing:.13em;text-transform:uppercase}h1{margin:0;font-size:clamp(36px,7vw,72px);line-height:.95;letter-spacing:-.055em}.lede{max-width:700px;color:var(--muted);font-size:17px}.summary{text-align:right}.summary b{display:block;color:var(--orange);font-size:34px}.summary span{display:block;color:var(--muted);font-size:12px}.control-block{margin:22px 0}.control-label{display:block;margin-bottom:8px;color:#777;font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}.controls{display:flex;gap:9px;flex-wrap:wrap}.toolbar{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:24px}.toolbar .control-block{margin:0}.sort-label{display:grid;gap:8px;color:#777;font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}button,select{border:1px solid var(--line);border-radius:999px;background:#111;color:var(--muted);padding:9px 15px;cursor:pointer}select{min-width:180px;border-radius:12px;color:var(--ink)}button.active,button:hover{border-color:var(--orange);color:#fff;background:#271509}.leaderboard{margin:10px 0 30px;padding:24px;background:linear-gradient(145deg,#171717,#0e0e0e);border:1px solid var(--line);border-radius:24px}.chart-heading{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:14px}.chart-heading h2{margin:0;font-size:clamp(23px,4vw,34px);letter-spacing:-.025em}.chart-heading p{margin:4px 0 0;color:var(--muted)}.score-formula{max-width:630px;text-align:right;color:var(--muted);font-size:12px}.score-formula b{color:var(--ink)}.score-plot{position:relative;padding-left:44px;margin-top:22px;overflow-x:auto}.score-grid{position:absolute;z-index:0;left:44px;right:0;top:0;height:320px;min-width:680px}.score-grid span{position:absolute;left:0;right:0;bottom:var(--grid-position);height:1px;background:var(--grid)}.score-grid i{position:absolute;right:calc(100% + 10px);top:-8px;color:var(--muted);font-size:10px;font-style:normal}.score-bars{position:relative;z-index:1;display:grid;grid-template-columns:repeat(var(--bar-count),minmax(82px,1fr));gap:10px;min-width:max(680px,calc(var(--bar-count) * 92px));min-height:430px}.score-entry{position:relative;display:grid;grid-template-rows:320px auto;text-decoration:none;color:var(--ink);min-width:0}.score-rank{position:absolute;z-index:2;top:6px;left:50%;transform:translateX(-50%);color:#ffffffb8;font-size:11px;font-weight:900}.score-bar-stage{display:flex;align-items:flex-end;justify-content:center;height:320px}.score-bar{display:flex;align-items:flex-start;justify-content:center;width:min(64px,78%);height:var(--bar-score);min-height:34px;padding-top:13px;border-radius:7px 7px 2px 2px;background:linear-gradient(180deg,color-mix(in srgb,var(--bar-color),white 13%),var(--bar-color));box-shadow:0 8px 30px color-mix(in srgb,var(--bar-color),transparent 75%);transition:filter .18s,transform .18s}.score-bar b{font-size:15px;color:#fff;text-shadow:0 1px 4px #0008}.score-copy{display:block;padding:12px 3px 0;text-align:center;line-height:1.25}.score-copy strong{display:block;font-size:12px;overflow-wrap:anywhere}.score-copy small{display:block;margin-top:5px;color:var(--muted);font-size:10px}.score-entry:hover .score-bar,.score-entry:focus .score-bar{filter:brightness(1.18);transform:translateY(-3px)}.score-entry:focus{outline:2px solid var(--orange2);outline-offset:3px;border-radius:5px}.score-entry.is-leader .score-bar{background:linear-gradient(180deg,#ffe5a3,var(--gold) 34%,var(--orange));box-shadow:0 0 32px #ffb13b66}.score-entry.is-leader .score-rank{color:var(--gold)}.score-note{margin:10px 0 0;color:var(--muted);font-size:12px}.results-heading{margin:8px 0 14px;font-size:25px}main{display:grid;gap:16px;min-width:0}.result-card{min-width:0;scroll-margin-top:16px;background:linear-gradient(145deg,#191919,#101010);border:1px solid var(--line);border-radius:22px;padding:22px;box-shadow:0 18px 60px #0006}.result-card:target{border-color:var(--orange);box-shadow:0 0 0 1px var(--orange),0 18px 60px #0006}.card-head{display:flex;justify-content:space-between;gap:18px;min-width:0}.card-head>div{min-width:0}.result-card h2{margin:0;font-size:21px;overflow-wrap:anywhere}.card-head p:not(.eyebrow){margin:4px 0;color:var(--muted);overflow-wrap:anywhere}.score{min-width:94px;text-align:center;color:var(--orange);font-size:36px;font-weight:900;line-height:1}.score small{display:block;color:var(--muted);font-size:10px;letter-spacing:.12em;text-transform:uppercase;margin-top:7px}.fitness-verdict{display:flex;align-items:center;gap:10px;margin-top:17px;padding:10px 12px;border:1px solid;border-radius:12px;background:#0c0c0c}.fitness-verdict b{flex:0 0 auto}.fitness-verdict span{color:var(--muted);font-size:12px}.fitness-ready{border-color:#317a4d}.fitness-ready b{color:#56d68b}.fitness-limited{border-color:#80672b}.fitness-limited b{color:#ffd166}.fitness-mixed{border-color:#2e6685}.fitness-mixed b{color:#42b8ff}.fitness-not_fit{border-color:#803b35}.fitness-not_fit b{color:#ff8278}.fitness-unrated{border-color:#494949}.fitness-unrated b{color:#aaa9a2}.metrics,.categories{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:10px;margin-top:19px;min-width:0}.metrics span,.categories span{min-width:0;background:#0c0c0c;border:1px solid #272622;border-radius:13px;padding:11px;color:var(--muted)}.metrics b,.categories b{display:block;color:var(--ink);font-size:15px}.run-details{margin-top:18px;border:1px solid #292824;border-radius:14px;background:#0b0b0b}.run-details summary{padding:12px 14px;cursor:pointer;color:var(--orange2);font-weight:750}.table-wrap{overflow-x:auto;padding:0 14px 14px}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:9px 12px;border-bottom:1px solid #24231f;text-align:left;font-size:12px}th{color:#777;text-transform:uppercase;letter-spacing:.08em}td{color:var(--muted)}footer{margin-top:18px;color:#777;font-size:12px}.empty{padding:70px;text-align:center;border:1px dashed var(--line);border-radius:20px;color:var(--muted)}
@media(max-width:820px){header{grid-template-columns:1fr}.summary{text-align:left}.leaderboard{padding:18px}.chart-heading{align-items:flex-start;flex-direction:column}.score-formula{text-align:left}.score-plot{padding-left:0}.score-grid{display:none}.score-bars{display:flex;flex-direction:column;gap:13px;min-height:0}.score-entry{display:grid;grid-template-columns:minmax(106px,1fr) minmax(140px,2fr);grid-template-rows:auto;height:auto;gap:12px;align-items:center}.score-copy{position:relative;grid-column:1;grid-row:1;padding:0 0 0 27px;text-align:left}.score-bar-stage{grid-column:2;grid-row:1;height:30px;justify-content:flex-start;background:#090909;border:1px solid var(--line);border-radius:6px;overflow:hidden}.score-bar{align-items:center;justify-content:flex-end;width:var(--bar-score);height:100%;min-width:48px;padding:0 8px;border-radius:4px}.score-rank{top:0;left:0;right:auto;transform:none;color:#ffffff88}.score-entry.is-leader .score-rank{color:var(--gold)}.brand img{width:68px;height:68px}.card-head{align-items:flex-start}}@media(max-width:430px){.shell{width:min(100% - 20px,1180px)}.leaderboard{padding:14px}.score-entry{grid-template-columns:minmax(94px,1fr) minmax(128px,1.55fr);gap:9px}.card-head{flex-direction:column}.score{text-align:left}.metrics,.categories{grid-template-columns:1fr 1fr}}
body{overflow-x:hidden}
@media(max-width:820px){.shell{width:calc(100% - 24px);padding:22px 0 48px}header{gap:18px;padding-bottom:22px}.summary b{display:inline;margin-right:7px;font-size:28px}.summary span{display:inline}.brand{align-items:flex-start;gap:13px}.brand img{width:64px;height:64px;flex:0 0 auto}.lede{font-size:15px}.controls{flex-wrap:nowrap;overflow-x:auto;padding:0 2px 5px;overscroll-behavior-inline:contain;scroll-snap-type:x proximity;-webkit-overflow-scrolling:touch}.controls button{flex:0 0 auto;min-height:44px;scroll-snap-align:start}.toolbar{align-items:stretch;flex-direction:column;margin-bottom:18px}.sort-label,select{width:100%}.leaderboard{padding:17px;margin-bottom:22px;border-radius:18px}.chart-heading{gap:9px}.score-plot{padding-left:0;overflow:visible}.score-bars{width:100%;min-width:0;gap:11px}.score-entry{grid-template-columns:minmax(0,1.05fr) minmax(0,1.8fr);width:100%;gap:10px}.score-copy{min-width:0;padding-left:25px}.score-copy small{overflow-wrap:anywhere}.score-bar-stage{min-width:0;height:34px}.result-card{padding:18px;border-radius:18px}.result-card h2,.card-head p{overflow-wrap:anywhere}.empty{padding:40px 16px}}
@media(max-width:560px){.shell{width:calc(100% - 18px)}h1{font-size:39px}.control-block{margin:18px 0}button,select{min-height:44px}.leaderboard{padding:14px}.chart-heading h2{font-size:24px}.score-entry{grid-template-columns:minmax(0,1.1fr) minmax(0,1.5fr);gap:8px}.score-copy strong{font-size:11px}.score-copy small{font-size:9px}.results-heading{font-size:22px}.result-card{padding:15px}.card-head{flex-direction:column;gap:12px}.result-card h2{font-size:18px}.score{min-width:0;text-align:left;font-size:32px}.metrics,.categories{grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}.metrics span,.categories span{padding:9px;font-size:12px}footer{margin-top:14px;overflow-wrap:anywhere}}
@media(max-width:360px){.brand img{width:56px;height:56px}h1{font-size:35px}.score-entry{grid-template-columns:minmax(0,1fr) minmax(110px,1.35fr)}.metrics,.categories{grid-template-columns:1fr}}
</style></head><body><div class="shell"><header><div><div class="brand"><img src="tater-mascot.png" alt="Tater mascot"><div><p class="eyebrow">Local model field test</p><h1>Tater Bench</h1></div></div><p class="lede">Tater-style accuracy and measured speed averaged fairly across devices, engines, and speculative decoding configurations.</p></div><div class="summary"><b>__MODEL_COUNT__</b>model configurations<span>__DEVICE_COUNT__ __DEVICE_NOUN__ · __RUN_COUNT__ unique __RUN_NOUN__ · __DUPLICATE_COUNT__ __DUPLICATE_NOUN__ omitted</span></div></header>
<div class="control-block"><span class="control-label">Hardware view</span><nav class="controls" aria-label="Choose hardware view">__SCOPES__</nav></div>
<div class="toolbar"><div class="control-block"><span class="control-label">Filter</span><nav class="controls" aria-label="Filter results">__FILTERS__</nav></div><label class="sort-label">Sort results<select id="sort-results"><option value="score-desc">Final Tater Score — high to low</option><option value="fitness-desc">Fitness, then Tater Score</option><option value="accuracy-desc">Accuracy — high to low</option><option value="speed-desc">Generation speed — high to low</option><option value="ttft-asc">TTFT — low to high</option><option value="memory-asc">Memory — low to high</option><option value="samples-desc">Most tested</option><option value="model-asc">Model name — A to Z</option></select></label></div>
<section class="leaderboard" aria-labelledby="leaderboard-title"><div class="chart-heading"><div><p class="eyebrow">Tater leaderboard</p><h2 id="leaderboard-title">Best models for Tater — all devices</h2><p id="leaderboard-subtitle">Hardware-type averages receive equal weight in the overall score.</p></div><p class="score-formula"><b>100-point raw formula:</b> 90 category-weighted accuracy (35 Astraeus routing/tool selection · 25 Thanatos execution · 15 Spudex · 10 synthesis · 5 chat) · 7 generation speed · 2 TTFT · 1 memory. Limited results cap at 79.9; Not Fit results cap at 49.9 before averaging.</p></div><div class="score-plot"><div class="score-grid" aria-hidden="true"><span style="--grid-position:0%"><i>0</i></span><span style="--grid-position:20%"><i>20</i></span><span style="--grid-position:40%"><i>40</i></span><span style="--grid-position:60%"><i>60</i></span><span style="--grid-position:80%"><i>80</i></span><span style="--grid-position:100%"><i>100</i></span></div><div class="score-bars" id="score-bars" style="--bar-count:__BAR_COUNT__">__LEADERBOARD__</div></div><p class="score-note">Bars are ordered by final score. Readiness gates are applied to each underlying hardware result before cross-device averaging, preserving hardware-specific verdicts while preventing Limited or Not Fit results from displaying misleadingly high scores.</p></section>
<h2 class="results-heading" id="results-heading">All Devices results — sorted by Final Tater Score</h2><main id="results-list">__CARDS__</main></div>
<script>
(() => {
  const filterButtons = [...document.querySelectorAll('button[data-filter]')];
  const scopeButtons = [...document.querySelectorAll('button[data-scope-button]')];
  const sortSelect = document.getElementById('sort-results');
  const scoreBars = document.getElementById('score-bars');
  const resultsList = document.getElementById('results-list');
  const leaderboardTitle = document.getElementById('leaderboard-title');
  const leaderboardSubtitle = document.getElementById('leaderboard-subtitle');
  const resultsHeading = document.getElementById('results-heading');
  let activeScope = 'overall';
  let activeFilter = 'all';
  const number = (node, key) => Number(node.dataset[key] || 0);
  function compare(a, b) {
    switch (sortSelect.value) {
      case 'fitness-desc': return number(a, 'fitnessRank') - number(b, 'fitnessRank') || number(b, 'score') - number(a, 'score');
      case 'score-desc': return number(b, 'score') - number(a, 'score') || number(b, 'accuracy') - number(a, 'accuracy');
      case 'accuracy-desc': return number(b, 'accuracy') - number(a, 'accuracy') || number(b, 'score') - number(a, 'score');
      case 'speed-desc': return number(b, 'speed') - number(a, 'speed') || number(b, 'score') - number(a, 'score');
      case 'ttft-asc': return number(a, 'ttft') - number(b, 'ttft') || number(b, 'score') - number(a, 'score');
      case 'memory-asc': return number(a, 'memory') - number(b, 'memory') || number(b, 'score') - number(a, 'score');
      case 'samples-desc': return number(b, 'samples') - number(a, 'samples') || number(b, 'score') - number(a, 'score');
      case 'model-asc': return a.dataset.model.localeCompare(b.dataset.model);
      default: return number(b, 'score') - number(a, 'score') || number(b, 'accuracy') - number(a, 'accuracy');
    }
  }
  const matches = node => node.dataset.scope === activeScope && (activeFilter === 'all' || node.dataset.mode === activeFilter || node.dataset.engine === activeFilter);
  function render() {
    const bars = [...document.querySelectorAll('.score-entry')].sort(compare);
    const cards = [...document.querySelectorAll('.result-card')].sort(compare);
    let rank = 0;
    bars.forEach(entry => {
      const visible = matches(entry);
      entry.hidden = !visible;
      entry.classList.remove('is-leader');
      scoreBars.appendChild(entry);
      if (visible) {
        rank += 1;
        entry.querySelector('.score-rank').textContent = `#${rank}`;
        entry.setAttribute('aria-label', `Rank ${rank}: ${entry.dataset.label}`);
        entry.classList.toggle('is-leader', rank === 1);
      }
    });
    scoreBars.style.setProperty('--bar-count', Math.max(rank, 1));
    let detailRank = 0;
    cards.forEach(card => {
      const visible = matches(card);
      card.hidden = !visible;
      resultsList.appendChild(card);
      if (visible) { detailRank += 1; card.querySelector('.detail-rank').textContent = `#${detailRank}`; }
    });
    const scopeLabel = scopeButtons.find(button => button.dataset.scopeButton === activeScope)?.textContent.trim() || 'All Devices';
    const filterLabel = filterButtons.find(button => button.dataset.filter === activeFilter)?.textContent.trim() || 'All';
    const titleScopeLabel = activeScope === 'overall' ? scopeLabel.toLowerCase() : scopeLabel;
    leaderboardTitle.textContent = `Best models for Tater — ${titleScopeLabel}`;
    leaderboardSubtitle.textContent = activeScope === 'overall' ? 'Outcome-distinct runs are averaged per hardware type, then every hardware type receives equal weight.' : 'Outcome-distinct runs on this hardware type are averaged into one result.';
    resultsHeading.textContent = `${scopeLabel}${activeFilter === 'all' ? '' : ` · ${filterLabel}`} — sorted ${sortSelect.options[sortSelect.selectedIndex].text.toLowerCase()}`;
  }
  filterButtons.forEach(button => button.addEventListener('click', () => {
    activeFilter = button.dataset.filter;
    filterButtons.forEach(item => { const selected = item === button; item.classList.toggle('active', selected); item.setAttribute('aria-pressed', selected ? 'true' : 'false'); });
    render();
  }));
  scopeButtons.forEach(button => button.addEventListener('click', () => {
    activeScope = button.dataset.scopeButton;
    scopeButtons.forEach(item => { const selected = item === button; item.classList.toggle('active', selected); item.setAttribute('aria-pressed', selected ? 'true' : 'false'); });
    render();
  }));
  sortSelect.addEventListener('change', render);
  render();
})();
</script></body></html>'''
    device_count = int(aggregate.get("device_count") or 0)
    run_count = int(aggregate.get("run_count") or 0)
    duplicate_count = int(aggregate.get("duplicate_run_count") or 0)
    return (
        template.replace("__RUN_COUNT__", str(run_count))
        .replace("__RUN_NOUN__", "run" if run_count == 1 else "runs")
        .replace("__DUPLICATE_COUNT__", str(duplicate_count))
        .replace("__DUPLICATE_NOUN__", "duplicate" if duplicate_count == 1 else "duplicates")
        .replace("__MODEL_COUNT__", str(int(aggregate.get("model_result_count") or 0)))
        .replace("__DEVICE_COUNT__", str(device_count))
        .replace("__DEVICE_NOUN__", "device" if device_count == 1 else "devices")
        .replace("__SCOPES__", scope_tabs)
        .replace("__FILTERS__", filters_html)
        .replace("__BAR_COUNT__", str(max(1, len(aggregate.get("leaderboard") or []))))
        .replace("__LEADERBOARD__", leaderboard_html)
        .replace("__CARDS__", cards_html)
    )


def write_reports(
    *,
    results_dir: str | Path = "results",
    markdown_path: str | Path = "RESULTS.md",
    docs_dir: str | Path = "docs",
) -> dict[str, Path]:
    aggregate = aggregate_payload(load_batches(results_dir))
    markdown = Path(markdown_path)
    docs = Path(docs_dir)
    docs.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(aggregate), encoding="utf-8")
    json_path = docs / "results.json"
    html_path = docs / "index.html"
    json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    html_path.write_text(render_html(aggregate), encoding="utf-8")
    return {"markdown": markdown, "json": json_path, "html": html_path}
