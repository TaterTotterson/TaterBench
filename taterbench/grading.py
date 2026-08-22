from __future__ import annotations

import json
import re
from typing import Any


def _extract_json(text: str) -> tuple[dict[str, Any] | None, bool]:
    raw = str(text or "").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(fenced)
        if isinstance(parsed, dict):
            return parsed, False
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return (parsed, False) if isinstance(parsed, dict) else (None, False)
        except json.JSONDecodeError:
            pass
    return None, False


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _match_value(actual: Any, expected: Any) -> float:
    if isinstance(expected, dict) and any(
        key in expected for key in ("equals", "contains", "one_of", "non_empty")
    ):
        if "equals" in expected:
            return _match_value(actual, expected["equals"])
        if "contains" in expected:
            return 1.0 if _normalized(expected["contains"]) in _normalized(actual) else 0.0
        if "non_empty" in expected:
            return 1.0 if _normalized(actual) else 0.0
        options = expected.get("one_of") or []
        return max((_match_value(actual, option) for option in options), default=0.0)
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not expected:
            return 0.0
        return sum(_match_value(actual.get(key), value) for key, value in expected.items()) / len(expected)
    if isinstance(expected, list):
        if not isinstance(actual, list) or not expected:
            return 0.0
        matched = sum(any(_match_value(item, target) == 1.0 for item in actual) for target in expected)
        return matched / len(expected)
    if isinstance(expected, str):
        return 1.0 if _normalized(actual) == _normalized(expected) else 0.0
    return 1.0 if actual == expected else 0.0


def _grade_astraeus(parsed: dict[str, Any] | None, strict: bool, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if parsed is None:
        return 0.0, {"valid_json": False}
    mode_ok = _normalized(parsed.get("mode")) == _normalized(expected.get("mode"))
    steps = parsed.get("steps") if isinstance(parsed.get("steps"), list) else []
    expected_tool_options: list[list[str]] = []
    expected_tool_partial_credit: list[dict[str, float]] = []
    for item in expected.get("tool_sequence") or []:
        if isinstance(item, dict):
            options = [str(value) for value in item.get("one_of") or [] if str(value)]
            partial_credit: dict[str, float] = {}
            for tool_id, credit in (item.get("partial_credit") or {}).items():
                try:
                    partial_credit[str(tool_id)] = max(0.0, min(1.0, float(credit)))
                except (TypeError, ValueError):
                    continue
        else:
            options = [str(item)] if str(item) else []
            partial_credit = {}
        expected_tool_options.append(options)
        expected_tool_partial_credit.append(partial_credit)
    expected_tools = [options[0] if options else "" for options in expected_tool_options]
    actual_tools = [str(item.get("tool_hint") or "") for item in steps if isinstance(item, dict)]
    expected_step_nl = expected.get("step_nl") or []
    actual_step_nl = [str(item.get("nl") or "") for item in steps if isinstance(item, dict)]
    if expected_step_nl:
        step_nl_scores = [
            _match_value(actual_step_nl[index], nl_expected)
            if index < len(actual_step_nl)
            else 0.0
            for index, nl_expected in enumerate(expected_step_nl)
        ]
        step_nl_score = sum(step_nl_scores) / len(expected_step_nl)
    else:
        step_nl_scores = []
        step_nl_score = 1.0
    if expected_tool_options:
        tool_score = sum(
            (
                1.0
                if actual_tools[index] in options
                else expected_tool_partial_credit[index].get(actual_tools[index], 0.0)
            )
            for index, options in enumerate(expected_tool_options)
            if index < len(actual_tools)
        ) / len(expected_tool_options)
        count_ok = len(actual_tools) == len(expected_tool_options)
    else:
        tool_score = 1.0 if not actual_tools else 0.0
        count_ok = not actual_tools
    equivalent_ids = {tool_id for options in expected_tool_options for tool_id in options}
    partial_ids = {tool_id for credits in expected_tool_partial_credit for tool_id in credits}
    valid_ids = set(expected.get("available_tools") or []) | equivalent_ids | partial_ids
    valid_score = 1.0 if all(tool in valid_ids for tool in actual_tools) else 0.0
    score = (0.15 if strict else 0.08) + (0.30 if mode_ok else 0.0) + 0.40 * tool_score
    score += 0.10 if count_ok else 0.0
    score += 0.05 * valid_score
    if expected_step_nl and step_nl_score < 1.0:
        score = min(score, 0.79)
    return min(1.0, score), {
        "valid_json": True,
        "strict_json": strict,
        "mode_ok": mode_ok,
        "expected_tools": expected_tools,
        "expected_tool_options": expected_tool_options,
        "expected_tool_partial_credit": expected_tool_partial_credit,
        "actual_tools": actual_tools,
        "tool_sequence_score": tool_score,
        "expected_step_nl": expected_step_nl,
        "actual_step_nl": actual_step_nl,
        "step_nl_scores": step_nl_scores,
        "step_nl_score": step_nl_score,
        "step_count_ok": count_ok,
    }


def _grade_thanatos(text: str, parsed: dict[str, Any] | None, strict: bool, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if expected.get("blocker"):
        contains = [str(item).lower() for item in expected.get("contains") or []]
        text_lower = text.lower()
        keyword_score = (
            sum(item in text_lower for item in contains) / len(contains)
            if contains
            else (1.0 if _normalized(text) else 0.0)
        )
        not_json = parsed is None
        return 0.6 * keyword_score + (0.4 if not_json else 0.0), {
            "blocker_expected": True,
            "not_json": not_json,
            "keyword_score": keyword_score,
        }
    if parsed is None:
        return 0.0, {"valid_json": False}
    shape_ok = set(parsed) == {"function", "arguments"}
    function_ok = _normalized(parsed.get("function")) == _normalized(expected.get("function"))
    arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    argument_score = _match_value(arguments, expected.get("arguments") or {})
    score = (0.15 if strict else 0.07) + (0.10 if shape_ok else 0.0) + (0.40 if function_ok else 0.0)
    score += 0.35 * argument_score
    return min(1.0, score), {
        "valid_json": True,
        "strict_json": strict,
        "shape_ok": shape_ok,
        "function_ok": function_ok,
        "argument_score": argument_score,
        "actual_function": parsed.get("function"),
        "actual_arguments": arguments,
    }


def _grade_spudex(parsed: dict[str, Any] | None, strict: bool, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    if parsed is None:
        return 0.0, {"valid_json": False}
    type_ok = _normalized(parsed.get("type")) == _normalized(expected.get("type"))
    checks: list[float] = []
    for key in ("outcome", "path", "argv", "query"):
        if key in expected:
            checks.append(_match_value(parsed.get(key), expected[key]))
    content_score = sum(checks) / len(checks) if checks else 1.0
    score = (0.2 if strict else 0.1) + (0.45 if type_ok else 0.0) + 0.35 * content_score
    return min(1.0, score), {
        "valid_json": True,
        "strict_json": strict,
        "type_ok": type_ok,
        "content_score": content_score,
        "actual_type": parsed.get("type"),
    }


def _grade_text(text: str, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    lowered = _normalized(text)
    required = [_normalized(item) for item in expected.get("contains") or []]
    any_of = [_normalized(item) for item in expected.get("any_of") or []]
    forbidden = [_normalized(item) for item in expected.get("forbidden") or []]
    required_score = sum(item in lowered for item in required) / len(required) if required else 1.0
    any_score = 1.0 if not any_of or any(item in lowered for item in any_of) else 0.0
    forbidden_score = 1.0 if not any(item in lowered for item in forbidden) else 0.0
    nonempty = 1.0 if lowered else 0.0
    score = 0.55 * required_score + 0.20 * any_score + 0.20 * forbidden_score + 0.05 * nonempty
    return score, {
        "required_score": required_score,
        "any_of_score": any_score,
        "forbidden_score": forbidden_score,
        "nonempty": bool(nonempty),
    }


def grade_response(scenario: dict[str, Any], text: str) -> dict[str, Any]:
    kind = str(scenario.get("kind") or "chat")
    expected = scenario.get("expected") if isinstance(scenario.get("expected"), dict) else {}
    parsed, strict = _extract_json(text)
    if kind == "astraeus":
        expected = {**expected, "available_tools": scenario.get("available_tools") or []}
        score, details = _grade_astraeus(parsed, strict, expected)
    elif kind == "thanatos":
        score, details = _grade_thanatos(text, parsed, strict, expected)
    elif kind == "spudex":
        score, details = _grade_spudex(parsed, strict, expected)
    else:
        score, details = _grade_text(text, expected)
    return {
        "score": round(max(0.0, min(1.0, score)), 6),
        "passed": score >= float(scenario.get("pass_threshold") or 0.8),
        "details": details,
    }
