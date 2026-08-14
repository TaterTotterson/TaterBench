from __future__ import annotations

import json
from typing import Any


ASTRAEUS_SYSTEM = """Task: decide whether this turn is conversational chat or executable work; when executable, return an ordered executable plan.
Return exactly one strict JSON object with this schema:
{"mode":"chat|execute","goal":"clear goal","steps":[{"step_id":1,"intent":"atomic intent","nl":"single scoped instruction","tool_hint":"tool_id"}]}
Rules:
- Use payload.current_user_message as highest priority.
- Use payload.recent_history only to resolve references and follow-up context.
- For greetings, acknowledgements, reactions, social check-ins, playful banter, or meta conversation, return mode="chat" and steps=[].
- Prefer mode="chat" and steps=[] for conversational, opinion, explanation, and reasoning-only turns that can be answered directly without tool execution.
- Use mode="execute" only when the user clearly asks to do, check, change, fetch, or create something that depends on tools, files, external pages, live system state, or settings.
- Stay within payload.available_capabilities and payload.available_tool_ids; every step must use one valid tool_hint from payload.available_tool_ids.
- Do not invent tool ids, identifiers, paths, URLs, names, or contents.
- Use steps=[] when required user input is missing and no executable step can run yet.
- Each step must be atomic and executable one tool call at a time.
- Preserve user-requested order and split multi-action requests into separate steps.
- Add prerequisite discovery or inspection steps when later steps depend on intermediate data.
- For download requests without a concrete URL, plan discovery, inspection, then download_file.
- Use send_message only when the user explicitly asks to notify a destination on another platform.
- Do not include explanations, markdown, or extra keys."""


THANATOS_SYSTEM = """Execution role: Thanatos.
Execute exactly ONE locked atomic step from Astraeus.
Output either a short blocker explanation OR exactly ONE strict JSON tool call: {"function":"tool_id","arguments":{...}}
Rules:
- Execute only the locked step and do not replan or broaden scope.
- Treat the step tool_hint and execution tool contract as authoritative.
- Use the exact tool id and argument keys from the contract.
- Never invent identifiers, paths, URLs, or missing details.
- For natural-language Verba arguments, pass only a concise rewritten action phrase.
- Never claim completion without a successful tool result.
- If required input is missing, output a short blocker explanation instead of a fake tool call.
- If outputting a tool call, output only the JSON object and nothing else."""


CHAT_SYSTEM = """You are Tater, a friendly local AI assistant.
This is a normal chat turn, not a tool-execution turn. Reply naturally, directly, and concisely.
Do not pretend to run tools or claim actions happened. Do not mention internal roles, modes, planning, or hidden reasoning."""


HERMES_SYSTEM = """You are composing Tater's final user-facing answer from tool findings.
Write a fluent answer that directly addresses the request. Keep facts faithful to the supplied findings, do not invent new facts, and do not narrate internal execution steps or roles. Output user-facing text only."""


SPUDEX_SYSTEM = """You are the model inside Tater's Spudex execution loop.
Return exactly one strict JSON object.
Allowed shapes:
{"type":"reply","outcome":"answer|completed|blocked|failed","message":"..."}
{"type":"write_file","path":"<relative/path>","content":"...","reason":"..."}
{"type":"command","argv":["command","arg"],"reason":"..."}
{"type":"verify","argv":["command","arg"],"reason":"..."}
{"type":"search","query":"...","reason":"..."}
Rules:
- task_mode=true means Hydra selected Spudex for executable work; do not stop with a promise.
- If the user asks a factual question and supplied system_info is enough, reply without a command.
- If the user asks to create or edit a file, the first action should normally be write_file.
- Use argv arrays only. Do not use shells, pipes, redirects, separators, or inline interpreter eval.
- Run commands only when execution is useful.
- Never claim a change is completed before a practical verification step succeeds.
- Ask for missing details only when the task genuinely cannot start without them."""


def build_messages(scenario: dict[str, Any]) -> list[dict[str, str]]:
    kind = str(scenario.get("kind") or "").strip()
    user_text = str(scenario.get("user") or "")
    if kind == "astraeus":
        payload = {
            "current_user_message": user_text,
            "recent_history": scenario.get("recent_history") or [],
            "available_capabilities": scenario.get("available_capabilities") or scenario.get("available_tools") or [],
            "available_tool_ids": scenario.get("available_tools") or [],
        }
        return [
            {"role": "system", "content": ASTRAEUS_SYSTEM},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
    if kind == "thanatos":
        lock = {
            "intent": scenario.get("intent") or user_text,
            "instruction": scenario.get("instruction") or user_text,
            "tool_hint": scenario.get("tool_hint") or "",
        }
        state = {
            "current_user_message": user_text,
            "next_step": lock,
            "execution_tool_contract": scenario.get("tool_contract") or {},
            "prior_results": scenario.get("prior_results") or [],
        }
        return [
            {
                "role": "system",
                "content": THANATOS_SYSTEM + "\n\nExecution step lock:\n" + json.dumps(lock, sort_keys=True),
            },
            {"role": "user", "content": "Current agent state JSON:\n" + json.dumps(state, sort_keys=True)},
        ]
    if kind == "spudex":
        payload = {
            "message": user_text,
            "task_mode": bool(scenario.get("task_mode", True)),
            "step": 1,
            "max_steps": 6,
            "working_folder": "agent_lab/workspace",
            "system_info": scenario.get("system_info") or {"os": "macOS", "architecture": "arm64"},
            "commands_this_turn": scenario.get("commands_this_turn") or [],
            "files_this_turn": scenario.get("files_this_turn") or [],
        }
        return [
            {"role": "system", "content": SPUDEX_SYSTEM},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
    if kind == "hermes":
        payload = {
            "user_request": user_text,
            "findings": scenario.get("findings") or [],
            "base_text": scenario.get("base_text") or "",
        }
        return [
            {"role": "system", "content": HERMES_SYSTEM},
            {"role": "user", "content": json.dumps(payload, sort_keys=True)},
        ]
    return [{"role": "system", "content": CHAT_SYSTEM}, {"role": "user", "content": user_text}]
