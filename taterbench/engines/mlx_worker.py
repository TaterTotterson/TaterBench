from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, separators=(",", ":"), default=str), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-size", type=int, default=8192)
    return parser.parse_args()


def _prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        for kwargs in (
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
        ):
            try:
                rendered = apply_template(messages, **kwargs)
                if isinstance(rendered, str) and rendered:
                    return rendered
            except (TypeError, ValueError):
                continue
    lines = [f"{item.get('role', 'user').upper()}: {item.get('content', '')}" for item in messages]
    return "\n".join(lines) + "\nASSISTANT:"


def main() -> int:
    args = _parse_args()
    try:
        import mlx
        from mlx_engine.generate import create_generator, load_model, tokenize
        from transformers import AutoTokenizer

        started = time.perf_counter()
        model_path = str(Path(args.model).expanduser().resolve())
        model_kit = load_model(
            model_path,
            max_kv_size=max(2048, int(args.context_size)),
            max_seq_nums=1,
            prefill_step_size=4096,
            trust_remote_code=False,
            seed=0,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        _emit(
            {
                "event": "ready",
                "load_seconds": time.perf_counter() - started,
                "python_version": platform.python_version(),
                "mlx_version": getattr(mlx, "__version__", ""),
            }
        )
    except Exception as exc:
        _emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
        return 1

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                shutdown = getattr(model_kit, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                return 0
            if request.get("command") != "generate":
                raise ValueError("unknown worker command")
            messages = request.get("messages") if isinstance(request.get("messages"), list) else []
            prompt = _prompt(tokenizer, messages)
            prompt_tokens = list(tokenize(model_kit, prompt) or [])
            started = time.perf_counter()
            first_token_at = 0.0
            text_parts: list[str] = []
            completion_tokens = 0
            stop_reason = ""
            for result in create_generator(
                model_kit,
                prompt_tokens,
                max_tokens=max(1, int(request.get("max_tokens") or 1)),
                temp=max(0.0, float(request.get("temperature") or 0.0)),
                seed=int(request.get("seed") or 0),
                request_id=f"taterbench-{uuid.uuid4().hex}",
                speculative_decoding_toggle=False,
            ):
                part = str(getattr(result, "text", "") or "")
                if part and not first_token_at:
                    first_token_at = time.perf_counter()
                text_parts.append(part)
                tokens = getattr(result, "tokens", None)
                try:
                    completion_tokens += len(tokens or [])
                except TypeError:
                    pass
                condition = getattr(result, "stop_condition", None)
                if condition is not None:
                    stop_reason = str(getattr(condition, "stop_reason", "") or condition)
                    break
            elapsed = time.perf_counter() - started
            if completion_tokens <= 0:
                try:
                    completion_tokens = len(tokenizer.encode("".join(text_parts)))
                except Exception:
                    completion_tokens = 0
            _emit(
                {
                    "event": "result",
                    "text": "".join(text_parts).strip(),
                    "elapsed_seconds": elapsed,
                    "ttft_seconds": (first_token_at - started) if first_token_at else elapsed,
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": completion_tokens,
                    "completion_tokens_per_second": completion_tokens / elapsed if elapsed and completion_tokens else 0.0,
                    "stop_reason": stop_reason,
                }
            )
        except Exception as exc:
            _emit({"event": "error", "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
