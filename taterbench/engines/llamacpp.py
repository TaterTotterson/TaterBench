from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..hardware import is_strix_halo, parse_llama_version
from ..paths import first_executable, llama_server_candidates, llama_supported_speculative_methods, tater_home
from ..types import GenerationResult, ModelCandidate, RunVariant
from .base import EngineSession, ProcessMemorySampler, terminate_process


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _first_env(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _gpu_layers_argument(raw: str, *, default: str) -> str:
    token = str(raw or "").strip().lower()
    if not token:
        return default
    if token == "auto":
        return "auto"
    if token in {"all", "gpu"}:
        return "all"
    if token in {"none", "off", "false", "cpu"}:
        return "0"
    try:
        parsed = int(token)
    except ValueError:
        return default
    if parsed == -1:
        return "auto"
    if parsed <= -2:
        return "all"
    return str(parsed)


def _strix_halo_full_offload_default() -> bool:
    override = _first_env(
        "TATER_BENCH_STRIX_HALO_FULL_OFFLOAD",
        "TATER_LLAMA_CPP_STRIX_HALO_FULL_OFFLOAD",
    ).lower()
    if override in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if override in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return is_strix_halo()


def _target_gpu_layers() -> str:
    raw = _first_env("TATER_BENCH_LLAMA_N_GPU_LAYERS", "TATER_LLAMA_CPP_N_GPU_LAYERS")
    default = "all" if _strix_halo_full_offload_default() else "auto"
    return _gpu_layers_argument(raw, default=default)


def _draft_gpu_layers(target: str) -> str:
    raw = _first_env(
        "TATER_BENCH_LLAMA_DRAFT_N_GPU_LAYERS",
        "TATER_LLAMA_CPP_DRAFT_N_GPU_LAYERS",
    )
    return _gpu_layers_argument(raw, default=target)


def _json_request(url: str, payload: dict[str, Any] | None = None, timeout: float = 10.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


class LlamaCppSession(EngineSession):
    def __init__(
        self,
        model: ModelCandidate,
        variant: RunVariant,
        *,
        home: str | Path | None = None,
        context_size: int = 8192,
        load_timeout: float = 900.0,
    ):
        self.model = model
        self.variant = variant
        self.home = tater_home(home)
        self.context_size = max(2048, int(context_size))
        self.load_timeout = max(30.0, float(load_timeout))
        self.process: subprocess.Popen[str] | None = None
        self.sampler: ProcessMemorySampler | None = None
        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.stderr_tail: list[str] = []
        self.stdout_tail: list[str] = []
        self.port = 0
        self.base_url = ""
        self.load_seconds = 0.0
        self.peak_rss_bytes = 0
        self.metadata: dict[str, Any] = {}
        self.target_gpu_layers = "auto"
        self.draft_gpu_layers = "auto"

    def _command(self, server: Path) -> list[str]:
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.target_gpu_layers = _target_gpu_layers()
        self.draft_gpu_layers = _draft_gpu_layers(self.target_gpu_layers)
        command = [
            str(server),
            "--model",
            str(self.model.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--batch-size",
            "512",
            "--n-gpu-layers",
            self.target_gpu_layers,
            "--alias",
            "tater-bench",
            "--parallel",
            "1",
            "--no-ui",
            "--jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--reasoning-format",
            "none",
            "--chat-template-kwargs",
            '{"enable_thinking":false,"reasoning_budget":0}',
            "--no-context-shift",
            "--ctx-checkpoints",
            "0",
            "--cache-reuse",
            "256",
            "--flash-attn",
            "on",
        ]
        if self.model.mmproj_path and self.model.mmproj_path.exists():
            command.extend(["--mmproj", str(self.model.mmproj_path)])
        if self.variant.speculative:
            command.extend(
                [
                    "--spec-type",
                    self.variant.speculative_method,
                    "--spec-draft-n-max",
                    str(self.variant.draft_tokens),
                    "--spec-draft-ngl",
                    self.draft_gpu_layers,
                ]
            )
            if self.variant.draft_path:
                command.extend(["--model-draft", str(self.variant.draft_path)])
        return command

    @staticmethod
    def _drain(stream: Any, target: list[str]) -> None:
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                compact = " ".join(str(line or "").split())
                if compact:
                    target.append(compact)
                    del target[:-120]
        except Exception:
            return

    def start(self) -> None:
        server = first_executable(llama_server_candidates(self.home))
        if server is None:
            raise RuntimeError("Tater's llama-server binary was not found. Run Tater setup first.")
        if self.variant.speculative and self.variant.speculative_method not in llama_supported_speculative_methods(self.home):
            raise RuntimeError(
                f"The installed llama-server does not support {self.variant.speculative_method}. "
                "Update Tater's llama.cpp runtime before benchmarking this mode."
            )
        version_result = subprocess.run(
            [str(server), "--version"], capture_output=True, text=True, timeout=15.0, check=False
        )
        version_text = " ".join((version_result.stdout or version_result.stderr or "").split())
        command = self._command(server)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="taterbench-llama-")
        started = time.perf_counter()
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.sampler = ProcessMemorySampler(self.process)
        self.sampler.start()
        threading.Thread(target=self._drain, args=(self.process.stdout, self.stdout_tail), daemon=True).start()
        threading.Thread(target=self._drain, args=(self.process.stderr, self.stderr_tail), daemon=True).start()
        deadline = time.monotonic() + self.load_timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                logs = " ".join(self.stderr_tail[-30:])[-4000:]
                raise RuntimeError(f"llama-server exited while loading (exit {self.process.returncode}). {logs}")
            try:
                health = _json_request(f"{self.base_url}/health", timeout=2.0)
                if str(health.get("status") or "").lower() == "ok":
                    break
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.25)
        else:
            logs = " ".join(self.stderr_tail[-30:])[-4000:]
            raise TimeoutError(f"llama-server was not ready: {logs or last_error}")
        self.load_seconds = time.perf_counter() - started
        props: dict[str, Any] = {}
        try:
            payload = _json_request(f"{self.base_url}/props", timeout=5.0)
            props = payload if isinstance(payload, dict) else {}
        except Exception:
            pass
        self.metadata = {
            "engine": "llama.cpp",
            "server_path": str(server),
            "server_version": parse_llama_version(version_text),
            "server_system_info": str(props.get("system_info") or version_text)[:1200],
            "context_size": self.context_size,
            "variant": self.variant.to_dict(),
            "target_gpu_layers": self.target_gpu_layers,
            "draft_gpu_layers": self.draft_gpu_layers if self.variant.speculative else "",
            "command_flags": command[1:],
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("llama.cpp engine is not running")
        payload = {
            "model": "tater-bench",
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max(1, int(max_tokens)),
            "temperature": max(0.0, float(temperature)),
            "seed": int(seed),
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 0},
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        started = time.perf_counter()
        first_token_at = 0.0
        parts: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        last_payload: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(request, timeout=600.0) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    last_payload = event
                    event_usage = event.get("usage")
                    if isinstance(event_usage, dict):
                        usage = event_usage
                    event_timings = event.get("timings")
                    if isinstance(event_timings, dict):
                        timings = event_timings
                    choices = event.get("choices")
                    choice = choices[0] if isinstance(choices, list) and choices else {}
                    delta = choice.get("delta") if isinstance(choice, dict) else {}
                    text = str(delta.get("content") or "") if isinstance(delta, dict) else ""
                    if text:
                        if not first_token_at:
                            first_token_at = time.perf_counter()
                        parts.append(text)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"llama.cpp generation failed with HTTP {exc.code}: {detail}") from exc
        elapsed = time.perf_counter() - started
        prompt_tokens = int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0)
        completion_tokens = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        completion_tps = float(timings.get("predicted_per_second") or 0.0)
        if not completion_tps and completion_tokens and elapsed:
            completion_tps = completion_tokens / elapsed
        return GenerationResult(
            text="".join(parts).strip(),
            elapsed_seconds=elapsed,
            ttft_seconds=(first_token_at - started) if first_token_at else elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_per_second=float(timings.get("prompt_per_second") or 0.0),
            completion_tokens_per_second=completion_tps,
            raw={"usage": usage, "timings": timings, "finish": last_payload.get("choices", [])},
        )

    def close(self) -> None:
        terminate_process(self.process)
        if self.sampler is not None:
            self.peak_rss_bytes = self.sampler.stop()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
        self.process = None
