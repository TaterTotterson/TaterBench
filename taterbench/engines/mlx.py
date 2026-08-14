from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from ..paths import first_executable, mlx_python_candidates, tater_home
from ..types import GenerationResult, ModelCandidate, RunVariant
from .base import EngineSession, ProcessMemorySampler, terminate_process


class MlxSession(EngineSession):
    def __init__(
        self,
        model: ModelCandidate,
        variant: RunVariant,
        *,
        home: str | Path | None = None,
        context_size: int = 8192,
        load_timeout: float = 900.0,
    ):
        if variant.speculative:
            raise ValueError("MTP/DFlash variants are llama.cpp GGUF features in Tater Bench")
        self.model = model
        self.variant = variant
        self.home = tater_home(home)
        self.context_size = max(2048, int(context_size))
        self.load_timeout = max(30.0, float(load_timeout))
        self.process: subprocess.Popen[str] | None = None
        self.sampler: ProcessMemorySampler | None = None
        self.load_seconds = 0.0
        self.peak_rss_bytes = 0
        self.metadata: dict[str, Any] = {}
        self.stderr_tail: list[str] = []
        self.stdout_noise: list[str] = []
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()

    def _drain_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in iter(self.process.stdout.readline, ""):
            compact = line.strip()
            if not compact:
                continue
            try:
                payload = json.loads(compact)
            except json.JSONDecodeError:
                self.stdout_noise.append(compact)
                del self.stdout_noise[:-80]
                continue
            if isinstance(payload, dict):
                self._responses.put(payload)

    def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in iter(self.process.stderr.readline, ""):
            compact = " ".join(line.split())
            if compact:
                self.stderr_tail.append(compact)
                del self.stderr_tail[:-120]

    def _receive(self, timeout: float) -> dict[str, Any]:
        try:
            return self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            if self.process is not None and self.process.poll() is not None:
                logs = " ".join(self.stderr_tail[-30:])[-4000:]
                raise RuntimeError(f"MLX worker exited (exit {self.process.returncode}). {logs}") from exc
            raise TimeoutError("MLX worker did not respond before the timeout") from exc

    def start(self) -> None:
        python = first_executable(mlx_python_candidates(self.home))
        if python is None:
            raise RuntimeError("Tater's Python environment was not found. Run Tater setup first.")
        project_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), str(self.home / "runtime" / "mlx-engine"), existing_pythonpath]
        ).strip(os.pathsep)
        command = [
            str(python),
            "-m",
            "taterbench.engines.mlx_worker",
            "--model",
            str(self.model.model_path),
            "--context-size",
            str(self.context_size),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self.sampler = ProcessMemorySampler(self.process)
        self.sampler.start()
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        ready = self._receive(self.load_timeout)
        if ready.get("event") != "ready":
            raise RuntimeError(str(ready.get("error") or "MLX worker failed during model load"))
        self.load_seconds = float(ready.get("load_seconds") or 0.0)
        self.metadata = {
            "engine": "mlx",
            "python_path": str(python),
            "python_version": str(ready.get("python_version") or ""),
            "mlx_version": str(ready.get("mlx_version") or ""),
            "mlx_engine_path": str(self.home / "runtime" / "mlx-engine"),
            "context_size": self.context_size,
            "variant": self.variant.to_dict(),
        }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        if self.process is None or self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("MLX engine is not running")
        payload = {
            "command": "generate",
            "messages": messages,
            "max_tokens": max(1, int(max_tokens)),
            "temperature": max(0.0, float(temperature)),
            "seed": int(seed),
        }
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        response = self._receive(600.0)
        if response.get("event") == "error":
            raise RuntimeError(str(response.get("error") or "MLX generation failed"))
        return GenerationResult(
            text=str(response.get("text") or "").strip(),
            elapsed_seconds=float(response.get("elapsed_seconds") or 0.0),
            ttft_seconds=float(response.get("ttft_seconds") or 0.0),
            prompt_tokens=int(response.get("prompt_tokens") or 0),
            completion_tokens=int(response.get("completion_tokens") or 0),
            prompt_tokens_per_second=float(response.get("prompt_tokens_per_second") or 0.0),
            completion_tokens_per_second=float(response.get("completion_tokens_per_second") or 0.0),
            raw={"stop_reason": response.get("stop_reason")},
        )

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None and self.process.stdin is not None:
            try:
                self.process.stdin.write('{"command":"shutdown"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=8.0)
            except Exception:
                pass
        terminate_process(self.process)
        if self.sampler is not None:
            self.peak_rss_bytes = self.sampler.stop()
        self.process = None
