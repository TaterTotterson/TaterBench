from __future__ import annotations

import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from ..types import GenerationResult


class ProcessMemorySampler:
    def __init__(self, process: subprocess.Popen[Any], interval: float = 0.2):
        self.process = process
        self.interval = interval
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="taterbench-rss")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> int:
        self._stop.set()
        self._thread.join(timeout=2.0)
        return self.peak_rss_bytes

    def _run(self) -> None:
        while not self._stop.is_set() and self.process.poll() is None:
            try:
                result = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(self.process.pid)],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                    check=False,
                )
                rss_kib = int((result.stdout or "0").strip() or 0)
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss_kib * 1024)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)


class EngineSession(ABC):
    load_seconds: float = 0.0
    peak_rss_bytes: int = 0
    metadata: dict[str, Any]

    def __enter__(self) -> "EngineSession":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5.0)
    except Exception:
        try:
            process.kill()
            process.wait(timeout=3.0)
        except Exception:
            pass


def wait_for_exit(process: subprocess.Popen[Any], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        time.sleep(0.1)
    return process.poll() is not None
