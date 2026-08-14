from __future__ import annotations

import os
import subprocess
from pathlib import Path


def tater_home(explicit: str | Path | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = os.getenv("TATER_HOME") or os.getenv("TATER_ASSISTANT_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".taterassistant").resolve()


def model_registry_path(home: Path) -> Path:
    return home / "agent_lab" / "models" / "llm" / "downloaded_models.json"


def llama_server_candidates(home: Path) -> list[Path]:
    configured = os.getenv("TATER_BENCH_LLAMA_SERVER") or os.getenv("TATER_LLAMA_CPP_SERVER_BIN")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            home / "runtime" / "llama.cpp" / "build" / "bin" / "llama-server",
            home / "runtime" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server",
            home / "runtime" / "llama.cpp" / "bin" / "llama-server",
            Path("/opt/homebrew/bin/llama-server"),
            Path("/usr/local/bin/llama-server"),
        ]
    )
    return candidates


def mlx_python_candidates(home: Path) -> list[Path]:
    configured = os.getenv("TATER_BENCH_MLX_PYTHON")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(
        [
            home / "venv" / "bin" / "python",
            home / "venv" / "Scripts" / "python.exe",
        ]
    )
    return candidates


def first_executable(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                # Preserve virtual-environment and cache-facing paths. Resolving
                # a venv's Python symlink would bypass its package environment.
                return candidate.absolute()
        except OSError:
            continue
    return None


def llama_supported_speculative_methods(home: Path) -> set[str]:
    server = first_executable(llama_server_candidates(home))
    if server is None:
        return set()
    supported: set[str] = set()
    for method in ("draft-mtp", "draft-dflash", "draft-dspark"):
        try:
            result = subprocess.run(
                [str(server), "--spec-type", method, "--version"],
                capture_output=True,
                text=True,
                timeout=12.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = " ".join((result.stdout + " " + result.stderr).lower().split())
        if result.returncode == 0 and "unknown speculative type" not in output:
            supported.add(method)
    return supported
