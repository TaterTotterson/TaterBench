from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any


def _command(args: list[str], timeout: float = 8.0) -> str:
    return " ".join(_command_raw(args, timeout=timeout).split())


def _command_raw(args: list[str], timeout: float = 8.0) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        return (result.stdout or result.stderr or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sysctl(name: str) -> str:
    return _command(["sysctl", "-n", name]) if platform.system() == "Darwin" else ""


def _memory_bytes() -> int:
    if platform.system() == "Darwin":
        try:
            return int(_sysctl("hw.memsize"))
        except ValueError:
            return 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (ValueError, OSError, AttributeError):
        return 0


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model")
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except (OSError, IndexError):
        pass
    return platform.processor() or platform.machine()


def _gpu_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if platform.system() == "Darwin":
        raw = _command_raw(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20.0)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {}
        for item in payload.get("SPDisplaysDataType", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "name": str(item.get("sppci_model") or item.get("_name") or "Apple GPU"),
                    "vendor": str(item.get("spdisplays_vendor") or "Apple"),
                    "vram": str(item.get("spdisplays_vram") or item.get("spdisplays_vram_shared") or "unified"),
                    "backend": "metal",
                }
            )
    nvidia = _command_raw(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]
    )
    for line in nvidia.splitlines() if nvidia else []:
        parts = [part.strip() for part in line.split(",")]
        if parts:
            rows.append(
                {
                    "name": parts[0],
                    "vram_mib": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0,
                    "driver": parts[2] if len(parts) > 2 else "",
                    "vendor": "NVIDIA",
                    "backend": "cuda",
                }
            )
    if not rows and platform.system() == "Linux":
        rocm = _command_raw(["rocminfo"], timeout=12.0)
        names: list[str] = []
        for match in re.finditer(r"Marketing Name:\s*(.+)", rocm):
            name = match.group(1).strip()
            if name and name not in names and "cpu" not in name.lower():
                names.append(name)
        for name in names:
            rows.append({"name": name, "vendor": "AMD", "backend": "rocm"})
    if not rows and platform.system() == "Linux":
        pci = _command_raw(["lspci"], timeout=8.0)
        for line in pci.splitlines():
            if re.search(r"VGA compatible controller|3D controller|Display controller", line, re.I):
                name = line.split(": ", 1)[-1].strip()
                rows.append({"name": name, "vendor": "", "backend": "system"})
    return rows


def capture_hardware(include_hostname: bool = False) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count() or 0,
        "physical_cores": int(_sysctl("hw.physicalcpu") or 0) if platform.system() == "Darwin" else 0,
        "memory_bytes": _memory_bytes(),
        "gpus": _gpu_rows(),
        "python": platform.python_version(),
    }
    if include_hostname:
        profile["hostname"] = platform.node()
    identity = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    profile["hardware_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return profile


def bytes_label(value: int | float) -> str:
    size = float(value or 0)
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or suffix == "TiB":
            return f"{size:.1f} {suffix}"
        size /= 1024
    return f"{size:.1f} TiB"


def parse_llama_version(text: str) -> str:
    compact = " ".join(str(text or "").split())
    match = re.search(r"version:\s*([^\s]+)", compact, re.I)
    return match.group(1) if match else compact[:240]
