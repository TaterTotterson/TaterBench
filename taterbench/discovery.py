from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .paths import model_registry_path, tater_home
from .types import DraftModel, ModelCandidate, RunVariant


DRAFT_TOKENS = {"mtp": 3, "dflash": 15, "dspark": 7}
_QUANT_RE = re.compile(r"(?:^|[-_.])((?:UD-)?Q\d(?:_[A-Z0-9]+)+|BF16|F16|FP16|NVFP4)(?:[-_.]|$)", re.I)


def _stable_id(provider: str, path: Path) -> str:
    digest = hashlib.sha256(f"{provider}:{path}".encode("utf-8")).hexdigest()[:12]
    return f"{provider}:{digest}"


def _existing_path(value: Any) -> Path | None:
    token = str(value or "").strip()
    if not token:
        return None
    path = Path(token).expanduser()
    try:
        # Hugging Face snapshots are symlink trees. Keep the human-facing
        # snapshot name so classification still sees mmproj/MTP/DFlash names.
        return path.absolute() if path.exists() else None
    except OSError:
        return None


def classify_gguf(path: Path) -> str:
    name = path.name.lower()
    if "mmproj" in name or "projector" in name:
        return "projector"
    for method in ("dflash", "dspark", "mtp"):
        if re.search(rf"(?:^|[-_.]){method}(?:[-_.]|$)", name):
            return method
    return "main"


def quantization_from_name(name: str) -> str:
    match = _QUANT_RE.search(str(name or ""))
    return match.group(1).upper() if match else ""


def _load_registry(home: Path) -> list[dict[str, Any]]:
    path = model_registry_path(home)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("models") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _paths_in_snapshot(directory: Path) -> Iterable[Path]:
    try:
        yield from (path.absolute() for path in directory.iterdir() if path.name.lower().endswith(".gguf"))
    except OSError:
        return


def _matching_sidecars(main_path: Path, row: dict[str, Any]) -> tuple[Path | None, list[DraftModel]]:
    mmproj = _existing_path(row.get("mmproj_path"))
    if mmproj is not None and (not mmproj.is_file() or classify_gguf(mmproj) != "projector"):
        mmproj = None
    drafts: list[DraftModel] = []
    seen: set[tuple[str, str]] = set()
    for candidate in _paths_in_snapshot(main_path.parent):
        kind = classify_gguf(candidate)
        if kind == "projector" and mmproj is None:
            mmproj = candidate
        elif kind in DRAFT_TOKENS:
            key = (kind, str(candidate))
            if key not in seen:
                seen.add(key)
                drafts.append(DraftModel(kind, candidate, DRAFT_TOKENS[kind]))
    drafts.sort(key=lambda item: ("mtp", "dflash", "dspark").index(item.method))
    return mmproj, drafts


def _candidate_from_row(row: dict[str, Any]) -> ModelCandidate | None:
    provider = str(row.get("provider") or "").strip()
    path = _existing_path(row.get("model_path"))
    if path is None or provider not in {"llama_cpp", "mlx_lm"}:
        return None
    filename = str(row.get("filename") or path.name).strip()
    if provider == "llama_cpp" and classify_gguf(path) != "main":
        return None
    repo_id = str(row.get("repo_id") or "").strip()
    label = str(row.get("model") or repo_id or filename or path.name).strip()
    mmproj: Path | None = None
    drafts: list[DraftModel] = []
    if provider == "llama_cpp":
        mmproj, drafts = _matching_sidecars(path, row)
    try:
        max_context = max(0, int(row.get("max_context_tokens") or 0))
    except (TypeError, ValueError):
        max_context = 0
    return ModelCandidate(
        id=_stable_id(provider, path),
        label=label,
        provider=provider,
        model_path=path,
        repo_id=repo_id,
        filename=filename,
        quantization=quantization_from_name(" ".join((filename, repo_id, label))),
        supports_vision=bool(row.get("supports_vision") or mmproj),
        mmproj_path=mmproj,
        max_context_tokens=max_context,
        drafts=drafts,
    )


def _fallback_scan(home: Path) -> list[ModelCandidate]:
    llm_root = home / "agent_lab" / "models" / "llm"
    found: list[ModelCandidate] = []
    llama_root = llm_root / "llama-cpp"
    if llama_root.is_dir():
        for path in llama_root.rglob("*.gguf"):
            logical = path.absolute()
            if classify_gguf(logical) != "main":
                continue
            mmproj, drafts = _matching_sidecars(logical, {})
            repo_id = ""
            for parent in logical.parents:
                if parent.name.startswith("models--"):
                    parts = parent.name[len("models--") :].split("--")
                    repo_id = "/".join(parts) if len(parts) >= 2 else parts[0]
                    break
            found.append(
                ModelCandidate(
                    id=_stable_id("llama_cpp", logical),
                    label=logical.name,
                    provider="llama_cpp",
                    model_path=logical,
                    repo_id=repo_id,
                    filename=logical.name,
                    quantization=quantization_from_name(logical.name),
                    supports_vision=bool(mmproj),
                    mmproj_path=mmproj,
                    drafts=drafts,
                )
            )
    mlx_root = llm_root / "mlx"
    if mlx_root.is_dir():
        for config_path in mlx_root.rglob("config.json"):
            snapshot = config_path.parent.absolute()
            found.append(
                ModelCandidate(
                    id=_stable_id("mlx_lm", snapshot),
                    label=snapshot.parent.parent.name.replace("models--", "").replace("--", "/"),
                    provider="mlx_lm",
                    model_path=snapshot,
                    repo_id=snapshot.parent.parent.name.replace("models--", "").replace("--", "/"),
                )
            )
    return found


def discover_models(home: str | Path | None = None) -> list[ModelCandidate]:
    root = tater_home(home)
    candidates = [item for row in _load_registry(root) if (item := _candidate_from_row(row)) is not None]
    candidates.extend(_fallback_scan(root))
    deduped: dict[str, ModelCandidate] = {}
    for candidate in candidates:
        identity = (
            f"{candidate.provider}:{candidate.repo_id.lower()}"
            if candidate.provider == "mlx_lm" and candidate.repo_id
            else f"{candidate.provider}:{candidate.repo_id.lower()}:{candidate.filename.lower()}"
            if candidate.repo_id and candidate.filename
            else f"{candidate.provider}:{candidate.model_path}"
        )
        existing = deduped.get(identity)
        if existing is None:
            deduped[identity] = candidate
            continue
        if not existing.repo_id and candidate.repo_id:
            existing.repo_id = candidate.repo_id
        if not existing.mmproj_path and candidate.mmproj_path:
            existing.mmproj_path = candidate.mmproj_path
            existing.supports_vision = True
        known = {(draft.method, draft.path.name.lower()) for draft in existing.drafts}
        for draft in candidate.drafts:
            key = (draft.method, draft.path.name.lower())
            if key not in known:
                known.add(key)
                existing.drafts.append(draft)
    for candidate in deduped.values():
        unique_drafts: dict[str, DraftModel] = {}
        for draft in candidate.drafts:
            unique_drafts.setdefault(draft.method, draft)
        candidate.drafts = sorted(
            unique_drafts.values(), key=lambda item: ("mtp", "dflash", "dspark").index(item.method)
        )
    return sorted(deduped.values(), key=lambda item: (item.provider, item.label.lower()))


def variants_for_model(model: ModelCandidate, include_speculative: bool = True) -> list[RunVariant]:
    variants = [RunVariant("baseline")]
    if model.provider != "llama_cpp" or not include_speculative:
        return variants
    for draft in model.drafts:
        variants.append(
            RunVariant(
                name=draft.method,
                speculative_method=f"draft-{draft.method}",
                draft_path=draft.path,
                draft_tokens=draft.tokens,
            )
        )
    identity = " ".join((model.label, model.repo_id, model.filename))
    has_mtp = any(variant.name == "mtp" for variant in variants)
    if not has_mtp and re.search(r"(?:^|[-_.])MTP(?:[-_.]|$)", identity, re.I):
        variants.insert(1, RunVariant(name="mtp", speculative_method="draft-mtp", draft_tokens=DRAFT_TOKENS["mtp"]))
    return variants


def select_models(models: list[ModelCandidate], selectors: list[str]) -> list[ModelCandidate]:
    if not selectors:
        return models
    selected: list[ModelCandidate] = []
    for model in models:
        haystack = " ".join((model.id, model.label, model.repo_id, model.filename, str(model.model_path))).lower()
        if any(selector.lower() in haystack for selector in selectors):
            selected.append(model)
    return selected
