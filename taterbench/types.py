from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DraftModel:
    method: str
    path: Path
    tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "path": str(self.path), "tokens": self.tokens}


@dataclass
class ModelCandidate:
    id: str
    label: str
    provider: str
    model_path: Path
    repo_id: str = ""
    filename: str = ""
    quantization: str = ""
    supports_vision: bool = False
    supports_video: bool = False
    supports_audio: bool = False
    mmproj_path: Path | None = None
    max_context_tokens: int = 0
    drafts: list[DraftModel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "model_path": str(self.model_path),
            "repo_id": self.repo_id,
            "filename": self.filename,
            "quantization": self.quantization,
            "supports_vision": self.supports_vision,
            "supports_video": self.supports_video,
            "supports_audio": self.supports_audio,
            "mmproj_path": str(self.mmproj_path) if self.mmproj_path else "",
            "max_context_tokens": self.max_context_tokens,
            "drafts": [draft.to_dict() for draft in self.drafts],
        }


@dataclass(frozen=True)
class RunVariant:
    name: str
    speculative_method: str = ""
    draft_path: Path | None = None
    draft_tokens: int = 0

    @property
    def speculative(self) -> bool:
        return bool(self.speculative_method)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "speculative_method": self.speculative_method,
            "draft_path": str(self.draft_path) if self.draft_path else "",
            "draft_tokens": self.draft_tokens,
        }


@dataclass
class GenerationResult:
    text: str
    elapsed_seconds: float
    ttft_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tokens_per_second: float = 0.0
    completion_tokens_per_second: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
