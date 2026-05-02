from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import time
from typing import Any


class PipelineStage(StrEnum):
    TRANSCRIPT = "transcript"
    NORMALIZED = "normalized"
    TRANSLATED = "translated"
    SYNTHESIZED = "synthesized"
    OUTPUT = "output"
    STATUS = "status"


@dataclass(slots=True)
class AudioFrame:
    pcm16: bytes
    sample_rate: int
    channels: int
    is_speech: bool
    rms: float = 0.0
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class TranscriptChunk:
    text: str
    sequence_id: int
    is_final: bool = False
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class NormalizedChunk:
    source_text: str
    normalized_text: str
    sequence_id: int
    is_final: bool
    fast_path: bool = False
    matched_terms: tuple[str, ...] = ()
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class TranslationChunk:
    source_text: str
    translated_text: str
    source_language: str
    target_language: str
    sequence_id: int
    is_final: bool
    cache_hit: bool = False
    cache_key: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SpeechChunk:
    text: str
    sequence_id: int
    sample_rate: int
    pcm16: bytes
    is_final: bool
    output_device: str | None = None
    cache_key: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class PipelineEvent:
    stage: PipelineStage
    sequence_id: int
    message: str
    payload: Any | None = None
    created_at: float = field(default_factory=time.time)
