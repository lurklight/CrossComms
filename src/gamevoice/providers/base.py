from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from ..models import AudioFrame, NormalizedChunk, SpeechChunk, TranscriptChunk, TranslationChunk


class StreamingSpeechRecognizer(Protocol):
    async def transcribe(
        self,
        frames: AsyncIterator[AudioFrame],
    ) -> AsyncIterator[TranscriptChunk]: ...


class TextTranslator(Protocol):
    async def stream_translate(
        self,
        chunk: NormalizedChunk,
        source_language: str,
        target_language: str,
    ) -> AsyncIterator[TranslationChunk]: ...


class StreamingSpeechSynthesizer(Protocol):
    async def stream_speech(self, chunk: TranslationChunk) -> AsyncIterator[SpeechChunk]: ...
