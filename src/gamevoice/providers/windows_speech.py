from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
import subprocess
import tempfile
import wave

from ..models import AudioFrame, TranscriptChunk
from .base import StreamingSpeechRecognizer
from .utterances import CapturedUtterance, UtteranceSegmenter


WINDOWS_CULTURE_MAP = {
    "en": "en-US",
    "en-us": "en-US",
    "zh": "zh-CN",
    "zh-cn": "zh-CN",
}


class WindowsSpeechSegmentRecognizer(StreamingSpeechRecognizer):
    def __init__(
        self,
        source_language: str = "en",
        frame_ms: int = 30,
        endpoint_silence_ms: int = 420,
        min_speech_ms: int = 240,
        max_utterance_ms: int = 4_500,
    ) -> None:
        self.requested_language = source_language
        self.culture = resolve_windows_culture(source_language)
        self.segmenter = UtteranceSegmenter(
            frame_ms=frame_ms,
            endpoint_silence_ms=endpoint_silence_ms,
            min_speech_ms=min_speech_ms,
            max_utterance_ms=max_utterance_ms,
        )
        self._sequence_id = 1
        self._script_path = Path(__file__).with_name("windows_speech_transcribe.ps1")

    async def transcribe(
        self,
        frames: AsyncIterator[AudioFrame],
    ) -> AsyncIterator[TranscriptChunk]:
        async for frame in frames:
            utterance = self.segmenter.push(frame)
            if utterance is not None:
                transcript = await self._transcribe_utterance(utterance)
                if transcript:
                    yield TranscriptChunk(
                        text=transcript,
                        sequence_id=self._next_sequence_id(),
                        is_final=True,
                    )

    async def _transcribe_utterance(self, utterance: CapturedUtterance) -> str | None:
        return await asyncio.to_thread(self._recognize_sync, utterance)

    def _recognize_sync(self, utterance: CapturedUtterance) -> str | None:
        audio_path = self._write_wave_file(utterance)
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(self._script_path),
                    "-AudioPath",
                    str(audio_path),
                    "-Culture",
                    self.culture,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
        finally:
            audio_path.unlink(missing_ok=True)

        if result.returncode != 0:
            return None

        transcript = result.stdout.strip()
        if not transcript:
            return None
        return transcript

    def _write_wave_file(self, utterance: CapturedUtterance) -> Path:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as handle:
            path = Path(handle.name)

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(utterance.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(utterance.sample_rate)
            wav_file.writeframes(utterance.pcm16)
        return path

    def _next_sequence_id(self) -> int:
        current = self._sequence_id
        self._sequence_id += 1
        return current


def resolve_windows_culture(source_language: str) -> str:
    key = source_language.strip().lower()
    return WINDOWS_CULTURE_MAP.get(key, source_language)
