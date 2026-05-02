from __future__ import annotations

from array import array
import asyncio
from pathlib import Path
import subprocess
from uuid import uuid4
import wave

from ..models import SpeechChunk, TranslationChunk


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_EDGE_VOICES = {
    "en": "en-US-AriaNeural",
    "es": "es-MX-DaliaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ja": "ja-JP-NanamiNeural",
}


class EdgeNeuralSpeechSynthesizer:
    def __init__(
        self,
        sample_rate: int = 48_000,
        node_binary: str = "node",
        helper_script: Path | None = None,
        voice_map: dict[str, str] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.node_binary = node_binary
        self.helper_script = helper_script or Path(__file__).with_name("edge_neural_tts.cjs")
        self.voice_map = {**DEFAULT_EDGE_VOICES, **(voice_map or {})}

    async def stream_speech(self, chunk: TranslationChunk):
        text = chunk.translated_text.strip()
        if not text:
            return

        sample_rate, pcm16 = await asyncio.to_thread(self._synthesize_sync, chunk)
        yield SpeechChunk(
            text=chunk.translated_text,
            sequence_id=chunk.sequence_id,
            sample_rate=sample_rate,
            pcm16=pcm16,
            is_final=chunk.is_final,
            cache_key=chunk.cache_key,
        )

    def _synthesize_sync(self, chunk: TranslationChunk) -> tuple[int, bytes]:
        temp_root = _project_root()
        suffix = uuid4().hex
        text_path = temp_root / f"gamevoice-edge-tts-{suffix}.txt"
        wav_path = temp_root / f"gamevoice-edge-tts-{suffix}.wav"
        voice_name = self._choose_voice(chunk)

        try:
            text_path.write_text(chunk.translated_text, encoding="utf-8")
            command = [
                self.node_binary,
                str(self.helper_script),
                "--text-path",
                str(text_path),
                "--output-path",
                str(wav_path),
                "--voice",
                voice_name,
                "--sample-rate",
                str(self.sample_rate),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                error_text = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(error_text or "Edge neural TTS failed.")

            sample_rate, pcm16 = self._read_wave_file(wav_path)
            if sample_rate != self.sample_rate:
                pcm16 = self._resample_pcm16_mono(
                    pcm16,
                    source_rate=sample_rate,
                    target_rate=self.sample_rate,
                )
                sample_rate = self.sample_rate
            return sample_rate, pcm16
        finally:
            text_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def _choose_voice(self, chunk: TranslationChunk) -> str:
        speech_language = chunk.target_language
        if (
            chunk.source_language
            and chunk.target_language
            and chunk.source_text.strip().casefold() == chunk.translated_text.strip().casefold()
        ):
            speech_language = chunk.source_language

        language_key = speech_language.strip().lower().split("-", maxsplit=1)[0]
        return self.voice_map.get(language_key, DEFAULT_EDGE_VOICES["en"])

    @staticmethod
    def _read_wave_file(path: Path) -> tuple[int, bytes]:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            pcm16 = handle.readframes(frame_count)

        if sample_width != 2:
            raise RuntimeError(
                f"Unsupported neural TTS sample width {sample_width * 8} bits; expected 16-bit PCM."
            )
        if channels != 1:
            raise RuntimeError(
                f"Unsupported neural TTS channel count {channels}; expected mono audio."
            )
        if not pcm16:
            raise RuntimeError("Neural TTS produced an empty WAV file.")
        return sample_rate, pcm16

    @staticmethod
    def _resample_pcm16_mono(
        pcm16: bytes,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        if source_rate == target_rate or not pcm16:
            return pcm16

        source_samples = array("h")
        source_samples.frombytes(pcm16)
        if len(source_samples) < 2:
            return pcm16

        target_length = max(1, int(round(len(source_samples) * target_rate / source_rate)))
        step = source_rate / target_rate
        resampled = array("h")

        for index in range(target_length):
            position = index * step
            left_index = int(position)
            if left_index >= len(source_samples) - 1:
                resampled.append(source_samples[-1])
                continue

            right_index = left_index + 1
            fraction = position - left_index
            left_sample = source_samples[left_index]
            right_sample = source_samples[right_index]
            interpolated = int(
                round(left_sample + (right_sample - left_sample) * fraction)
            )
            resampled.append(max(-32_768, min(32_767, interpolated)))

        return resampled.tobytes()
