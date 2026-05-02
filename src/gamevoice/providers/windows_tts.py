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


class WindowsSapiSpeechSynthesizer:
    def __init__(
        self,
        sample_rate: int | None = None,
        script_path: Path | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.script_path = script_path or Path(__file__).with_name("windows_sapi_tts.ps1")

    async def stream_speech(self, chunk: TranslationChunk):
        text = chunk.translated_text.strip()
        if not text:
            return

        sample_rate, pcm16 = await asyncio.to_thread(
            self._synthesize_sync,
            text,
            chunk.target_language,
        )

        yield SpeechChunk(
            text=chunk.translated_text,
            sequence_id=chunk.sequence_id,
            sample_rate=sample_rate,
            pcm16=pcm16,
            is_final=chunk.is_final,
            cache_key=chunk.cache_key,
        )

    def _synthesize_sync(self, text: str, language: str) -> tuple[int, bytes]:
        temp_root = _project_root()
        suffix = uuid4().hex
        text_path = temp_root / f"gamevoice-tts-{suffix}.txt"
        output_path = temp_root / f"gamevoice-tts-{suffix}.wav"
        try:
            text_path.write_text(text, encoding="utf-8")

            command = [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(self.script_path),
                "-TextPath",
                str(text_path),
                "-OutputPath",
                str(output_path),
                "-Language",
                language,
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
                raise RuntimeError(error_text or "Windows speech synthesis failed.")

            sample_rate, pcm16 = self._read_wave_file(output_path)
            if self.sample_rate is not None and sample_rate != self.sample_rate:
                pcm16 = self._resample_pcm16_mono(
                    pcm16,
                    source_rate=sample_rate,
                    target_rate=self.sample_rate,
                )
                sample_rate = self.sample_rate
            return sample_rate, pcm16
        finally:
            text_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

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
                f"Unsupported TTS sample width {sample_width * 8} bits; expected 16-bit PCM."
            )
        if channels != 1:
            raise RuntimeError(
                f"Unsupported TTS channel count {channels}; expected mono audio."
            )
        if not pcm16:
            raise RuntimeError("Windows speech synthesis produced an empty WAV file.")
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
