from __future__ import annotations

from array import array
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from uuid import uuid4
import wave

from ..models import SpeechChunk, TranslationChunk


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


DEFAULT_PIPER_MODELS = {
    "en": "en_US-lessac-high.onnx",
    "es": "es_MX-claude-high.onnx",
    "fr": "fr_FR-siwis-medium.onnx",
    "de": "de_DE-thorsten-high.onnx",
    "it": "it_IT-paola-medium.onnx",
    "pt": "pt_PT-tugao-medium.onnx",
    "pt-br": "pt_BR-faber-medium.onnx",
    "ru": "ru_RU-irina-medium.onnx",
    "vi": "vi_VN-vais1000-medium.onnx",
    "zh": "zh_CN-huayan-medium.onnx",
    "zh-cn": "zh_CN-huayan-medium.onnx",
}


@dataclass(frozen=True, slots=True)
class PiperVoiceModel:
    file_name: str
    label: str
    language_family: str
    locale_code: str


def list_installed_piper_voices(runtime_root: Path | None = None) -> list[PiperVoiceModel]:
    root = runtime_root or (_project_root() / ".piper-runtime")
    voice_root = root / "voices"
    if not voice_root.exists():
        return []

    options: list[PiperVoiceModel] = []
    for path in sorted(voice_root.glob("*.onnx")):
        stem = path.stem
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        locale = parts[0]
        quality = parts[-1]
        voice_name = "-".join(parts[1:-1])
        language_family = locale.split("_", maxsplit=1)[0].lower()
        label = f"{locale} / {voice_name} / {quality}"
        options.append(
            PiperVoiceModel(
                file_name=path.name,
                label=label,
                language_family=language_family,
                locale_code=locale,
            )
        )
    return options


class PiperSpeechSynthesizer:
    def __init__(
        self,
        sample_rate: int = 48_000,
        runtime_root: Path | None = None,
        model_map: dict[str, str] | None = None,
        source_voice_model: str | None = None,
        target_voice_model: str | None = None,
        cache_size: int = 64,
    ) -> None:
        self.sample_rate = sample_rate
        self.runtime_root = runtime_root or (_project_root() / ".piper-runtime")
        self.voice_root = self.runtime_root / "voices"
        self.model_map = {**DEFAULT_PIPER_MODELS, **(model_map or {})}
        self.source_voice_model = source_voice_model
        self.target_voice_model = target_voice_model
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple[str, str, int], tuple[int, bytes]] = OrderedDict()
        self._model_sample_rate_cache: dict[str, int] = {}

    def validate_runtime(self) -> None:
        if not self.piper_binary.exists():
            raise RuntimeError(
                "Piper runtime is not installed yet. Expected "
                f"`{self.runtime_root}` to contain `piper.exe`."
            )
        required_models = {
            "en": self.source_voice_model or DEFAULT_PIPER_MODELS["en"],
            "es": self.target_voice_model or DEFAULT_PIPER_MODELS["es"],
        }
        for language, model_name in required_models.items():
            model_path = self.voice_root / model_name
            if not model_path.exists():
                raise RuntimeError(
                    f"Missing Piper voice model for `{language}` at `{model_path}`."
                )

    async def stream_speech(self, chunk: TranslationChunk):
        text = chunk.translated_text.strip()
        if not text:
            return

        model_path = self._choose_model_path(chunk)
        cache_key = (str(model_path), text, self.sample_rate)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            sample_rate, pcm16 = cached
            yield SpeechChunk(
                text=chunk.translated_text,
                sequence_id=chunk.sequence_id,
                sample_rate=sample_rate,
                pcm16=pcm16,
                is_final=chunk.is_final,
                cache_key=chunk.cache_key,
            )
            return

        sample_rate = self._model_sample_rate(model_path)
        buffered_audio = bytearray()
        pending = bytearray()
        emit_buffer = bytearray()
        first_chunk_sent = False
        first_emit_threshold = max(2048, int(sample_rate * 2 * 0.08))
        process = await asyncio.create_subprocess_exec(
            str(self.piper_binary),
            "--model",
            str(model_path),
            "--output_raw",
            "--sentence_silence",
            "0.0",
            "-q",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.piper_binary.parent),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None

        process.stdin.write((text + "\n").encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        while True:
            block = await process.stdout.read(4096)
            if not block:
                break
            buffered_audio.extend(block)
            pending.extend(block)
            even_length = len(pending) - (len(pending) % 2)
            if even_length <= 0:
                continue
            emit_buffer.extend(pending[:even_length])
            del pending[:even_length]
            if not first_chunk_sent and len(emit_buffer) >= first_emit_threshold:
                chunk_pcm16 = bytes(emit_buffer)
                emit_buffer.clear()
                first_chunk_sent = True
                yield SpeechChunk(
                    text=chunk.translated_text,
                    sequence_id=chunk.sequence_id,
                    sample_rate=sample_rate,
                    pcm16=chunk_pcm16,
                    is_final=False,
                    cache_key=chunk.cache_key,
                )

        stderr_text = (await process.stderr.read()).decode("utf-8", errors="ignore").strip()
        return_code = await process.wait()
        if return_code != 0:
            raise RuntimeError(stderr_text or "Piper TTS failed.")

        if pending:
            if len(pending) % 2 != 0:
                pending = pending[:-1]
            emit_buffer.extend(pending)
        if not emit_buffer and not buffered_audio:
            raise RuntimeError("Piper produced no audio.")

        final_pcm16 = bytes(emit_buffer)
        if final_pcm16:
            yield SpeechChunk(
                text=chunk.translated_text,
                sequence_id=chunk.sequence_id,
                sample_rate=sample_rate,
                pcm16=final_pcm16,
                is_final=chunk.is_final,
                cache_key=chunk.cache_key,
            )

        cached_sample_rate = sample_rate
        cached_pcm16 = bytes(buffered_audio)
        if sample_rate != self.sample_rate:
            cached_pcm16 = self._resample_pcm16_mono(
                cached_pcm16,
                source_rate=sample_rate,
                target_rate=self.sample_rate,
            )
            cached_sample_rate = self.sample_rate
        resolved = (cached_sample_rate, cached_pcm16)
        self._cache[cache_key] = resolved
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _synthesize_sync(self, chunk: TranslationChunk) -> tuple[int, bytes]:
        model_path = self._choose_model_path(chunk)
        cache_key = (str(model_path), chunk.translated_text.strip(), self.sample_rate)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        wav_path = _project_root() / f"gamevoice-piper-tts-{uuid4().hex}.wav"

        try:
            completed = subprocess.run(
                [
                    str(self.piper_binary),
                    "--model",
                    str(model_path),
                    "--output_file",
                    str(wav_path),
                ],
                input=chunk.translated_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
                cwd=self.piper_binary.parent,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if completed.returncode != 0:
                error_text = (completed.stderr or completed.stdout).strip()
                raise RuntimeError(error_text or "Piper TTS failed.")

            sample_rate, pcm16 = self._read_wave_file(wav_path)
            if sample_rate != self.sample_rate:
                pcm16 = self._resample_pcm16_mono(
                    pcm16,
                    source_rate=sample_rate,
                    target_rate=self.sample_rate,
                )
                sample_rate = self.sample_rate
            resolved = (sample_rate, pcm16)
            self._cache[cache_key] = resolved
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return resolved
        finally:
            wav_path.unlink(missing_ok=True)

    @property
    def piper_binary(self) -> Path:
        candidates = (
            self.runtime_root / "piper.exe",
            self.runtime_root / "piper" / "piper.exe",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _choose_model_path(self, chunk: TranslationChunk) -> Path:
        use_source_voice = (
            chunk.source_language
            and chunk.target_language
            and chunk.source_text.strip().casefold() == chunk.translated_text.strip().casefold()
        )
        speech_language = chunk.target_language
        if use_source_voice:
            speech_language = chunk.source_language

        explicit_model = self.source_voice_model if use_source_voice else self.target_voice_model
        if explicit_model:
            explicit_path = self.voice_root / explicit_model
            if explicit_path.exists():
                return explicit_path

        normalized_language = speech_language.strip().replace("_", "-").lower()
        language_family = normalized_language.split("-", maxsplit=1)[0]
        model_name = (
            self.model_map.get(normalized_language)
            or self.model_map.get(language_family)
            or self.model_map["en"]
        )
        model_path = self.voice_root / model_name
        if model_path.exists():
            return model_path

        fallback = self.voice_root / self.model_map["en"]
        if fallback.exists():
            return fallback

        raise RuntimeError(
            f"Piper voice model not found for language `{normalized_language}`. "
            f"Expected `{model_path}`."
        )

    def _model_sample_rate(self, model_path: Path) -> int:
        cache_key = str(model_path)
        cached = self._model_sample_rate_cache.get(cache_key)
        if cached is not None:
            return cached

        config_path = model_path.with_suffix(model_path.suffix + ".json")
        sample_rate = self.sample_rate
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                sample_rate = int(
                    payload.get("audio", {}).get("sample_rate", self.sample_rate)
                )
            except Exception:
                sample_rate = self.sample_rate
        self._model_sample_rate_cache[cache_key] = sample_rate
        return sample_rate

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
                f"Unsupported Piper sample width {sample_width * 8} bits; expected 16-bit PCM."
            )
        if channels != 1:
            raise RuntimeError(
                f"Unsupported Piper channel count {channels}; expected mono audio."
            )
        if not pcm16:
            raise RuntimeError("Piper produced an empty WAV file.")
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
