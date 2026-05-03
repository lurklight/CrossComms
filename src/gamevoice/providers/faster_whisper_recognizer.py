from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterable
import importlib.util
from pathlib import Path
import re
import threading

from ..models import AudioFrame, TranscriptChunk
from .base import StreamingSpeechRecognizer
from .utterances import CapturedUtterance, UtteranceSegmenter


def default_whisper_model_for_language(source_language: str) -> str:
    return "small.en" if source_language.strip().lower().startswith("en") else "small"


def build_whisper_hotwords(phrases: Iterable[str]) -> str | None:
    cleaned = [phrase.strip() for phrase in phrases if phrase.strip()]
    if not cleaned:
        return None
    return ", ".join(dict.fromkeys(cleaned))


FILLER_NOISE_TOKENS = {
    "a",
    "ah",
    "b",
    "eh",
    "er",
    "hm",
    "hmm",
    "mhm",
    "mm",
    "oh",
    "uh",
    "um",
}
SHORT_ALLOWED_TOKENS = {"gg", "go", "hi", "no", "ok", "yo"}


class FasterWhisperSegmentRecognizer(StreamingSpeechRecognizer):
    def __init__(
        self,
        source_language: str = "en",
        frame_ms: int = 20,
        endpoint_silence_ms: int = 160,
        min_speech_ms: int = 180,
        max_utterance_ms: int = 4_500,
        speech_trigger_ms: int = 60,
        min_speech_ratio: float = 0.38,
        min_peak_rms: float = 420.0,
        model_name: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
        hotwords: str | None = None,
        min_average_speech_rms: float = 340.0,
        beam_size: int = 1,
        best_of: int = 1,
    ) -> None:
        self.source_language = source_language
        self.model_name = model_name or default_whisper_model_for_language(source_language)
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.hotwords = hotwords
        self.min_average_speech_rms = min_average_speech_rms
        self.beam_size = max(1, int(beam_size))
        self.best_of = max(1, int(best_of))
        self.segmenter = UtteranceSegmenter(
            frame_ms=frame_ms,
            endpoint_silence_ms=endpoint_silence_ms,
            min_speech_ms=min_speech_ms,
            max_utterance_ms=max_utterance_ms,
            speech_trigger_ms=speech_trigger_ms,
            min_speech_ratio=min_speech_ratio,
            min_peak_rms=min_peak_rms,
        )
        self._sequence_id = 1
        self._model = None
        self._model_lock = threading.Lock()

    async def transcribe(
        self,
        frames: AsyncIterator[AudioFrame],
        transmit_active: Callable[[], bool] | None = None,
    ) -> AsyncIterator[TranscriptChunk]:
        transmit_was_active = False
        async for frame in frames:
            if transmit_active is not None:
                active = bool(transmit_active())
                if not active:
                    if transmit_was_active:
                        transcript = await self._flush_pending()
                        if transcript is not None:
                            yield transcript
                    transmit_was_active = False
                    continue
                transmit_was_active = True
            utterance = self.segmenter.push(frame)
            if utterance is None:
                continue
            transcript = await self._transcript_chunk_from_utterance(utterance)
            if transcript is not None:
                yield transcript
        if transmit_active is not None and transmit_was_active:
            transcript = await self._flush_pending()
            if transcript is not None:
                yield transcript

    def validate_runtime(self) -> None:
        missing = [
            module_name
            for module_name in ("numpy", "faster_whisper")
            if importlib.util.find_spec(module_name) is None
        ]
        if missing:
            raise RuntimeError(
                "Missing local STT dependencies: "
                + ", ".join(missing)
                + ". Install the `local-stt` extras first."
            )

    def ensure_runtime_ready(self) -> None:
        self.validate_runtime()
        try:
            if self.device == "cuda":
                self._validate_cuda_runtime()
                self._get_model()
        except Exception as exc:
            message = self._friendly_cuda_runtime_error(str(exc))
            raise RuntimeError(message) from exc

    async def _transcribe_utterance(self, utterance: CapturedUtterance) -> str | None:
        return await asyncio.to_thread(self._recognize_sync, utterance)

    async def _flush_pending(self) -> TranscriptChunk | None:
        utterance = self.segmenter.flush()
        if utterance is None:
            return None
        return await self._transcript_chunk_from_utterance(utterance)

    async def _transcript_chunk_from_utterance(
        self,
        utterance: CapturedUtterance,
    ) -> TranscriptChunk | None:
        transcript = await self._transcribe_utterance(utterance)
        if not transcript:
            return None
        return TranscriptChunk(
            text=transcript,
            sequence_id=self._next_sequence_id(),
            is_final=True,
        )

    def _recognize_sync(self, utterance: CapturedUtterance) -> str | None:
        model = self._get_model()
        np = self._require_numpy()
        audio = self._pcm16_to_float32_mono(np, utterance)

        sample_rate = getattr(model.feature_extractor, "sampling_rate", 16_000)
        if utterance.sample_rate != sample_rate:
            audio = self._resample_audio(np, audio, utterance.sample_rate, sample_rate)

        segment_iter, _info = model.transcribe(
            audio,
            language=self._language_hint(),
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            hotwords=self.hotwords,
        )
        segments = [segment for segment in segment_iter if segment.text.strip()]
        text = self._sanitize_transcript_text(
            " ".join(segment.text.strip() for segment in segments).strip()
        )
        if not text:
            return None
        if not self._is_transcript_usable(utterance, text, segments):
            return None
        return text

    def _is_transcript_usable(
        self,
        utterance: CapturedUtterance,
        text: str,
        segments,
    ) -> bool:
        cleaned = text.strip()
        lowered = cleaned.lower()
        if not cleaned:
            return False
        if not re.search(r"[a-z0-9]", lowered):
            return False
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", lowered)
        if not tokens:
            return False
        if utterance.average_speech_rms < self.min_average_speech_rms and len(cleaned) <= 6:
            return False

        if all(token in FILLER_NOISE_TOKENS for token in tokens):
            return False
        if (
            len(tokens) >= 2
            and len(set(tokens)) == 1
            and len(tokens[0]) <= 2
        ):
            return False
        if len(tokens) <= 3 and all(len(token) <= 2 for token in tokens):
            if any(token not in SHORT_ALLOWED_TOKENS for token in tokens):
                return False
            if utterance.speech_ratio < 0.62:
                return False

        if lowered in FILLER_NOISE_TOKENS and utterance.peak_rms < self.min_average_speech_rms * 1.5:
            return False

        max_no_speech_prob = max(
            (
                float(getattr(segment, "no_speech_prob", 0.0) or 0.0)
                for segment in segments
            ),
            default=0.0,
        )
        avg_logprob_values = [
            float(getattr(segment, "avg_logprob", 0.0) or 0.0)
            for segment in segments
        ]
        avg_logprob = (
            sum(avg_logprob_values) / len(avg_logprob_values)
            if avg_logprob_values
            else 0.0
        )

        if max_no_speech_prob > 0.72 and utterance.average_speech_rms < self.min_average_speech_rms * 1.2:
            return False
        if len(tokens) <= 2 and max_no_speech_prob > 0.45:
            return False
        if avg_logprob < -1.1 and utterance.average_speech_rms < self.min_average_speech_rms * 1.35:
            return False
        return True

    def _get_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install local STT dependencies first."
                ) from exc

            kwargs: dict[str, object] = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.download_root is not None:
                kwargs["download_root"] = str(self.download_root)

            self._model = WhisperModel(self.model_name, **kwargs)
            return self._model

    def _validate_cuda_runtime(self) -> None:
        try:
            import ctranslate2
        except ImportError as exc:
            raise RuntimeError(
                "GPU Only mode requires a CUDA-enabled faster-whisper / ctranslate2 install."
            ) from exc

        get_cuda_device_count = getattr(ctranslate2, "get_cuda_device_count", None)
        if callable(get_cuda_device_count):
            try:
                device_count = int(get_cuda_device_count())
            except Exception as exc:
                raise RuntimeError(
                    "GPU Only mode could not verify CUDA device availability."
                ) from exc
            if device_count < 1:
                raise RuntimeError(
                    "GPU Only mode is selected, but no CUDA GPU was detected."
                )

    @staticmethod
    def _friendly_cuda_runtime_error(message: str) -> str:
        lowered = message.lower()
        if "cublas64_12.dll" in lowered:
            return (
                "GPU Only mode needs the CUDA 12 cuBLAS runtime, but `cublas64_12.dll` "
                "was not found. Install the NVIDIA CUDA 12 GPU runtime libraries "
                "(cuBLAS + cuDNN) or install the optional local GPU dependency wheels, "
                "then relaunch CrossComms."
            )
        if "cudnn" in lowered and ".dll" in lowered:
            return (
                "GPU Only mode needs the cuDNN runtime for CUDA 12, but the required "
                "cuDNN DLL was not found. Install cuDNN 9 for CUDA 12 or the optional "
                "local GPU dependency wheels, then relaunch CrossComms."
            )
        if "no cuda gpu" in lowered or "no cuda-capable device" in lowered:
            return "GPU Only mode is selected, but no CUDA-capable GPU is available."
        return f"GPU Only mode failed: {message}"

    def warm_up(self) -> None:
        model = self._get_model()
        np = self._require_numpy()
        audio = np.zeros(1_600, dtype=np.float32)
        segment_iter, _info = model.transcribe(
            audio,
            language=self._language_hint(),
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            hotwords=self.hotwords,
        )
        list(segment_iter)

    def _language_hint(self) -> str | None:
        lowered = self.source_language.strip().lower()
        if lowered in {"auto", ""}:
            return None
        return lowered

    def _next_sequence_id(self) -> int:
        current = self._sequence_id
        self._sequence_id += 1
        return current

    @staticmethod
    def _sanitize_transcript_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return ""

        tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", cleaned.lower())
        if len(tokens) < 4:
            return cleaned

        for phrase_len in range(1, min(4, len(tokens) // 2) + 1):
            if len(tokens) % phrase_len != 0:
                continue
            phrase = tokens[:phrase_len]
            repeats = len(tokens) // phrase_len
            if repeats < 3:
                continue
            if all(
                tokens[index:index + phrase_len] == phrase
                for index in range(0, len(tokens), phrase_len)
            ):
                return " ".join(phrase * min(repeats, 2))

        collapsed: list[str] = []
        last_token = None
        repeat_count = 0
        for token in tokens:
            if token == last_token:
                repeat_count += 1
            else:
                last_token = token
                repeat_count = 1
            if repeat_count <= 2:
                collapsed.append(token)
        if len(collapsed) != len(tokens):
            return " ".join(collapsed)
        return cleaned

    @staticmethod
    def _require_numpy():
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "numpy is not installed. Install local STT dependencies first."
            ) from exc
        return np

    @staticmethod
    def _pcm16_to_float32_mono(np, utterance: CapturedUtterance):
        audio = np.frombuffer(utterance.pcm16, dtype=np.int16).astype(np.float32)
        if utterance.channels > 1:
            usable = (audio.size // utterance.channels) * utterance.channels
            audio = audio[:usable].reshape(-1, utterance.channels).mean(axis=1)
        return audio / 32768.0

    @staticmethod
    def _resample_audio(np, audio, source_rate: int, target_rate: int):
        if source_rate == target_rate:
            return audio
        duration = audio.shape[0] / float(source_rate)
        target_length = max(1, int(round(duration * target_rate)))
        source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
        target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
