from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.models import AudioFrame
from gamevoice.providers.faster_whisper_recognizer import (
    FasterWhisperSegmentRecognizer,
    build_whisper_hotwords,
    default_whisper_model_for_language,
)
from gamevoice.providers.utterances import CapturedUtterance


def _frame(
    is_speech: bool,
    sample_rate: int = 16_000,
    rms: float | None = None,
) -> AudioFrame:
    pcm = (b"\x10\x00" if is_speech else b"\x00\x00") * int(sample_rate * 0.03)
    return AudioFrame(
        pcm16=pcm,
        sample_rate=sample_rate,
        channels=1,
        is_speech=is_speech,
        rms=(500.0 if is_speech else 0.0) if rms is None else rms,
    )


class TestRecognizer(FasterWhisperSegmentRecognizer):
    def __init__(self) -> None:
        super().__init__(
            source_language="en",
            frame_ms=30,
            endpoint_silence_ms=90,
            min_speech_ms=60,
            max_utterance_ms=600,
            min_peak_rms=100.0,
            min_average_speech_rms=100.0,
        )

    def validate_runtime(self) -> None:
        return

    async def _transcribe_utterance(self, utterance):  # type: ignore[override]
        self.last_duration_ms = utterance.duration_ms
        return "enemy low"


class FasterWhisperRecognizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_recognizer_yields_transcript_for_segmented_audio(self) -> None:
        recognizer = TestRecognizer()

        async def frames() -> AsyncIterator[AudioFrame]:
            for frame in [
                _frame(False),
                _frame(True),
                _frame(True),
                _frame(True),
                _frame(False),
                _frame(False),
                _frame(False),
            ]:
                yield frame

        results = []
        async for transcript in recognizer.transcribe(frames()):
            results.append(transcript.text)

        self.assertEqual(results, ["enemy low"])

    def test_default_model_prefers_english_optimized_variant(self) -> None:
        self.assertEqual(default_whisper_model_for_language("en"), "small.en")
        self.assertEqual(default_whisper_model_for_language("en-US"), "small.en")
        self.assertEqual(default_whisper_model_for_language("es"), "small")

    def test_hotwords_deduplicate_and_preserve_order(self) -> None:
        hotwords = build_whisper_hotwords(
            ["one shot", "rotate b", "one shot", " ", "mid"]
        )
        self.assertEqual(hotwords, "one shot, rotate b, mid")

    async def test_short_noise_spike_does_not_start_an_utterance(self) -> None:
        recognizer = TestRecognizer()

        async def frames() -> AsyncIterator[AudioFrame]:
            for frame in [
                _frame(False),
                _frame(True, rms=900.0),
                _frame(False),
                _frame(False),
                _frame(False),
            ]:
                yield frame

        results = []
        async for transcript in recognizer.transcribe(frames()):
            results.append(transcript.text)

        self.assertEqual(results, [])

    def test_repeated_filler_transcript_is_rejected(self) -> None:
        recognizer = TestRecognizer()
        utterance = CapturedUtterance(
            pcm16=b"\x10\x00" * 1_600,
            sample_rate=16_000,
            channels=1,
            duration_ms=420,
            speech_frame_count=10,
            speech_ratio=0.7,
            average_speech_rms=650.0,
            peak_rms=900.0,
        )
        segment = type("Segment", (), {"no_speech_prob": 0.02, "avg_logprob": -0.2})()

        self.assertFalse(
            recognizer._is_transcript_usable(utterance, "oh oh oh oh", [segment])
        )

    def test_short_clear_greeting_is_kept(self) -> None:
        recognizer = TestRecognizer()
        utterance = CapturedUtterance(
            pcm16=b"\x10\x00" * 1_600,
            sample_rate=16_000,
            channels=1,
            duration_ms=420,
            speech_frame_count=10,
            speech_ratio=0.82,
            average_speech_rms=780.0,
            peak_rms=1_050.0,
        )
        segment = type("Segment", (), {"no_speech_prob": 0.04, "avg_logprob": -0.15})()

        self.assertTrue(
            recognizer._is_transcript_usable(utterance, "hi", [segment])
        )


if __name__ == "__main__":
    unittest.main()
