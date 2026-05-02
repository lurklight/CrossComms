from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.models import AudioFrame
from gamevoice.providers.windows_speech import (
    UtteranceSegmenter,
    WindowsSpeechSegmentRecognizer,
    resolve_windows_culture,
)


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


class TestRecognizer(WindowsSpeechSegmentRecognizer):
    def __init__(self) -> None:
        super().__init__(
            source_language="en",
            frame_ms=30,
            endpoint_silence_ms=90,
            min_speech_ms=60,
            max_utterance_ms=600,
        )

    async def _transcribe_utterance(self, utterance):  # type: ignore[override]
        self.last_duration_ms = utterance.duration_ms
        return "enemy low"


class WindowsSpeechTests(unittest.IsolatedAsyncioTestCase):
    def test_segmenter_flushes_after_speech_then_silence(self) -> None:
        segmenter = UtteranceSegmenter(
            frame_ms=30,
            endpoint_silence_ms=90,
            min_speech_ms=60,
            max_utterance_ms=600,
            min_peak_rms=100.0,
        )

        result = None
        frames = [
            _frame(False),
            _frame(True),
            _frame(True),
            _frame(True),
            _frame(False),
            _frame(False),
            _frame(False),
        ]
        for frame in frames:
            result = segmenter.push(frame) or result

        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.duration_ms, 180)
        self.assertEqual(result.sample_rate, 16_000)

    def test_segmenter_ignores_single_loud_click(self) -> None:
        segmenter = UtteranceSegmenter(
            frame_ms=30,
            endpoint_silence_ms=90,
            min_speech_ms=60,
            max_utterance_ms=600,
            min_peak_rms=100.0,
        )

        result = None
        for frame in [
            _frame(False),
            _frame(True, rms=950.0),
            _frame(False),
            _frame(False),
            _frame(False),
        ]:
            result = segmenter.push(frame) or result

        self.assertIsNone(result)

    async def test_recognizer_yields_transcript_for_utterance(self) -> None:
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

    def test_windows_culture_mapping_prefers_installed_tags(self) -> None:
        self.assertEqual(resolve_windows_culture("en"), "en-US")
        self.assertEqual(resolve_windows_culture("zh"), "zh-CN")
        self.assertEqual(resolve_windows_culture("fr"), "fr")


if __name__ == "__main__":
    unittest.main()
