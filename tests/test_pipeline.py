from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.audio.output import NullAudioSink
from gamevoice.config import RuntimeConfig
from gamevoice.comms.normalizer import GameCommsNormalizer, SlangPack
from gamevoice.pipeline import RealtimeVoicePipeline
from gamevoice.providers.mock import MockSpeechSynthesizer, MockTranslator


TEST_PACK = SlangPack(
    name="Test Pack",
    replacements={
        "enemy low": "enemy is very low health",
    },
)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_phrase_flows_through_pipeline_twice(self) -> None:
        sink = NullAudioSink()
        normalizer = GameCommsNormalizer(TEST_PACK)
        pipeline = RealtimeVoicePipeline(
            config=RuntimeConfig(target_language="es"),
            normalizer=normalizer,
            translator=MockTranslator(step_delay=0.001),
            synthesizer=MockSpeechSynthesizer(step_delay=0.001),
            sink=sink,
        )

        await pipeline.start()
        await pipeline.submit_text("enemy low", is_final=True)
        await pipeline.wait_for_idle()

        self.assertGreaterEqual(len(sink.chunks), 1)
        first_text = sink.chunks[-1].text

        await pipeline.submit_text("enemy low", is_final=True)
        await pipeline.wait_for_idle()

        self.assertEqual(sink.chunks[-1].text, first_text)
        self.assertGreaterEqual(len(sink.chunks), 2)
        await pipeline.stop()

    async def test_unknown_phrases_pass_through_in_english(self) -> None:
        sink = NullAudioSink()
        normalizer = GameCommsNormalizer(TEST_PACK)
        pipeline = RealtimeVoicePipeline(
            config=RuntimeConfig(target_language="es"),
            normalizer=normalizer,
            translator=MockTranslator(step_delay=0.001),
            synthesizer=MockSpeechSynthesizer(step_delay=0.001),
            sink=sink,
        )

        await pipeline.start()
        await pipeline.submit_text("can you give him that bobcat?", is_final=True)
        await pipeline.wait_for_idle()

        self.assertGreaterEqual(len(sink.chunks), 1)
        self.assertEqual(sink.chunks[-1].text, "can you give him that bobcat?")
        await pipeline.stop()


if __name__ == "__main__":
    unittest.main()
