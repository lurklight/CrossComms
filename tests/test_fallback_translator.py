from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.models import NormalizedChunk, TranslationChunk
from gamevoice.providers.fallback import FallbackTextTranslator


class _PrimaryOk:
    async def stream_translate(self, chunk, source_language, target_language):
        yield TranslationChunk(
            source_text=chunk.normalized_text,
            translated_text="local-ok",
            source_language=source_language,
            target_language=target_language,
            sequence_id=chunk.sequence_id,
            is_final=chunk.is_final,
        )


class _PrimaryFail:
    async def stream_translate(self, chunk, source_language, target_language):
        raise RuntimeError("broken")
        yield


class _Secondary:
    async def stream_translate(self, chunk, source_language, target_language):
        yield TranslationChunk(
            source_text=chunk.normalized_text,
            translated_text="web-fallback",
            source_language=source_language,
            target_language=target_language,
            sequence_id=chunk.sequence_id,
            is_final=chunk.is_final,
        )


class FallbackTextTranslatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_primary_when_primary_succeeds(self) -> None:
        translator = FallbackTextTranslator(_PrimaryOk(), _Secondary())
        chunk = NormalizedChunk(
            source_text="enemy low",
            normalized_text="enemy low",
            sequence_id=1,
            is_final=True,
        )

        results = [
            translated.translated_text
            async for translated in translator.stream_translate(chunk, "en", "es")
        ]

        self.assertEqual(results, ["local-ok"])

    async def test_falls_back_when_primary_raises(self) -> None:
        translator = FallbackTextTranslator(_PrimaryFail(), _Secondary())
        chunk = NormalizedChunk(
            source_text="enemy low",
            normalized_text="enemy low",
            sequence_id=1,
            is_final=True,
        )

        results = [
            translated.translated_text
            async for translated in translator.stream_translate(chunk, "en", "es")
        ]

        self.assertEqual(results, ["web-fallback"])


if __name__ == "__main__":
    unittest.main()
