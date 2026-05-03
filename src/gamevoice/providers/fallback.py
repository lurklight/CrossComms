from __future__ import annotations

from collections.abc import AsyncIterator

from ..models import NormalizedChunk, TranslationChunk


class FallbackTextTranslator:
    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary

    def validate_runtime(self) -> None:
        validate_runtime = getattr(self.primary, "validate_runtime", None)
        if callable(validate_runtime):
            validate_runtime()

    async def stream_translate(
        self,
        chunk: NormalizedChunk,
        source_language: str,
        target_language: str,
    ) -> AsyncIterator[TranslationChunk]:
        try:
            async for translated in self.primary.stream_translate(
                chunk,
                source_language,
                target_language,
            ):
                yield translated
                return
        except Exception:
            pass

        async for translated in self.secondary.stream_translate(
            chunk,
            source_language,
            target_language,
        ):
            yield translated
