from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..models import NormalizedChunk, TranslationChunk


DEFAULT_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Referer": "https://translate.google.com/",
}


class FreeWebTextTranslator:
    def __init__(
        self,
        timeout_s: float = 8.0,
    ) -> None:
        self.timeout_s = timeout_s

    async def stream_translate(
        self,
        chunk: NormalizedChunk,
        source_language: str,
        target_language: str,
    ) -> AsyncIterator[TranslationChunk]:
        translated = await self._translate_text(
            chunk.normalized_text,
            source_language,
            target_language,
        )
        yield TranslationChunk(
            source_text=chunk.normalized_text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            sequence_id=chunk.sequence_id,
            is_final=chunk.is_final,
            cache_key=chunk.normalized_text,
        )

    async def _translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        return await asyncio.to_thread(
            self._translate_sync,
            text,
            source_language,
            target_language,
        )

    def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        cleaned = text.strip()
        if not cleaned:
            return text
        if source_language.strip().lower() == target_language.strip().lower():
            return text

        request = Request(
            "https://translate.googleapis.com/translate_a/single?"
            + urlencode(
                {
                    "client": "gtx",
                    "sl": source_language or "auto",
                    "tl": target_language,
                    "dt": "t",
                    "q": cleaned,
                }
            ),
            headers=DEFAULT_WEB_HEADERS,
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        translated = self._extract_translation(payload)
        return translated or text

    @staticmethod
    def _extract_translation(payload: Any) -> str:
        if not isinstance(payload, list) or not payload:
            return ""

        segments = payload[0]
        if not isinstance(segments, list):
            return ""

        parts: list[str] = []
        for segment in segments:
            if not isinstance(segment, list) or not segment:
                continue
            translated = segment[0]
            if isinstance(translated, str):
                parts.append(translated)
        return "".join(parts).strip()
