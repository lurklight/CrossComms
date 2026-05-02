from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class CachedPhrase:
    translated_text: str
    audio_bytes: bytes = b""
    sample_rate: int = 24_000
    updated_at: float = field(default_factory=time.time)


class PhraseCache:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], CachedPhrase] = {}

    def get(self, normalized_text: str, target_language: str) -> CachedPhrase | None:
        return self._items.get((normalized_text, target_language))

    def put(
        self,
        normalized_text: str,
        target_language: str,
        translated_text: str,
        audio_bytes: bytes = b"",
        sample_rate: int = 24_000,
    ) -> None:
        self._items[(normalized_text, target_language)] = CachedPhrase(
            translated_text=translated_text,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )

    def __len__(self) -> int:
        return len(self._items)
