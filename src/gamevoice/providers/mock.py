from __future__ import annotations

import asyncio
import math
import struct

from ..models import NormalizedChunk, SpeechChunk, TranslationChunk


SPANISH_PHRASES = {
    "enemy is very low health": "el enemigo tiene muy poca vida",
    "armor broken": "armadura rota",
    "moving to b": "moviendose hacia b",
    "moving to a": "moviendose hacia a",
    "attack now": "ataquen ahora",
    "retreat now": "retrocedan ahora",
    "enemy behind us": "enemigo detras de nosotros",
    "need help mid": "necesito ayuda en medio",
}

FRENCH_PHRASES = {
    "enemy is very low health": "ennemi presque a terre",
    "armor broken": "armure brisee",
    "moving to b": "rotation vers b",
    "moving to a": "rotation vers a",
    "attack now": "attaque maintenant",
    "retreat now": "repliez vous",
    "enemy behind us": "ennemi derriere nous",
    "need help mid": "besoin d aide au milieu",
}

GERMAN_PHRASES = {
    "enemy is very low health": "gegner hat kaum leben",
    "armor broken": "ruestung ist gebrochen",
    "moving to b": "bewege mich zu b",
    "moving to a": "bewege mich zu a",
    "attack now": "jetzt angreifen",
    "retreat now": "zurueckziehen",
    "enemy behind us": "gegner hinter uns",
    "need help mid": "brauche hilfe in der mitte",
}


class MockTranslator:
    def __init__(self, step_delay: float = 0.04) -> None:
        self.step_delay = step_delay
        self._phrasebooks = {
            "es": SPANISH_PHRASES,
            "fr": FRENCH_PHRASES,
            "de": GERMAN_PHRASES,
        }

    async def stream_translate(
        self,
        chunk: NormalizedChunk,
        source_language: str,
        target_language: str,
    ):
        translated = self._translate_text(chunk.normalized_text, target_language)
        partials = self._partials(translated)
        for index, partial in enumerate(partials):
            await asyncio.sleep(self.step_delay)
            yield TranslationChunk(
                source_text=chunk.normalized_text,
                translated_text=partial,
                source_language=source_language,
                target_language=target_language,
                sequence_id=chunk.sequence_id,
                is_final=index == len(partials) - 1 and chunk.is_final,
                cache_key=chunk.normalized_text,
            )

    def _translate_text(self, text: str, target_language: str) -> str:
        phrasebook = self._phrasebooks.get(target_language, {})
        translated = text
        for source_phrase, target_phrase in sorted(
            phrasebook.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            translated = translated.replace(source_phrase, target_phrase)
        if translated == text:
            return text
        return translated

    def _partials(self, translated: str) -> list[str]:
        words = translated.split()
        if len(words) <= 3:
            return [translated]
        midpoint = max(2, len(words) // 2)
        return [" ".join(words[:midpoint]), translated]


class MockSpeechSynthesizer:
    def __init__(self, sample_rate: int = 24_000, step_delay: float = 0.02) -> None:
        self.sample_rate = sample_rate
        self.step_delay = step_delay

    async def stream_speech(self, chunk: TranslationChunk):
        if not chunk.translated_text:
            return
        await asyncio.sleep(self.step_delay)
        duration_ms = max(180, min(900, len(chunk.translated_text) * 24))
        tone = _sine_wave_pcm16(
            duration_ms=duration_ms,
            sample_rate=self.sample_rate,
            frequency=420 + (chunk.sequence_id % 5) * 40,
        )
        yield SpeechChunk(
            text=chunk.translated_text,
            sequence_id=chunk.sequence_id,
            sample_rate=self.sample_rate,
            pcm16=tone,
            is_final=chunk.is_final,
            cache_key=chunk.cache_key,
        )


def _sine_wave_pcm16(
    duration_ms: int,
    sample_rate: int,
    frequency: float,
    amplitude: float = 0.2,
) -> bytes:
    total_samples = int(sample_rate * (duration_ms / 1_000))
    frame_data = bytearray()
    for index in range(total_samples):
        sample = amplitude * math.sin(2 * math.pi * frequency * (index / sample_rate))
        frame_data.extend(struct.pack("<h", int(sample * 32_767)))
    return bytes(frame_data)
