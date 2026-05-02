from __future__ import annotations

from dataclasses import dataclass, field
import re

from ..models import NormalizedChunk, TranscriptChunk


def _canonicalize(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


@dataclass(slots=True)
class SlangPack:
    name: str
    source_language: str = "en"
    replacements: dict[str, str] = field(default_factory=dict)
    fast_commands: dict[str, str] = field(default_factory=dict)


class GameCommsNormalizer:
    def __init__(
        self,
        slang_pack: SlangPack,
        custom_replacements: dict[str, str] | None = None,
    ) -> None:
        self.slang_pack = slang_pack
        merged = dict(slang_pack.replacements)
        if custom_replacements:
            merged.update(
                {
                    _canonicalize(key): _canonicalize(value)
                    for key, value in custom_replacements.items()
                }
            )
        self._compiled_rules = [
            (
                term,
                re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", flags=re.IGNORECASE),
                replacement,
            )
            for term, replacement in sorted(
                merged.items(),
                key=lambda item: len(item[0]),
                reverse=True,
            )
        ]

    def normalize(self, chunk: TranscriptChunk) -> NormalizedChunk:
        canonical_text = _canonicalize(chunk.text)
        matched_terms: list[str] = []

        fast_match = self.slang_pack.fast_commands.get(canonical_text)
        if fast_match is not None:
            return NormalizedChunk(
                source_text=chunk.text,
                normalized_text=fast_match,
                sequence_id=chunk.sequence_id,
                is_final=chunk.is_final,
                fast_path=True,
                matched_terms=(canonical_text,),
            )

        normalized = canonical_text
        for term, pattern, replacement in self._compiled_rules:
            normalized, count = pattern.subn(replacement, normalized)
            if count:
                matched_terms.append(term)

        normalized = re.sub(r"\s+", " ", normalized).strip(" ,.")
        final_fast_match = self.slang_pack.fast_commands.get(normalized)
        if final_fast_match is not None:
            normalized = final_fast_match
            matched_terms.append(normalized)

        return NormalizedChunk(
            source_text=chunk.text,
            normalized_text=normalized,
            sequence_id=chunk.sequence_id,
            is_final=chunk.is_final,
            fast_path=final_fast_match is not None,
            matched_terms=tuple(dict.fromkeys(matched_terms)),
        )
