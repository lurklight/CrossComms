from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_LANGUAGE_DEFINITIONS = [
    {
        "code": "en",
        "name": "English",
        "voice_family": "en",
        "default_source_voice": "en_US-amy-medium.onnx",
        "default_target_voice": "en_US-amy-medium.onnx",
    },
    {
        "code": "es",
        "name": "Spanish",
        "voice_family": "es",
        "default_source_voice": "es_ES-davefx-medium.onnx",
        "default_target_voice": "es_ES-davefx-medium.onnx",
    },
    {
        "code": "fr",
        "name": "French",
        "voice_family": "fr",
        "default_source_voice": "fr_FR-siwis-medium.onnx",
        "default_target_voice": "fr_FR-siwis-medium.onnx",
    },
    {
        "code": "de",
        "name": "German",
        "voice_family": "de",
        "default_source_voice": "de_DE-thorsten-high.onnx",
        "default_target_voice": "de_DE-thorsten-high.onnx",
    },
    {
        "code": "it",
        "name": "Italian",
        "voice_family": "it",
        "default_source_voice": "it_IT-paola-medium.onnx",
        "default_target_voice": "it_IT-paola-medium.onnx",
    },
    {
        "code": "pt",
        "name": "Portuguese",
        "voice_family": "pt",
        "default_source_voice": "pt_PT-tugao-medium.onnx",
        "default_target_voice": "pt_PT-tugao-medium.onnx",
    },
    {
        "code": "pt-BR",
        "name": "Portuguese (Brazil)",
        "voice_family": "pt",
        "default_source_voice": "pt_BR-faber-medium.onnx",
        "default_target_voice": "pt_BR-faber-medium.onnx",
    },
    {
        "code": "ru",
        "name": "Russian",
        "voice_family": "ru",
        "default_source_voice": "ru_RU-irina-medium.onnx",
        "default_target_voice": "ru_RU-irina-medium.onnx",
    },
    {
        "code": "vi",
        "name": "Vietnamese",
        "voice_family": "vi",
        "default_source_voice": "vi_VN-vais1000-medium.onnx",
        "default_target_voice": "vi_VN-vais1000-medium.onnx",
    },
    {
        "code": "zh-CN",
        "name": "Chinese (Simplified)",
        "voice_family": "zh",
        "default_source_voice": "zh_CN-huayan-medium.onnx",
        "default_target_voice": "zh_CN-huayan-medium.onnx",
    },
]


@dataclass(frozen=True, slots=True)
class LanguageOption:
    code: str
    name: str
    voice_family: str
    default_source_voice: str | None = None
    default_target_voice: str | None = None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.code})"


def load_language_options(path: Path | None = None) -> list[LanguageOption]:
    definitions = DEFAULT_LANGUAGE_DEFINITIONS
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            raw = raw.get("languages", [])
        if isinstance(raw, list) and raw:
            definitions = raw

    options: list[LanguageOption] = []
    for item in definitions:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        name = str(item.get("name", "")).strip()
        if not code or not name:
            continue
        voice_family = str(item.get("voice_family", code.split("-", maxsplit=1)[0])).strip()
        options.append(
            LanguageOption(
                code=code,
                name=name,
                voice_family=voice_family,
                default_source_voice=_optional_text(item.get("default_source_voice")),
                default_target_voice=_optional_text(item.get("default_target_voice")),
            )
        )
    return options


def _optional_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
