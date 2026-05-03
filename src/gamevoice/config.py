from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

DeviceRef: TypeAlias = str | int | None

STT_MODE_CPU_ONLY = "CPU Only"
STT_MODE_GPU_ONLY = "GPU Only"
TRANSLATION_MODE_WEB = "Web"
TRANSLATION_MODE_LOCAL_OPUS = "Local OPUS"


def whisper_runtime_for_mode(mode: str) -> tuple[str, str]:
    normalized = mode.strip().casefold()
    if normalized == STT_MODE_GPU_ONLY.casefold():
        return "cuda", "float16"
    return "cpu", "int8"


def translation_mode_for_label(mode: str) -> str:
    normalized = mode.strip().casefold()
    if normalized == TRANSLATION_MODE_LOCAL_OPUS.casefold():
        return TRANSLATION_MODE_LOCAL_OPUS
    return TRANSLATION_MODE_WEB


@dataclass(slots=True)
class RuntimeConfig:
    source_language: str = "en"
    target_language: str = "es"
    sample_rate: int = 48_000
    input_sample_rate: int = 48_000
    frame_ms: int = 20
    vad_threshold: int = 320
    vad_start_multiplier: float = 2.6
    vad_continue_multiplier: float = 1.8
    min_speech_ms: int = 180
    speech_trigger_ms: int = 60
    min_speech_ratio: float = 0.5
    min_peak_rms: float = 600.0
    min_average_speech_rms: float = 450.0
    endpoint_silence_ms: int = 160
    max_utterance_ms: int = 4_500
    queue_size: int = 64
    input_mic_name: DeviceRef = None
    virtual_mic_name: DeviceRef = None
    source_voice_model: str | None = None
    target_voice_model: str | None = None
    whisper_model: str | None = None
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_beam_size: int = 1
    whisper_best_of: int = 1
    stt_mode_label: str = STT_MODE_CPU_ONLY
    translation_mode_label: str = TRANSLATION_MODE_WEB
    push_to_talk_enabled: bool = True
    push_to_talk_key_code: int | None = 0x06
    push_to_talk_key_label: str = "Mouse 5"
