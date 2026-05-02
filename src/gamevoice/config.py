from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

DeviceRef: TypeAlias = str | int | None


@dataclass(slots=True)
class RuntimeConfig:
    source_language: str = "en"
    target_language: str = "es"
    sample_rate: int = 48_000
    input_sample_rate: int = 48_000
    frame_ms: int = 30
    vad_threshold: int = 320
    vad_start_multiplier: float = 2.6
    vad_continue_multiplier: float = 1.8
    min_speech_ms: int = 300
    speech_trigger_ms: int = 120
    min_speech_ratio: float = 0.5
    min_peak_rms: float = 600.0
    min_average_speech_rms: float = 450.0
    endpoint_silence_ms: int = 480
    max_utterance_ms: int = 4_500
    queue_size: int = 64
    input_mic_name: DeviceRef = None
    virtual_mic_name: DeviceRef = None
    source_voice_model: str | None = None
    target_voice_model: str | None = None
    whisper_model: str | None = None
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
