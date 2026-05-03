from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..models import AudioFrame


@dataclass(slots=True)
class CapturedUtterance:
    pcm16: bytes
    sample_rate: int
    channels: int
    duration_ms: int
    speech_frame_count: int
    speech_ratio: float
    average_speech_rms: float
    peak_rms: float


class UtteranceSegmenter:
    def __init__(
        self,
        frame_ms: int,
        endpoint_silence_ms: int = 160,
        min_speech_ms: int = 180,
        max_utterance_ms: int = 4_500,
        pre_roll_frames: int = 6,
        speech_trigger_ms: int = 60,
        min_speech_ratio: float = 0.38,
        min_peak_rms: float = 420.0,
    ) -> None:
        self.frame_ms = frame_ms
        self._silence_frames_needed = max(1, endpoint_silence_ms // frame_ms)
        self._min_speech_frames = max(1, min_speech_ms // frame_ms)
        self._max_frames = max(1, max_utterance_ms // frame_ms)
        self._speech_trigger_frames = max(1, speech_trigger_ms // frame_ms)
        self._min_speech_ratio = min_speech_ratio
        self._min_peak_rms = min_peak_rms
        self._pre_roll: deque[AudioFrame] = deque(maxlen=pre_roll_frames)
        self._active_frames: list[AudioFrame] = []
        self._speaking = False
        self._pending_speech_frames = 0
        self._speech_frame_count = 0
        self._speech_rms_sum = 0.0
        self._peak_rms = 0.0
        self._trailing_silence_frames = 0
        self._sample_rate = 16_000
        self._channels = 1

    def push(self, frame: AudioFrame) -> CapturedUtterance | None:
        self._sample_rate = frame.sample_rate
        self._channels = frame.channels

        if not self._speaking:
            self._pre_roll.append(frame)
            if frame.is_speech:
                self._pending_speech_frames += 1
            else:
                self._pending_speech_frames = 0

            if self._pending_speech_frames >= self._speech_trigger_frames:
                self._speaking = True
                self._active_frames = list(self._pre_roll)
                self._speech_frame_count = sum(
                    1 for active_frame in self._active_frames if active_frame.is_speech
                )
                self._speech_rms_sum = sum(
                    active_frame.rms
                    for active_frame in self._active_frames
                    if active_frame.is_speech
                )
                self._peak_rms = max(
                    (active_frame.rms for active_frame in self._active_frames),
                    default=0.0,
                )
                self._trailing_silence_frames = 0
            return None

        self._active_frames.append(frame)
        if frame.is_speech:
            self._speech_frame_count += 1
            self._speech_rms_sum += frame.rms
            self._peak_rms = max(self._peak_rms, frame.rms)
            self._trailing_silence_frames = 0
        else:
            self._trailing_silence_frames += 1

        if len(self._active_frames) >= self._max_frames:
            return self._finalize()
        if self._trailing_silence_frames >= self._silence_frames_needed:
            return self._finalize()
        return None

    def flush(self) -> CapturedUtterance | None:
        if not self._speaking:
            return None
        return self._finalize()

    def _finalize(self) -> CapturedUtterance | None:
        utterance = self._captured_utterance_from_active_frames()
        self._reset_state()
        return utterance

    def _captured_utterance_from_active_frames(self) -> CapturedUtterance | None:
        total_frames = len(self._active_frames)
        speech_ratio = (
            self._speech_frame_count / total_frames if total_frames else 0.0
        )
        average_speech_rms = (
            self._speech_rms_sum / self._speech_frame_count
            if self._speech_frame_count
            else 0.0
        )
        if (
            self._speech_frame_count >= self._min_speech_frames
            and speech_ratio >= self._min_speech_ratio
            and self._peak_rms >= self._min_peak_rms
        ):
            return CapturedUtterance(
                pcm16=b"".join(frame.pcm16 for frame in self._active_frames),
                sample_rate=self._sample_rate,
                channels=self._channels,
                duration_ms=total_frames * self.frame_ms,
                speech_frame_count=self._speech_frame_count,
                speech_ratio=speech_ratio,
                average_speech_rms=average_speech_rms,
                peak_rms=self._peak_rms,
            )
        return None

    def _reset_state(self) -> None:
        self._pre_roll.clear()
        self._active_frames = []
        self._speaking = False
        self._pending_speech_frames = 0
        self._speech_frame_count = 0
        self._speech_rms_sum = 0.0
        self._peak_rms = 0.0
        self._trailing_silence_frames = 0
