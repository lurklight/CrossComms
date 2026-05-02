from __future__ import annotations

import asyncio
import math

from ..models import AudioFrame


class EnergyVoiceActivityDetector:
    def __init__(
        self,
        threshold: int = 260,
        sample_width: int = 2,
        min_noise_floor: float = 90.0,
        max_noise_floor: float = 1_400.0,
        start_multiplier: float = 2.0,
        continue_multiplier: float = 1.45,
        adaptation: float = 0.08,
    ) -> None:
        self.threshold = threshold
        self.sample_width = sample_width
        self.min_noise_floor = min_noise_floor
        self.max_noise_floor = max_noise_floor
        self.start_multiplier = start_multiplier
        self.continue_multiplier = continue_multiplier
        self.adaptation = adaptation
        self._noise_floor = float(max(threshold * 0.45, min_noise_floor))
        self._speaking = False

    def is_speech(self, pcm16: bytes) -> bool:
        return self.analyze(pcm16)[0]

    def analyze(self, pcm16: bytes) -> tuple[bool, float]:
        if not pcm16:
            self._speaking = False
            return False, 0.0

        rms = self._rms(pcm16)
        multiplier = (
            self.continue_multiplier if self._speaking else self.start_multiplier
        )
        effective_threshold = max(self.threshold, self._noise_floor * multiplier)
        is_speech = rms >= effective_threshold

        if not is_speech:
            observed_floor = min(
                self.max_noise_floor,
                max(self.min_noise_floor, rms),
            )
            self._noise_floor = (
                (1.0 - self.adaptation) * self._noise_floor
                + self.adaptation * observed_floor
            )
        else:
            self._noise_floor = max(
                self.min_noise_floor,
                min(self._noise_floor, rms * 0.8),
            )

        self._speaking = is_speech
        return is_speech, rms

    def _rms(self, pcm16: bytes) -> float:
        if self.sample_width != 2:
            raise ValueError("EnergyVoiceActivityDetector currently expects 16-bit PCM input.")

        sample_bytes = len(pcm16) - (len(pcm16) % self.sample_width)
        if sample_bytes == 0:
            return 0.0

        samples = memoryview(pcm16[:sample_bytes]).cast("h")
        mean_square = sum(sample * sample for sample in samples) / len(samples)
        return math.sqrt(mean_square)


class SoundDeviceMicrophoneSource:
    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        frame_ms: int = 30,
        device: str | int | None = None,
        vad: EnergyVoiceActivityDetector | None = None,
        queue_size: int = 128,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.device = device
        self.vad = vad or EnergyVoiceActivityDetector()
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=queue_size)
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. Run `python -m pip install -e .[audio]`."
            ) from exc

        if self._stream is not None:
            return

        self.sample_rate = self._resolve_supported_sample_rate(sd)
        self._loop = asyncio.get_running_loop()
        frame_size = int(self.sample_rate * (self.frame_ms / 1_000))

        def callback(indata: bytes, frames: int, time_info, status) -> None:
            del frames, time_info, status
            assert self._loop is not None
            chunk = bytes(indata)
            is_speech, rms = self.vad.analyze(chunk)
            frame = AudioFrame(
                pcm16=chunk,
                sample_rate=self.sample_rate,
                channels=self.channels,
                is_speech=is_speech,
                rms=rms,
            )
            self._loop.call_soon_threadsafe(self._enqueue_frame, frame)

        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=frame_size,
            channels=self.channels,
            dtype="int16",
            callback=callback,
            device=self.device,
        )
        self._stream.start()

    def _enqueue_frame(self, frame: AudioFrame) -> None:
        if self._queue.full():
            return
        self._queue.put_nowait(frame)

    async def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    async def frames(self):
        while True:
            yield await self._queue.get()

    def _resolve_supported_sample_rate(self, sd) -> int:
        desired_rate = int(self.sample_rate)
        try:
            sd.check_input_settings(
                device=self.device,
                channels=self.channels,
                dtype="int16",
                samplerate=desired_rate,
            )
            return desired_rate
        except Exception as first_error:
            device_info = sd.query_devices(self.device, "input")
            fallback_rate = int(float(device_info.get("default_samplerate", desired_rate)))
            if fallback_rate == desired_rate:
                raise RuntimeError(
                    f"Input device rejected sample rate {desired_rate} Hz: {first_error}"
                ) from first_error

            try:
                sd.check_input_settings(
                    device=self.device,
                    channels=self.channels,
                    dtype="int16",
                    samplerate=fallback_rate,
                )
            except Exception as second_error:
                raise RuntimeError(
                    "Input device rejected both the requested sample rate "
                    f"({desired_rate} Hz) and the device default ({fallback_rate} Hz): {second_error}"
                ) from second_error
            return fallback_rate
