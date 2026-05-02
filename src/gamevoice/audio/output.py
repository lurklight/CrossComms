from __future__ import annotations

from array import array
from typing import Protocol

from ..models import SpeechChunk


class AudioSink(Protocol):
    async def write(self, chunk: SpeechChunk) -> None: ...

    async def close(self) -> None: ...


class NullAudioSink:
    def __init__(self) -> None:
        self.chunks: list[SpeechChunk] = []

    async def write(self, chunk: SpeechChunk) -> None:
        self.chunks.append(chunk)

    async def close(self) -> None:
        return


class SoundDeviceVirtualMicSink:
    def __init__(
        self,
        device_name: str | int | None,
        sample_rate: int = 24_000,
        channels: int = 1,
    ) -> None:
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
        self._stream_sample_rate = sample_rate
        self._stream_channels = channels

    async def write(self, chunk: SpeechChunk) -> None:
        if not chunk.pcm16:
            return

        if self._stream is None:
            self._open_stream(chunk.sample_rate)

        pcm16 = chunk.pcm16
        if chunk.sample_rate != self._stream_sample_rate:
            pcm16 = self._resample_pcm16_mono(
                pcm16,
                source_rate=chunk.sample_rate,
                target_rate=self._stream_sample_rate,
            )
        if self._stream_channels > 1:
            pcm16 = self._expand_mono_pcm16(pcm16, self._stream_channels)

        self._stream.write(pcm16)

    def _open_stream(self, sample_rate: int) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is not installed. Run `python -m pip install -e .[audio]`."
            ) from exc

        resolved_rate, resolved_channels = self._resolve_output_format(sd, sample_rate)
        self._stream_sample_rate = resolved_rate
        self._stream_channels = resolved_channels
        self._stream = sd.RawOutputStream(
            samplerate=resolved_rate,
            channels=resolved_channels,
            dtype="int16",
            device=self.device_name,
        )
        self._stream.start()

    async def close(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _resolve_output_format(self, sd, sample_rate: int) -> tuple[int, int]:
        device_info = sd.query_devices(self.device_name, "output")
        preferred_channels = self._preferred_output_channels(device_info)
        desired_rate = int(sample_rate)
        try:
            sd.check_output_settings(
                device=self.device_name,
                channels=preferred_channels,
                dtype="int16",
                samplerate=desired_rate,
            )
            return desired_rate, preferred_channels
        except Exception as first_error:
            fallback_rate = int(float(device_info.get("default_samplerate", desired_rate)))
            try:
                sd.check_output_settings(
                    device=self.device_name,
                    channels=preferred_channels,
                    dtype="int16",
                    samplerate=fallback_rate,
                )
                return fallback_rate, preferred_channels
            except Exception:
                pass

            mono_channels = 1
            try:
                sd.check_output_settings(
                    device=self.device_name,
                    channels=mono_channels,
                    dtype="int16",
                    samplerate=desired_rate,
                )
                return desired_rate, mono_channels
            except Exception:
                pass

            try:
                sd.check_output_settings(
                    device=self.device_name,
                    channels=mono_channels,
                    dtype="int16",
                    samplerate=fallback_rate,
                )
                return fallback_rate, mono_channels
            except Exception as second_error:
                raise RuntimeError(
                    "Output device rejected the tested stream formats "
                    f"({preferred_channels}ch/{desired_rate}Hz, "
                    f"{preferred_channels}ch/{fallback_rate}Hz, "
                    f"1ch/{desired_rate}Hz, 1ch/{fallback_rate}Hz): {second_error}"
                ) from first_error

    @staticmethod
    def _preferred_output_channels(device_info) -> int:
        max_output_channels = int(device_info.get("max_output_channels", 0) or 0)
        if max_output_channels >= 2:
            return 2
        if max_output_channels >= 1:
            return 1
        return 1

    @staticmethod
    def _expand_mono_pcm16(pcm16: bytes, channels: int) -> bytes:
        if channels <= 1 or not pcm16:
            return pcm16

        mono_samples = array("h")
        mono_samples.frombytes(pcm16)
        expanded = array("h")
        for sample in mono_samples:
            for _ in range(channels):
                expanded.append(sample)
        return expanded.tobytes()

    @staticmethod
    def _resample_pcm16_mono(
        pcm16: bytes,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        if source_rate == target_rate or not pcm16:
            return pcm16

        source_samples = array("h")
        source_samples.frombytes(pcm16)
        if len(source_samples) < 2:
            return pcm16

        target_length = max(1, int(round(len(source_samples) * target_rate / source_rate)))
        step = source_rate / target_rate
        resampled = array("h")

        for index in range(target_length):
            position = index * step
            left_index = int(position)
            if left_index >= len(source_samples) - 1:
                resampled.append(source_samples[-1])
                continue

            right_index = left_index + 1
            fraction = position - left_index
            left_sample = source_samples[left_index]
            right_sample = source_samples[right_index]
            interpolated = int(
                round(left_sample + (right_sample - left_sample) * fraction)
            )
            resampled.append(max(-32_768, min(32_767, interpolated)))

        return resampled.tobytes()
