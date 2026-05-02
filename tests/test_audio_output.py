from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.audio.output import SoundDeviceVirtualMicSink


class SoundDeviceVirtualMicSinkTests(unittest.TestCase):
    def test_expand_mono_pcm16_duplicates_samples_for_stereo(self) -> None:
        mono = b"\x01\x00\x02\x00"
        stereo = SoundDeviceVirtualMicSink._expand_mono_pcm16(mono, channels=2)

        self.assertEqual(stereo, b"\x01\x00\x01\x00\x02\x00\x02\x00")

    def test_resample_pcm16_mono_changes_length_for_new_rate(self) -> None:
        mono = b"\x00\x00\x10\x00\x20\x00\x10\x00"
        resampled = SoundDeviceVirtualMicSink._resample_pcm16_mono(
            mono,
            source_rate=22_050,
            target_rate=48_000,
        )

        self.assertGreater(len(resampled), len(mono))
        self.assertEqual(len(resampled) % 2, 0)


if __name__ == "__main__":
    unittest.main()
