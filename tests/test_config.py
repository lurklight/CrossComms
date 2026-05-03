from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.config import (
    STT_MODE_CPU_ONLY,
    STT_MODE_GPU_ONLY,
    TRANSLATION_MODE_LOCAL_OPUS,
    TRANSLATION_MODE_WEB,
    translation_mode_for_label,
    whisper_runtime_for_mode,
)


class RuntimeConfigTests(unittest.TestCase):
    def test_cpu_only_mode_maps_to_cpu_int8(self) -> None:
        self.assertEqual(
            whisper_runtime_for_mode(STT_MODE_CPU_ONLY),
            ("cpu", "int8"),
        )

    def test_gpu_only_mode_maps_to_cuda_float16(self) -> None:
        self.assertEqual(
            whisper_runtime_for_mode(STT_MODE_GPU_ONLY),
            ("cuda", "float16"),
        )

    def test_unknown_mode_defaults_to_cpu(self) -> None:
        self.assertEqual(
            whisper_runtime_for_mode("whatever"),
            ("cpu", "int8"),
        )

    def test_translation_mode_normalizes_local_opus(self) -> None:
        self.assertEqual(
            translation_mode_for_label(TRANSLATION_MODE_LOCAL_OPUS),
            TRANSLATION_MODE_LOCAL_OPUS,
        )

    def test_unknown_translation_mode_defaults_to_web(self) -> None:
        self.assertEqual(
            translation_mode_for_label("whatever"),
            TRANSLATION_MODE_WEB,
        )


if __name__ == "__main__":
    unittest.main()
