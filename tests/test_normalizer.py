from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.comms.normalizer import GameCommsNormalizer, SlangPack
from gamevoice.models import TranscriptChunk


TEST_PACK = SlangPack(
    name="Test Pack",
    replacements={
        "one shot": "enemy is very low health",
        "cracked": "armor broken",
        "rotating b": "moving to b",
    },
    fast_commands={
        "enemy low": "enemy is very low health",
    },
)


class NormalizerTests(unittest.TestCase):
    def test_rewrites_common_shooter_slang(self) -> None:
        normalizer = GameCommsNormalizer(TEST_PACK)

        chunk = TranscriptChunk(
            text="he's one shot, cracked, rotating B",
            sequence_id=1,
            is_final=True,
        )
        normalized = normalizer.normalize(chunk)

        self.assertIn("enemy is very low health", normalized.normalized_text)
        self.assertIn("armor broken", normalized.normalized_text)
        self.assertIn("moving to b", normalized.normalized_text)

    def test_exact_fast_command_uses_fast_path(self) -> None:
        normalizer = GameCommsNormalizer(TEST_PACK)

        chunk = TranscriptChunk(text="enemy low", sequence_id=2, is_final=True)
        normalized = normalizer.normalize(chunk)

        self.assertTrue(normalized.fast_path)
        self.assertEqual(normalized.normalized_text, "enemy is very low health")

    def test_custom_replacement_is_applied(self) -> None:
        normalizer = GameCommsNormalizer(
            TEST_PACK,
            custom_replacements={"ratting": "moving slowly and hiding"},
        )

        chunk = TranscriptChunk(text="ratting near mid", sequence_id=3, is_final=True)
        normalized = normalizer.normalize(chunk)

        self.assertEqual(normalized.normalized_text, "moving slowly and hiding near mid")


if __name__ == "__main__":
    unittest.main()
