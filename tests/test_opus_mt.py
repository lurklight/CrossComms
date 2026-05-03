from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.providers.opus_mt import OpusMtTextTranslator


class OpusMtTextTranslatorTests(unittest.TestCase):
    def test_language_key_normalizes_locale_variants(self) -> None:
        self.assertEqual(OpusMtTextTranslator._opus_language_key("pt-BR"), "pt")
        self.assertEqual(OpusMtTextTranslator._opus_language_key("zh-CN"), "zh")
        self.assertEqual(OpusMtTextTranslator._opus_language_key("es"), "es")

    def test_translation_plan_uses_direct_model_when_available(self) -> None:
        self.assertEqual(
            OpusMtTextTranslator._translation_plan("en", "es"),
            [("en", "es")],
        )

    def test_translation_plan_pivots_through_english(self) -> None:
        self.assertEqual(
            OpusMtTextTranslator._translation_plan("ru", "es"),
            [("ru", "en"), ("en", "es")],
        )

    def test_translation_plan_rejects_unsupported_language(self) -> None:
        with self.assertRaises(RuntimeError):
            OpusMtTextTranslator._translation_plan("ja", "es")

    def test_translate_sync_pivots_and_caches(self) -> None:
        translator = OpusMtTextTranslator(download_root=Path("unused"))
        with patch.object(
            translator,
            "_translate_once",
            side_effect=["need ammo", "necesito municion"],
        ) as mocked_translate_once:
            first = translator._translate_sync("need ammo", "ru", "es")
            second = translator._translate_sync("need ammo", "ru", "es")

        self.assertEqual(first, "necesito municion")
        self.assertEqual(second, "necesito municion")
        self.assertEqual(mocked_translate_once.call_count, 2)
        self.assertEqual(
            mocked_translate_once.call_args_list[0].args,
            ("need ammo", "ru", "en"),
        )
        self.assertEqual(
            mocked_translate_once.call_args_list[1].args,
            ("need ammo", "en", "es"),
        )


if __name__ == "__main__":
    unittest.main()
