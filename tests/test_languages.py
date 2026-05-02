from __future__ import annotations

from pathlib import Path
import sys
import unittest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.languages import load_language_options


class LanguageOptionTests(unittest.TestCase):
    def test_loader_reads_languages_from_repo_json(self) -> None:
        path = Path(__file__).resolve().parents[1] / "languages.json"

        options = load_language_options(path)

        self.assertGreaterEqual(len(options), 8)
        self.assertEqual(options[0].code, "en")
        self.assertEqual(options[0].label, "English (en)")

    def test_loader_uses_voice_family_fallback_from_code_prefix(self) -> None:
        path = Path(__file__).resolve().parents[1] / "tmp-test-languages.json"
        try:
            path.write_text(
                '{"languages":[{"code":"pt-BR","name":"Portuguese (Brazil)"}]}',
                encoding="utf-8",
            )
            options = load_language_options(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(len(options), 1)
        self.assertEqual(options[0].voice_family, "pt")


if __name__ == "__main__":
    unittest.main()
