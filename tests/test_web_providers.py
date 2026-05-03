from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gamevoice.models import NormalizedChunk, TranslationChunk
from gamevoice.providers.edge_neural_tts import EdgeNeuralSpeechSynthesizer
from gamevoice.providers.piper_tts import PiperSpeechSynthesizer, list_installed_piper_voices
from gamevoice.providers.web import FreeWebTextTranslator


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _wav_bytes(sample_rate: int = 22_050, frame_count: int = 1_600) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((b"\x20\x00" * frame_count))
    return buffer.getvalue()


class FreeWebTextTranslatorTests(unittest.IsolatedAsyncioTestCase):
    def test_translator_extracts_joined_segments(self) -> None:
        payload = [[["hola", "hello", None, None, 1], [" amigo", " friend", None, None, 1]]]
        self.assertEqual(FreeWebTextTranslator._extract_translation(payload), "hola amigo")

    async def test_translator_uses_web_response(self) -> None:
        translator = FreeWebTextTranslator()
        chunk = NormalizedChunk(
            source_text="hello friend",
            normalized_text="hello friend",
            sequence_id=1,
            is_final=True,
        )

        with patch(
            "gamevoice.providers.web.urlopen",
            return_value=_FakeResponse(b'[[["hola amigo","hello friend",null,null,1]]]'),
        ) as mocked_urlopen:
            results = []
            async for translated in translator.stream_translate(chunk, "en", "es"):
                results.append(translated.translated_text)

        self.assertEqual(results, ["hola amigo"])
        self.assertEqual(mocked_urlopen.call_count, 1)

    async def test_translator_reuses_cached_result_for_repeat_phrase(self) -> None:
        translator = FreeWebTextTranslator()
        chunk = NormalizedChunk(
            source_text="need ammo",
            normalized_text="need ammo",
            sequence_id=1,
            is_final=True,
        )

        with patch(
            "gamevoice.providers.web.urlopen",
            return_value=_FakeResponse(b'[[["necesito municion","need ammo",null,null,1]]]'),
        ) as mocked_urlopen:
            first = [
                translated.translated_text
                async for translated in translator.stream_translate(chunk, "en", "es")
            ]
            second = [
                translated.translated_text
                async for translated in translator.stream_translate(chunk, "en", "es")
            ]

        self.assertEqual(first, ["necesito municion"])
        self.assertEqual(second, ["necesito municion"])
        self.assertEqual(mocked_urlopen.call_count, 1)


class EdgeNeuralSpeechSynthesizerTests(unittest.TestCase):
    def test_read_wave_file_returns_sample_rate_and_pcm(self) -> None:
        path = ROOT / "tmp-test-wave.wav"
        try:
            path.write_bytes(_wav_bytes())
            sample_rate, pcm16 = EdgeNeuralSpeechSynthesizer._read_wave_file(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(sample_rate, 22_050)
        self.assertGreater(len(pcm16), 0)
        self.assertEqual(len(pcm16) % 2, 0)

    def test_resample_pcm16_mono_changes_length_for_new_rate(self) -> None:
        original = b"\x00\x00\x10\x00\x20\x00\x10\x00"
        resampled = EdgeNeuralSpeechSynthesizer._resample_pcm16_mono(
            original,
            source_rate=22_050,
            target_rate=44_100,
        )
        self.assertGreater(len(resampled), len(original))
        self.assertEqual(len(resampled) % 2, 0)

    def test_passthrough_translation_uses_source_voice_language(self) -> None:
        synthesizer = EdgeNeuralSpeechSynthesizer()
        chunk = TranslationChunk(
            source_text="can you give him that bobcat?",
            translated_text="can you give him that bobcat?",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )

        self.assertEqual(synthesizer._choose_voice(chunk), "en-US-AriaNeural")

    def test_translated_phrase_uses_target_voice_language(self) -> None:
        synthesizer = EdgeNeuralSpeechSynthesizer()
        chunk = TranslationChunk(
            source_text="hi, how are you doing?",
            translated_text="hola como estas",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )

        self.assertEqual(synthesizer._choose_voice(chunk), "es-MX-DaliaNeural")


class PiperSpeechSynthesizerTests(unittest.TestCase):
    def test_list_installed_piper_voices_parses_labels(self) -> None:
        runtime_root = ROOT / "fake-piper-runtime"
        fake_paths = [
            runtime_root / "voices" / "en_US-amy-medium.onnx",
            runtime_root / "voices" / "es_ES-davefx-medium.onnx",
        ]
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.glob",
            return_value=fake_paths,
        ):
            voices = list_installed_piper_voices(runtime_root)

        self.assertEqual(
            [
                (
                    voice.file_name,
                    voice.label,
                    voice.language_family,
                    voice.locale_code,
                )
                for voice in voices
            ],
            [
                ("en_US-amy-medium.onnx", "en_US / amy / medium", "en", "en_US"),
                ("es_ES-davefx-medium.onnx", "es_ES / davefx / medium", "es", "es_ES"),
            ],
        )

    def test_read_wave_file_returns_sample_rate_and_pcm(self) -> None:
        path = ROOT / "tmp-test-piper-wave.wav"
        try:
            path.write_bytes(_wav_bytes())
            sample_rate, pcm16 = PiperSpeechSynthesizer._read_wave_file(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(sample_rate, 22_050)
        self.assertGreater(len(pcm16), 0)
        self.assertEqual(len(pcm16) % 2, 0)

    def test_passthrough_translation_uses_source_language_model(self) -> None:
        synthesizer = PiperSpeechSynthesizer(
            runtime_root=ROOT / "fake-piper-runtime",
            model_map={
                "en": "en_US-lessac-high.onnx",
                "es": "es_MX-claude-high.onnx",
            },
        )
        chunk = TranslationChunk(
            source_text="can you give him that bobcat?",
            translated_text="can you give him that bobcat?",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )

        with patch("pathlib.Path.exists", return_value=True):
            self.assertEqual(
                synthesizer._choose_model_path(chunk).name,
                "en_US-lessac-high.onnx",
            )

    def test_translated_phrase_uses_target_language_model(self) -> None:
        synthesizer = PiperSpeechSynthesizer(
            runtime_root=ROOT / "fake-piper-runtime",
            model_map={
                "en": "en_US-lessac-high.onnx",
                "es": "es_MX-claude-high.onnx",
            },
        )
        chunk = TranslationChunk(
            source_text="hi, how are you doing?",
            translated_text="hola como estas",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )

        with patch("pathlib.Path.exists", return_value=True):
            self.assertEqual(
                synthesizer._choose_model_path(chunk).name,
                "es_MX-claude-high.onnx",
            )

    def test_explicit_target_voice_model_overrides_default_mapping(self) -> None:
        synthesizer = PiperSpeechSynthesizer(
            runtime_root=ROOT / "fake-piper-runtime",
            target_voice_model="es_ES-davefx-medium.onnx",
        )
        chunk = TranslationChunk(
            source_text="how are you doing?",
            translated_text="como estas?",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )

        with patch("pathlib.Path.exists", return_value=True):
            self.assertEqual(
                synthesizer._choose_model_path(chunk).name,
                "es_ES-davefx-medium.onnx",
            )

    def test_exact_language_model_mapping_beats_family_fallback(self) -> None:
        synthesizer = PiperSpeechSynthesizer(
            runtime_root=ROOT / "fake-piper-runtime",
            model_map={
                "en": "en_US-lessac-high.onnx",
                "pt": "pt_PT-tugao-medium.onnx",
                "pt-br": "pt_BR-faber-medium.onnx",
            },
        )
        chunk = TranslationChunk(
            source_text="hi",
            translated_text="oi",
            source_language="en",
            target_language="pt-BR",
            sequence_id=1,
            is_final=True,
        )

        with patch("pathlib.Path.exists", return_value=True):
            self.assertEqual(
                synthesizer._choose_model_path(chunk).name,
                "pt_BR-faber-medium.onnx",
            )

    def test_synthesizer_reuses_cached_audio_for_repeat_phrase(self) -> None:
        synthesizer = PiperSpeechSynthesizer(
            runtime_root=ROOT / "fake-piper-runtime",
            model_map={
                "en": "en_US-lessac-high.onnx",
                "es": "es_MX-claude-high.onnx",
            },
        )
        chunk = TranslationChunk(
            source_text="hi",
            translated_text="hola",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )

        with patch("pathlib.Path.exists", return_value=True), patch(
            "gamevoice.providers.piper_tts.subprocess.run"
        ) as mocked_run, patch.object(
            PiperSpeechSynthesizer,
            "_read_wave_file",
            return_value=(22_050, b"\x20\x00" * 128),
        ):
            mocked_run.return_value.returncode = 0
            mocked_run.return_value.stderr = ""
            mocked_run.return_value.stdout = ""

            first_rate, first_pcm16 = synthesizer._synthesize_sync(chunk)
            second_rate, second_pcm16 = synthesizer._synthesize_sync(chunk)

        self.assertEqual(mocked_run.call_count, 1)
        self.assertEqual(first_rate, second_rate)
        self.assertEqual(first_pcm16, second_pcm16)

    def test_model_sample_rate_reads_voice_json(self) -> None:
        runtime_root = ROOT / "fake-piper-runtime"
        voice_dir = runtime_root / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = voice_dir / "es_ES-davefx-medium.onnx"
        config_path = voice_dir / "es_ES-davefx-medium.onnx.json"
        try:
            model_path.write_bytes(b"")
            config_path.write_text('{"audio":{"sample_rate":22050}}', encoding="utf-8")
            synthesizer = PiperSpeechSynthesizer(runtime_root=runtime_root)
            self.assertEqual(synthesizer._model_sample_rate(model_path), 22_050)
        finally:
            config_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)
            voice_dir.rmdir()
            runtime_root.rmdir()


class PiperSpeechSynthesizerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_speech_uses_cached_audio_without_spawning(self) -> None:
        synthesizer = PiperSpeechSynthesizer(runtime_root=ROOT / "fake-piper-runtime")
        model_path = ROOT / "fake-piper-runtime" / "voices" / "es_MX-claude-high.onnx"
        chunk = TranslationChunk(
            source_text="hi",
            translated_text="hola",
            source_language="en",
            target_language="es",
            sequence_id=1,
            is_final=True,
        )
        synthesizer._cache[(str(model_path), "hola", synthesizer.sample_rate)] = (
            synthesizer.sample_rate,
            b"\x20\x00" * 64,
        )

        with patch.object(
            PiperSpeechSynthesizer,
            "_choose_model_path",
            return_value=model_path,
        ), patch(
            "asyncio.create_subprocess_exec"
        ) as mocked_exec:
            chunks = [speech async for speech in synthesizer.stream_speech(chunk)]

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].pcm16, b"\x20\x00" * 64)
        mocked_exec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
