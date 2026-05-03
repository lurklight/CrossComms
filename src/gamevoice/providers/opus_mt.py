from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
import threading

from ..models import NormalizedChunk, TranslationChunk


SUPPORTED_OPUS_KEYS = {"en", "es", "fr", "de", "it", "pt", "ru", "vi", "zh"}


@dataclass(slots=True)
class _OpusModelBundle:
    repo_id: str
    model_dir: Path
    source_processor: object
    target_processor: object


class OpusMtTextTranslator:
    def __init__(
        self,
        download_root: Path,
        cache_size: int = 256,
        device: str = "cpu",
        model_quantization: str = "int8",
    ) -> None:
        self.download_root = Path(download_root)
        self.cache_size = max(1, int(cache_size))
        self.device = device.strip().lower() or "cpu"
        self.model_quantization = model_quantization.strip().lower() or "int8"
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._bundle_cache: dict[str, _OpusModelBundle] = {}
        self._translator_cache: dict[str, object] = {}
        self._cache_lock = threading.Lock()
        self._model_lock = threading.Lock()

    def validate_runtime(self) -> None:
        self.download_root.mkdir(parents=True, exist_ok=True)
        self._require_sentencepiece()
        self._require_hf_hub()
        self._require_ctranslate2()

    async def stream_translate(
        self,
        chunk: NormalizedChunk,
        source_language: str,
        target_language: str,
    ) -> AsyncIterator[TranslationChunk]:
        translated = await asyncio.to_thread(
            self._translate_sync,
            chunk.normalized_text,
            source_language,
            target_language,
        )
        yield TranslationChunk(
            source_text=chunk.normalized_text,
            translated_text=translated,
            source_language=source_language,
            target_language=target_language,
            sequence_id=chunk.sequence_id,
            is_final=chunk.is_final,
            cache_key=chunk.normalized_text,
        )

    def _translate_sync(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        cleaned = text.strip()
        if not cleaned:
            return text

        source_key = self._opus_language_key(source_language)
        target_key = self._opus_language_key(target_language)
        if source_key == target_key:
            return text

        cache_key = (source_key, target_key, cleaned)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        translated = cleaned
        for pair in self._translation_plan(source_key, target_key):
            translated = self._translate_once(translated, pair[0], pair[1])

        resolved = translated.strip() or text
        self._cache_put(cache_key, resolved)
        return resolved

    def _translate_once(self, text: str, source_key: str, target_key: str) -> str:
        bundle = self._ensure_model_bundle(source_key, target_key)
        translator = self._ensure_translator(bundle)
        source_tokens = bundle.source_processor.encode(text, out_type=str)
        if not source_tokens or source_tokens[-1] != "</s>":
            source_tokens.append("</s>")
        results = translator.translate_batch([source_tokens], beam_size=1)
        if not results or not results[0].hypotheses:
            return text
        output_tokens = results[0].hypotheses[0]
        translated = self._decode_pieces(bundle.target_processor, output_tokens)
        return translated.strip() or text

    def _ensure_model_bundle(self, source_key: str, target_key: str) -> _OpusModelBundle:
        pair_key = f"{source_key}-{target_key}"
        cached = self._bundle_cache.get(pair_key)
        if cached is not None:
            return cached

        with self._model_lock:
            cached = self._bundle_cache.get(pair_key)
            if cached is not None:
                return cached

            pair_root = self.download_root / pair_key
            model_dir, repo_id = self._download_direct_model(source_key, target_key, pair_root)
            bundle = _OpusModelBundle(
                repo_id=repo_id,
                model_dir=model_dir,
                source_processor=self._load_sentencepiece(model_dir / "source.spm"),
                target_processor=self._load_sentencepiece(model_dir / "target.spm"),
            )
            self._bundle_cache[pair_key] = bundle
            return bundle

    def _ensure_translator(self, bundle: _OpusModelBundle):
        cached = self._translator_cache.get(bundle.repo_id)
        if cached is not None:
            return cached

        with self._model_lock:
            cached = self._translator_cache.get(bundle.repo_id)
            if cached is not None:
                return cached

            ctranslate2 = self._require_ctranslate2()
            translator = ctranslate2.Translator(str(bundle.model_dir), device=self.device)
            self._translator_cache[bundle.repo_id] = translator
            return translator

    def _download_direct_model(
        self,
        source_key: str,
        target_key: str,
        pair_root: Path,
    ) -> tuple[Path, str]:
        pair_root.mkdir(parents=True, exist_ok=True)
        for repo_id in self._candidate_repo_ids(source_key, target_key):
            model_dir = pair_root / repo_id.replace("/", "--")
            if self._has_direct_model_files(model_dir):
                return model_dir, repo_id
            try:
                self._download_source_model(repo_id, model_dir)
            except Exception:
                continue
            if self._has_direct_model_files(model_dir):
                return model_dir, repo_id
        raise RuntimeError(
            f"Local OPUS could not find a converted CTranslate2 model for {source_key} -> {target_key}."
        )

    def _download_source_model(self, repo_id: str, model_dir: Path) -> None:
        snapshot_download = self._require_hf_hub()
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_dir),
        )

    @staticmethod
    def _has_direct_model_files(model_dir: Path) -> bool:
        return all(
            (model_dir / file_name).exists()
            for file_name in ("model.bin", "source.spm", "target.spm")
        )

    @classmethod
    def _candidate_repo_ids(cls, source_key: str, target_key: str) -> list[str]:
        slug = f"opus-mt-{source_key}-{target_key}"
        return [
            f"Sams200/{slug}",
            f"gaudi/{slug}-ctranslate2",
        ]

    def _load_sentencepiece(self, model_path: Path):
        sentencepiece = self._require_sentencepiece_module()
        processor = sentencepiece.SentencePieceProcessor()
        if not processor.load(str(model_path)):
            raise RuntimeError(f"Local OPUS could not load tokenizer file: {model_path.name}")
        return processor

    @staticmethod
    def _decode_pieces(processor, pieces: list[str]) -> str:
        decode_pieces = getattr(processor, "decode_pieces", None)
        if callable(decode_pieces):
            return decode_pieces(pieces)
        return processor.decode(pieces)

    @staticmethod
    def _opus_language_key(language_code: str) -> str:
        cleaned = language_code.strip().lower()
        if cleaned.startswith("pt"):
            return "pt"
        if cleaned.startswith("zh"):
            return "zh"
        return cleaned.split("-", maxsplit=1)[0]

    @classmethod
    def _translation_plan(cls, source_key: str, target_key: str) -> list[tuple[str, str]]:
        if source_key not in SUPPORTED_OPUS_KEYS or target_key not in SUPPORTED_OPUS_KEYS:
            raise RuntimeError(
                f"Local OPUS translation does not support {source_key} -> {target_key} yet."
            )
        if source_key == "en" or target_key == "en":
            return [(source_key, target_key)]
        if source_key != target_key:
            return [(source_key, "en"), ("en", target_key)]
        return [(source_key, target_key)]

    def _cache_get(self, cache_key: tuple[str, str, str]) -> str | None:
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
            return cached

    def _cache_put(self, cache_key: tuple[str, str, str], value: str) -> None:
        with self._cache_lock:
            self._cache[cache_key] = value
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    @staticmethod
    def _require_hf_hub():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Local OPUS translation needs huggingface-hub. Run setup again to install it."
            ) from exc
        return snapshot_download

    @staticmethod
    def _require_sentencepiece() -> None:
        OpusMtTextTranslator._require_sentencepiece_module()

    @staticmethod
    def _require_sentencepiece_module():
        try:
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError(
                "Local OPUS translation needs sentencepiece. Run setup again to install it."
            ) from exc
        return sentencepiece

    @staticmethod
    def _require_ctranslate2():
        try:
            import ctranslate2
        except ImportError as exc:
            raise RuntimeError(
                "Local OPUS translation needs ctranslate2. Run setup again to install it."
            ) from exc
        if getattr(ctranslate2, "Translator", None) is None:
            raise RuntimeError(
                "Local OPUS translation could not load the ctranslate2 Translator runtime."
            )
        return ctranslate2
