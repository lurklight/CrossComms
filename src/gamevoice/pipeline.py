from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress

from .audio.output import AudioSink
from .config import RuntimeConfig
from .comms.normalizer import GameCommsNormalizer
from .models import (
    NormalizedChunk,
    PipelineEvent,
    PipelineStage,
    TranscriptChunk,
    TranslationChunk,
)
from .providers.base import StreamingSpeechSynthesizer, TextTranslator


EventHandler = Callable[[PipelineEvent], None]


class RealtimeVoicePipeline:
    def __init__(
        self,
        config: RuntimeConfig,
        normalizer: GameCommsNormalizer,
        translator: TextTranslator,
        synthesizer: StreamingSpeechSynthesizer,
        sink: AudioSink,
        event_handler: EventHandler | None = None,
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.translator = translator
        self.synthesizer = synthesizer
        self.sink = sink
        self._event_handler = event_handler
        self._transcript_queue: asyncio.Queue[TranscriptChunk] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._normalized_queue: asyncio.Queue[NormalizedChunk] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._translation_queue: asyncio.Queue[TranslationChunk] = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._speech_queue: asyncio.Queue = asyncio.Queue(maxsize=config.queue_size)
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._next_sequence_id = 1

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._normalize_loop(), name="normalize"),
            asyncio.create_task(self._translate_loop(), name="translate"),
            asyncio.create_task(self._synthesize_loop(), name="synthesize"),
            asyncio.create_task(self._output_loop(), name="output"),
        ]
        self._emit(PipelineStage.STATUS, 0, "Pipeline started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for task in self._tasks:
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*self._tasks, return_exceptions=False)
        self._tasks.clear()
        await self.sink.close()
        self._emit(PipelineStage.STATUS, 0, "Pipeline stopped")

    async def submit_text(self, text: str, is_final: bool = True) -> int:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Cannot submit an empty transcript chunk.")

        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1

        chunk = TranscriptChunk(text=cleaned, sequence_id=sequence_id, is_final=is_final)
        await self._transcript_queue.put(chunk)
        return sequence_id

    async def wait_for_idle(self) -> None:
        while True:
            await self._transcript_queue.join()
            await self._normalized_queue.join()
            await self._translation_queue.join()
            await self._speech_queue.join()
            if all(
                queue.empty()
                for queue in (
                    self._transcript_queue,
                    self._normalized_queue,
                    self._translation_queue,
                    self._speech_queue,
                )
            ):
                return

    async def _normalize_loop(self) -> None:
        while True:
            chunk = await self._transcript_queue.get()
            try:
                self._emit(
                    PipelineStage.TRANSCRIPT,
                    chunk.sequence_id,
                    f"Transcript received: {chunk.text}",
                    chunk,
                )
                normalized = self.normalizer.normalize(chunk)
                self._emit(
                    PipelineStage.NORMALIZED,
                    chunk.sequence_id,
                    f"Normalized: {normalized.normalized_text}",
                    normalized,
                )
                await self._normalized_queue.put(normalized)
            finally:
                self._transcript_queue.task_done()

    async def _translate_loop(self) -> None:
        while True:
            normalized = await self._normalized_queue.get()
            try:
                try:
                    async for translated in self.translator.stream_translate(
                        normalized,
                        self.config.source_language,
                        self.config.target_language,
                    ):
                        self._emit(
                            PipelineStage.TRANSLATED,
                            normalized.sequence_id,
                            f"Translated: {translated.translated_text}",
                            translated,
                        )
                        await self._translation_queue.put(translated)
                except Exception as exc:
                    fallback = TranslationChunk(
                        source_text=normalized.normalized_text,
                        translated_text=normalized.normalized_text,
                        source_language=self.config.source_language,
                        target_language=self.config.target_language,
                        sequence_id=normalized.sequence_id,
                        is_final=normalized.is_final,
                        cache_key=normalized.normalized_text,
                    )
                    self._emit(
                        PipelineStage.STATUS,
                        normalized.sequence_id,
                        f"Translation backend failed, using source text: {exc}",
                    )
                    self._emit(
                        PipelineStage.TRANSLATED,
                        normalized.sequence_id,
                        f"Translated: {fallback.translated_text}",
                        fallback,
                    )
                    await self._translation_queue.put(fallback)
            finally:
                self._normalized_queue.task_done()

    async def _synthesize_loop(self) -> None:
        while True:
            translated = await self._translation_queue.get()
            try:
                try:
                    async for speech in self.synthesizer.stream_speech(translated):
                        if speech.is_final:
                            self._emit(
                                PipelineStage.SYNTHESIZED,
                                translated.sequence_id,
                                f"Synthesized audio for: {speech.text}",
                                speech,
                            )
                        await self._speech_queue.put(speech)
                except Exception as exc:
                    self._emit(
                        PipelineStage.STATUS,
                        translated.sequence_id,
                        f"Speech synthesis failed: {exc}",
                    )
            finally:
                self._translation_queue.task_done()

    async def _output_loop(self) -> None:
        while True:
            speech = await self._speech_queue.get()
            try:
                await self.sink.write(speech)
                if speech.is_final:
                    self._emit(
                        PipelineStage.OUTPUT,
                        speech.sequence_id,
                        f"Sent audio to output for: {speech.text}",
                        speech,
                    )
            finally:
                self._speech_queue.task_done()

    def _emit(
        self,
        stage: PipelineStage,
        sequence_id: int,
        message: str,
        payload=None,
    ) -> None:
        if self._event_handler is None:
            return
        self._event_handler(
            PipelineEvent(
                stage=stage,
                sequence_id=sequence_id,
                message=message,
                payload=payload,
            )
        )
