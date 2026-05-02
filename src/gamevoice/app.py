from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path as _Path

    _FILE = _Path(__file__).resolve()
    _SRC_ROOT = _FILE.parents[1]
    _PROJECT_ROOT = _FILE.parents[2]
    _PACKAGES = _PROJECT_ROOT / ".packages"

    for _entry in (_PACKAGES, _SRC_ROOT):
        if _entry.exists() and str(_entry) not in sys.path:
            sys.path.insert(0, str(_entry))
    __package__ = "gamevoice"

import asyncio
from contextlib import suppress
import ctypes
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from .audio.devices import (
    DeviceOption,
    audio_backend_available,
    list_input_device_options,
    list_output_device_options,
)
from .audio.input import EnergyVoiceActivityDetector, SoundDeviceMicrophoneSource
from .audio.output import NullAudioSink, SoundDeviceVirtualMicSink
from .config import RuntimeConfig
from .comms.normalizer import GameCommsNormalizer, SlangPack
from .languages import LanguageOption, load_language_options
from .models import PipelineEvent, PipelineStage
from .pipeline import RealtimeVoicePipeline
from .providers.faster_whisper_recognizer import (
    FasterWhisperSegmentRecognizer,
    build_whisper_hotwords,
    default_whisper_model_for_language,
)
from .providers.piper_tts import PiperSpeechSynthesizer, list_installed_piper_voices
from .providers.web import FreeWebTextTranslator


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class PipelineController:
    def __init__(self, event_queue: Queue[PipelineEvent]) -> None:
        self.event_queue = event_queue
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._pipeline: RealtimeVoicePipeline | None = None
        self._microphone_source: SoundDeviceMicrophoneSource | None = None
        self._live_capture_task: asyncio.Task | None = None
        self._running = False

    def start(self, config: RuntimeConfig) -> None:
        if self._running:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._start_pipeline(config), self._loop)
        future.result(timeout=5)
        self._running = True

    def stop(self) -> None:
        if not self._running or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._stop_pipeline(), self._loop)
        future.result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._loop = None
        self._thread = None
        self._pipeline = None
        self._running = False

    def submit_text(self, text: str, is_final: bool) -> None:
        if not self._running or self._loop is None or self._pipeline is None:
            raise RuntimeError("Start the pipeline before sending transcripts.")
        future = asyncio.run_coroutine_threadsafe(
            self._pipeline.submit_text(text, is_final=is_final),
            self._loop,
        )
        future.result(timeout=3)

    def _run_loop(self) -> None:
        assert self._loop is not None
        co_initialized = False
        if hasattr(ctypes, "windll"):
            try:
                hr = ctypes.windll.ole32.CoInitialize(None)
                co_initialized = hr in (0, 1)
            except Exception:
                co_initialized = False
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            if co_initialized:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    pass

    async def _start_pipeline(self, config: RuntimeConfig) -> None:
        normalizer = GameCommsNormalizer(SlangPack(name="Built-in"))
        sink = self._build_sink(config)
        synthesizer = PiperSpeechSynthesizer(
            sample_rate=config.sample_rate,
            source_voice_model=config.source_voice_model,
            target_voice_model=config.target_voice_model,
        )
        synthesizer.validate_runtime()
        self._pipeline = RealtimeVoicePipeline(
            config=config,
            normalizer=normalizer,
            translator=FreeWebTextTranslator(),
            synthesizer=synthesizer,
            sink=sink,
            event_handler=self.event_queue.put,
        )
        await self._pipeline.start()
        self._emit_status(
            "Using free web translation plus a local Piper voice backend. "
            f"Source voice: {config.source_voice_model or 'default'} | "
            f"Target voice: {config.target_voice_model or 'default'}. "
            "Translation still needs internet, but speech synthesis is now fully local."
        )
        await self._start_live_capture(config)

    async def _stop_pipeline(self) -> None:
        if self._live_capture_task is not None:
            self._live_capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._live_capture_task
            self._live_capture_task = None

        if self._microphone_source is not None:
            await self._microphone_source.stop()
            self._microphone_source = None

        if self._pipeline is None:
            return
        await self._pipeline.stop()

    async def _start_live_capture(self, config: RuntimeConfig) -> None:
        if config.input_mic_name is None:
            self._emit_status("Live mic capture is off because no input device is selected.")
            return

        self._microphone_source = SoundDeviceMicrophoneSource(
            sample_rate=config.input_sample_rate,
            channels=1,
            frame_ms=config.frame_ms,
            device=config.input_mic_name,
            vad=EnergyVoiceActivityDetector(
                threshold=config.vad_threshold,
                start_multiplier=config.vad_start_multiplier,
                continue_multiplier=config.vad_continue_multiplier,
            ),
            queue_size=config.queue_size,
        )
        await self._microphone_source.start()

        recognizer = FasterWhisperSegmentRecognizer(
            source_language=config.source_language,
            frame_ms=config.frame_ms,
            endpoint_silence_ms=config.endpoint_silence_ms,
            min_speech_ms=config.min_speech_ms,
            max_utterance_ms=config.max_utterance_ms,
            speech_trigger_ms=config.speech_trigger_ms,
            min_speech_ratio=config.min_speech_ratio,
            min_peak_rms=config.min_peak_rms,
            model_name=config.whisper_model
            or default_whisper_model_for_language(config.source_language),
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            download_root=project_root() / ".model-cache",
            min_average_speech_rms=config.min_average_speech_rms,
            hotwords=build_whisper_hotwords(
                ["hi", "mid", "rotate b", "rotate a", "one shot", "cracked"]
            ),
        )
        recognizer.validate_runtime()
        self._emit_status(
            "Live mic capture started with faster-whisper local STT. "
            f"Model: {recognizer.model_name} on {recognizer.device}/{recognizer.compute_type}. "
            "The first spoken phrase may pause while the model loads. "
            "Noise gating is running with stricter background filtering."
        )
        self._live_capture_task = asyncio.create_task(
            self._forward_live_transcripts(recognizer),
            name="live-mic-capture",
        )

    async def _forward_live_transcripts(
        self,
        recognizer,
    ) -> None:
        assert self._pipeline is not None
        assert self._microphone_source is not None
        try:
            async for transcript in recognizer.transcribe(self._microphone_source.frames()):
                await self._pipeline.submit_text(
                    transcript.text,
                    is_final=transcript.is_final,
                )
                self._emit_status(f"Recognized from mic: {transcript.text}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_status(f"Live mic capture stopped: {exc}")

    def _build_sink(self, config: RuntimeConfig):
        if not config.virtual_mic_name:
            return NullAudioSink()
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            self.event_queue.put(
                PipelineEvent(
                    stage=PipelineStage.STATUS,
                    sequence_id=0,
                    message="Audio extras are not installed. Falling back to a dry-run sink.",
                )
            )
            return NullAudioSink()
        return SoundDeviceVirtualMicSink(
            device_name=config.virtual_mic_name,
            sample_rate=config.sample_rate,
        )

    def _emit_status(self, message: str) -> None:
        self.event_queue.put(
            PipelineEvent(
                stage=PipelineStage.STATUS,
                sequence_id=0,
                message=message,
            )
        )


class TranslatorApp:
    NO_VOICE_LABEL = "No matching Piper voice installed"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CrossComms")
        self.root.geometry("1120x780")
        self.root.minsize(980, 720)

        self.colors = {
            "bg_dark": "#0a0a0a",
            "bg_panel": "#121212",
            "bg_card": "#181818",
            "bg_card_alt": "#1f1f1f",
            "bg_input": "#252525",
            "bg_hover": "#2d2d2d",
            "accent_red": "#e63946",
            "accent_red_hover": "#ff4d5a",
            "text_primary": "#f5f5f5",
            "text_secondary": "#9ea2a8",
            "border": "#2d2d2d",
            "success": "#2ecc71",
            "warning": "#f39c12",
        }

        self.event_queue: Queue[PipelineEvent] = Queue()
        self.controller = PipelineController(self.event_queue)
        self.language_options = load_language_options(project_root() / "languages.json")
        self.audio_backend_ready = audio_backend_available()

        input_options = list_input_device_options()
        output_options = list_output_device_options()
        piper_voices = list_installed_piper_voices(project_root() / ".piper-runtime")
        default_input_device = self._preferred_input_label(input_options)
        default_output_device = self._preferred_output_label(output_options)
        self.language_map = {
            option.label: option for option in self.language_options
        }
        default_source_language = self._preferred_language_label("en")
        default_target_language = self._preferred_language_label("es")
        source_language_option = self.language_map.get(default_source_language)
        target_language_option = self.language_map.get(default_target_language)
        default_source_voice = self._preferred_piper_voice_label(
            self._voice_options_for_language(source_language_option, piper_voices),
            source_language_option.default_source_voice if source_language_option else None,
        )
        default_target_voice = self._preferred_piper_voice_label(
            self._voice_options_for_language(target_language_option, piper_voices),
            target_language_option.default_target_voice if target_language_option else None,
        )

        self.source_language = tk.StringVar(value=default_source_language)
        self.target_language = tk.StringVar(value=default_target_language)
        self.input_mic_name = tk.StringVar(value=default_input_device)
        self.virtual_mic_name = tk.StringVar(value=default_output_device)
        self.source_voice_name = tk.StringVar(value=default_source_voice)
        self.target_voice_name = tk.StringVar(value=default_target_voice)
        self.input_text = tk.StringVar(
            value="he's one shot, cracked, rotating B"
        )
        self.audio_status = tk.StringVar(
            value=self._audio_status_message(
                self.audio_backend_ready,
                input_options,
                output_options,
            )
        )

        self.transcript_value = tk.StringVar(value="Waiting for input")
        self.normalized_value = tk.StringVar(value="Waiting for normalization")
        self.translated_value = tk.StringVar(value="Waiting for translation")
        self.output_value = tk.StringVar(value="Waiting for synthesized output")

        self.input_device_options = input_options
        self.output_device_options = output_options
        self.piper_voice_options = piper_voices
        self.input_device_map = {option.label: option for option in input_options}
        self.output_device_map = {option.label: option for option in output_options}
        self.source_voice_options = self._voice_options_for_language(
            source_language_option,
            self.piper_voice_options,
        )
        self.target_voice_options = self._voice_options_for_language(
            target_language_option,
            self.piper_voice_options,
        )
        self.source_voice_map = self._voice_map_for_options(self.source_voice_options)
        self.target_voice_map = self._voice_map_for_options(self.target_voice_options)
        self.stage_expanded = tk.BooleanVar(value=False)
        self.log_expanded = tk.BooleanVar(value=False)
        self.status_text = tk.StringVar(value="Ready")
        self.status_color = self.colors["text_secondary"]
        self.root.configure(bg=self.colors["bg_dark"])
        self._build_ui()
        self.source_language.trace_add("write", self._on_source_language_changed)
        self.target_language.trace_add("write", self._on_target_language_changed)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._poll_events)

    def _build_ui(self) -> None:
        self._configure_theme()

        self.main_container = tk.Frame(
            self.root,
            bg=self.colors["bg_dark"],
        )
        self.main_container.pack(fill="both", expand=True, padx=20, pady=20)

        self._build_header(self.main_container)
        self._build_controls_card(self.main_container)
        self._build_manual_card(self.main_container)
        self._build_stage_card(self.main_container)
        self._build_log_card(self.main_container)

    def _configure_theme(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(
            "Game.TCombobox",
            fieldbackground=self.colors["bg_input"],
            background=self.colors["bg_input"],
            foreground=self.colors["text_primary"],
            borderwidth=0,
            relief="flat",
            padding=8,
            arrowcolor=self.colors["text_secondary"],
            insertcolor=self.colors["text_primary"],
            selectbackground=self.colors["bg_hover"],
            selectforeground=self.colors["text_primary"],
        )
        style.map(
            "Game.TCombobox",
            fieldbackground=[("readonly", self.colors["bg_input"])],
            background=[("readonly", self.colors["bg_input"])],
            foreground=[("readonly", self.colors["text_primary"])],
        )

    def _build_header(self, parent: tk.Widget) -> None:
        header = tk.Frame(parent, bg=self.colors["bg_dark"])
        header.pack(fill="x", pady=(0, 18))

        title_block = tk.Frame(header, bg=self.colors["bg_dark"])
        title_block.pack(side="left")

        tk.Label(
            title_block,
            text="CrossComms",
            font=("Segoe UI", 24, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_dark"],
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Real-time translated voice comms for games.",
            font=("Segoe UI", 10),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_dark"],
        ).pack(anchor="w", pady=(4, 0))

        status_frame = tk.Frame(
            header,
            bg=self.colors["bg_card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        status_frame.pack(side="right", padx=(16, 0))

        status_inner = tk.Frame(status_frame, bg=self.colors["bg_card"])
        status_inner.pack(padx=14, pady=10)

        self.status_dot = tk.Canvas(
            status_inner,
            width=12,
            height=12,
            bg=self.colors["bg_card"],
            highlightthickness=0,
        )
        self.status_dot.pack(side="left", padx=(0, 8))
        self.status_circle = self.status_dot.create_oval(
            2,
            2,
            10,
            10,
            fill=self.status_color,
            outline="",
        )
        self.status_label = tk.Label(
            status_inner,
            textvariable=self.status_text,
            font=("Segoe UI", 10, "bold"),
            fg=self.status_color,
            bg=self.colors["bg_card"],
        )
        self.status_label.pack(side="left")

    def _build_controls_card(self, parent: tk.Widget) -> None:
        card, inner = self._create_card(parent)
        card.pack(fill="x", pady=(0, 18))

        tk.Label(
            inner,
            text="Session Controls",
            font=("Segoe UI", 12, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_card"],
        ).pack(anchor="w")
        tk.Label(
            inner,
            textvariable=self.audio_status,
            font=("Segoe UI", 9),
            fg=self.colors["warning"],
            bg=self.colors["bg_card"],
            justify="left",
            wraplength=980,
        ).pack(anchor="w", pady=(6, 18))

        row1 = tk.Frame(inner, bg=self.colors["bg_card"])
        row1.pack(fill="x", pady=(0, 14))
        self.source_language_box = self._create_combo_field(
            row1,
            "Source Language",
            self.source_language,
            [option.label for option in self.language_options],
            width=28,
        )
        self.target_language_box = self._create_combo_field(
            row1,
            "Target Language",
            self.target_language,
            [option.label for option in self.language_options],
            width=28,
            padx=(18, 0),
        )

        row2 = tk.Frame(inner, bg=self.colors["bg_card"])
        row2.pack(fill="x", pady=(0, 14))
        self.input_device_box = self._create_combo_field(
            row2,
            "Input Microphone",
            self.input_mic_name,
            [option.label for option in self.input_device_options],
            width=36,
        )
        self.device_box = self._create_combo_field(
            row2,
            "Virtual Cable Out",
            self.virtual_mic_name,
            [option.label for option in self.output_device_options],
            width=48,
            padx=(18, 0),
        )

        row3 = tk.Frame(inner, bg=self.colors["bg_card"])
        row3.pack(fill="x", pady=(0, 18))
        self.source_voice_box = self._create_combo_field(
            row3,
            "Source Voice",
            self.source_voice_name,
            self._voice_display_labels(self.source_voice_options),
            width=30,
        )
        self.target_voice_box = self._create_combo_field(
            row3,
            "Target Voice",
            self.target_voice_name,
            self._voice_display_labels(self.target_voice_options),
            width=30,
            padx=(18, 0),
        )

        button_row = tk.Frame(inner, bg=self.colors["bg_card"])
        button_row.pack(fill="x")
        self.start_button = self._create_button(
            button_row,
            "Start",
            self._start_pipeline,
            primary=True,
        )
        self.start_button.pack(side="left")
        self.stop_button = self._create_button(
            button_row,
            "Stop",
            self._stop_pipeline,
        )
        self.stop_button.pack(side="left", padx=(10, 0))
        self.refresh_button = self._create_button(
            button_row,
            "Refresh Lists",
            self._refresh_devices,
        )
        self.refresh_button.pack(side="left", padx=(10, 0))

    def _build_manual_card(self, parent: tk.Widget) -> None:
        card, inner = self._create_card(parent)
        card.pack(fill="x", pady=(0, 18))

        tk.Label(
            inner,
            text="Manual Test",
            font=("Segoe UI", 12, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_card"],
        ).pack(anchor="w")
        tk.Label(
            inner,
            text=(
                "Use this box for quick text testing while live microphone capture is running."
            ),
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
            wraplength=980,
            justify="left",
        ).pack(anchor="w", pady=(6, 12))

        self.input_entry = tk.Entry(
            inner,
            textvariable=self.input_text,
            font=("Segoe UI", 11),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_input"],
            insertbackground=self.colors["text_primary"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent_red"],
            bd=0,
        )
        self.input_entry.pack(fill="x", ipady=10)

        button_row = tk.Frame(inner, bg=self.colors["bg_card"])
        button_row.pack(fill="x", pady=(12, 0))
        self.final_button = self._create_button(
            button_row,
            "Translate",
            lambda: self._send_text(True),
        )
        self.final_button.pack(side="left")
        self.partial_button = self._create_button(
            button_row,
            "Preview",
            lambda: self._send_text(False),
        )
        self.partial_button.pack(side="left", padx=(10, 0))
        self.example_button = self._create_button(
            button_row,
            "Use Example",
            lambda: self.input_text.set("enemy low, push now"),
        )
        self.example_button.pack(side="left", padx=(10, 0))

    def _build_stage_card(self, parent: tk.Widget) -> None:
        header = tk.Frame(
            parent,
            bg=self.colors["bg_card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            cursor="hand2",
        )
        header.pack(fill="x", pady=(0, 2))
        header.bind("<Button-1>", lambda _event: self._toggle_stage_output())

        header_inner = tk.Frame(header, bg=self.colors["bg_card"])
        header_inner.pack(fill="x", padx=16, pady=12)
        header_inner.bind("<Button-1>", lambda _event: self._toggle_stage_output())

        self.stage_toggle_label = tk.Label(
            header_inner,
            text=">",
            font=("Consolas", 11, "bold"),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
        )
        self.stage_toggle_label.pack(side="left", padx=(0, 10))
        self.stage_toggle_label.bind("<Button-1>", lambda _event: self._toggle_stage_output())

        title = tk.Label(
            header_inner,
            text="Latest Stage Output",
            font=("Segoe UI", 10, "bold"),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_card"],
        )
        title.pack(side="left")
        title.bind("<Button-1>", lambda _event: self._toggle_stage_output())

        self.stage_frame, inner = self._create_card(parent)

        self._stage_row(inner, "Transcript", self.transcript_value)
        self._stage_row(inner, "Normalized", self.normalized_value)
        self._stage_row(inner, "Translated", self.translated_value)
        self._stage_row(inner, "Output", self.output_value)

        if self.stage_expanded.get():
            self.stage_frame.pack(fill="x", pady=(0, 18))
            self.stage_toggle_label.config(text="v")
        else:
            self.stage_toggle_label.config(text=">")

    def _build_log_card(self, parent: tk.Widget) -> None:
        header = tk.Frame(
            parent,
            bg=self.colors["bg_card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            cursor="hand2",
        )
        header.pack(fill="x")
        header.bind("<Button-1>", lambda _event: self._toggle_log())

        header_inner = tk.Frame(header, bg=self.colors["bg_card"])
        header_inner.pack(fill="x", padx=16, pady=12)
        header_inner.bind("<Button-1>", lambda _event: self._toggle_log())

        self.log_toggle_label = tk.Label(
            header_inner,
            text=">",
            font=("Consolas", 11, "bold"),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
        )
        self.log_toggle_label.pack(side="left", padx=(0, 10))
        self.log_toggle_label.bind("<Button-1>", lambda _event: self._toggle_log())

        title = tk.Label(
            header_inner,
            text="Pipeline Log",
            font=("Segoe UI", 10, "bold"),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
        )
        title.pack(side="left")
        title.bind("<Button-1>", lambda _event: self._toggle_log())

        self.log_frame = tk.Frame(
            parent,
            bg=self.colors["bg_card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )

        log_inner = tk.Frame(self.log_frame, bg=self.colors["bg_card"])
        log_inner.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        scrollbar = tk.Scrollbar(
            log_inner,
            troughcolor=self.colors["bg_input"],
            activebackground=self.colors["bg_hover"],
            bg=self.colors["bg_input"],
            highlightthickness=0,
            relief="flat",
        )
        scrollbar.pack(side="right", fill="y")

        self.log_output = tk.Text(
            log_inner,
            wrap="word",
            height=12,
            bg=self.colors["bg_input"],
            fg=self.colors["text_secondary"],
            insertbackground=self.colors["text_primary"],
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=12,
            font=("Consolas", 9),
            yscrollcommand=scrollbar.set,
        )
        self.log_output.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.log_output.yview)
        self.log_output.config(state="disabled")
        if self.log_expanded.get():
            self.log_frame.pack(fill="both", expand=True, pady=(2, 0))
            self.log_toggle_label.config(text="v")
        else:
            self.log_toggle_label.config(text=">")

    def _create_card(self, parent: tk.Widget) -> tuple[tk.Frame, tk.Frame]:
        card = tk.Frame(
            parent,
            bg=self.colors["bg_card"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        inner = tk.Frame(card, bg=self.colors["bg_card"])
        inner.pack(fill="both", expand=True, padx=18, pady=18)
        return card, inner

    def _create_combo_field(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        width: int,
        padx: tuple[int, int] = (0, 0),
    ):
        section = tk.Frame(parent, bg=self.colors["bg_card"])
        section.pack(side="left", padx=padx)

        tk.Label(
            section,
            text=label,
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
        ).pack(anchor="w", pady=(0, 6))

        combo = ttk.Combobox(
            section,
            textvariable=variable,
            values=values,
            state="readonly",
            width=width,
            style="Game.TCombobox",
            font=("Segoe UI", 10),
        )
        combo.pack(anchor="w")
        return combo

    def _create_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        primary: bool = False,
    ) -> tk.Button:
        bg_color = self.colors["accent_red"] if primary else self.colors["bg_input"]
        hover_color = self.colors["accent_red_hover"] if primary else self.colors["bg_hover"]
        button = tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI", 10, "bold" if primary else "normal"),
            fg=self.colors["text_primary"],
            bg=bg_color,
            activeforeground=self.colors["text_primary"],
            activebackground=hover_color,
            relief="flat",
            borderwidth=0,
            padx=18,
            pady=10,
            cursor="hand2",
        )
        button.bind("<Enter>", lambda _event: button.config(bg=hover_color))
        button.bind("<Leave>", lambda _event: button.config(bg=bg_color))
        return button

    def _stage_row(self, parent: tk.Widget, label: str, value: tk.StringVar) -> None:
        row = tk.Frame(parent, bg=self.colors["bg_card"])
        row.pack(fill="x", pady=(0, 10))

        label_widget = tk.Label(
            row,
            text=label,
            font=("Segoe UI", 9, "bold"),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg_card"],
            width=12,
            anchor="w",
        )
        label_widget.pack(side="left", padx=(0, 14))

        value_shell = tk.Frame(
            row,
            bg=self.colors["bg_card_alt"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        value_shell.pack(side="left", fill="x", expand=True)

        value_inner = value_shell
        tk.Label(
            value_inner,
            textvariable=value,
            font=("Segoe UI", 11),
            fg=self.colors["text_primary"],
            bg=self.colors["bg_card_alt"],
            anchor="w",
            justify="left",
            wraplength=760,
            padx=12,
            pady=10,
        ).pack(fill="x")

    def _build_config(self) -> RuntimeConfig:
        input_choice = self.input_device_map.get(self.input_mic_name.get().strip())
        output_choice = self.output_device_map.get(self.virtual_mic_name.get().strip())
        input_sample_rate = (
            input_choice.default_sample_rate if input_choice is not None else 48_000
        )
        output_sample_rate = (
            output_choice.default_sample_rate if output_choice is not None else 48_000
        )
        source_language = self._selected_language_option(self.source_language.get())
        target_language = self._selected_language_option(self.target_language.get())
        source_voice_model = self.source_voice_map.get(self.source_voice_name.get().strip())
        target_voice_model = self.target_voice_map.get(self.target_voice_name.get().strip())
        return RuntimeConfig(
            source_language=source_language.code if source_language is not None else "en",
            target_language=target_language.code if target_language is not None else "es",
            sample_rate=output_sample_rate,
            input_sample_rate=input_sample_rate,
            input_mic_name=input_choice.index if input_choice is not None else None,
            virtual_mic_name=output_choice.index if output_choice is not None else None,
            source_voice_model=source_voice_model,
            target_voice_model=target_voice_model,
        )

    def _start_pipeline(self) -> None:
        try:
            self.controller.start(self._build_config())
        except Exception as exc:
            self._set_status("Error", self.colors["warning"])
            messagebox.showerror("Start failed", str(exc))
            return
        self._set_status("Listening", self.colors["accent_red"])
        self._append_log("Pipeline started.")

    def _stop_pipeline(self) -> None:
        try:
            self.controller.stop()
        except Exception as exc:
            self._set_status("Error", self.colors["warning"])
            messagebox.showerror("Stop failed", str(exc))
            return
        self._set_status("Ready", self.colors["text_secondary"])
        self._append_log("Pipeline stopped.")

    def _send_text(self, is_final: bool) -> None:
        try:
            self.controller.submit_text(self.input_text.get(), is_final=is_final)
        except Exception as exc:
            messagebox.showwarning("Cannot send transcript", str(exc))

    def _refresh_devices(self) -> None:
        self.audio_backend_ready = audio_backend_available()
        self.input_device_options = list_input_device_options()
        self.output_device_options = list_output_device_options()
        self.language_options = load_language_options(project_root() / "languages.json")
        self.language_map = {
            option.label: option for option in self.language_options
        }
        self.piper_voice_options = list_installed_piper_voices(project_root() / ".piper-runtime")
        self.input_device_map = {
            option.label: option for option in self.input_device_options
        }
        self.output_device_map = {
            option.label: option for option in self.output_device_options
        }
        self.input_device_box["values"] = [
            option.label for option in self.input_device_options
        ]
        self.device_box["values"] = [
            option.label for option in self.output_device_options
        ]
        self.source_language_box["values"] = [
            option.label for option in self.language_options
        ]
        self.target_language_box["values"] = [
            option.label for option in self.language_options
        ]
        if self.input_device_options and not self.input_mic_name.get():
            self.input_mic_name.set(self.input_device_options[0].label)
        if self.output_device_options and not self.virtual_mic_name.get():
            self.virtual_mic_name.set(self.output_device_options[0].label)
        self._sync_voice_options("source")
        self._sync_voice_options("target")
        self.audio_status.set(
            self._audio_status_message(
                self.audio_backend_ready,
                self.input_device_options,
                self.output_device_options,
            )
        )
        self._append_log("Refreshed languages, voices, and audio devices.")

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except Empty:
            pass
        self.root.after(120, self._poll_events)

    def _handle_event(self, event: PipelineEvent) -> None:
        timestamp = datetime.fromtimestamp(event.created_at).strftime("%H:%M:%S")
        self._append_log(f"[{timestamp}] [{event.stage}] {event.message}")

        if event.stage == PipelineStage.TRANSCRIPT:
            self.transcript_value.set(event.message.removeprefix("Transcript received: "))
        elif event.stage == PipelineStage.NORMALIZED:
            self.normalized_value.set(event.message.removeprefix("Normalized: "))
        elif event.stage == PipelineStage.TRANSLATED:
            self.translated_value.set(event.message.removeprefix("Translated: "))
        elif event.stage in {PipelineStage.SYNTHESIZED, PipelineStage.OUTPUT}:
            payload = event.payload
            if payload is not None:
                self.output_value.set(payload.text)
        elif event.stage == PipelineStage.STATUS:
            lowered = event.message.lower()
            if "stopped" in lowered and "pipeline" in lowered:
                self._set_status("Ready", self.colors["text_secondary"])
            elif "failed" in lowered or "error" in lowered:
                self._set_status("Warning", self.colors["warning"])
            elif "started" in lowered or "recognized from mic" in lowered:
                self._set_status("Listening", self.colors["accent_red"])

    def _append_log(self, line: str) -> None:
        self.log_output.config(state="normal")
        self.log_output.insert(tk.END, f"{line}\n")
        self.log_output.see(tk.END)
        self.log_output.config(state="disabled")

    def _toggle_log(self) -> None:
        if self.log_expanded.get():
            self.log_frame.pack_forget()
            self.log_toggle_label.config(text=">")
            self.log_expanded.set(False)
            return

        self.log_frame.pack(fill="both", expand=True, pady=(2, 0))
        self.log_toggle_label.config(text="v")
        self.log_expanded.set(True)

    def _toggle_stage_output(self) -> None:
        if self.stage_expanded.get():
            self.stage_frame.pack_forget()
            self.stage_toggle_label.config(text=">")
            self.stage_expanded.set(False)
            return

        self.stage_frame.pack(fill="x", pady=(0, 18))
        self.stage_toggle_label.config(text="v")
        self.stage_expanded.set(True)

    def _set_status(self, text: str, color: str) -> None:
        self.status_text.set(text)
        self.status_label.config(fg=color)
        self.status_dot.itemconfig(self.status_circle, fill=color)

    def _audio_status_message(
        self,
        backend_ready: bool,
        input_devices: list[DeviceOption],
        output_devices: list[DeviceOption],
    ) -> str:
        if not backend_ready:
            return (
                "Audio device discovery is unavailable because `sounddevice` is not installed yet. "
                "Run `python -m pip install -e .[audio]` from this project folder. "
                "For VB-CABLE, pick `CABLE Input` as the app output device. "
                "Discord or your game should use `CABLE Output` as the microphone."
            )
        if not input_devices and not output_devices:
            return (
                "The audio backend is installed, but no devices were returned. "
                "If VB-CABLE is installed, it should usually appear as `CABLE Input` in the output list."
            )
        return (
            f"Detected {len(input_devices)} input device(s) and {len(output_devices)} output device(s). "
            "Start will listen to the selected input mic. For VB-CABLE, route translated speech to a `CABLE Input` entry; then select `CABLE Output` in Discord or the game."
        )

    def _preferred_input_label(self, options: list[DeviceOption]) -> str:
        for option in options:
            label = option.label.lower()
            if "microphone" in label and "wasapi" in label and "cable output" not in label:
                return option.label
        for option in options:
            label = option.label.lower()
            if "microphone" in label and "cable output" not in label:
                return option.label
        return options[0].label if options else ""

    def _preferred_output_label(self, options: list[DeviceOption]) -> str:
        for option in options:
            label = option.label.lower()
            if "cable input" in label and "wasapi" in label:
                return option.label
        for option in options:
            if "cable input" in option.label.lower():
                return option.label
        return options[0].label if options else ""

    def _preferred_piper_voice_label(self, options, preferred_files) -> str:
        if not options:
            return self.NO_VOICE_LABEL
        if isinstance(preferred_files, str):
            preferred_files = (preferred_files,)
        for preferred in preferred_files or ():
            for option in options:
                if option.file_name == preferred:
                    return self._voice_display_label(option)
        return self._voice_display_label(options[0]) if options else ""

    def _selected_language_option(self, label: str) -> LanguageOption | None:
        return self.language_map.get(label.strip())

    def _preferred_language_label(self, code: str) -> str:
        for option in self.language_options:
            if option.code == code:
                return option.label
        return self.language_options[0].label if self.language_options else code

    def _voice_options_for_language(
        self,
        language: LanguageOption | None,
        installed_voices,
    ) -> list:
        if language is None:
            return list(installed_voices)
        matching = [
            voice
            for voice in installed_voices
            if voice.language_family == language.voice_family
        ]
        return matching

    def _voice_display_labels(self, options) -> list[str]:
        return [self._voice_display_label(option) for option in options]

    def _voice_map_for_options(self, options) -> dict[str, str]:
        return {
            self._voice_display_label(option): option.file_name for option in options
        }

    def _voice_display_label(self, voice) -> str:
        best_languages = self._language_names_for_voice(voice)
        if not best_languages:
            return voice.label
        return f"{voice.label} | Best for: {best_languages}"

    def _language_names_for_voice(self, voice) -> str:
        normalized_locale = voice.locale_code.replace("_", "-").lower()
        exact_names = [
            option.name
            for option in self.language_options
            if option.code.replace("_", "-").lower() == normalized_locale
        ]
        if exact_names:
            return ", ".join(dict.fromkeys(exact_names))

        names = [
            option.name
            for option in self.language_options
            if option.voice_family == voice.language_family
        ]
        return ", ".join(dict.fromkeys(names))

    def _on_source_language_changed(self, *_args) -> None:
        self._sync_voice_options("source")

    def _on_target_language_changed(self, *_args) -> None:
        self._sync_voice_options("target")

    def _sync_voice_options(self, kind: str) -> None:
        if not hasattr(self, "source_voice_box"):
            return

        if kind == "source":
            language = self._selected_language_option(self.source_language.get())
            options = self._voice_options_for_language(language, self.piper_voice_options)
            self.source_voice_options = options
            self.source_voice_map = self._voice_map_for_options(options)
            self.source_voice_box["values"] = (
                self._voice_display_labels(options)
                if options
                else [self.NO_VOICE_LABEL]
            )
            if self.source_voice_name.get() not in self.source_voice_map:
                preferred = (
                    language.default_source_voice if language is not None else None
                )
                self.source_voice_name.set(
                    self._preferred_piper_voice_label(options, preferred)
                )
            return

        language = self._selected_language_option(self.target_language.get())
        options = self._voice_options_for_language(language, self.piper_voice_options)
        self.target_voice_options = options
        self.target_voice_map = self._voice_map_for_options(options)
        self.target_voice_box["values"] = (
            self._voice_display_labels(options)
            if options
            else [self.NO_VOICE_LABEL]
        )
        if self.target_voice_name.get() not in self.target_voice_map:
            preferred = (
                language.default_target_voice if language is not None else None
            )
            self.target_voice_name.set(
                self._preferred_piper_voice_label(options, preferred)
            )

    def _on_close(self) -> None:
        if self.controller is not None:
            try:
                self.controller.stop()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = TranslatorApp(root)
    root.app = app
    root.mainloop()


if __name__ == "__main__":
    main()
