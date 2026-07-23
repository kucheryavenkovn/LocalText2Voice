from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QThread, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.video_dubbing.models import (
    Alignment,
    CueStatus,
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    DuckingSettings,
    ExportSettings,
    OriginalAudioMode,
    OutputContainer,
    PreviewSettings,
    SyncMode,
)
from app.core.video_dubbing.project_store import DUBBING_MANIFEST_NAME
from app.core.video_dubbing.service import (
    VideoDubbingService,
    VideoDubbingServiceError,
)
from app.core.video_dubbing.voice_catalog import VoiceCatalogService
from app.tts.base import BaseTTSEngine
from app.tts.engine_registry import TTS_ENGINES
from app.workers.video_dubbing_worker import VideoDubbingWorker

from .cue_timeline_widget import CueTimelineWidget
from .icons import ui_icon
from .widgets import FilePicker, LogView, PathPicker


LISTENING_ORIGINAL = "original"
LISTENING_TRANSLATION = "translation"
LISTENING_MIX = "mix"

_TABLE_COLUMNS = (
    ("seq", "#"),
    ("start", "Start"),
    ("end", "End"),
    ("budget", "Budget"),
    ("text", "Text"),
    ("raw", "Raw ms"),
    ("required", "Required"),
    ("applied", "Applied"),
    ("fitted", "Fitted ms"),
    ("diff", "Diff ms"),
    ("overflow", "Overflow"),
    ("status", "Status"),
    ("warnings", "Warnings"),
)


class VideoDubbingPage(QWidget):
    """Self-contained «Озвучка видео» page."""

    projectOpened = Signal(str)

    def __init__(
        self,
        tr: Callable[[str, str], str],
        tts_engine: BaseTTSEngine | None = None,
        ffmpeg_path: str = "ffmpeg/ffmpeg.exe",
        piper_path: str = "engines/piper/piper.exe",
        default_output_dir: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.tr = tr
        self._tts_engine = tts_engine
        self._ffmpeg_path = ffmpeg_path
        self._piper_path = piper_path
        self._default_output_dir = default_output_dir
        self._service: VideoDubbingService | None = None
        self._project: DubbingProject | None = None
        self._worker_thread: QThread | None = None
        self._worker: VideoDubbingWorker | None = None
        self._listening_mode = LISTENING_ORIGINAL
        self._loop_cue = False
        self._selected_sequence: int | None = None
        self._voice_config_override: dict[str, Any] | None = None
        self._voices: list = []
        self._loading_ui = False
        self.voice_catalog = VoiceCatalogService(piper_path=piper_path)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave)
        self._build_ui()
        self._install_shortcuts()

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_setup_panel())
        splitter.addWidget(self._build_player_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 2)

        root.addWidget(self._build_actions_row())
        self.timeline_widget = CueTimelineWidget()
        root.addWidget(self.timeline_widget)
        root.addWidget(self._build_table(), 3)
        root.addWidget(self._build_progress_row())
        self.log_view = LogView()
        self.log_view.setFixedHeight(110)
        root.addWidget(self.log_view)

    def _build_setup_panel(self) -> QWidget:
        box = QGroupBox(self.tr("video_dubbing_setup", "Project Setup"))
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.video_picker = FilePicker(
            self.tr("video_dubbing_browse_video", "Browse…"),
            "Video (*.mp4 *.mkv *.mov *.webm);;All files (*.*)",
        )
        self.srt_picker = FilePicker(
            self.tr("video_dubbing_browse_srt", "Browse…"),
            "SubRip (*.srt);;All files (*.*)",
        )
        self.project_dir_picker = PathPicker(
            self.tr("video_dubbing_browse_dir", "Browse…"),
            self._default_output_dir,
        )
        self.title_edit = QLineEdit(self.tr("video_dubbing_default_title", "Video Dubbing"))
        self.language_edit = QLineEdit("ru")

        self.tts_engine_combo = QComboBox()
        for engine in TTS_ENGINES:
            self.tts_engine_combo.addItem(engine.display_name, engine.engine_id)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_engine_changed)

        self.voice_combo = QComboBox()
        self.voice_combo.setEditable(True)
        self.voice_refresh_button = QPushButton(self.tr("video_dubbing_refresh_voices", "Refresh"))
        self.voice_refresh_button.clicked.connect(self._refresh_voices)
        self.voices_page_button = QPushButton(self.tr("video_dubbing_voices_mgr", "Voices…"))
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_combo, 1)
        voice_row.addWidget(self.voice_refresh_button)
        voice_row.addWidget(self.voices_page_button)
        voice_widget = QWidget()
        voice_widget.setLayout(voice_row)
        voice_widget.layout().setContentsMargins(0, 0, 0, 0)

        self.preferred_speed_spin = QDoubleSpinBox()
        self.preferred_speed_spin.setRange(1.05, 4.0)
        self.preferred_speed_spin.setSingleStep(0.05)
        self.preferred_speed_spin.setValue(1.35)
        self.hard_speed_spin = QDoubleSpinBox()
        self.hard_speed_spin.setRange(1.1, 8.0)
        self.hard_speed_spin.setSingleStep(0.1)
        self.hard_speed_spin.setValue(2.50)
        self.guard_gap_spin = QSpinBox()
        self.guard_gap_spin.setRange(0, 1000)
        self.guard_gap_spin.setValue(20)
        self.guard_gap_spin.setSuffix(" ms")
        self.compress_pauses_check = QCheckBox(
            self.tr("video_dubbing_compress_pauses", "Compress internal pauses")
        )
        self.compress_pauses_check.setChecked(False)

        self.sync_combo = QComboBox()
        self.sync_combo.addItem(self.tr("video_dubbing_strict", "Strict"), SyncMode.STRICT.value)
        self.sync_combo.addItem(
            self.tr("video_dubbing_best_effort", "Best effort"),
            SyncMode.BEST_EFFORT.value,
        )

        self.original_mode_combo = QComboBox()
        for mode in OriginalAudioMode:
            self.original_mode_combo.addItem(mode.value, mode.value)
        self.original_outside_spin = QSpinBox()
        self.original_outside_spin.setRange(0, 100)
        self.original_outside_spin.setValue(100)
        self.original_during_spin = QSpinBox()
        self.original_during_spin.setRange(0, 100)
        self.original_during_spin.setValue(15)
        self.narration_volume_spin = QSpinBox()
        self.narration_volume_spin.setRange(0, 200)
        self.narration_volume_spin.setValue(100)
        self.attack_spin = QSpinBox()
        self.attack_spin.setRange(0, 2000)
        self.attack_spin.setValue(100)
        self.attack_spin.setSuffix(" ms")
        self.release_spin = QSpinBox()
        self.release_spin.setRange(0, 5000)
        self.release_spin.setValue(250)
        self.release_spin.setSuffix(" ms")

        self.container_combo = QComboBox()
        self.container_combo.addItem("MKV", OutputContainer.MKV.value)
        self.container_combo.addItem("MP4", OutputContainer.MP4.value)
        self.narration_only_check = QCheckBox(
            self.tr("video_dubbing_add_narration_only", "Add Narration Only track")
        )
        self.narration_only_check.setChecked(True)
        self.embed_srt_check = QCheckBox(
            self.tr("video_dubbing_embed_srt", "Embed subtitles")
        )

        form.addRow(self.tr("video_dubbing_video", "Video"), self.video_picker)
        form.addRow(self.tr("video_dubbing_srt", "SRT"), self.srt_picker)
        form.addRow(self.tr("video_dubbing_project_dir", "Project folder"), self.project_dir_picker)
        form.addRow(self.tr("video_dubbing_title", "Title"), self.title_edit)
        form.addRow(self.tr("video_dubbing_language", "Language"), self.language_edit)
        form.addRow(self.tr("video_dubbing_engine", "TTS engine"), self.tts_engine_combo)
        form.addRow(self.tr("video_dubbing_voice", "Voice"), voice_widget)
        form.addRow(self.tr("video_dubbing_preferred_speed", "Preferred speed"), self.preferred_speed_spin)
        form.addRow(self.tr("video_dubbing_hard_speed", "Hard speed limit"), self.hard_speed_spin)
        form.addRow(self.tr("video_dubbing_guard_gap", "Guard gap"), self.guard_gap_spin)
        form.addRow("", self.compress_pauses_check)
        form.addRow(self.tr("video_dubbing_sync_mode", "Sync mode"), self.sync_combo)
        form.addRow(self.tr("video_dubbing_original_mode", "Original audio"), self.original_mode_combo)
        form.addRow(self.tr("video_dubbing_original_outside", "Original outside (%)"), self.original_outside_spin)
        form.addRow(self.tr("video_dubbing_original_during", "Original during (%)"), self.original_during_spin)
        form.addRow(self.tr("video_dubbing_narration_volume", "Narration volume (%)"), self.narration_volume_spin)
        form.addRow(self.tr("video_dubbing_attack", "Attack"), self.attack_spin)
        form.addRow(self.tr("video_dubbing_release", "Release"), self.release_spin)
        form.addRow(self.tr("video_dubbing_container", "Container"), self.container_combo)
        form.addRow("", self.narration_only_check)
        form.addRow("", self.embed_srt_check)

        for picker in (self.video_picker, self.srt_picker, self.project_dir_picker):
            picker.path_changed.connect(self._schedule_autosave)
        for widget in (
            self.title_edit, self.language_edit, self.preferred_speed_spin,
            self.hard_speed_spin, self.guard_gap_spin, self.compress_pauses_check,
            self.sync_combo, self.original_mode_combo, self.original_outside_spin,
            self.original_during_spin, self.narration_volume_spin, self.attack_spin,
            self.release_spin, self.container_combo, self.narration_only_check,
            self.embed_srt_check,
        ):
            self._connect_change_signal(widget, self._schedule_autosave)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(box)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    def _build_player_panel(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(220)
        layout.addWidget(self.video_widget, 1)

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.setVideoOutput(self.video_widget)
        self.audio_output.setVolume(0.9)
        self.media_player.positionChanged.connect(self._on_position_changed)
        self.media_player.durationChanged.connect(self._on_duration_changed)

        controls = QHBoxLayout()
        self.play_button = QPushButton(self.tr("video_dubbing_play", "Play"))
        self.play_button.setIcon(ui_icon("play"))
        self.pause_button = QPushButton(self.tr("video_dubbing_pause", "Pause"))
        self.stop_button = QPushButton(self.tr("video_dubbing_stop", "Stop"))
        self.prev_cue_button = QPushButton(self.tr("video_dubbing_prev", "Prev"))
        self.next_cue_button = QPushButton(self.tr("video_dubbing_next", "Next"))
        self.loop_button = QPushButton(self.tr("video_dubbing_loop", "Loop cue"))
        self.loop_button.setCheckable(True)
        self.play_button.clicked.connect(self._play)
        self.pause_button.clicked.connect(self.media_player.pause)
        self.stop_button.clicked.connect(self.media_player.stop)
        self.prev_cue_button.clicked.connect(self._goto_prev_cue)
        self.next_cue_button.clicked.connect(self._goto_next_cue)
        self.loop_button.toggled.connect(self._on_loop_toggled)
        for button in (
            self.play_button, self.pause_button, self.stop_button,
            self.prev_cue_button, self.next_cue_button, self.loop_button,
        ):
            controls.addWidget(button)
        controls.addStretch(1)

        self.original_button = QPushButton(self.tr("video_dubbing_listen_original", "Original"))
        self.translation_button = QPushButton(self.tr("video_dubbing_listen_translation", "Translation"))
        self.mix_button = QPushButton(self.tr("video_dubbing_listen_mix", "Mix"))
        for button in (self.original_button, self.translation_button, self.mix_button):
            button.setCheckable(True)
        self.original_button.clicked.connect(lambda: self._set_listening_mode(LISTENING_ORIGINAL))
        self.translation_button.clicked.connect(lambda: self._set_listening_mode(LISTENING_TRANSLATION))
        self.mix_button.clicked.connect(lambda: self._set_listening_mode(LISTENING_MIX))
        controls.addWidget(self.original_button)
        controls.addWidget(self.translation_button)
        controls.addWidget(self.mix_button)
        layout.addLayout(controls)

        position_row = QHBoxLayout()
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self._seek_slider)
        self.time_label = QLabel("00:00:00.000 / 00:00:00.000")
        self.time_label.setMinimumWidth(220)
        position_row.addWidget(self.position_slider, 1)
        position_row.addWidget(self.time_label)
        layout.addLayout(position_row)
        return wrapper

    def _build_actions_row(self) -> QWidget:
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        actions = (
            (self.tr("video_dubbing_create", "Create project"), self._create_project),
            (self.tr("video_dubbing_open", "Open project"), self._open_project),
            (self.tr("video_dubbing_open_recent", "Open recent"), self._open_recent),
            (self.tr("video_dubbing_save", "Save"), self._save_project),
            (self.tr("video_dubbing_import_srt", "Import SRT"), self._import_srt),
            (self.tr("video_dubbing_analyze", "Analyze"), self._analyze),
            (self.tr("video_dubbing_generate_missing", "Generate missing"), lambda: self._run_generation("missing")),
            (self.tr("video_dubbing_generate_selected", "Generate selected"), lambda: self._run_generation("selected")),
            (self.tr("video_dubbing_force_all", "Force regenerate all"), self._confirm_force_all),
            (self.tr("video_dubbing_refit", "Re-fit existing"), self._refit_existing),
            (self.tr("video_dubbing_render_narration", "Render narration"), self._render_narration),
            (self.tr("video_dubbing_render_mix", "Render mix"), self._render_mix),
            (self.tr("video_dubbing_render_cue_preview", "Cue preview"), self._render_cue_preview),
            (self.tr("video_dubbing_render_full_preview", "Full preview"), self._render_full_preview),
            (self.tr("video_dubbing_export", "Export video"), self._export_video),
            (self.tr("video_dubbing_cancel", "Cancel"), self._cancel),
            (self.tr("video_dubbing_open_folder", "Open folder"), self._open_folder),
        )
        for label, handler in actions:
            button = QPushButton(label)
            button.clicked.connect(handler)
            row.addWidget(button)
        row.addStretch(1)
        return wrapper

    def _build_table(self) -> QWidget:
        box = QGroupBox(self.tr("video_dubbing_cues", "Cues"))
        layout = QVBoxLayout(box)
        self.cue_table = QTableWidget(0, len(_TABLE_COLUMNS))
        self.cue_table.setHorizontalHeaderLabels([header for _key, header in _TABLE_COLUMNS])
        self.cue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cue_table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.cue_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.cue_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.cue_table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.cue_table)
        return box

    def _build_progress_row(self) -> QWidget:
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        self.stage_label = QLabel("—")
        self.stage_label.setMinimumWidth(140)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_status = QLabel("")
        row.addWidget(self.stage_label)
        row.addWidget(self.progress_bar, 1)
        row.addWidget(self.progress_status)
        return wrapper

    def _install_shortcuts(self) -> None:
        save_sc = QShortcut(QKeySequence("Ctrl+S"), self)
        save_sc.activated.connect(self._save_project)
        open_sc = QShortcut(QKeySequence("Ctrl+O"), self)
        open_sc.activated.connect(self._open_project)

    @staticmethod
    def _connect_change_signal(widget: QWidget, slot: Callable[..., None]) -> None:
        for signal_name in ("valueChanged", "textChanged", "stateChanged", "currentIndexChanged"):
            signal = getattr(widget, signal_name, None)
            if signal is not None:
                signal.connect(slot)
                return

    # ------------------------------------------------------------------ service

    def set_tts_engine(self, engine: BaseTTSEngine) -> None:
        self._tts_engine = engine
        if self._service is not None:
            self._service.tts_engine = engine

    def set_ffmpeg_path(self, ffmpeg_path: str) -> None:
        self._ffmpeg_path = ffmpeg_path

    def set_engine_context(
        self,
        engine: BaseTTSEngine | None,
        voice_config: dict[str, Any] | None,
        ffmpeg_path: str | None = None,
    ) -> None:
        if engine is not None:
            self._tts_engine = engine
        if voice_config is not None:
            self._voice_config_override = dict(voice_config)
            engine_id = str(voice_config.get("engine", ""))
            if engine_id:
                index = self.tts_engine_combo.findData(engine_id)
                if index >= 0:
                    self._loading_ui = True
                    self.tts_engine_combo.setCurrentIndex(index)
                    self._loading_ui = False
            voice_value = voice_config.get("voice") or voice_config.get("speaker")
            if voice_value:
                self.voice_combo.setCurrentText(str(voice_value))
        if ffmpeg_path:
            self._ffmpeg_path = ffmpeg_path
        self._refresh_voices()

    def _ensure_service(self) -> VideoDubbingService:
        if self._tts_engine is None:
            raise VideoDubbingServiceError(
                "No TTS engine available. Select an engine in Settings."
            )
        if self._service is None:
            self._service = VideoDubbingService(
                self._tts_engine,
                log_callback=self.log_view.append_event,
                progress_callback=self._on_progress,
                cue_updated_callback=self._on_cue_updated,
            )
        else:
            self._service.tts_engine = self._tts_engine
            self._service.log_callback = self.log_view.append_event
            self._service.progress_callback = self._on_progress
            self._service.cue_updated_callback = self._on_cue_updated
        return self._service

    # ------------------------------------------------------------------ UI <-> project

    def _build_settings(self) -> DubbingProjectSettings:
        ducking = DuckingSettings(
            mode=OriginalAudioMode(self.original_mode_combo.currentData()),
            original_outside_percent=float(self.original_outside_spin.value()),
            original_during_percent=float(self.original_during_spin.value()),
            narration_volume_percent=float(self.narration_volume_spin.value()),
            attack_ms=int(self.attack_spin.value()),
            release_ms=int(self.release_spin.value()),
        )
        export = ExportSettings(
            container=OutputContainer(self.container_combo.currentData()),
            include_narration_only=self.narration_only_check.isChecked(),
            embed_subtitles=self.embed_srt_check.isChecked(),
        )
        voice_config = self._current_voice_config()
        return DubbingProjectSettings(
            language=self.language_edit.text().strip(),
            tts_engine=str(self.tts_engine_combo.currentData() or "piper"),
            voice=self.voice_combo.currentText().strip(),
            voice_config=voice_config,
            preferred_speed_limit=float(self.preferred_speed_spin.value()),
            hard_speed_limit=float(self.hard_speed_spin.value()),
            guard_gap_ms=int(self.guard_gap_spin.value()),
            compress_internal_pauses=self.compress_pauses_check.isChecked(),
            sync_mode=SyncMode(self.sync_combo.currentData()),
            ffmpeg_path=self._ffmpeg_path,
            ducking=ducking,
            preview=PreviewSettings(alignment=Alignment.START),
            export=export,
        )

    def _current_voice_config(self) -> dict[str, Any]:
        engine_id = str(self.tts_engine_combo.currentData() or "piper")
        voice_id = self.voice_combo.currentText().strip()
        if self._voice_config_override and self._voice_config_override.get("engine") == engine_id:
            config = dict(self._voice_config_override)
        else:
            config = {"engine": engine_id}
        try:
            config = self.voice_catalog.resolve_voice_config(engine_id, voice_id, config)
        except Exception:
            config.setdefault("voice", voice_id)
        return config

    def _apply_ui_to_project(self) -> None:
        if self._project is None:
            return
        self._project.title = self.title_edit.text().strip() or "Video Dubbing"
        self._project.settings = self._build_settings()
        self._project.selected_sequence = self._selected_sequence
        self._project.last_player_position_ms = int(self.media_player.position())

    def _apply_project_to_ui(self) -> None:
        if self._project is None:
            return
        self._loading_ui = True
        try:
            self.title_edit.setText(self._project.title)
            if self._project.video_path:
                self.video_picker.set_path(self._project.video_path)
            if self._project.srt_source_path:
                self.srt_picker.set_path(self._project.srt_source_path)
            self.project_dir_picker.set_path(str(self._project.project_dir))
            s = self._project.settings
            self.language_edit.setText(s.language or "")
            idx = self.tts_engine_combo.findData(s.tts_engine)
            if idx >= 0:
                self.tts_engine_combo.setCurrentIndex(idx)
            self.preferred_speed_spin.setValue(s.preferred_speed_limit)
            self.hard_speed_spin.setValue(s.hard_speed_limit)
            self.guard_gap_spin.setValue(s.guard_gap_ms)
            self.compress_pauses_check.setChecked(s.compress_internal_pauses)
            self.sync_combo.setCurrentIndex(
                0 if s.sync_mode == SyncMode.STRICT else 1
            )
            oidx = self.original_mode_combo.findData(s.ducking.mode.value)
            if oidx >= 0:
                self.original_mode_combo.setCurrentIndex(oidx)
            self.original_outside_spin.setValue(int(s.ducking.original_outside_percent))
            self.original_during_spin.setValue(int(s.ducking.original_during_percent))
            self.narration_volume_spin.setValue(int(s.ducking.narration_volume_percent))
            self.attack_spin.setValue(s.ducking.attack_ms)
            self.release_spin.setValue(s.ducking.release_ms)
            cidx = self.container_combo.findData(s.export.container.value)
            if cidx >= 0:
                self.container_combo.setCurrentIndex(cidx)
            self.narration_only_check.setChecked(s.export.include_narration_only)
            self.embed_srt_check.setChecked(s.export.embed_subtitles)
            self._voice_config_override = dict(s.voice_config) if s.voice_config else None
        finally:
            self._loading_ui = False
        self._refresh_voices()
        if self._project.settings.voice:
            self.voice_combo.setCurrentText(self._project.settings.voice)
        self._refresh_ui_from_project()
        self.timeline_widget.set_cues(self._project.cues, self._project.duration_ms)
        if self._project.selected_sequence is not None:
            self._selected_sequence = self._project.selected_sequence
            self.timeline_widget.set_selected(self._selected_sequence)

    # ------------------------------------------------------------------ project lifecycle

    def _create_project(self) -> None:
        project_dir = self.project_dir_picker.path()
        manifest = project_dir / DUBBING_MANIFEST_NAME
        if manifest.is_file():
            choice = QMessageBox.question(
                self,
                self.tr("video_dubbing_project_exists", "Project exists"),
                self.tr(
                    "video_dubbing_project_exists_msg",
                    "A project already exists in this folder. Open it instead?",
                ),
                QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
            )
            if choice == QMessageBox.StandardButton.Open:
                self._load_project_from_manifest(manifest)
                return
            return
        try:
            service = self._ensure_service()
            settings = self._build_settings()
            self._project = service.create_project(
                project_dir, title=self.title_edit.text().strip(), settings=settings
            )
            video_path = self.video_picker.path()
            if video_path and video_path.is_file():
                service.attach_video(self._project, video_path)
            self._apply_project_to_ui()
            self._set_listening_mode(LISTENING_ORIGINAL)
            self.projectOpened.emit(self._project.project_id)
        except VideoDubbingServiceError as exc:
            self._show_error(exc)

    def _open_project(self) -> None:
        manifest_path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("video_dubbing_open", "Open project"),
            str(self.project_dir_picker.path()),
            "Video dubbing project (dubbing_project.json)",
        )
        if manifest_path:
            self._load_project_from_manifest(Path(manifest_path))

    def _open_recent(self) -> None:
        service = self._ensure_service()
        recent = service.list_recent(10)
        if not recent:
            self._show_info(self.tr("video_dubbing_no_recent", "No recent projects."))
            return
        menu = QMenu(self)
        for entry in recent:
            label = f"{entry['title']} — {entry['cue_count']} cues — {entry['updated_at']}"
            action = menu.addAction(label)
            action.setData(entry["project_id"])
        chosen = menu.exec(self.mapToGlobal(self.rect().center()))
        if chosen is not None:
            try:
                service = self._ensure_service()
                self._project = service.load_project(chosen.data())
                self._apply_project_to_ui()
                self.projectOpened.emit(self._project.project_id)
            except VideoDubbingServiceError as exc:
                self._show_error(exc)

    def _load_project_from_manifest(self, manifest_path: Path) -> None:
        try:
            service = self._ensure_service()
            self._project = service.open_project_manifest(manifest_path)
            self._apply_project_to_ui()
            self.projectOpened.emit(self._project.project_id)
        except VideoDubbingServiceError as exc:
            self._show_error(exc)

    def _save_project(self) -> None:
        if self._project is None:
            self._show_info(self.tr("video_dubbing_no_project", "Create or open a project first."))
            return
        self._apply_ui_to_project()
        service = self._ensure_service()
        service.save_project(self._project)
        import time
        self.log_view.append_event(
            self.tr("video_dubbing_saved", "Project saved")
            + f": {self._project.project_dir / DUBBING_MANIFEST_NAME} "
            f"({time.strftime('%H:%M:%S')})"
        )

    def _autosave(self) -> None:
        if self._project is None or self._loading_ui:
            return
        try:
            self._apply_ui_to_project()
            service = self._ensure_service()
            service.save_project(self._project)
        except VideoDubbingServiceError as exc:
            self.log_view.append_event(f"Autosave skipped: {exc}")

    def _schedule_autosave(self, *args: Any) -> None:
        if self._loading_ui or self._project is None:
            return
        self._autosave_timer.start(700)

    # ------------------------------------------------------------------ import/generate

    def _import_srt(self) -> None:
        if self._project is None:
            self._show_info(self.tr("video_dubbing_no_project", "Create or open a project first."))
            return
        srt_path = self.srt_picker.path()
        if srt_path is None or not srt_path.is_file():
            self._show_info(self.tr("video_dubbing_no_srt", "Choose an SRT file first."))
            return
        try:
            service = self._ensure_service()
            result = service.import_srt(self._project, srt_path)
            self._refresh_ui_from_project()
            if result.warnings:
                self.log_view.append_event(f"SRT warnings: {len(result.warnings)} (see table).")
        except VideoDubbingServiceError as exc:
            self._show_error(exc)

    def _analyze(self) -> None:
        if self._project is None:
            return
        service = self._ensure_service()
        warnings = service.analyze_project(self._project)
        self._refresh_ui_from_project()
        self.log_view.append_event(f"Analysis: {len(warnings)} warning(s).")

    def _run_generation(self, mode: str) -> None:
        if self._project is None:
            return
        self._apply_ui_to_project()
        self._ensure_service()
        if mode == "missing":

            def op(svc):
                return svc.generate_all(self._project, force=False)

        elif mode == "selected":
            sequences = [
                int(self.cue_table.item(row, 0).text())
                for row in sorted(set(i.row() for i in self.cue_table.selectedIndexes()))
            ]
            if not sequences:
                self._show_info(self.tr("video_dubbing_select_cue", "Select at least one cue."))
                return

            def op(svc):  # noqa: F811
                return svc.generate_selected(self._project, sequences, force=True)
        else:
            return

        def on_done(result):
            self._refresh_ui_from_project()

        self._start_worker(op, on_done)

    def _confirm_force_all(self) -> None:
        if self._project is None:
            return
        choice = QMessageBox.question(
            self,
            self.tr("video_dubbing_force_all", "Force regenerate all"),
            self.tr(
                "video_dubbing_force_all_msg",
                "Re-synthesize every enabled cue? This cannot be undone.",
            ),
        )
        if choice != QMessageBox.StandardButton.Yes:
            return
        self._apply_ui_to_project()

        def op(svc):
            return svc.generate_all(self._project, force=True)

        def on_done(result):
            self._refresh_ui_from_project()

        self._start_worker(op, on_done)

    def _refit_existing(self) -> None:
        if self._project is None:
            return
        self._apply_ui_to_project()

        def op(svc):
            return svc.re_fit_existing(self._project)

        def on_done(result):
            self._refresh_ui_from_project()

        self._start_worker(op, on_done)

    def _render_narration(self) -> None:
        if self._project is None:
            return
        self._apply_ui_to_project()
        self._start_worker(
            lambda svc: svc.render_narration(self._project),
            lambda result: self._refresh_ui_from_project(),
        )

    def _render_mix(self) -> None:
        if self._project is None:
            return
        self._apply_ui_to_project()
        self._start_worker(
            lambda svc: svc.render_dubbed_mix(self._project),
            lambda result: self._refresh_ui_from_project(),
        )

    def _render_cue_preview(self) -> None:
        if self._project is None or self._selected_sequence is None:
            return
        sequence = self._selected_sequence

        def on_done(path):
            self._refresh_ui_from_project()
            if path and Path(str(path)).is_file():
                self._load_media(Path(str(path)))

        self._start_worker(
            lambda svc: svc.render_cue_preview(self._project, sequence), on_done
        )

    def _render_full_preview(self) -> None:
        if self._project is None:
            return

        def on_done(path):
            self._refresh_ui_from_project()
            if path and self._listening_mode == LISTENING_MIX:
                self._load_media(Path(str(path)))

        self._start_worker(lambda svc: svc.render_full_preview(self._project), on_done)

    def _export_video(self) -> None:
        if self._project is None:
            return
        service = self._ensure_service()
        can, blockers = service.can_export(self._project)
        if not can:
            self._show_error_message(
                "\n".join(blockers),
                title=self.tr("video_dubbing_export_blocked", "Export blocked"),
            )
            return
        self._apply_ui_to_project()
        self._start_worker(
            lambda svc: svc.export_video(self._project),
            lambda path: self.log_view.append_event(f"Exported: {path}"),
        )

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
        elif self._service is not None:
            self._service.cancel()

    def _open_folder(self) -> None:
        if self._project is None:
            return
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._project.render_dir())))

    # ------------------------------------------------------------------ worker

    def _start_worker(
        self,
        operation: Callable[[VideoDubbingService], Any],
        on_finished: Callable[[Any], None],
    ) -> None:
        if self._worker_thread is not None:
            return
        service = self._ensure_service()
        service.reset_cancel()
        thread = QThread(self)
        worker = VideoDubbingWorker(service, operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.cue_updated.connect(self._on_cue_updated)
        worker.log.connect(self.log_view.append_event)
        worker.finished.connect(self._on_worker_finished)
        worker.finished.connect(lambda result: on_finished(result))
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker_thread = thread
        self._worker = worker
        self._set_busy(True)
        thread.start()

    def _clear_worker(self) -> None:
        self._worker_thread = None
        self._worker = None
        self._set_busy(False)

    def _on_worker_finished(self, _result: Any) -> None:
        self._set_busy(False)

    def _on_worker_failed(self, message: str) -> None:
        self._set_busy(False)
        self.log_view.append_event(message)
        self._show_error_message(message)

    def _on_worker_cancelled(self) -> None:
        self._set_busy(False)
        self.log_view.append_event(self.tr("video_dubbing_cancelled", "Cancelled."))
        self._refresh_ui_from_project()

    def _on_progress(self, stage: str, current: int, total: int, message: str) -> None:
        self.stage_label.setText(stage)
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))
        else:
            self.progress_bar.setValue(0)
        self.progress_status.setText(
            f"{current}/{total} — {message}" if total > 0 else message
        )

    def _on_cue_updated(
        self, sequence: int, status: str, raw_ms: int, fitted_ms: int, index: int
    ) -> None:
        if self._project is None:
            return
        cue = next((c for c in self._project.cues if c.sequence == sequence), None)
        if cue is None:
            return
        cue.status = status
        if raw_ms:
            cue.raw_duration_ms = raw_ms
        if fitted_ms:
            cue.fitted_duration_ms = fitted_ms
        self._update_cue_row(cue)
        self.timeline_widget.set_cues(self._project.cues, self._project.duration_ms)

    # ------------------------------------------------------------------ player

    def _play(self) -> None:
        if self._project is None:
            return
        if self._listening_mode == LISTENING_ORIGINAL and self._project.video_path:
            self._load_media(Path(self._project.video_path))
        elif self._listening_mode == LISTENING_MIX and self._project.full_preview_path:
            self._load_media(Path(self._project.full_preview_path))
        elif self._listening_mode == LISTENING_TRANSLATION and self._project.narration_wav:
            self._load_media(Path(self._project.narration_wav))
        self.media_player.play()

    def _load_media(self, path: Path) -> None:
        if path and path.is_file():
            self.media_player.setSource(QUrl.fromLocalFile(str(path)))

    def _seek_slider(self, position: int) -> None:
        self.media_player.setPosition(position)

    def _on_position_changed(self, position: int) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        self.timeline_widget.set_position_ms(int(position))
        self._update_time_label(position, self.media_player.duration())
        if (
            self._loop_cue
            and self._selected_sequence is not None
            and self._project
        ):
            cue = next(
                (c for c in self._project.cues if c.sequence == self._selected_sequence),
                None,
            )
            if cue is not None and position > cue.end_ms + self._project.settings.preview.post_roll_ms:
                self.media_player.setPosition(
                    max(0, cue.start_ms - self._project.settings.preview.pre_roll_ms)
                )

    def _on_duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, duration)
        self._update_time_label(self.media_player.position(), duration)

    def _update_time_label(self, position: int, duration: int) -> None:
        self.time_label.setText(f"{_format_ms(position)} / {_format_ms(duration)}")

    def _set_listening_mode(self, mode: str) -> None:
        self._listening_mode = mode
        self.original_button.setChecked(mode == LISTENING_ORIGINAL)
        self.translation_button.setChecked(mode == LISTENING_TRANSLATION)
        self.mix_button.setChecked(mode == LISTENING_MIX)
        if self._project is None:
            return
        if mode == LISTENING_ORIGINAL and self._project.video_path:
            self._load_media(Path(self._project.video_path))
        elif mode == LISTENING_MIX and self._project.full_preview_path:
            self._load_media(Path(self._project.full_preview_path))
        elif mode == LISTENING_TRANSLATION and self._project.narration_wav:
            self._load_media(Path(self._project.narration_wav))

    def _on_loop_toggled(self, checked: bool) -> None:
        self._loop_cue = checked

    def _goto_prev_cue(self) -> None:
        if not self._project or not self._project.cues:
            return
        ordered = sorted(self._project.cues, key=lambda c: c.start_ms)
        current_ms = self.media_player.position()
        prev = None
        for cue in ordered:
            if cue.start_ms < current_ms - 200:
                prev = cue
            else:
                break
        if prev is not None:
            self._jump_to_cue(prev.sequence)

    def _goto_next_cue(self) -> None:
        if not self._project or not self._project.cues:
            return
        ordered = sorted(self._project.cues, key=lambda c: c.start_ms)
        current_ms = self.media_player.position()
        for cue in ordered:
            if cue.start_ms > current_ms + 200:
                self._jump_to_cue(cue.sequence)
                return

    def _jump_to_cue(self, sequence: int) -> None:
        if self._project is None:
            return
        cue = next((c for c in self._project.cues if c.sequence == sequence), None)
        if cue is None:
            return
        pre_roll = self._project.settings.preview.pre_roll_ms
        self.media_player.setPosition(max(0, cue.start_ms - pre_roll))

    # ------------------------------------------------------------------ voices

    def _on_engine_changed(self, _index: int) -> None:
        if self._loading_ui:
            return
        engine_id = str(self.tts_engine_combo.currentData() or "piper")
        try:
            engine = self.voice_catalog.create_engine(engine_id, self._piper_path)
            self._tts_engine = engine
            if self._service is not None:
                self._service.tts_engine = engine
        except Exception as exc:
            self.log_view.append_event(f"Could not create engine {engine_id}: {exc}")
        self._refresh_voices()
        self._schedule_autosave()

    def _refresh_voices(self) -> None:
        engine_id = str(self.tts_engine_combo.currentData() or "piper")
        previous = self.voice_combo.currentText()
        self._loading_ui = True
        self.voice_combo.clear()
        try:
            self._voices = self.voice_catalog.list_voices(engine_id)
        except Exception as exc:
            self._voices = []
            self.log_view.append_event(f"Voice list failed: {exc}")
        for voice in self._voices:
            self.voice_combo.addItem(voice.display_name, voice.voice_id)
        if previous:
            self.voice_combo.setCurrentText(previous)
        self._loading_ui = False
        if not self._voices:
            self.log_view.append_event(
                self.tr("video_dubbing_no_voices", "No voices listed for this engine; type a value manually.")
            )

    # ------------------------------------------------------------------ table

    def _refresh_ui_from_project(self) -> None:
        if self._project is None:
            return
        self.cue_table.blockSignals(True)
        self.cue_table.setRowCount(0)
        for cue in self._project.cues:
            self._append_cue_row(cue)
        self.cue_table.blockSignals(False)
        self.timeline_widget.set_cues(self._project.cues, self._project.duration_ms)

    def _append_cue_row(self, cue: DubbingCue) -> int:
        row = self.cue_table.rowCount()
        self.cue_table.insertRow(row)
        self._set_row_values(row, cue)
        return row

    def _update_cue_row(self, cue: DubbingCue) -> None:
        for row in range(self.cue_table.rowCount()):
            item = self.cue_table.item(row, 0)
            if item is not None and item.text() == str(cue.sequence):
                self.cue_table.blockSignals(True)
                self._set_row_values(row, cue)
                self.cue_table.blockSignals(False)
                return
        # not found: append
        self.cue_table.blockSignals(True)
        self._append_cue_row(cue)
        self.cue_table.blockSignals(False)

    def _set_row_values(self, row: int, cue: DubbingCue) -> None:
        values = (
            str(cue.sequence),
            _format_ms(cue.start_ms),
            _format_ms(cue.end_ms),
            f"{cue.duration_budget_ms} ms",
            cue.spoken_text,
            f"{cue.raw_duration_ms} ms" if cue.raw_duration_ms else "—",
            f"{cue.required_speed_factor:.2f}" if cue.required_speed_factor else "—",
            f"{cue.applied_speed_factor:.2f}" if cue.applied_speed_factor else "1.00",
            f"{cue.fitted_duration_ms} ms" if cue.fitted_duration_ms else "—",
            f"{cue.timing_diff_ms} ms" if cue.timing_diff_ms is not None else "—",
            f"{cue.overflow_ms} ms" if cue.overflow_ms else "—",
            cue.status,
            ", ".join(cue.warning_codes),
        )
        for col, text in enumerate(values):
            existing = self.cue_table.item(row, col)
            editable = col == 4
            if existing is None:
                item = QTableWidgetItem(text)
                if not editable:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.cue_table.setItem(row, col, item)
            else:
                existing.setText(text)
        # color the status cell by severity
        status_item = self.cue_table.item(row, 11)
        if status_item is not None:
            color = _STATUS_UI_COLORS.get(cue.status, QColor("#94a3b8"))
            status_item.setForeground(QBrush(color))

    def _on_table_selection_changed(self) -> None:
        rows = sorted(set(index.row() for index in self.cue_table.selectedIndexes()))
        if not rows:
            return
        seq_item = self.cue_table.item(rows[0], 0)
        if seq_item is None:
            return
        try:
            sequence = int(seq_item.text())
        except ValueError:
            return
        self._selected_sequence = sequence
        self.timeline_widget.set_selected(sequence)
        self._jump_to_cue(sequence)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if self._loading_ui or self._project is None or item.column() != 4:
            return
        row = item.row()
        seq_item = self.cue_table.item(row, 0)
        if seq_item is None:
            return
        try:
            sequence = int(seq_item.text())
        except ValueError:
            return
        service = self._ensure_service()
        try:
            service.update_cue_text(self._project, sequence, item.text())
            self._schedule_autosave()
        except VideoDubbingServiceError as exc:
            self._show_error(exc)

    # ------------------------------------------------------------------ misc

    def _set_busy(self, busy: bool) -> None:
        self.progress_bar.setEnabled(True)
        for button in self.findChildren(QPushButton):
            if button is self._cancel_button_for_busy():
                continue
            # Only disable the action buttons row, keep player + cancel.
        cancel = self._cancel_button_for_busy()
        if cancel is not None:
            cancel.setEnabled(True)

    def _cancel_button_for_busy(self) -> QPushButton | None:
        return None  # Cancel stays enabled via its own connection; simplified.

    def _show_error(self, exc: VideoDubbingServiceError) -> None:
        self._show_error_message(str(exc))

    def _show_error_message(self, message: str, title: str | None = None) -> None:
        QMessageBox.critical(
            self,
            title or self.tr("video_dubbing_error", "Video dubbing error"),
            message,
        )

    def _show_info(self, message: str) -> None:
        QMessageBox.information(self, "Video dubbing", message)

    def cleanup(self) -> None:
        self._cancel()
        if self.media_player is not None:
            self.media_player.stop()


_STATUS_UI_COLORS = {
    CueStatus.RENDERED.value: QColor("#16a34a"),
    CueStatus.FITTED.value: QColor("#16a34a"),
    CueStatus.SPEED_UP.value: QColor("#2563eb"),
    CueStatus.STRONG_SPEED_UP.value: QColor("#ea580c"),
    CueStatus.EXTREME_SPEED_REQUIRED.value: QColor("#dc2626"),
    CueStatus.FAILED.value: QColor("#7f1d1d"),
    CueStatus.STALE.value: QColor("#ca8a04"),
    CueStatus.DISABLED.value: QColor("#9ca3af"),
    CueStatus.PENDING.value: QColor("#94a3b8"),
    CueStatus.RENDERING.value: QColor("#0891b2"),
}


def _format_ms(total_ms: int) -> str:
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(int(total_ms), 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, milliseconds = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
