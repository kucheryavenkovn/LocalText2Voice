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
    QGridLayout,
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
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.video_dubbing.generation import GenerationRunResult, GenerationRunStatus
from app.core.video_dubbing.models import (
    Alignment,
    CueStatus,
    CueTimingMode,
    DubbingCue,
    DubbingProject,
    DubbingProjectSettings,
    DuckingSettings,
    ElasticTimingSettings,
    ExportSettings,
    ORIGINAL_AUDIO_MODE_LABELS_RU,
    OriginalAudioMode,
    OutputContainer,
    PreviewSettings,
    SyncMode,
    TEMPO_SMOOTHING_MODE_LABELS_RU,
    TempoSmoothingMode,
    TempoSmoothingSettings,
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
    ("seq", "№"),
    ("start", "Начало"),
    ("end", "Конец"),
    ("plan", "План"),
    ("text", "Текст"),
    ("raw", "TTS"),
    ("applied", "Скорость"),
    ("shift", "Сдвиг"),
    ("group", "Группа"),
    ("fitted", "Fitted"),
    ("status", "Статус"),
    ("warnings", "Предупр."),
)


class VideoDubbingPage(QWidget):
    """Self-contained «Озвучка видео» page."""

    projectOpened = Signal(str)
    openReviewRequested = Signal(object)  # DubbingProject

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
        self._generation_busy = False
        self._action_buttons: list[QPushButton] = []
        self._settings_lock_widgets: list[QWidget] = []
        self._preview_global_offset_ms = 0
        self._cue_preview_end_ms: int | None = None
        self._pending_seek_ms: int | None = None
        self._pending_play = False
        self._was_playing = False
        self.voice_catalog = VoiceCatalogService(piper_path=piper_path)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave)
        self._build_ui()
        self._install_shortcuts()

    # ------------------------------------------------------------------ build

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.top_splitter.addWidget(self._build_player_panel())
        self.top_splitter.addWidget(self._build_setup_panel())
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 1)
        self.top_splitter.setSizes([960, 960])
        root.addWidget(self.top_splitter, 4)

        root.addWidget(self._build_actions_row())
        self.timeline_widget = CueTimelineWidget()
        self.timeline_widget.cueSelected.connect(self._on_timeline_cue_selected)
        self.timeline_widget.positionSeeked.connect(self._on_timeline_seek)
        self.timeline_widget.cueActivated.connect(self._on_timeline_cue_activated)
        root.addWidget(self.timeline_widget)
        root.addWidget(self._build_table(), 5)
        root.addWidget(self._build_progress_row())
        self.log_view = LogView()
        self.log_view.setFixedHeight(90)
        root.addWidget(self.log_view)

    def _build_setup_panel(self) -> QWidget:
        box = QGroupBox(self.tr("video_dubbing_setup", "Настройки проекта"))
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 6, 6, 6)
        tabs = QTabWidget()
        self.settings_tabs = tabs

        self.video_picker = FilePicker(
            self.tr("video_dubbing_browse_video", "Обзор…"),
            "Video (*.mp4 *.mkv *.mov *.webm);;All files (*.*)",
        )
        self.srt_picker = FilePicker(
            self.tr("video_dubbing_browse_srt", "Обзор…"),
            "SubRip (*.srt);;All files (*.*)",
        )
        self.project_dir_picker = PathPicker(
            self.tr("video_dubbing_browse_dir", "Обзор…"),
            self._default_output_dir,
        )
        self.title_edit = QLineEdit(self.tr("video_dubbing_default_title", "Озвучка видео"))
        self.language_edit = QLineEdit("ru")

        self.tts_engine_combo = QComboBox()
        for engine in TTS_ENGINES:
            self.tts_engine_combo.addItem(engine.display_name, engine.engine_id)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_engine_changed)

        self.voice_combo = QComboBox()
        self.voice_combo.setEditable(True)
        self.voice_combo.currentIndexChanged.connect(self._on_voice_changed)
        self.voice_combo.editTextChanged.connect(self._on_voice_edit_text)
        self.voice_refresh_button = QPushButton(self.tr("video_dubbing_refresh_voices", "Обновить"))
        self.voice_refresh_button.clicked.connect(self._refresh_voices)
        self.voice_preview_button = QPushButton(self.tr("video_dubbing_preview_ref", "▶ Ref"))
        self.voice_preview_button.setToolTip(
            self.tr(
                "video_dubbing_preview_ref_tip",
                "Прослушать reference-голос, который реально уйдёт в OmniVoice",
            )
        )
        self.voice_preview_button.clicked.connect(self._preview_reference_voice)
        self.voices_page_button = QPushButton(self.tr("video_dubbing_voices_mgr", "Голоса…"))
        voice_row = QHBoxLayout()
        voice_row.addWidget(self.voice_combo, 1)
        voice_row.addWidget(self.voice_preview_button)
        voice_row.addWidget(self.voice_refresh_button)
        voice_row.addWidget(self.voices_page_button)
        voice_widget = QWidget()
        voice_widget.setLayout(voice_row)
        voice_widget.layout().setContentsMargins(0, 0, 0, 0)
        self.voice_status_label = QLabel("")
        self.voice_status_label.setWordWrap(True)
        self.voice_status_label.setObjectName("helperLabel")

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
            self.tr("video_dubbing_compress_pauses", "Сжимать внутренние паузы")
        )
        self.compress_pauses_check.setChecked(False)
        self.pause_keep_spin = QSpinBox()
        self.pause_keep_spin.setRange(20, 500)
        self.pause_keep_spin.setValue(90)
        self.pause_keep_spin.setSuffix(" ms")

        self.sync_combo = QComboBox()
        self.sync_combo.addItem(self.tr("video_dubbing_strict", "Строгий"), SyncMode.STRICT.value)
        self.sync_combo.addItem(
            self.tr("video_dubbing_best_effort", "Best effort"),
            SyncMode.BEST_EFFORT.value,
        )
        self.timing_mode_combo = QComboBox()
        self.timing_mode_combo.addItem(
            self.tr("video_dubbing_timing_strict", "Жёсткие интервалы"),
            CueTimingMode.STRICT.value,
        )
        self.timing_mode_combo.addItem(
            self.tr("video_dubbing_timing_elastic", "Эластичные группы"),
            CueTimingMode.ELASTIC_GROUP.value,
        )
        self.elastic_max_cues_spin = QSpinBox()
        self.elastic_max_cues_spin.setRange(1, 5)
        self.elastic_max_cues_spin.setValue(3)
        self.elastic_max_speed_spin = QDoubleSpinBox()
        self.elastic_max_speed_spin.setRange(1.05, 3.0)
        self.elastic_max_speed_spin.setSingleStep(0.05)
        self.elastic_max_speed_spin.setValue(1.35)
        self.elastic_min_gap_spin = QSpinBox()
        self.elastic_min_gap_spin.setRange(0, 1000)
        self.elastic_min_gap_spin.setValue(120)
        self.elastic_min_gap_spin.setSuffix(" ms")
        self.elastic_max_shift_spin = QSpinBox()
        self.elastic_max_shift_spin.setRange(0, 20000)
        self.elastic_max_shift_spin.setValue(2000)
        self.elastic_max_shift_spin.setSuffix(" ms")
        self.elastic_prefer_small_check = QCheckBox(
            self.tr("video_dubbing_prefer_smaller_group", "Предпочитать меньшую группу")
        )
        self.tempo_mode_combo = QComboBox()
        for mode in TempoSmoothingMode:
            self.tempo_mode_combo.addItem(
                self.tr(
                    f"video_dubbing_tempo_{mode.value}",
                    TEMPO_SMOOTHING_MODE_LABELS_RU.get(mode.value, mode.value),
                ),
                mode.value,
            )
        self.tempo_neighbors_spin = QSpinBox()
        self.tempo_neighbors_spin.setRange(2, 5)
        self.tempo_neighbors_spin.setValue(3)
        self.tempo_jump_spin = QDoubleSpinBox()
        self.tempo_jump_spin.setRange(0.01, 1.0)
        self.tempo_jump_spin.setSingleStep(0.01)
        self.tempo_jump_spin.setValue(0.10)
        self.tempo_delta_spin = QDoubleSpinBox()
        self.tempo_delta_spin.setRange(0.01, 1.0)
        self.tempo_delta_spin.setSingleStep(0.01)
        self.tempo_delta_spin.setValue(0.08)
        self.tempo_optional_spin = QDoubleSpinBox()
        self.tempo_optional_spin.setRange(0.0, 1.0)
        self.tempo_optional_spin.setSingleStep(0.05)
        self.tempo_optional_spin.setValue(0.20)
        self.autoplay_select_check = QCheckBox(
            self.tr(
                "video_dubbing_autoplay_on_select",
                "Воспроизводить реплику после выбора",
            )
        )

        self.original_mode_combo = QComboBox()
        for mode in OriginalAudioMode:
            label = self.tr(
                f"video_dubbing_mode_{mode.value}",
                ORIGINAL_AUDIO_MODE_LABELS_RU.get(mode.value, mode.value),
            )
            self.original_mode_combo.addItem(label, mode.value)
            tip = {
                OriginalAudioMode.REPLACE.value: self.tr(
                    "video_dubbing_mode_replace_tip",
                    "В дублированной дорожке остаётся только голос перевода.",
                ),
                OriginalAudioMode.CONSTANT.value: self.tr(
                    "video_dubbing_mode_constant_tip",
                    "Оригинал с постоянной громкостью, поверх — перевод.",
                ),
                OriginalAudioMode.DUCKING.value: self.tr(
                    "video_dubbing_mode_ducking_tip",
                    "Во время речи перевода громкость оригинала снижается.",
                ),
                OriginalAudioMode.NARRATION_ONLY.value: self.tr(
                    "video_dubbing_mode_narration_tip",
                    "Отдельная дорожка перевода без оригинала.",
                ),
            }.get(mode.value, "")
            self.original_mode_combo.setItemData(
                self.original_mode_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
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
            self.tr("video_dubbing_add_narration_only", "Добавить дорожку «Только перевод»")
        )
        self.narration_only_check.setChecked(True)
        self.embed_srt_check = QCheckBox(
            self.tr("video_dubbing_embed_srt", "Встроить субтитры")
        )

        def _form(*rows: tuple) -> QWidget:
            w = QWidget()
            f = QFormLayout(w)
            f.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            f.setContentsMargins(4, 4, 4, 4)
            f.setSpacing(6)
            for row in rows:
                if len(row) == 1:
                    f.addRow(row[0])
                else:
                    f.addRow(row[0], row[1])
            return w

        tabs.addTab(
            _form(
                (self.tr("video_dubbing_video", "Видео"), self.video_picker),
                (self.tr("video_dubbing_srt", "SRT"), self.srt_picker),
                (self.tr("video_dubbing_project_dir", "Каталог проекта"), self.project_dir_picker),
                (self.tr("video_dubbing_title", "Название"), self.title_edit),
                (self.tr("video_dubbing_language", "Язык"), self.language_edit),
            ),
            self.tr("video_dubbing_tab_project", "Проект"),
        )
        tabs.addTab(
            _form(
                (self.tr("video_dubbing_engine", "TTS-движок"), self.tts_engine_combo),
                (self.tr("video_dubbing_voice", "Голос"), voice_widget),
                ("", self.voice_status_label),
                (self.tr("video_dubbing_preferred_speed", "Предпочтительная скорость"), self.preferred_speed_spin),
                (self.tr("video_dubbing_hard_speed", "Жёсткий предел скорости"), self.hard_speed_spin),
                (self.compress_pauses_check,),
                (self.tr("video_dubbing_pause_keep", "Оставлять паузу"), self.pause_keep_spin),
            ),
            self.tr("video_dubbing_tab_voice", "Голос"),
        )
        tabs.addTab(
            _form(
                (self.tr("video_dubbing_timing_mode", "Режим тайминга"), self.timing_mode_combo),
                (self.tr("video_dubbing_sync_mode", "Режим синхронизации"), self.sync_combo),
                (self.tr("video_dubbing_guard_gap", "Защитный зазор"), self.guard_gap_spin),
                (self.tr("video_dubbing_elastic_max_cues", "Макс. реплик в группе"), self.elastic_max_cues_spin),
                (self.tr("video_dubbing_elastic_max_speed", "Макс. общий коэффициент"), self.elastic_max_speed_spin),
                (self.tr("video_dubbing_elastic_min_gap", "Мин. интервал"), self.elastic_min_gap_spin),
                (self.tr("video_dubbing_elastic_max_shift", "Макс. сдвиг"), self.elastic_max_shift_spin),
                (self.elastic_prefer_small_check,),
                (self.tr("video_dubbing_tempo_mode", "Сглаживание темпа"), self.tempo_mode_combo),
                (self.tr("video_dubbing_tempo_neighbors", "Соседних реплик"), self.tempo_neighbors_spin),
                (self.tr("video_dubbing_tempo_jump", "Порог скачка"), self.tempo_jump_spin),
                (self.tr("video_dubbing_tempo_delta", "Макс. разница соседей"), self.tempo_delta_spin),
                (self.tr("video_dubbing_tempo_optional", "Макс. ускорение «влезающих»"), self.tempo_optional_spin),
                (self.autoplay_select_check,),
            ),
            self.tr("video_dubbing_tab_timing", "Тайминг"),
        )
        self.timing_mode_combo.setToolTip(
            "Жёсткие интервалы — только своё start–end SRT, без сдвига.\n"
            "Эластичные группы — можно занять паузу справа и сдвинуть соседей "
            "только вправо с общим коэффициентом скорости.\n"
            "Исходный SRT не меняется (пишутся planned start/end).\n"
            "После смены → Пересчитать (не TTS)."
        )
        self.sync_combo.setToolTip(
            "Строгий — жёстче к переполнению окна.\n"
            "Best effort — мягче, чаще ускоряет до hard limit.\n"
            "Это про fit одной реплики, не про эластичные группы."
        )
        self.guard_gap_spin.setToolTip(
            "Минимальный зазор до следующей реплики при расчёте safe_end."
        )
        self.elastic_max_cues_spin.setToolTip(
            "Сколько соседних cue (1–5) можно объединить в эластичную группу."
        )
        self.elastic_max_speed_spin.setToolTip(
            "Максимальный общий коэффициент скорости внутри эластичной группы."
        )
        self.elastic_min_gap_spin.setToolTip(
            "Минимальная пауза между planned-концом и planned-началом соседа."
        )
        self.elastic_max_shift_spin.setToolTip(
            "Максимальный сдвиг одной реплики вправо (мс)."
        )
        self.tempo_mode_combo.setToolTip(
            "Сглаживание темпа — отдельно от эластики.\n"
            "В жёстком режиме не сдвигает таймкоды, только выравнивает скорости.\n"
            "После смены → Пересчитать (не TTS)."
        )
        self.autoplay_select_check.setToolTip(
            "Если включено — одиночный клик по шкале/таблице сразу играет реплику.\n"
            "Иначе клик только перематывает; двойной клик — play."
        )
        tabs.addTab(
            _form(
                (self.tr("video_dubbing_original_mode", "Режим наложения"), self.original_mode_combo),
                (self.tr("video_dubbing_original_outside", "Оригинал вне реплик (%)"), self.original_outside_spin),
                (self.tr("video_dubbing_original_during", "Оригинал во время (%)"), self.original_during_spin),
                (self.tr("video_dubbing_narration_volume", "Громкость перевода (%)"), self.narration_volume_spin),
                (self.tr("video_dubbing_attack", "Attack"), self.attack_spin),
                (self.tr("video_dubbing_release", "Release"), self.release_spin),
            ),
            self.tr("video_dubbing_tab_audio", "Звук"),
        )
        tabs.addTab(
            _form(
                (self.tr("video_dubbing_container", "Контейнер"), self.container_combo),
                (self.narration_only_check,),
                (self.embed_srt_check,),
            ),
            self.tr("video_dubbing_tab_export", "Экспорт"),
        )
        outer.addWidget(tabs)

        for picker in (self.video_picker, self.srt_picker, self.project_dir_picker):
            picker.path_changed.connect(self._schedule_autosave)
        for widget in (
            self.title_edit, self.language_edit, self.preferred_speed_spin,
            self.hard_speed_spin, self.guard_gap_spin, self.compress_pauses_check,
            self.pause_keep_spin, self.sync_combo, self.timing_mode_combo,
            self.elastic_max_cues_spin, self.elastic_max_speed_spin,
            self.elastic_min_gap_spin, self.elastic_max_shift_spin,
            self.elastic_prefer_small_check,
            self.tempo_mode_combo, self.tempo_neighbors_spin, self.tempo_jump_spin,
            self.tempo_delta_spin, self.tempo_optional_spin, self.autoplay_select_check,
            self.original_mode_combo, self.original_outside_spin,
            self.original_during_spin, self.narration_volume_spin, self.attack_spin,
            self.release_spin, self.container_combo, self.narration_only_check,
            self.embed_srt_check, self.tts_engine_combo, self.voice_combo,
        ):
            self._connect_change_signal(widget, self._schedule_autosave)

        self._settings_lock_widgets = [
            self.video_picker, self.srt_picker, self.project_dir_picker,
            self.title_edit, self.language_edit, self.tts_engine_combo,
            self.voice_combo, self.voice_refresh_button, self.voices_page_button,
            self.preferred_speed_spin, self.hard_speed_spin, self.compress_pauses_check,
            self.pause_keep_spin, self.timing_mode_combo,
        ]

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
        controls.setSpacing(4)
        self.play_button = QPushButton()
        self.play_button.setIcon(ui_icon("play"))
        self.play_button.setToolTip(self.tr("video_dubbing_play", "Воспроизведение"))
        self.pause_button = QPushButton()
        self.pause_button.setIcon(ui_icon("pause"))
        self.pause_button.setToolTip(self.tr("video_dubbing_pause", "Пауза"))
        self.stop_button = QPushButton()
        self.stop_button.setIcon(ui_icon("stop"))
        self.stop_button.setToolTip(self.tr("video_dubbing_stop", "Стоп"))
        self.prev_cue_button = QPushButton()
        self.prev_cue_button.setIcon(ui_icon("prev"))
        self.prev_cue_button.setToolTip(self.tr("video_dubbing_prev", "Предыдущая реплика"))
        self.next_cue_button = QPushButton()
        self.next_cue_button.setIcon(ui_icon("next"))
        self.next_cue_button.setToolTip(self.tr("video_dubbing_next", "Следующая реплика"))
        self.loop_button = QPushButton()
        self.loop_button.setIcon(ui_icon("loop"))
        self.loop_button.setToolTip(self.tr("video_dubbing_loop", "Зациклить реплику"))
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
            button.setFixedSize(36, 32)
            button.setIconSize(button.size() * 0.55)
            button.setFlat(False)
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
        outer = QVBoxLayout(wrapper)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
        self.pipeline_status_label = QLabel("")
        self.pipeline_status_label.setWordWrap(True)
        self.pipeline_status_label.setObjectName("helperLabel")
        outer.addWidget(self.pipeline_status_label)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        # name, label, handler, base_tooltip
        primary = (
            ("analyze", self.tr("video_dubbing_analyze", "Анализ"), self._analyze,
             "Проверить SRT, пустые тексты, пересечения. TTS не запускает."),
            ("generate", self.tr("video_dubbing_generate_missing", "Создать речь"),
             lambda: self._run_generation("missing"),
             "TTS только для реплик без raw / с устаревшим голосом или текстом.\n"
             "Не сдвигает таймкоды. После смены голоса/текста — обязательно."),
            ("selected", self.tr("video_dubbing_generate_selected", "Выбранные"),
             lambda: self._run_generation("selected"),
             "Принудительный TTS для выделенных строк таблицы."),
            ("refit", self.tr("video_dubbing_refit", "Пересчитать"), self._refit_existing,
             "Refit без нового TTS: скорость, паузы, эластичные группы, сдвиг planned.\n"
             "Нужен после смены тайминга/скорости/эластики. Исходный SRT не меняет."),
            ("preview", self.tr("video_dubbing_render_full_preview", "Предпросмотр"),
             self._render_full_preview,
             "Собрать полный preview (может сначала обновить mix)."),
        )
        secondary = (
            ("narration", self.tr("video_dubbing_render_narration", "Дорожка"),
             self._render_narration,
             "Собрать narration.wav по planned/start таймкодам fitted WAV.\n"
             "Нужна после TTS или Пересчитать. TTS не вызывает."),
            ("mix", self.tr("video_dubbing_render_mix", "Микс"), self._render_mix,
             "Собрать dubbed mix (оригинал + перевод + ducking).\n"
             "Нужен после Дорожки или смены настроек Звук."),
            ("export", self.tr("video_dubbing_export", "Видео"), self._export_video,
             "Финальный MP4/MKV. Нужен актуальный микс."),
            ("cancel", self.tr("video_dubbing_cancel", "Отмена"), self._cancel,
             "Остановить текущую фоновую операцию."),
            ("folder", self.tr("video_dubbing_open_folder", "Папка"), self._open_folder,
             "Открыть каталог render/ проекта."),
        )
        self._action_buttons = []
        self._named_action_buttons: dict[str, QPushButton] = {}
        self.cancel_button = None
        self._action_base_tips: dict[str, str] = {}
        for col, (name, label, handler, tip) in enumerate(primary):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.setMinimumHeight(30)
            button.clicked.connect(handler)
            grid.addWidget(button, 0, col)
            self._action_buttons.append(button)
            self._named_action_buttons[name] = button
            self._action_base_tips[name] = tip
        for col, (name, label, handler, tip) in enumerate(secondary):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.setMinimumHeight(30)
            button.clicked.connect(handler)
            grid.addWidget(button, 1, col)
            self._action_buttons.append(button)
            self._named_action_buttons[name] = button
            self._action_base_tips[name] = tip
            if name == "cancel":
                self.cancel_button = button
        more = QToolButton()
        more.setText(self.tr("video_dubbing_more", "Ещё"))
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(more)
        for label, handler in (
            (self.tr("video_dubbing_create", "Создать проект"), self._create_project),
            (self.tr("video_dubbing_open", "Открыть проект"), self._open_project),
            (self.tr("video_dubbing_open_recent", "Недавние"), self._open_recent),
            (self.tr("video_dubbing_save", "Сохранить"), self._save_project),
            (self.tr("video_dubbing_import_srt", "Импорт SRT"), self._import_srt),
            (self.tr("video_dubbing_force_all", "Пересоздать всё"), self._confirm_force_all),
            (self.tr("video_dubbing_render_cue_preview", "Превью реплики"), self._render_cue_preview),
            (self.tr("video_dubbing_export_adjusted_srt", "Экспорт adjusted SRT"), self._export_adjusted_srt),
            (self.tr("video_dubbing_plan_elastic", "Рассчитать эластичную группу"), self._plan_elastic),
            (self.tr("video_dubbing_open_review", "Обзор реплик (Whisper)"), self._open_review),
        ):
            menu.addAction(label, handler)
        more.setMenu(menu)
        grid.addWidget(more, 0, len(primary))
        self.more_button = more
        outer.addLayout(grid)
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
        """Inject engine from main window without clobbering page/project voice.

        Previously every visit to the dubbing page overwrote the selected gallery
        voice with the main TTS panel selection (often a male reference), so
        choosing Russian Woman here had no lasting effect.
        """
        if engine is not None:
            self._tts_engine = engine
            if self._service is not None:
                self._service.tts_engine = engine
        if ffmpeg_path:
            self._ffmpeg_path = ffmpeg_path

        # Prefer an already opened project's voice over the global main-window voice.
        project_voice = ""
        project_engine = ""
        project_cfg: dict[str, Any] | None = None
        if self._project is not None:
            project_voice = str(self._project.settings.voice or "")
            project_engine = str(self._project.settings.tts_engine or "")
            if self._project.settings.voice_config:
                project_cfg = dict(self._project.settings.voice_config)

        page_voice = self._selected_voice_id() if hasattr(self, "voice_combo") else ""
        keep_page_voice = bool(page_voice) and page_voice not in {"", "piper"}

        if voice_config is not None:
            engine_id = str(voice_config.get("engine", "") or "")
            if engine_id:
                index = self.tts_engine_combo.findData(engine_id)
                if index >= 0 and (
                    self._project is None
                    or not project_engine
                    or project_engine == engine_id
                ):
                    self._loading_ui = True
                    self.tts_engine_combo.setCurrentIndex(index)
                    self._loading_ui = False

            # Only adopt main-window voice when the page/project has none yet.
            if project_cfg and project_voice:
                self._voice_config_override = project_cfg
            elif keep_page_voice and self._voice_config_override:
                pass  # keep user's current dubbing-page selection
            else:
                self._voice_config_override = dict(voice_config)

        self._refresh_voices()
        # Restore authoritative voice after refresh.
        restore = project_voice or (
            self._selected_voice_id() if keep_page_voice else ""
        )
        if not restore and voice_config is not None:
            restore = str(
                voice_config.get("voice")
                or voice_config.get("speaker")
                or voice_config.get("reference_voice_name")
                or ""
            )
        if restore:
            self._loading_ui = True
            try:
                idx = self.voice_combo.findData(restore)
                if idx < 0:
                    bare = restore.split(" — ", 1)[0].strip()
                    idx = self.voice_combo.findData(bare)
                if idx >= 0:
                    self.voice_combo.setCurrentIndex(idx)
                else:
                    self.voice_combo.setCurrentIndex(-1)
                    self.voice_combo.setEditText(restore)
            finally:
                self._loading_ui = False
            # Re-resolve so reference_audio_path matches the restored voice.
            try:
                self._voice_config_override = self._current_voice_config()
            except Exception:
                pass

    def _sync_tts_engine(self) -> BaseTTSEngine:
        """Create/attach the engine that matches the page combo (not a stale Piper)."""
        engine_id = str(self.tts_engine_combo.currentData() or "piper")
        current_id = ""
        if self._tts_engine is not None:
            current_id = str(
                getattr(self._tts_engine, "engine_id", "")
                or type(self._tts_engine).__name__
            ).casefold()
        need_new = self._tts_engine is None
        if not need_new:
            # Map class names → ids when engine_id attr is absent.
            class_map = {
                "piperttsengine": "piper",
                "omnivoicettsengine": "omnivoice",
                "chatterboxttsengine": "chatterbox",
                "kokoropythonttsengine": "kokoro_python",
                "qwenttsengine": "qwen",
            }
            resolved = class_map.get(current_id, current_id)
            if engine_id.casefold() not in resolved and resolved not in engine_id.casefold():
                need_new = True
        if need_new:
            try:
                self._tts_engine = self.voice_catalog.create_engine(
                    engine_id, self._piper_path
                )
                self.log_view.append_event(
                    f"TTS engine ready: {engine_id} ({type(self._tts_engine).__name__})"
                )
            except Exception as exc:
                raise VideoDubbingServiceError(
                    f"Could not create TTS engine '{engine_id}': {exc}"
                ) from exc
        if self._service is not None:
            self._service.tts_engine = self._tts_engine
        return self._tts_engine  # type: ignore[return-value]

    def _ensure_service(self) -> VideoDubbingService:
        self._sync_tts_engine()
        if self._tts_engine is None:
            raise VideoDubbingServiceError(
                "No TTS engine available. Select an engine on the Voice tab."
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
            # Never overwrite worker signal bridges while a background job runs —
            # autosave used to point callbacks at UI methods and crash with AV.
            if self._worker is None:
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
        timing_mode = CueTimingMode(
            str(self.timing_mode_combo.currentData() or CueTimingMode.STRICT.value)
        )
        elastic = ElasticTimingSettings(
            enabled=timing_mode == CueTimingMode.ELASTIC_GROUP,
            max_cues_per_group=int(self.elastic_max_cues_spin.value()),
            max_common_speed_factor=float(self.elastic_max_speed_spin.value()),
            min_inter_cue_gap_ms=int(self.elastic_min_gap_spin.value()),
            max_shift_per_cue_ms=int(self.elastic_max_shift_spin.value()),
            prefer_smaller_group=self.elastic_prefer_small_check.isChecked(),
        )
        try:
            tempo_mode = TempoSmoothingMode(
                str(self.tempo_mode_combo.currentData() or TempoSmoothingMode.SMOOTH.value)
            )
        except ValueError:
            tempo_mode = TempoSmoothingMode.SMOOTH
        tempo = TempoSmoothingSettings(
            mode=tempo_mode,
            max_cues_per_group=int(self.tempo_neighbors_spin.value()),
            speed_jump_threshold=float(self.tempo_jump_spin.value()),
            max_neighbor_speed_delta=float(self.tempo_delta_spin.value()),
            max_optional_speedup=float(self.tempo_optional_spin.value()),
            max_speed_factor=float(self.preferred_speed_spin.value()),
        )
        return DubbingProjectSettings(
            language=self.language_edit.text().strip(),
            tts_engine=str(self.tts_engine_combo.currentData() or "piper"),
            voice=self._selected_voice_id(),
            voice_config=voice_config,
            preferred_speed_limit=float(self.preferred_speed_spin.value()),
            hard_speed_limit=float(self.hard_speed_spin.value()),
            guard_gap_ms=int(self.guard_gap_spin.value()),
            compress_internal_pauses=self.compress_pauses_check.isChecked(),
            internal_pause_keep_ms=int(self.pause_keep_spin.value()),
            sync_mode=SyncMode(self.sync_combo.currentData()),
            cue_timing_mode=timing_mode,
            elastic_timing=elastic,
            tempo_smoothing=tempo,
            ffmpeg_path=self._ffmpeg_path,
            ducking=ducking,
            preview=PreviewSettings(
                alignment=Alignment.START,
                autoplay_on_select=self.autoplay_select_check.isChecked(),
            ),
            export=export,
            cue_error_policy="continue",
        )

    def _selected_voice_id(self) -> str:
        """Return gallery voice id, never the localized display label."""
        data = self.voice_combo.currentData()
        if data is not None and str(data).strip():
            return str(data).strip()
        text = self.voice_combo.currentText().strip()
        # Display labels look like "Russian Woman — ru".
        if " — " in text:
            text = text.split(" — ", 1)[0].strip()
        return text

    def _current_voice_config(self) -> dict[str, Any]:
        engine_id = str(self.tts_engine_combo.currentData() or "piper")
        voice_id = self._selected_voice_id()
        if self._voice_config_override and self._voice_config_override.get("engine") == engine_id:
            config = dict(self._voice_config_override)
        else:
            config = {"engine": engine_id}
        # Changing the gallery voice must never keep a previous reference clip.
        if engine_id in {"omnivoice", "chatterbox"}:
            for key in (
                "reference_audio_path",
                "reference_text",
                "reference_voice_name",
                "reference_voice_content_hash",
                "ref_audio",
                "ref_text",
                "voice",
            ):
                config.pop(key, None)
        try:
            config = self.voice_catalog.resolve_voice_config(engine_id, voice_id, config)
        except Exception:
            config.setdefault("voice", voice_id)
        config["voice"] = voice_id
        config["engine"] = engine_id
        # Keep override in sync so subsequent saves use the newly selected voice.
        self._voice_config_override = dict(config)
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
            self.pause_keep_spin.setValue(int(s.internal_pause_keep_ms or 90))
            self.sync_combo.setCurrentIndex(
                0 if s.sync_mode == SyncMode.STRICT else 1
            )
            tidx = self.timing_mode_combo.findData(s.cue_timing_mode.value)
            if tidx >= 0:
                self.timing_mode_combo.setCurrentIndex(tidx)
            elastic = s.elastic_timing
            self.elastic_max_cues_spin.setValue(int(elastic.max_cues_per_group))
            self.elastic_max_speed_spin.setValue(float(elastic.max_common_speed_factor))
            self.elastic_min_gap_spin.setValue(int(elastic.min_inter_cue_gap_ms))
            self.elastic_max_shift_spin.setValue(int(elastic.max_shift_per_cue_ms))
            self.elastic_prefer_small_check.setChecked(bool(elastic.prefer_smaller_group))
            tm = self.tempo_mode_combo.findData(s.tempo_smoothing.mode.value)
            if tm >= 0:
                self.tempo_mode_combo.setCurrentIndex(tm)
            self.tempo_neighbors_spin.setValue(int(s.tempo_smoothing.max_cues_per_group))
            self.tempo_jump_spin.setValue(float(s.tempo_smoothing.speed_jump_threshold))
            self.tempo_delta_spin.setValue(float(s.tempo_smoothing.max_neighbor_speed_delta))
            self.tempo_optional_spin.setValue(float(s.tempo_smoothing.max_optional_speedup))
            self.autoplay_select_check.setChecked(bool(s.preview.autoplay_on_select))
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
            voice_name = self._project.settings.voice
            index = self.voice_combo.findData(voice_name)
            if index < 0:
                index = self.voice_combo.findText(voice_name)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
            else:
                self.voice_combo.setCurrentIndex(-1)
                self.voice_combo.setEditText(voice_name)
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
            # During generation do not overwrite the immutable GenerationContext
            # snapshot with live UI voice/engine edits.
            if not self._generation_busy:
                self._apply_ui_to_project()
            else:
                # Persist only playback/selection metadata.
                self._project.selected_sequence = self._selected_sequence
                self._project.last_player_position_ms = int(self.media_player.position())
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
        # Force a fresh resolve of the currently selected gallery voice so a
        # stale main-window override cannot keep a previous male reference.
        try:
            self._sync_tts_engine()
        except VideoDubbingServiceError as exc:
            self._show_error(exc)
            return
        self._apply_voice_selection(invalidate=False)
        self._apply_ui_to_project()
        cfg = self._project.settings.voice_config or {}
        self.log_view.append_event(
            "TTS start: "
            f"engine={self._project.settings.tts_engine} "
            f"runtime={type(self._tts_engine).__name__ if self._tts_engine else '—'} "
            f"voice={self._project.settings.voice} "
            f"ref={Path(str(cfg.get('reference_audio_path') or '')).name or '—'} "
            f"lang={self._project.settings.language or '—'}"
        )
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
            self._report_generation_result(result)

        self._start_worker(op, on_done, generation=True)

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
        # Same prep as normal generation: sync engine + resolve voice + clear cancel.
        try:
            self._sync_tts_engine()
        except VideoDubbingServiceError as exc:
            self._show_error(exc)
            return
        self._apply_voice_selection(invalidate=False)
        self._apply_ui_to_project()
        service = self._ensure_service()
        service.reset_cancel()
        cfg = self._project.settings.voice_config or {}
        self.log_view.append_event(
            "Force TTS: "
            f"engine={self._project.settings.tts_engine} "
            f"runtime={type(self._tts_engine).__name__ if self._tts_engine else '—'} "
            f"voice={self._project.settings.voice} "
            f"ref={Path(str(cfg.get('reference_audio_path') or '')).name or '—'} "
            f"cues={sum(1 for c in self._project.cues if c.enabled)}"
        )

        def op(svc):
            svc.reset_cancel()
            return svc.generate_all(self._project, force=True)

        def on_done(result):
            self._refresh_ui_from_project()
            self._report_generation_result(result)

        self._start_worker(op, on_done, generation=True)

    def _refit_existing(self) -> None:
        if self._project is None:
            return
        self._apply_ui_to_project()

        def op(svc):
            return svc.re_fit_existing(self._project)

        def on_done(result):
            self._refresh_ui_from_project()
            self._report_generation_result(result)

        self._start_worker(op, on_done, generation=True)

    def _export_adjusted_srt(self) -> None:
        if self._project is None:
            return
        try:
            path = self._ensure_service().export_adjusted_srt(self._project)
            self.log_view.append_event(f"Adjusted SRT: {path}")
        except Exception as exc:
            self._show_error_message(str(exc))

    def _plan_elastic(self) -> None:
        if self._project is None:
            return
        self._apply_ui_to_project()
        service = self._ensure_service()
        plans = service.plan_elastic_groups(self._project)
        if not plans:
            self._show_info(self.tr("video_dubbing_no_elastic", "Эластичные группы не требуются."))
            return
        for plan in plans:
            service.apply_elastic_plan(self._project, plan)
            self.log_view.append_event(
                f"{plan.group_id}: cue {plan.sequences[0]}–{plan.sequences[-1]}, "
                f"коэф. {plan.common_speed_factor:.3f}, сдвиг макс. {plan.max_shift_ms} ms"
            )
        self._refresh_ui_from_project()

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

    def _open_review(self) -> None:
        if self._project is None:
            self._show_info(
                self.tr("video_dubbing_no_project", "Сначала создайте или откройте проект.")
            )
            return
        self.openReviewRequested.emit(self._project)

    def _pipeline_needs(self) -> dict[str, bool]:
        """What pipeline steps are still required for a coherent export."""
        needs = {
            "generate": False,
            "refit": False,
            "narration": False,
            "mix": False,
            "export": False,
            "preview": False,
        }
        if self._project is None:
            return needs
        project = self._project
        missing_raw = 0
        stale_cues = 0
        for cue in project.cues:
            if not cue.enabled:
                continue
            raw_ok = (
                cue.raw_audio_path is not None
                and Path(cue.raw_audio_path).is_file()
                and (cue.raw_duration_ms or 0) > 0
            )
            if not raw_ok or cue.is_stale or cue.status in {
                CueStatus.PENDING.value,
                CueStatus.STALE.value,
                CueStatus.FAILED.value,
                CueStatus.MISSING_AUDIO.value,
            }:
                missing_raw += 1
            fitted_ok = (
                cue.fitted_audio_path is not None
                and Path(cue.fitted_audio_path).is_file()
                and (cue.fitted_duration_ms or 0) > 0
            )
            if raw_ok and not fitted_ok:
                stale_cues += 1
        needs["generate"] = missing_raw > 0
        needs["refit"] = (not needs["generate"]) and (
            stale_cues > 0 or bool(project.stale.cues)
        )
        needs["narration"] = bool(project.stale.narration) or not (
            project.narration_wav and Path(project.narration_wav).is_file()
        )
        needs["mix"] = bool(project.stale.mix) or not (
            project.dubbed_mix_wav and Path(project.dubbed_mix_wav).is_file()
        )
        needs["preview"] = bool(project.stale.preview)
        needs["export"] = bool(project.stale.video) or not (
            project.final_video_path and Path(project.final_video_path).is_file()
        )
        # If earlier steps needed, later ones are also needed.
        if needs["generate"] or needs["refit"]:
            needs["narration"] = True
            needs["mix"] = True
            needs["export"] = True
            needs["preview"] = True
        elif needs["narration"]:
            needs["mix"] = True
            needs["export"] = True
            needs["preview"] = True
        elif needs["mix"]:
            needs["export"] = True
            needs["preview"] = True
        return needs

    def _update_pipeline_indicators(self) -> None:
        if not hasattr(self, "_named_action_buttons"):
            return
        needs = self._pipeline_needs()
        warn_style = (
            "QPushButton { border: 2px solid #ea580c; font-weight: 600; }"
        )
        ok_style = ""
        mapping = {
            "generate": needs["generate"],
            "refit": needs["refit"] and not needs["generate"],
            "narration": needs["narration"] and not needs["generate"],
            "mix": needs["mix"] and not needs["generate"],
            "export": needs["export"] and not needs["generate"],
            "preview": needs["preview"] and not needs["generate"],
        }
        warn_suffix = {
            "generate": " ⚠ TTS",
            "refit": " ⚠ fit",
            "narration": " ⚠",
            "mix": " ⚠",
            "export": " ⚠",
            "preview": " ⚠",
        }
        base_labels = {
            "generate": self.tr("video_dubbing_generate_missing", "Создать речь"),
            "refit": self.tr("video_dubbing_refit", "Пересчитать"),
            "narration": self.tr("video_dubbing_render_narration", "Дорожка"),
            "mix": self.tr("video_dubbing_render_mix", "Микс"),
            "export": self.tr("video_dubbing_export", "Видео"),
            "preview": self.tr("video_dubbing_render_full_preview", "Предпросмотр"),
        }
        for name, needed in mapping.items():
            button = self._named_action_buttons.get(name)
            if button is None:
                continue
            base = base_labels.get(name, button.text().replace(" ⚠ TTS", "").replace(" ⚠ fit", "").replace(" ⚠", ""))
            tip = self._action_base_tips.get(name, "")
            if needed:
                button.setText(base + warn_suffix.get(name, " ⚠"))
                button.setStyleSheet(warn_style)
                button.setToolTip(
                    tip
                    + "\n\n⚠ Требуется выполнить этот шаг — данные устарели или отсутствуют."
                )
            else:
                button.setText(base)
                button.setStyleSheet(ok_style)
                button.setToolTip(tip)

        steps: list[str] = []
        if needs["generate"]:
            steps.append("1) Создать речь (TTS)")
        if needs["refit"] and not needs["generate"]:
            steps.append("2) Пересчитать (fit/эластика, без TTS)")
        if needs["narration"]:
            steps.append("3) Дорожка")
        if needs["mix"]:
            steps.append("4) Микс")
        if needs["export"]:
            steps.append("5) Видео")
        if not steps:
            self.pipeline_status_label.setText(
                "✓ Конвейер актуален. Можно слушать Перевод/Микс или экспортировать."
            )
            self.pipeline_status_label.setStyleSheet("color: #15803d;")
        else:
            self.pipeline_status_label.setText(
                "Дальше: " + " → ".join(steps)
                + "  |  Сдвиг/эластика = Пересчитать, не TTS."
            )
            self.pipeline_status_label.setStyleSheet("color: #c2410c; font-weight: 600;")

    # ------------------------------------------------------------------ worker

    def _start_worker(
        self,
        operation: Callable[[VideoDubbingService], Any],
        on_finished: Callable[[Any], None],
        *,
        generation: bool = False,
    ) -> None:
        if self._worker_thread is not None:
            if self._worker_thread.isRunning():
                self.log_view.append_event(
                    self.tr(
                        "video_dubbing_busy",
                        "Уже выполняется операция. Дождитесь окончания или нажмите Отмена.",
                    )
                )
                return
            # Stale finished thread handle — drop it so a new run can start.
            self._clear_worker()
        service = self._ensure_service()
        service.reset_cancel()
        thread = QThread(self)
        worker = VideoDubbingWorker(service, operation)
        worker.moveToThread(thread)
        # Always queue worker → UI so slots never run on the QThread.
        queued = Qt.ConnectionType.QueuedConnection
        self._pending_on_finished = on_finished
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress, queued)
        worker.cue_updated.connect(self._on_cue_updated, queued)
        worker.log.connect(self.log_view.append_event, queued)
        worker.finished.connect(self._on_worker_finished, queued)
        worker.failed.connect(self._on_worker_failed, queued)
        worker.cancelled.connect(self._on_worker_cancelled, queued)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._worker_thread = thread
        self._worker = worker
        self._set_busy(True, generation=generation)
        thread.start()

    def _clear_worker(self) -> None:
        self._worker_thread = None
        self._worker = None
        # Restore direct UI callbacks only after the worker is gone.
        if self._service is not None:
            self._service.log_callback = self.log_view.append_event
            self._service.progress_callback = self._on_progress
            self._service.cue_updated_callback = self._on_cue_updated
        self._set_busy(False)

    def _on_worker_finished(self, result: Any) -> None:
        self._set_busy(False)
        callback = getattr(self, "_pending_on_finished", None)
        self._pending_on_finished = None
        if callback is not None:
            callback(result)

    def _on_worker_failed(self, message: str) -> None:
        self._set_busy(False)
        self._pending_on_finished = None
        self.log_view.append_event(message)
        self._show_error_message(
            message,
            title=self.tr("video_dubbing_stopped_error", "Генерация остановлена из-за ошибки"),
        )

    def _on_worker_cancelled(self) -> None:
        self._set_busy(False)
        self._pending_on_finished = None
        msg = self.tr("video_dubbing_cancelled", "Генерация отменена")
        self.log_view.append_event(msg)
        self.progress_status.setText(msg)
        self._refresh_ui_from_project()

    def _report_generation_result(self, result: Any) -> None:
        if not isinstance(result, GenerationRunResult):
            return
        counts = result.summary_counts()
        detail = (
            f"обработано {counts['total']}, успешно {counts['completed']}, "
            f"ошибок {counts['failed']}, пропущено {counts['skipped']}, "
            f"отменено {counts['cancelled']}"
        )
        if result.status == GenerationRunStatus.COMPLETED:
            title = self.tr("video_dubbing_gen_completed", "Генерация завершена")
        elif result.status == GenerationRunStatus.COMPLETED_WITH_ERRORS:
            title = self.tr(
                "video_dubbing_gen_completed_errors",
                "Генерация завершена с ошибками",
            )
        elif result.status == GenerationRunStatus.CANCELLED:
            title = self.tr("video_dubbing_cancelled", "Генерация отменена")
        else:
            title = self.tr(
                "video_dubbing_stopped_error",
                "Генерация остановлена из-за ошибки",
            )
        self.log_view.append_event(f"{title}: {detail}")
        self.progress_status.setText(f"{title} — {detail}")
        # Defer modal dialogs to the next event-loop tick so the worker thread
        # can fully unwind before a blocking QMessageBox appears.
        if result.status == GenerationRunStatus.COMPLETED_WITH_ERRORS:
            QTimer.singleShot(0, lambda: self._show_info(f"{title}\n{detail}"))
        elif result.status == GenerationRunStatus.FAILED and result.error_message:
            message = f"{title}\n{result.error_message}\n{detail}"
            QTimer.singleShot(0, lambda m=message: self._show_error_message(m))

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

    def _global_position_ms(self) -> int:
        return int(self._preview_global_offset_ms + max(0, self.media_player.position()))

    def _cue_start_ms(self, cue: DubbingCue) -> int:
        return int(cue.effective_start_ms())

    def _cue_end_ms(self, cue: DubbingCue) -> int:
        end = cue.effective_end_ms()
        if cue.fitted_duration_ms:
            end = max(end, self._cue_start_ms(cue) + int(cue.fitted_duration_ms))
        return int(end)

    def _media_path_for_mode(self, mode: str) -> Path | None:
        if self._project is None:
            return None
        if mode == LISTENING_ORIGINAL and self._project.video_path:
            path = Path(self._project.video_path)
            return path if path.is_file() else None
        if mode == LISTENING_MIX:
            for candidate in (
                self._project.full_preview_path,
                self._project.final_video_path,
                self._project.dubbed_mix_wav,
            ):
                if candidate and Path(candidate).is_file():
                    return Path(candidate)
            return None
        if mode == LISTENING_TRANSLATION:
            # Prefer the selected cue's own WAV so voice gender is obvious.
            if self._selected_sequence is not None:
                cue = next(
                    (c for c in self._project.cues if c.sequence == self._selected_sequence),
                    None,
                )
                if cue is not None:
                    for candidate in (cue.raw_audio_path, cue.fitted_audio_path):
                        if candidate and Path(candidate).is_file():
                            return Path(candidate)
            if self._project.narration_wav and Path(self._project.narration_wav).is_file():
                return Path(self._project.narration_wav)
        return None

    def _ensure_media_for_mode(self, *, play: bool = False) -> bool:
        path = self._media_path_for_mode(self._listening_mode)
        if path is None:
            if self._listening_mode == LISTENING_TRANSLATION:
                self.log_view.append_event(
                    self.tr(
                        "video_dubbing_no_translation_media",
                        "Нет дорожки перевода. Сначала «Создать речь» / «Дорожка».",
                    )
                )
            elif self._listening_mode == LISTENING_MIX:
                self.log_view.append_event(
                    self.tr(
                        "video_dubbing_no_mix_media",
                        "Нет микса/preview. Сначала «Микс» или «Предпросмотр».",
                    )
                )
            return False
        self._preview_global_offset_ms = 0
        self._load_media(path, seek_ms=self._global_position_ms(), play=play)
        return True

    def _play(self) -> None:
        if self._project is None:
            return
        if not self._ensure_media_for_mode(play=True):
            # Still try to play whatever is loaded.
            self.media_player.play()

    def _load_media(
        self,
        path: Path,
        *,
        seek_ms: int | None = None,
        play: bool = False,
        global_offset_ms: int = 0,
    ) -> None:
        if not path or not path.is_file():
            return
        self._preview_global_offset_ms = max(0, int(global_offset_ms))
        current = ""
        try:
            current = self.media_player.source().toLocalFile()
        except Exception:
            current = ""
        need_reload = Path(current) != path if current else True
        if seek_ms is not None:
            self._pending_seek_ms = max(0, int(seek_ms) - self._preview_global_offset_ms)
        self._pending_play = play
        if need_reload:
            self.media_player.setSource(QUrl.fromLocalFile(str(path)))
        else:
            self._apply_pending_seek_and_play()

    def _apply_pending_seek_and_play(self) -> None:
        if self._pending_seek_ms is not None:
            self.media_player.setPosition(int(self._pending_seek_ms))
            self._pending_seek_ms = None
        if self._pending_play:
            self._pending_play = False
            self.media_player.play()

    def _seek_slider(self, position: int) -> None:
        self.media_player.setPosition(position)

    def _on_position_changed(self, position: int) -> None:
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)
        global_ms = self._global_position_ms()
        self.timeline_widget.set_position_ms(int(global_ms))
        self._update_time_label(global_ms, self._display_duration_ms())
        if self._cue_preview_end_ms is not None and position >= self._cue_preview_end_ms:
            if self._loop_cue and self._selected_sequence is not None and self._project:
                cue = next(
                    (c for c in self._project.cues if c.sequence == self._selected_sequence),
                    None,
                )
                if cue is not None:
                    self._play_cue_preview(cue.sequence, loop_restart=True)
                    return
            self.media_player.pause()
            self._cue_preview_end_ms = None
            return
        if (
            self._loop_cue
            and self._selected_sequence is not None
            and self._project
            and self._cue_preview_end_ms is None
        ):
            cue = next(
                (c for c in self._project.cues if c.sequence == self._selected_sequence),
                None,
            )
            if cue is not None:
                end = self._cue_end_ms(cue) + self._project.settings.preview.post_roll_ms
                if global_ms > end:
                    self._jump_to_cue(cue.sequence, play=True)

    def _on_duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, max(0, duration))
        self._apply_pending_seek_and_play()
        self._update_time_label(self._global_position_ms(), self._display_duration_ms())

    def _display_duration_ms(self) -> int:
        if self._project and self._project.duration_ms:
            return int(self._project.duration_ms)
        return int(self.media_player.duration() + self._preview_global_offset_ms)

    def _update_time_label(self, position: int, duration: int) -> None:
        self.time_label.setText(f"{_format_ms(position)} / {_format_ms(duration)}")

    def _set_listening_mode(self, mode: str) -> None:
        if mode == self._listening_mode and self.media_player.source().isValid():
            self.original_button.setChecked(mode == LISTENING_ORIGINAL)
            self.translation_button.setChecked(mode == LISTENING_TRANSLATION)
            self.mix_button.setChecked(mode == LISTENING_MIX)
            return
        keep_ms = self._global_position_ms()
        was_playing = (
            self.media_player.playbackState()
            == self.media_player.PlaybackState.PlayingState
        )
        self._listening_mode = mode
        self.original_button.setChecked(mode == LISTENING_ORIGINAL)
        self.translation_button.setChecked(mode == LISTENING_TRANSLATION)
        self.mix_button.setChecked(mode == LISTENING_MIX)
        if self._project is None:
            return
        path = self._media_path_for_mode(mode)
        if path is None:
            self.log_view.append_event(
                self.tr(
                    "video_dubbing_mode_media_missing",
                    "Для выбранного режима нет готового медиа.",
                )
            )
            return
        self._cue_preview_end_ms = None
        self._load_media(path, seek_ms=keep_ms, play=was_playing, global_offset_ms=0)

    def _on_loop_toggled(self, checked: bool) -> None:
        self._loop_cue = checked

    def _ordered_cues(self) -> list:
        if not self._project:
            return []
        return sorted(
            self._project.cues,
            key=lambda c: (self._cue_start_ms(c), c.sequence),
        )

    def _goto_prev_cue(self) -> None:
        ordered = self._ordered_cues()
        if not ordered:
            return
        # Prefer navigation by selected sequence (stable in Translation mode
        # where the player position is local to a short cue WAV).
        if self._selected_sequence is not None:
            for index, cue in enumerate(ordered):
                if cue.sequence == self._selected_sequence:
                    if index > 0:
                        self._select_and_jump(ordered[index - 1].sequence, play=False)
                    return
        current_ms = self._global_position_ms()
        prev = None
        for cue in ordered:
            if self._cue_start_ms(cue) < current_ms - 80:
                prev = cue
            else:
                break
        if prev is not None:
            self._select_and_jump(prev.sequence, play=False)

    def _goto_next_cue(self) -> None:
        ordered = self._ordered_cues()
        if not ordered:
            return
        if self._selected_sequence is not None:
            for index, cue in enumerate(ordered):
                if cue.sequence == self._selected_sequence:
                    if index + 1 < len(ordered):
                        self._select_and_jump(ordered[index + 1].sequence, play=False)
                    return
        current_ms = self._global_position_ms()
        for cue in ordered:
            if self._cue_start_ms(cue) > current_ms + 80:
                self._select_and_jump(cue.sequence, play=False)
                return
        # Already past/on last by time: stay on last.
        self._select_and_jump(ordered[-1].sequence, play=False)

    def _on_timeline_cue_selected(self, sequence: int) -> None:
        autoplay = bool(
            self._project
            and self._project.settings.preview.autoplay_on_select
        )
        self._select_and_jump(sequence, play=autoplay)

    def _on_timeline_cue_activated(self, sequence: int) -> None:
        self._select_and_jump(sequence, play=True)

    def _on_timeline_seek(self, position_ms: int) -> None:
        # Seek only (empty area click) without changing selection.
        self._cue_preview_end_ms = None
        if not self._ensure_media_for_mode(play=False):
            self.media_player.setPosition(max(0, int(position_ms)))
            return
        self.media_player.setPosition(max(0, int(position_ms)))

    def _select_and_jump(self, sequence: int, *, play: bool) -> None:
        if self._project is None:
            return
        self._selected_sequence = sequence
        self.timeline_widget.set_selected(sequence)
        # Sync table selection without re-entry loops.
        self.cue_table.blockSignals(True)
        for row in range(self.cue_table.rowCount()):
            item = self.cue_table.item(row, 0)
            if item is not None and item.text() == str(sequence):
                self.cue_table.selectRow(row)
                self.cue_table.scrollToItem(item)
                break
        self.cue_table.blockSignals(False)
        if play:
            self._play_cue_preview(sequence)
        else:
            self._jump_to_cue(sequence, play=False)

    def select_cues_by_sequence(self, sequences) -> None:
        """Select every cue row whose sequence is in ``sequences``.

        Used by Generation Review to highlight cues that failed transcript
        validation so the user can regenerate them. Clears any current
        selection, selects all matching rows, and scrolls to the first one.
        Safe to call before a project is loaded (no-op).
        """
        targets = {str(int(seq)) for seq in sequences}
        if not targets:
            return
        self.cue_table.blockSignals(True)
        self.cue_table.clearSelection()
        first_item = None
        for row in range(self.cue_table.rowCount()):
            item = self.cue_table.item(row, 0)
            if item is not None and item.text() in targets:
                self.cue_table.selectRow(row)
                if first_item is None:
                    first_item = item
        if first_item is not None:
            self.cue_table.scrollToItem(first_item)
        self.cue_table.blockSignals(False)

    def _jump_to_cue(self, sequence: int, *, play: bool = False) -> None:
        if self._project is None:
            return
        cue = next((c for c in self._project.cues if c.sequence == sequence), None)
        if cue is None:
            return
        pre_roll = int(self._project.settings.preview.pre_roll_ms)
        seek = max(0, self._cue_start_ms(cue) - pre_roll)
        self._cue_preview_end_ms = None
        path = self._media_path_for_mode(self._listening_mode)
        if path is None and self._listening_mode == LISTENING_TRANSLATION:
            # Per-cue audio fallback: local file starts at 0.
            for candidate in (cue.fitted_audio_path, cue.raw_audio_path):
                if candidate and Path(candidate).is_file():
                    self._load_media(
                        Path(candidate),
                        seek_ms=0,
                        play=play,
                        global_offset_ms=self._cue_start_ms(cue),
                    )
                    if play:
                        post = int(self._project.settings.preview.post_roll_ms)
                        # Local duration approx.
                        dur = int(cue.fitted_duration_ms or cue.raw_duration_ms or 0)
                        self._cue_preview_end_ms = max(1, dur + post)
                    return
            self.log_view.append_event(
                self.tr(
                    "video_dubbing_no_cue_audio",
                    f"У реплики #{sequence} ещё нет WAV перевода.",
                )
            )
            return
        if path is None:
            self.log_view.append_event(
                self.tr(
                    "video_dubbing_mode_media_missing",
                    "Нет медиа для текущего режима прослушивания.",
                )
            )
            return
        self._load_media(path, seek_ms=seek, play=play, global_offset_ms=0)

    def _play_cue_preview(self, sequence: int, *, loop_restart: bool = False) -> None:
        if self._project is None:
            return
        cue = next((c for c in self._project.cues if c.sequence == sequence), None)
        if cue is None:
            return
        pre = int(self._project.settings.preview.pre_roll_ms)
        post = int(self._project.settings.preview.post_roll_ms)
        start = max(0, self._cue_start_ms(cue) - pre)
        end = self._cue_end_ms(cue) + post
        path = self._media_path_for_mode(self._listening_mode)
        if path is None and self._listening_mode == LISTENING_TRANSLATION:
            self._jump_to_cue(sequence, play=True)
            return
        if path is None:
            self._jump_to_cue(sequence, play=False)
            return
        # Absolute timeline media.
        local_end = end
        self._cue_preview_end_ms = local_end
        self._load_media(path, seek_ms=start, play=True, global_offset_ms=0)

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
        # Drop stale speaker clip from previous engine.
        self._voice_config_override = {"engine": engine_id}
        self._refresh_voices()
        self._schedule_autosave()

    def _on_voice_edit_text(self, _text: str) -> None:
        if self._loading_ui:
            return
        # Debounced via autosave; full resolve happens on index change / generate.
        self._schedule_autosave()

    def _on_voice_changed(self, _index: int) -> None:
        if self._loading_ui:
            return
        self._apply_voice_selection(invalidate=True)

    def _update_voice_status_label(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or self._voice_config_override or {}
        name = str(
            cfg.get("reference_voice_name")
            or cfg.get("voice")
            or self._selected_voice_id()
            or "—"
        )
        ref = str(cfg.get("reference_audio_path") or "")
        ref_path = Path(ref) if ref else None
        exists = bool(ref_path and ref_path.is_file())
        folder = ref_path.parent.name if ref_path else "—"
        hash8 = str(cfg.get("reference_voice_content_hash") or "")[:8]
        mark = "✓" if exists else "✗"
        self.voice_status_label.setText(
            f"{mark} Активный голос: {name} | папка: {folder}"
            + (f" | hash {hash8}…" if hash8 else "")
            + ("" if exists else " — файл reference не найден!")
        )

    def _preview_reference_voice(self) -> None:
        try:
            cfg = self._current_voice_config()
        except Exception as exc:
            self._show_error_message(str(exc))
            return
        self._update_voice_status_label(cfg)
        ref = str(cfg.get("reference_audio_path") or "")
        path = Path(ref) if ref else None
        if path is None or not path.is_file():
            self._show_info(
                self.tr(
                    "video_dubbing_ref_missing",
                    "Reference-файл голоса не найден. Выберите голос заново.",
                )
            )
            return
        self._cue_preview_end_ms = None
        self._load_media(path, seek_ms=0, play=True, global_offset_ms=0)
        self.log_view.append_event(f"Preview reference: {path}")

    def _apply_voice_selection(self, *, invalidate: bool = False) -> None:
        """Resolve the combo selection into a fresh voice_config and project settings."""
        try:
            config = self._current_voice_config()
        except Exception as exc:
            self.log_view.append_event(f"Voice resolve failed: {exc}")
            return
        self._voice_config_override = dict(config)
        voice_name = str(
            config.get("reference_voice_name")
            or config.get("voice")
            or self._selected_voice_id()
        )
        ref = str(config.get("reference_audio_path") or "")
        ref_hash = str(config.get("reference_voice_content_hash") or "")[:12]
        self.log_view.append_event(
            f"Голос: {voice_name}"
            + (f" | ref={Path(ref).name}" if ref else "")
            + (f" | dir={Path(ref).parent.name}" if ref else "")
            + (f" | hash={ref_hash}…" if ref_hash else "")
        )
        self._update_voice_status_label(config)
        if self._project is not None:
            previous_hash = str(
                (self._project.settings.voice_config or {}).get(
                    "reference_voice_content_hash", ""
                )
                or ""
            )
            previous_voice = str(self._project.settings.voice or "")
            self._project.settings.voice = voice_name
            self._project.settings.tts_engine = str(
                config.get("engine") or self._project.settings.tts_engine
            )
            self._project.settings.voice_config = dict(config)
            lang = str(config.get("language") or "").strip()
            if lang and lang.lower() not in {"auto", "default"}:
                self._project.settings.language = lang
                self.language_edit.setText(lang)
            changed = (
                invalidate
                or previous_voice != voice_name
                or (ref_hash and previous_hash and not previous_hash.startswith(ref_hash)
                    and previous_hash != str(config.get("reference_voice_content_hash") or ""))
                or (ref_hash and previous_hash != str(config.get("reference_voice_content_hash") or ""))
            )
            if changed and self._project.cues:
                for cue in self._project.cues:
                    if not cue.enabled:
                        continue
                    cue.mark_stale()
                    cue.generation_fingerprint = ""
                    cue.fit_fingerprint = ""
                    cue.raw_wav_hash = ""
                try:
                    from app.core.video_dubbing.stale_state import invalidate_for_text_change

                    invalidate_for_text_change(self._project.stale)
                except Exception:
                    self._project.stale.cues = True
                    self._project.stale.narration = True
                    self._project.stale.mix = True
                    self._project.stale.preview = True
                    self._project.stale.video = True
                self.log_view.append_event(
                    "Голос изменён — существующие raw/fit помечены как устаревшие. "
                    "Нужна перегенерация речи."
                )
                self._refresh_ui_from_project()
        self._schedule_autosave()

    def _refresh_voices(self) -> None:
        engine_id = str(self.tts_engine_combo.currentData() or "piper")
        previous_id = ""
        if self.voice_combo.currentData() is not None:
            previous_id = str(self.voice_combo.currentData() or "").strip()
        if not previous_id:
            previous_id = self._selected_voice_id()
        self._loading_ui = True
        self.voice_combo.clear()
        try:
            self._voices = self.voice_catalog.list_voices(engine_id)
        except Exception as exc:
            self._voices = []
            self.log_view.append_event(f"Voice list failed: {exc}")
        for voice in self._voices:
            self.voice_combo.addItem(voice.display_name, voice.voice_id)
        if previous_id:
            index = self.voice_combo.findData(previous_id)
            if index < 0:
                bare = previous_id.split(" — ", 1)[0].strip()
                index = self.voice_combo.findData(bare)
            if index < 0:
                # Also match against display labels.
                index = self.voice_combo.findText(previous_id)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
            else:
                # Editable free-text: clear item data so currentData() is not stale.
                self.voice_combo.setCurrentIndex(-1)
                self.voice_combo.setEditText(previous_id)
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
        self._update_pipeline_indicators()

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
        plan = "—"
        if cue.planned_start_ms is not None and cue.planned_end_ms is not None:
            plan = f"{_format_ms(cue.planned_start_ms)}–{_format_ms(cue.planned_end_ms)}"
        speed = (
            cue.planned_speed_factor
            or cue.common_speed_factor
            or cue.applied_speed_factor
            or 1.0
        )
        speed_tip = (
            f"Требуется: {cue.required_speed_factor:.2f}"
            if cue.required_speed_factor
            else "Требуется: —"
        )
        speed_tip += (
            f"\nЗапланировано: {cue.planned_speed_factor:.2f}"
            if cue.planned_speed_factor
            else "\nЗапланировано: —"
        )
        speed_tip += f"\nПрименено: {cue.applied_speed_factor:.2f}"
        if cue.smoothing_group_id:
            speed_tip += f"\nГруппа: {cue.smoothing_group_id}"
        if cue.smoothing_reason:
            speed_tip += f"\nПричина: {cue.smoothing_reason}"
        group_label = cue.smoothing_group_id or cue.timing_group_id or "—"
        values = (
            str(cue.sequence),
            _format_ms(cue.start_ms),
            _format_ms(cue.end_ms),
            plan,
            cue.spoken_text,
            f"{cue.raw_duration_ms} ms" if cue.raw_duration_ms else "—",
            f"{speed:.2f}",
            f"{cue.start_shift_ms} ms" if cue.start_shift_ms else "—",
            group_label,
            f"{cue.fitted_duration_ms} ms" if cue.fitted_duration_ms else "—",
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
                item = existing
            if col == 6:
                item.setToolTip(speed_tip)
        status_item = self.cue_table.item(row, 10)
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
        if sequence == self._selected_sequence:
            return
        autoplay = bool(
            self._project and self._project.settings.preview.autoplay_on_select
        )
        self._select_and_jump(sequence, play=autoplay)

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

    def _set_busy(self, busy: bool, *, generation: bool = False) -> None:
        self._generation_busy = bool(busy and generation)
        self.progress_bar.setEnabled(True)
        for button in self._action_buttons:
            if button is self.cancel_button:
                button.setEnabled(True)
            else:
                button.setEnabled(not busy)
        if hasattr(self, "more_button"):
            self.more_button.setEnabled(not busy)
        if generation or not busy:
            for widget in self._settings_lock_widgets:
                widget.setEnabled(not busy)

    def _cancel_button_for_busy(self) -> QPushButton | None:
        return getattr(self, "cancel_button", None)

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
