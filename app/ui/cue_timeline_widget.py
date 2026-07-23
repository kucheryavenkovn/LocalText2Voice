from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.core.video_dubbing.models import CueStatus, DubbingCue


_STATUS_COLORS: dict[str, QColor] = {
    CueStatus.RENDERED.value: QColor("#16a34a"),
    CueStatus.FITTED.value: QColor("#16a34a"),
    CueStatus.SPEED_UP.value: QColor("#2563eb"),
    CueStatus.NEEDS_SHORTENING.value: QColor("#dc2626"),
    CueStatus.FAILED.value: QColor("#7f1d1d"),
    CueStatus.STALE.value: QColor("#ca8a04"),
    CueStatus.DISABLED.value: QColor("#9ca3af"),
    CueStatus.PENDING.value: QColor("#94a3b8"),
    CueStatus.RENDERING.value: QColor("#0891b2"),
}


@dataclass
class TimelineSelectionRequest:
    sequence: int


class CueTimelineWidget(QWidget):
    """Painted, non-interactive (beyond click-to-select) timeline.

    Each cue renders as a block positioned at its absolute start_ms with a
    width matching its budget. Problematic cues are visually marked. A vertical
    playhead reflects the current player position.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._cues: list[DubbingCue] = []
        self._duration_ms: int = 0
        self._position_ms: int = 0
        self._selected_sequence: int | None = None

    def set_cues(self, cues: list[DubbingCue], duration_ms: int) -> None:
        self._cues = list(cues)
        self._duration_ms = max(1, duration_ms)
        self.update()

    def set_position_ms(self, position_ms: int) -> None:
        self._position_ms = max(0, int(position_ms))
        self.update()

    def set_selected(self, sequence: int | None) -> None:
        self._selected_sequence = sequence
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(400, 84)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._cues or self._duration_ms <= 0:
            return
        x = event.position().x()
        width = max(1, self.width() - 16)
        ratio = max(0.0, min(1.0, (x - 8) / width))
        clicked_ms = int(ratio * self._duration_ms)
        for cue in self._cues:
            if cue.start_ms <= clicked_ms <= cue.end_ms:
                self._selected_sequence = cue.sequence
                top_level = self.window()
                if top_level is not None:
                    top_level.setProperty(
                        "_dubbing_timeline_selection", cue.sequence
                    )
                self.update()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        if self._duration_ms <= 0:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "—")
            return
        margin = 8
        track_top = 18
        track_height = self.height() - 28
        usable_width = max(1, self.width() - margin * 2)
        scale = usable_width / self._duration_ms

        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawLine(
            margin, track_top + track_height, self.width() - margin, track_top + track_height
        )
        for cue in self._cues:
            x = margin + cue.start_ms * scale
            width = max(2.0, cue.duration_budget_ms * scale)
            color = _STATUS_COLORS.get(cue.status, QColor("#475569"))
            if not cue.enabled:
                color = _STATUS_COLORS[CueStatus.DISABLED.value]
            rect_x = int(x)
            rect_y = track_top
            rect_w = max(2, int(width))
            rect_h = track_height
            if cue.overflow_ms > 0:
                overflow_w = max(1, int(cue.overflow_ms * scale))
                painter.setBrush(QBrush(QColor("#7f1d1d")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect_x, rect_y, rect_w + overflow_w, rect_h)
            painter.setBrush(QBrush(color))
            if self._selected_sequence == cue.sequence:
                painter.setPen(QPen(QColor("#fde047"), 2))
            else:
                painter.setPen(QPen(QColor("#0f172a"), 1))
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)

        if self._position_ms > 0:
            play_x = margin + self._position_ms * scale
            painter.setPen(QPen(QColor("#f8fafc"), 1))
            painter.drawLine(int(play_x), 6, int(play_x), self.height() - 6)
        painter.end()
