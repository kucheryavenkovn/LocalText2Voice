from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from app.core.video_dubbing.models import CueStatus, DubbingCue


_STATUS_COLORS: dict[str, QColor] = {
    CueStatus.RENDERED.value: QColor("#16a34a"),
    CueStatus.FITTED.value: QColor("#16a34a"),
    CueStatus.SPEED_UP.value: QColor("#2563eb"),
    CueStatus.STRONG_SPEED_UP.value: QColor("#ea580c"),
    CueStatus.NEEDS_SHORTENING.value: QColor("#dc2626"),
    CueStatus.ELASTIC_GROUP_FAILED.value: QColor("#dc2626"),
    CueStatus.FAILED.value: QColor("#7f1d1d"),
    CueStatus.STALE.value: QColor("#ca8a04"),
    CueStatus.DISABLED.value: QColor("#9ca3af"),
    CueStatus.PENDING.value: QColor("#94a3b8"),
    CueStatus.RENDERING.value: QColor("#0891b2"),
    CueStatus.CANCELLED.value: QColor("#9ca3af"),
}


class CueTimelineWidget(QWidget):
    """Timeline with click-to-seek and cue selection."""

    cueSelected = Signal(int)  # sequence
    positionSeeked = Signal(int)  # absolute ms
    cueActivated = Signal(int)  # double-click sequence

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
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

    def _ms_at_x(self, x: float) -> int:
        margin = 8
        usable_width = max(1, self.width() - margin * 2)
        ratio = max(0.0, min(1.0, (x - margin) / usable_width))
        return int(ratio * self._duration_ms)

    def _cue_at_ms(self, clicked_ms: int) -> DubbingCue | None:
        hit: DubbingCue | None = None
        for cue in self._cues:
            start = cue.effective_start_ms()
            end = cue.effective_end_ms()
            # Also allow hit on source window so empty gaps still pick a cue.
            src_start = cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
            src_end = cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms
            if start <= clicked_ms <= max(end, src_end):
                hit = cue
            elif src_start <= clicked_ms <= src_end:
                hit = cue
        return hit

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._duration_ms <= 0 or not self._cues:
            self.setToolTip("")
            return
        cue = self._cue_at_ms(self._ms_at_x(event.position().x()))
        if cue is None:
            self.setToolTip(
                "Клик — перемотка. Цвета: зелёный=ok, синий=лёгкое ускорение, "
                "оранжевый=сильное, красный=проблема. Пунктир=группа/сдвиг."
            )
            return
        src_s = cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
        src_e = cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms
        plan_s = cue.effective_start_ms()
        plan_e = cue.effective_end_ms()
        shift = int(cue.start_shift_ms or 0)
        speed = cue.planned_speed_factor or cue.applied_speed_factor or 1.0
        lines = [
            f"Реплика #{cue.sequence} — {cue.status}",
            f"SRT: {src_s}–{src_e} ms",
            f"План: {plan_s}–{plan_e} ms",
            f"Скорость: {speed:.2f}"
            + (f" (req {cue.required_speed_factor:.2f})" if cue.required_speed_factor else ""),
        ]
        if shift:
            lines.append(f"Сдвиг вправо: {shift} ms (эластика; TTS не нужен → Пересчитать)")
        if cue.timing_group_id:
            lines.append(f"Эластичная группа: {cue.timing_group_id}")
        if cue.smoothing_group_id:
            lines.append(f"Группа темпа: {cue.smoothing_group_id}")
        if cue.is_stale:
            lines.append("⚠ Устарела — нужен TTS или Пересчитать")
        lines.append("Клик = выбор/seek · Двойной клик = play")
        self.setToolTip("\n".join(lines))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._duration_ms <= 0:
            super().mousePressEvent(event)
            return
        clicked_ms = self._ms_at_x(event.position().x())
        cue = self._cue_at_ms(clicked_ms) if self._cues else None
        if cue is not None:
            self._selected_sequence = cue.sequence
            self.update()
            self.cueSelected.emit(cue.sequence)
            seek_ms = max(0, cue.effective_start_ms())
            self.positionSeeked.emit(seek_ms)
            return
        self.positionSeeked.emit(clicked_ms)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if self._duration_ms <= 0:
            super().mouseDoubleClickEvent(event)
            return
        clicked_ms = self._ms_at_x(event.position().x())
        cue = self._cue_at_ms(clicked_ms) if self._cues else None
        if cue is not None:
            self._selected_sequence = cue.sequence
            self.update()
            self.cueSelected.emit(cue.sequence)
            self.cueActivated.emit(cue.sequence)
            return
        super().mouseDoubleClickEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0f172a"))
        if self._duration_ms <= 0:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "—")
            painter.end()
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
        # Source intervals (dim) then planned/fitted intervals (bright).
        for cue in self._cues:
            src_start = cue.source_start_ms if cue.source_start_ms is not None else cue.start_ms
            src_end = cue.source_end_ms if cue.source_end_ms is not None else cue.end_ms
            src_x = margin + src_start * scale
            src_w = max(2.0, (src_end - src_start) * scale)
            painter.setBrush(QBrush(QColor(51, 65, 85, 90)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(int(src_x), track_top + 4, max(2, int(src_w)), track_height - 8)

        for cue in self._cues:
            start = cue.effective_start_ms()
            end = cue.effective_end_ms()
            duration = max(1, end - start)
            if cue.fitted_duration_ms:
                duration = max(duration, cue.fitted_duration_ms)
            x = margin + start * scale
            width = max(2.0, duration * scale)
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
            if cue.timing_group_id:
                # Bracket for elastic group membership.
                painter.setBrush(QBrush(QColor(37, 99, 235, 40)))
                painter.setPen(QPen(QColor("#38bdf8"), 1, Qt.PenStyle.DashLine))
                painter.drawRect(rect_x - 1, rect_y - 3, rect_w + 2, rect_h + 6)
            painter.setBrush(QBrush(color))
            if self._selected_sequence == cue.sequence:
                painter.setPen(QPen(QColor("#fde047"), 2))
            else:
                painter.setPen(QPen(QColor("#0f172a"), 1))
            painter.drawRect(rect_x, rect_y, rect_w, rect_h)

        if self._position_ms >= 0:
            play_x = margin + self._position_ms * scale
            painter.setPen(QPen(QColor("#f8fafc"), 2))
            painter.drawLine(int(play_x), 6, int(play_x), self.height() - 6)
        painter.end()
