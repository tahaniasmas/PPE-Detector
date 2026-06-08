"""
Statistics page.

Holds in-memory statistics for the CURRENT session (no database). Updated live
as frames are processed and reset whenever a new video is loaded, exactly as
required by the spec.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class StatisticsWidget(QWidget):
    """Displays running totals computed entirely in memory."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._total_persons = 0
        self._total_violations = 0
        self._frames = 0
        self._max_persons = 0
        self._build_ui()
        self._render()


    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        title = QLabel("Session Statistics")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        root.addWidget(title)

        box = QGroupBox("Current video (resets when a new video is loaded)")
        grid = QGridLayout(box)

        self.lbl_video = QLabel("-")
        self.val_frames = QLabel("0")
        self.val_persons = QLabel("0")
        self.val_max = QLabel("0")
        self.val_violations = QLabel("0")
        self.val_rate = QLabel("100.0 %")

        rows = [
            ("Video:", self.lbl_video),
            ("Frames processed:", self.val_frames),
            ("Total person detections:", self.val_persons),
            ("Max persons in a frame:", self.val_max),
            ("Total violations:", self.val_violations),
            ("Compliance rate:", self.val_rate),
        ]
        for i, (caption, value) in enumerate(rows):
            cap = QLabel(caption)
            cap.setStyleSheet("color:#9aa4b2;")
            value.setStyleSheet("font-weight:600;")
            grid.addWidget(cap, i, 0, alignment=Qt.AlignmentFlag.AlignLeft)
            grid.addWidget(value, i, 1, alignment=Qt.AlignmentFlag.AlignRight)

        root.addWidget(box)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setFormat("Compliance %p%")
        root.addWidget(self.bar)

        root.addStretch(1)


    def reset(self, video_name: str = "-") -> None:
        """Clear all counters. Called when a new video is loaded."""
        self._total_persons = 0
        self._total_violations = 0
        self._frames = 0
        self._max_persons = 0
        self.lbl_video.setText(video_name)
        self._render()

    def update_live(self, summary: Dict[str, Any]) -> None:
        """Accumulate a per-frame summary emitted by the worker."""
        self._frames += 1
        self._total_persons += summary.get("persons", 0)
        self._total_violations += summary.get("violations", 0)
        self._max_persons = max(self._max_persons, summary.get("persons", 0))
        self._render()


    def _render(self) -> None:
        self.val_frames.setText(str(self._frames))
        self.val_persons.setText(str(self._total_persons))
        self.val_max.setText(str(self._max_persons))
        self.val_violations.setText(str(self._total_violations))

        if self._total_persons > 0:
            compliant = self._total_persons - self._total_violations
            rate = max(0.0, 100.0 * compliant / self._total_persons)
        else:
            rate = 100.0
        self.val_rate.setText(f"{rate:.1f} %")
        self.bar.setValue(int(round(rate)))
