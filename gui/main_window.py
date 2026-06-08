"""
Main application window.

Light, professional theme with a top header, a sidebar (QListWidget) and a
QStackedWidget for the three pages. Page changes fade in via a property
animation; hover/selection states are styled for a polished feel.
"""
from __future__ import annotations

from typing import Any, Dict

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QSize, Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from detector import config
from utils import JSONStorage
from .export_widget import ExportWidget
from .statistics_widget import StatisticsWidget
from .video_widget import VideoWidget


APP_STYLE = """
QMainWindow, QWidget { background:#f4f6fb; color:#1f2937;
    font-family:'Segoe UI','Inter',Arial,sans-serif; font-size:14px; }

QFrame#header { background:#ffffff; border-bottom:1px solid #e6eaf2; }
QLabel#headerTitle { font-size:18px; font-weight:700; color:#1e293b; }
QLabel#headerSub   { font-size:13px; color:#94a3b8; }
QFrame#accentDot   { background:#3b82f6; border-radius:7px; }

QListWidget#sidebar { background:#ffffff; color:#475569; border:none;
    border-right:1px solid #e6eaf2; outline:0; padding-top:12px; font-size:14px; }
QListWidget#sidebar::item { padding:13px 18px; margin:3px 10px; border-radius:10px; }
QListWidget#sidebar::item:hover { background:#eef3ff; color:#1d4ed8; }
QListWidget#sidebar::item:selected { background:#3b82f6; color:#ffffff; font-weight:600; }

QGroupBox { background:#ffffff; border:1px solid #e6eaf2; border-radius:14px;
    margin-top:14px; padding:12px; font-weight:600; }
QGroupBox::title { subcontrol-origin: margin; left:14px; padding:0 6px; color:#475569; }

QLabel#muted { color:#94a3b8; }
QLabel#speedValue { color:#2563eb; font-weight:700; min-width:38px; }

QPushButton { background:#ffffff; color:#1f2937; border:1px solid #d6dbe6;
    border-radius:10px; padding:9px 16px; font-weight:500; }
QPushButton:hover { background:#eef2ff; border-color:#bcccf5; }
QPushButton:pressed { background:#e0e7ff; }
QPushButton:disabled { color:#aeb6c4; background:#f3f4f7; border-color:#e8eaf0; }
QPushButton#primary { background:#3b82f6; color:#ffffff; border:none; font-weight:600; }
QPushButton#primary:hover { background:#2563eb; }
QPushButton#primary:pressed { background:#1d4ed8; }
QPushButton#primary:disabled { background:#bcd2fb; color:#eaf1ff; }

QLabel#videoCanvas { background:#0f172a; color:#94a3b8; border:1px solid #e6eaf2;
    border-radius:14px; font-size:15px; }

QTextEdit, QTableWidget { background:#ffffff; border:1px solid #e6eaf2;
    border-radius:10px; selection-background-color:#dbeafe; selection-color:#1e293b; }
QHeaderView::section { background:#f0f3fa; color:#475569; border:none;
    padding:9px; font-weight:600; }
QTableWidget { gridline-color:#eef1f7; }

QProgressBar { border:none; background:#e7ecf6; border-radius:9px; text-align:center;
    height:18px; color:#334155; }
QProgressBar::chunk { background:#3b82f6; border-radius:9px; }

QSlider::groove:horizontal { height:6px; background:#d9e0ee; border-radius:3px; }
QSlider::sub-page:horizontal { background:#3b82f6; border-radius:3px; }
QSlider::add-page:horizontal { background:#d9e0ee; border-radius:3px; }
QSlider::handle:horizontal { background:#ffffff; border:2px solid #3b82f6;
    width:16px; height:16px; margin:-7px 0; border-radius:10px; }
QSlider::handle:horizontal:hover { border-color:#2563eb; }

QScrollBar:vertical { background:transparent; width:10px; margin:2px; }
QScrollBar::handle:vertical { background:#c9d2e3; border-radius:5px; min-height:30px; }
QScrollBar::handle:vertical:hover { background:#aebbd4; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
"""


class MainWindow(QMainWindow):
    """Top-level window holding the header, sidebar and the three pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PPE Detection - Construction Site Safety")
        self.resize(1200, 740)
        self.setStyleSheet(APP_STYLE)

        config.ensure_dirs()
        self.storage = JSONStorage()
        self._page_anim: QPropertyAnimation | None = None

        self.video_page = VideoWidget()
        self.stats_page = StatisticsWidget()
        self.export_page = ExportWidget(self.storage)

        self._build_ui()
        self._connect_signals()
        self._fade_in_window()


    def _build_ui(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)


        header = QFrame()
        header.setObjectName("header")
        header.setFixedHeight(62)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(22, 0, 22, 0)
        dot = QFrame()
        dot.setObjectName("accentDot")
        dot.setFixedSize(14, 14)
        title = QLabel("PPE Detection")
        title.setObjectName("headerTitle")
        sub = QLabel("Construction Site Safety")
        sub.setObjectName("headerSub")
        hl.addWidget(dot)
        hl.addSpacing(10)
        hl.addWidget(title)
        hl.addSpacing(10)
        hl.addWidget(sub)
        hl.addStretch(1)
        outer.addWidget(header)


        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(210)
        for name in ("Analyse vidéo", "Statistiques", "Résultats / Export"):
            item = QListWidgetItem(name)
            item.setSizeHint(QSize(0, 50))
            self.sidebar.addItem(item)
        self.sidebar.setCurrentRow(0)

        self.pages = QStackedWidget()
        self.pages.addWidget(self.video_page)
        self.pages.addWidget(self.stats_page)
        self.pages.addWidget(self.export_page)

        body.addWidget(self.sidebar)
        body.addWidget(self.pages, stretch=1)
        outer.addLayout(body, stretch=1)

        self.setCentralWidget(central)

        self.sidebar.currentRowChanged.connect(self._switch_page)

    def _switch_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self._fade_page(self.pages.currentWidget())

    def _fade_page(self, widget: QWidget) -> None:
        """Fade a page in when it becomes visible."""
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity")
        anim.setDuration(260)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        anim.start()
        self._page_anim = anim

    def _fade_in_window(self) -> None:
        self.setWindowOpacity(0.0)
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(350)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start()
        self._window_anim = anim


    def _connect_signals(self) -> None:
        self.video_page.video_loaded.connect(self._on_video_loaded)
        self.video_page.stats_update.connect(self.stats_page.update_live)
        self.video_page.record_ready.connect(self._on_record_ready)

    def _on_video_loaded(self, video_path: str) -> None:
        name = video_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        self.stats_page.reset(name)

    def _on_record_ready(self, record: Dict[str, Any]) -> None:
        self.storage.save_record(record)
        self.export_page.refresh()