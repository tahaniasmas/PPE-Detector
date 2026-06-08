"""
Video Analysis page.

Lets the user choose a model and a video, then start/pause/stop processing and
adjust playback speed live. Shows the annotated stream, a progress bar and a
live status panel. Emits signals upward so the main window can update
statistics and storage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from detector import config
from .processing_worker import ProcessingWorker

VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)"
MODEL_FILTER = "YOLOv8 weights (*.pt);;All files (*)"

SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
DEFAULT_SPEED_INDEX = SPEEDS.index(1.0)


class VideoWidget(QWidget):
    """The main video analysis view."""

    record_ready = pyqtSignal(dict)
    stats_update = pyqtSignal(dict)
    video_loaded = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.video_path: Optional[Path] = None
        self.model_path: Path = config.DEFAULT_MODEL_PATH
        self.worker: Optional[ProcessingWorker] = None
        self._build_ui()
        self._refresh_buttons()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        self.video_label = QLabel("Charger une vidéo pour commencer")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setObjectName("videoCanvas")
        left.addWidget(self.video_label, stretch=1)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        left.addWidget(self.progress)

        speed_row = QHBoxLayout()
        speed_caption = QLabel("Vitesse :")
        speed_caption.setObjectName("muted")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(len(SPEEDS) - 1)
        self.speed_slider.setValue(DEFAULT_SPEED_INDEX)
        self.speed_slider.setFixedWidth(220)
        self.speed_value = QLabel("1.0x")
        self.speed_value.setObjectName("speedValue")
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(speed_caption)
        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_value)
        speed_row.addStretch(1)
        left.addLayout(speed_row)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.btn_load = QPushButton("Charger une vidéo")
        self.btn_model = QPushButton("Choisir le modèle")
        self.btn_start = QPushButton("Démarrer")
        self.btn_start.setObjectName("primary")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Arrêter")
        for btn in (self.btn_load, self.btn_model, self.btn_start, self.btn_pause, self.btn_stop):
            controls.addWidget(btn)
        left.addLayout(controls)

        self.btn_load.clicked.connect(self._choose_video)
        self.btn_model.clicked.connect(self._choose_model)
        self.btn_start.clicked.connect(self._start)
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self._stop)

        root.addLayout(left, stretch=3)

        panel = QGroupBox("État en direct")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setSpacing(8)

        self.lbl_model = QLabel()
        self.lbl_model.setWordWrap(True)
        self.lbl_video = QLabel("Vidéo : -")
        self.lbl_video.setWordWrap(True)
        self.lbl_frame = QLabel("Image : -")
        self.lbl_persons = QLabel("Personnes (image) : -")
        self.lbl_violations = QLabel("Violations (image) : -")
        self.lbl_state = QLabel("État : inactif")
        for lbl in (self.lbl_model, self.lbl_video, self.lbl_frame,
                    self.lbl_persons, self.lbl_violations, self.lbl_state):
            panel_layout.addWidget(lbl)

        log_caption = QLabel("Journal :")
        log_caption.setObjectName("muted")
        panel_layout.addWidget(log_caption)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        panel_layout.addWidget(self.log, stretch=1)

        root.addWidget(panel, stretch=2)
        self._update_model_label()

    def _update_model_label(self) -> None:
        exists = "OK" if self.model_path.exists() else "MANQUANT"
        self.lbl_model.setText(f"Modèle : {self.model_path.name} [{exists}]")

    def _log(self, text: str) -> None:
        self.log.append(text)

    def _is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _current_speed(self) -> float:
        return SPEEDS[self.speed_slider.value()]

    def _refresh_buttons(self) -> None:
        running = self._is_running()
        self.btn_load.setEnabled(not running)
        self.btn_model.setEnabled(not running)
        self.btn_start.setEnabled(self.video_path is not None and not running)
        self.btn_pause.setEnabled(running)
        self.btn_stop.setEnabled(running)

    def _on_speed_changed(self, index: int) -> None:
        speed = SPEEDS[index]
        self.speed_value.setText(f"{speed:g}x")
        if self._is_running():
            self.worker.set_speed(speed)

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir une vidéo", "", VIDEO_FILTER)
        if not path:
            return
        self.video_path = Path(path)
        self.lbl_video.setText(f"Vidéo : {self.video_path.name}")
        self.progress.setValue(0)
        self._log(f"Vidéo chargée : {self.video_path.name}")
        self.video_loaded.emit(str(self.video_path))
        self._refresh_buttons()

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choisir les poids YOLOv8", "", MODEL_FILTER)
        if not path:
            return
        self.model_path = Path(path)
        self._update_model_label()
        self._log(f"Modèle sélectionné : {self.model_path.name}")

    def _start(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "Aucune vidéo", "Veuillez d'abord charger une vidéo.")
            return
        if not self.model_path.exists():
            QMessageBox.critical(
                self,
                "Modèle manquant",
                f"Aucun fichier de poids à :\n{self.model_path}\n\n"
                "Placez le modèle dans le dossier models/ ou utilisez 'Choisir le modèle'.",
            )
            return

        self.worker = ProcessingWorker(
            self.video_path, self.model_path, speed=self._current_speed()
        )
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.progress.connect(self._on_progress)
        self.worker.status.connect(self._on_status)
        self.worker.error.connect(self._on_error)
        self.worker.finished_processing.connect(self._on_finished)
        self.worker.finished.connect(self._refresh_buttons)

        self.btn_pause.setText("Pause")
        self.lbl_state.setText("État : en cours")
        self.worker.start()
        self._refresh_buttons()

    def _toggle_pause(self) -> None:
        if not self._is_running():
            return
        if self.btn_pause.text() == "Pause":
            self.worker.pause()
            self.btn_pause.setText("Reprendre")
            self.lbl_state.setText("État : en pause")
        else:
            self.worker.resume()
            self.btn_pause.setText("Pause")
            self.lbl_state.setText("État : en cours")

    def _stop(self) -> None:
        if self._is_running():
            self.worker.stop()
            self.worker.wait(3000)
            self.lbl_state.setText("État : arrêté")
        self._refresh_buttons()

    def _on_frame(self, frame: np.ndarray, summary: Dict[str, Any]) -> None:
        self._display(frame)
        self.lbl_frame.setText(f"Image : {summary['frame_number']}")
        self.lbl_persons.setText(f"Personnes (image) : {summary['persons']}")
        self.lbl_violations.setText(f"Violations (image) : {summary['violations']}")
        self.stats_update.emit(summary)

    def _display(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy()).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress.setMaximum(total)
            self.progress.setValue(current)
        else:
            self.progress.setMaximum(0)

    def _on_status(self, text: str) -> None:
        self._log(text)

    def _on_error(self, message: str) -> None:
        self.lbl_state.setText("État : erreur")
        self._log(f"ERREUR : {message}")
        QMessageBox.critical(self, "Erreur de traitement", message)
        self._refresh_buttons()

    def _on_finished(self, record: Dict[str, Any]) -> None:
        self.lbl_state.setText("État : terminé")
        rate = record["compliance_rate"] * 100
        self._log(
            f"Terminé : {record['frames_processed']} images, "
            f"{record['total_violations']} violations, "
            f"conformité {rate:.1f}%."
        )
        self.record_ready.emit(record)