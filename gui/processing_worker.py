"""
Background processing worker.

Runs YOLOv8 frame-by-frame off the GUI thread. For every frame:

    detect -> evaluate compliance -> annotate -> emit to GUI
            -> optionally write to output video / save violation screenshots

Pause/resume/stop use a mutex + wait condition. A live, thread-safe speed
control adjusts the delay between frames so the user can slow down or speed up
playback (capped by inference speed).
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from PyQt6.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal

from detector import ComplianceChecker, PPEDetector, config
from utils.visualization import draw_frame


class ProcessingWorker(QThread):
    """Processes a video off the GUI thread and emits annotated frames."""

    frame_ready = pyqtSignal(np.ndarray, dict)   # annotated frame, frame summary
    progress = pyqtSignal(int, int)              # current frame, total frames
    finished_processing = pyqtSignal(dict)       # aggregated JSON record
    error = pyqtSignal(str)
    status = pyqtSignal(str)                      # human-readable status text

    def __init__(
        self,
        video_path: str | Path,
        model_path: str | Path | None,
        save_output: bool = True,
        save_screenshots: bool = True,
        speed: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.model_path = model_path
        self.save_output = save_output
        self.save_screenshots = save_screenshots

        self._mutex = QMutex()
        self._pause_cond = QWaitCondition()
        self._paused = False
        self._stopped = False
        self._speed = max(0.1, float(speed))

    # ------------------------------------------------------------------ #
    # Public controls (called from the GUI thread)
    # ------------------------------------------------------------------ #
    def pause(self) -> None:
        self._mutex.lock()
        self._paused = True
        self._mutex.unlock()

    def resume(self) -> None:
        self._mutex.lock()
        self._paused = False
        self._mutex.unlock()
        self._pause_cond.wakeAll()

    def stop(self) -> None:
        self._mutex.lock()
        self._stopped = True
        self._paused = False
        self._mutex.unlock()
        self._pause_cond.wakeAll()

    def set_speed(self, speed: float) -> None:
        """Change playback speed live (thread-safe). 1.0 = native FPS."""
        self._mutex.lock()
        self._speed = max(0.1, float(speed))
        self._mutex.unlock()

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        try:
            self.status.emit("Loading model...")
            detector = PPEDetector(self.model_path)
            checker = ComplianceChecker(detectable_ppe=detector.available_ppe)
            if checker.skipped_required:
                self.status.emit(
                    "Note: model cannot detect "
                    + ", ".join(checker.skipped_required)
                    + " -> those rules are skipped."
                )
            self._process(detector, checker)
        except Exception as exc:
            self.error.emit(str(exc))

    # ------------------------------------------------------------------ #
    def _should_continue(self) -> bool:
        """Block while paused; return False if a stop was requested."""
        self._mutex.lock()
        try:
            if self._stopped:
                return False
            while self._paused and not self._stopped:
                self._pause_cond.wait(self._mutex)
            return not self._stopped
        finally:
            self._mutex.unlock()

    def _current_speed(self) -> float:
        self._mutex.lock()
        try:
            return self._speed
        finally:
            self._mutex.unlock()

    def _process(self, detector: PPEDetector, checker: ComplianceChecker) -> None:
        config.ensure_dirs()

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.error.emit(f"Could not open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        processed_path: Path | None = None
        if self.save_output and width and height:
            processed_path = config.PROCESSED_DIR / f"{self.video_path.stem}_processed.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(processed_path), fourcc, fps, (width, height))

        total_person_detections = 0
        compliant_detections = 0
        max_persons = 0
        violations: List[Dict[str, Any]] = []
        frame_idx = 0

        self.status.emit("Processing...")
        while self._should_continue():
            frame_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.detect(frame)
            compliance = checker.evaluate_frame(detections)

            n_persons = len(compliance)
            total_person_detections += n_persons
            max_persons = max(max_persons, n_persons)

            frame_violations: List[Dict[str, Any]] = []
            for res in compliance:
                if res["compliant"]:
                    compliant_detections += 1
                else:
                    frame_violations.append(
                        {
                            "frame_number": frame_idx,
                            "timestamp_sec": round(frame_idx / fps, 2) if fps else 0.0,
                            "missing_ppe": res["missing_ppe"],
                            "bbox": list(res["bbox"]),
                            "screenshot_path": "",
                        }
                    )

            annotated = draw_frame(frame, detections, compliance)

            if self.save_screenshots and frame_violations:
                shot = config.SCREENSHOTS_DIR / f"{self.video_path.stem}_frame{frame_idx}.jpg"
                cv2.imwrite(str(shot), annotated)
                for v in frame_violations:
                    v["screenshot_path"] = str(shot)

            violations.extend(frame_violations)

            if writer is not None:
                writer.write(annotated)

            self.frame_ready.emit(
                annotated,
                {
                    "frame_number": frame_idx,
                    "persons": n_persons,
                    "violations": len(frame_violations),
                    "compliance_results": compliance,
                },
            )
            self.progress.emit(frame_idx + 1, total_frames)
            frame_idx += 1

            # --- speed control: wait so playback matches the chosen speed --- #
            speed = self._current_speed()
            target_interval = (1.0 / fps) / speed if fps else 0.0
            remaining = target_interval - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

        cap.release()
        if writer is not None:
            writer.release()

        compliance_rate = (
            compliant_detections / total_person_detections
            if total_person_detections
            else 1.0
        )
        record = {
            "video_name": self.video_path.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "frames_processed": frame_idx,
            "fps": round(float(fps), 2),
            "total_persons_detected": total_person_detections,
            "max_persons_in_frame": max_persons,
            "total_violations": len(violations),
            "compliance_rate": round(compliance_rate, 4),
            "violations": violations,
            "processed_video_path": str(processed_path) if processed_path else "",
        }

        if self._stopped:
            self.status.emit("Stopped.")
        else:
            self.status.emit("Done.")
            self.finished_processing.emit(record)