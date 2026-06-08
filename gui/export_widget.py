"""
Export Results page.

Loads previously saved analysis records from the JSON store, lists them in a
table, shows the full JSON of a selected record, and exports the results file
to a location of the user's choosing.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from utils import JSONStorage

COLUMNS = ["Video", "Timestamp", "Persons", "Violations", "Compliance %"]


class ExportWidget(QWidget):
    """View saved JSON results and export them."""

    def __init__(self, storage: JSONStorage, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.storage = storage
        self._records: List[Dict[str, Any]] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        title = QLabel("Saved Results")
        title.setStyleSheet("font-size:18px; font-weight:600;")
        header.addWidget(title)
        header.addStretch(1)

        self.btn_refresh = QPushButton("Reload")
        self.btn_export = QPushButton("Export JSON...")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_export.clicked.connect(self._export)
        header.addWidget(self.btn_refresh)
        header.addWidget(self.btn_export)
        root.addLayout(header)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.itemSelectionChanged.connect(self._show_detail)
        root.addWidget(self.table, stretch=2)

        root.addWidget(QLabel("Record detail:"))
        self.detail = QTextEdit()
        self.detail.setReadOnly(True)
        root.addWidget(self.detail, stretch=3)

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Reload all records from disk and repopulate the table."""
        self._records = self.storage.load()
        self.table.setRowCount(len(self._records))
        for row, rec in enumerate(self._records):
            rate = f"{rec.get('compliance_rate', 0) * 100:.1f}"
            values = [
                rec.get("video_name", "-"),
                rec.get("timestamp", "-"),
                str(rec.get("total_persons_detected", 0)),
                str(rec.get("total_violations", 0)),
                rate,
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, item)
        self.detail.clear()

    def _show_detail(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self._records):
            self.detail.setPlainText(
                json.dumps(self._records[idx], indent=2, ensure_ascii=False)
            )

    def _export(self) -> None:
        if not self._records:
            QMessageBox.information(self, "Nothing to export", "There are no saved results yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results", "ppe_results.json", "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.storage.export_to(path)
            QMessageBox.information(self, "Exported", f"Results exported to:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
