"""
JSON storage layer (no database, by design).

All analysis records are persisted to a single JSON file as a list of objects.
Writes are atomic (write to a temp file then replace) so a crash mid-write
cannot corrupt previous results.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from detector import config


class JSONStorage:
    """Read/append/export analysis records stored in a JSON file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else config.RESULTS_JSON
        self.path.parent.mkdir(parents=True, exist_ok=True)


    def load(self) -> List[Dict[str, Any]]:
        """Load all records. Returns [] if the file is missing or corrupt."""
        if not self.path.exists():
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []


    def save_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Append a single record and persist the whole list atomically."""
        records = self.load()
        records.append(record)
        self._write(records)
        return record

    def overwrite(self, records: List[Dict[str, Any]]) -> None:
        """Replace the entire file with `records`."""
        self._write(records)

    def export_to(self, destination: str | Path) -> Path:
        """Copy the current results file to `destination` and return its path."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():

            self._write([])
        shutil.copyfile(self.path, destination)
        return destination


    def _write(self, records: List[Dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, ensure_ascii=False)
        tmp.replace(self.path)  # atomic on the same filesystem
