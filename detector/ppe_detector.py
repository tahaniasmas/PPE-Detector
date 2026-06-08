"""
YOLOv8 detector wrapper.

`PPEDetector` hides the Ultralytics API behind a tiny, stable interface:
load a model once, then call `detect(frame)` to get a list of normalized
detection dicts. Raw model class names are translated to the canonical
categories defined in `config.CLASS_ALIASES`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

from detector import config


class PPEDetector:
    """Thin wrapper around an Ultralytics YOLOv8 model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf: float = config.CONF_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        device: str = config.DEVICE,
    ) -> None:
        self.model_path = Path(model_path) if model_path else config.DEFAULT_MODEL_PATH
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = None
        self._names: Dict[int, str] = {}      # raw id -> raw name
        self._load_model()


    def _load_model(self) -> None:
        """Import Ultralytics lazily and load the weights, with clear errors."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "The 'ultralytics' package is required. Install it with:\n"
                "    pip install ultralytics"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at: {self.model_path}\n"
                "Train a model on your PPE dataset and place the .pt file there, "
                "or choose a different file from the GUI."
            )

        self.model = YOLO(str(self.model_path))

        self._names = dict(self.model.names)


    @staticmethod
    def canonical(raw_name: str) -> str:
        """Translate a raw model class name to a canonical category."""
        key = raw_name.strip().lower()
        return config.CLASS_ALIASES.get(key, key)

    @property
    def available_categories(self) -> Set[str]:
        """All canonical categories this model can emit (excluding '_ignore')."""
        cats = {self.canonical(name) for name in self._names.values()}
        cats.discard("_ignore")
        return cats

    @property
    def available_ppe(self) -> Set[str]:
        """Canonical PPE categories this model can actually detect."""
        return {c for c in self.available_categories if c in config.PPE_CLASSES}


    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame.

        Returns a list of dicts:
            {class, raw_class, confidence, bbox=(x1, y1, x2, y2)}
        '_ignore' classes (cones, vehicles, ...) are dropped.
        """
        if self.model is None:
            return []

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device or None,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        detections: List[Dict[str, Any]] = []
        for box in boxes:
            cls_id = int(box.cls[0])
            raw_name = self._names.get(cls_id, str(cls_id))
            category = self.canonical(raw_name)
            if category == "_ignore":
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            detections.append(
                {
                    "class": category,
                    "raw_class": raw_name,
                    "confidence": float(box.conf[0]),
                    "bbox": (x1, y1, x2, y2),
                }
            )
        return detections
