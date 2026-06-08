"""
Visualization helpers.

Pure OpenCV drawing functions that turn a frame + detections + compliance
results into an annotated frame. Kept free of any GUI dependency so they can
be reused for the saved output video and the screenshots.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from detector import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw `text` with a filled background rectangle for readability."""
    x, y = org
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    y = max(y, th + baseline + 2)
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 4, y), bg_color, -1)
    cv2.putText(img, text, (x + 2, y - baseline), FONT, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    compliance_results: List[Dict[str, Any]],
) -> np.ndarray:
    """Return a copy of `frame` annotated with PPE boxes and compliance status."""
    out = frame.copy()


    for det in detections:
        cls = det["class"]
        if cls == config.PERSON_CLASS:
            continue
        x1, y1, x2, y2 = det["bbox"]
        color = config.COLORS.get(cls, config.COLORS["default"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        _put_label(out, f"{cls} {det['confidence']:.2f}", (x1, y1), color, scale=0.4)


    for res in compliance_results:
        x1, y1, x2, y2 = res["bbox"]
        if res["compliant"]:
            color = config.COLORS["compliant"]
            label = "COMPLIANT"
        else:
            color = config.COLORS["violation"]
            label = "NON-COMPLIANT: missing " + ", ".join(res["missing_ppe"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _put_label(out, label, (x1, max(y1, 18)), color, scale=0.5)


    total = len(compliance_results)
    violations = sum(1 for r in compliance_results if not r["compliant"])
    summary = f"Persons: {total}  |  Violations: {violations}"
    _put_label(out, summary, (8, 22), (40, 40, 40), scale=0.6, thickness=2)

    return out
