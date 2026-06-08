"""
PPE compliance evaluation.

`ComplianceChecker` takes the per-frame detections produced by `PPEDetector`,
associates each detected PPE item (and each NO-* negative detection) with the
person it belongs to, and decides whether that person is compliant.

Association heuristic (no model training required): a PPE/negative box belongs
to a person if its center falls inside the person box, or if a large enough
fraction of the PPE box is contained within the person box. This is simple and
robust for typical construction-site footage; swap in a tracker/pose model for
production use.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from detector import config

BBox = Tuple[int, int, int, int]


def _center(box: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _containment(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner`'s area that lies inside `outer` (0..1)."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    inner_area = _area(inner)
    return inter / inner_area if inner_area else 0.0


class ComplianceChecker:
    """Decide PPE compliance for each detected person in a frame."""

    def __init__(
        self,
        required_ppe: Optional[Sequence[str]] = None,
        detectable_ppe: Optional[Iterable[str]] = None,
        association_thresh: float = 0.30,
    ) -> None:
        requested = list(required_ppe if required_ppe is not None else config.REQUIRED_PPE)


        if detectable_ppe is not None:
            detectable = set(detectable_ppe)
            self.effective_required = [p for p in requested if p in detectable]
            self.skipped_required = [p for p in requested if p not in detectable]
        else:
            self.effective_required = requested
            self.skipped_required = []

        self.association_thresh = association_thresh


    def _is_associated(self, item_box: BBox, person_box: BBox) -> bool:
        cx, cy = _center(item_box)
        inside_center = (
            person_box[0] <= cx <= person_box[2]
            and person_box[1] <= cy <= person_box[3]
        )
        if inside_center:
            return True
        return _containment(item_box, person_box) >= self.association_thresh


    def evaluate_frame(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Return one compliance result per detected person:
            {person_index, bbox, worn_ppe, missing_ppe, compliant}
        """
        persons = [d for d in detections if d["class"] == config.PERSON_CLASS]
        ppe_items = [d for d in detections if d["class"] in config.PPE_CLASSES]
        negatives = [d for d in detections if d["class"] in config.NEGATIVE_CLASSES]

        results: List[Dict[str, Any]] = []
        for idx, person in enumerate(persons):
            pbox = person["bbox"]

            worn: Set[str] = {
                item["class"] for item in ppe_items if self._is_associated(item["bbox"], pbox)
            }
            # negative class "no_helmet" -> explicit missing "helmet"
            explicit_missing: Set[str] = {
                neg["class"].replace("no_", "", 1)
                for neg in negatives
                if self._is_associated(neg["bbox"], pbox)
            }

            missing: List[str] = []
            for req in self.effective_required:
                if req in worn:
                    continue
                missing.append(req)

            missing = list(dict.fromkeys(missing))

            results.append(
                {
                    "person_index": idx,
                    "bbox": pbox,
                    "worn_ppe": sorted(worn),
                    "missing_ppe": missing,
                    "explicit_violations": sorted(explicit_missing & set(self.effective_required)),
                    "compliant": len(missing) == 0,
                }
            )
        return results
