"""
Central configuration for the PPE Detection desktop application.

Everything that you might want to tweak (model path, confidence threshold,
which PPE items are mandatory, how raw model class names map to the canonical
categories used by the app) lives here so the rest of the code stays generic.
"""
from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_JSON = DATA_DIR / "results.json"


DEFAULT_MODEL_PATH = MODELS_DIR / "ppe_yolov8.pt"


CONF_THRESHOLD = 0.35      # minimum detection confidence
IOU_THRESHOLD = 0.50       # NMS IoU threshold
DEVICE = ""                # "" -> auto, "cpu" -> force CPU, "0" -> first GPU

PERSON_CLASS = "person"
PPE_CLASSES = ["helmet", "vest", "gloves", "goggles", "mask"]

NEGATIVE_CLASSES = ["no_helmet", "no_vest", "no_mask", "no_gloves", "no_goggles"]

CLASS_ALIASES = {
    # person
    "person": "person",
    "worker": "person",
    # helmet
    "helmet": "helmet",
    "hardhat": "helmet",
    "hard hat": "helmet",
    "hard-hat": "helmet",
    # vest
    "vest": "vest",
    "safety vest": "vest",
    "safety-vest": "vest",
    "safetyvest": "vest",
    # gloves
    "gloves": "gloves",
    "glove": "gloves",
    # goggles
    "goggles": "goggles",
    "glasses": "goggles",
    "safety glasses": "goggles",
    # mask
    "mask": "mask",
    "face mask": "mask",
    "face-mask": "mask",
    # negative classes
    "no-hardhat": "no_helmet",
    "no_hardhat": "no_helmet",
    "no-helmet": "no_helmet",
    "no-safety vest": "no_vest",
    "no-safety-vest": "no_vest",
    "no-vest": "no_vest",
    "no-mask": "no_mask",
    "no-gloves": "no_gloves",
    "no-goggles": "no_goggles",
    # classes we simply ignore
    "safety cone": "_ignore",
    "machinery": "_ignore",
    "vehicle": "_ignore",
}


REQUIRED_PPE = ["helmet", "vest", "gloves"]


COLORS = {
    "person": (255, 190, 0),
    "helmet": (0, 200, 0),
    "vest": (0, 200, 0),
    "gloves": (0, 200, 0),
    "goggles": (0, 200, 0),
    "mask": (0, 200, 0),
    "no_helmet": (0, 0, 255),
    "no_vest": (0, 0, 255),
    "no_mask": (0, 0, 255),
    "no_gloves": (0, 0, 255),
    "no_goggles": (0, 0, 255),
    "compliant": (0, 180, 0),
    "violation": (0, 0, 235),
    "default": (200, 200, 200),
}


def ensure_dirs() -> None:
    """Create all output directories if they do not exist yet."""
    for directory in (MODELS_DIR, DATA_DIR, SCREENSHOTS_DIR, PROCESSED_DIR):
        directory.mkdir(parents=True, exist_ok=True)
