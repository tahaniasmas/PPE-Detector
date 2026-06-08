"""AI detection package: YOLOv8 wrapper, compliance logic and configuration."""
from detector import config
from detector.compliance import ComplianceChecker
from detector.ppe_detector import PPEDetector

__all__ = ["config", "PPEDetector", "ComplianceChecker"]
