"""Field detectors package."""

from pitch_engine.detectors.base import BaseFieldDetector
from pitch_engine.detectors.color_threshold import ColorThresholdDetector
from pitch_engine.detectors.factory import create_detector, register_detector
from pitch_engine.detectors.mock import MockFieldDetector

__all__ = [
    "BaseFieldDetector",
    "ColorThresholdDetector",
    "MockFieldDetector",
    "create_detector",
    "register_detector",
]
