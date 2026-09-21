"""Unit tests for field detector seam and implementations (Part 1, Requirement 3)."""

import numpy as np
import pytest
from pitch_engine.config.schema import FieldDetectorConfig, SportType
from pitch_engine.core.exceptions import ConfigurationError
from pitch_engine.core.models import DetectionStatus, FieldDetection
from pitch_engine.detectors.base import BaseFieldDetector
from pitch_engine.detectors.color_threshold import ColorThresholdDetector
from pitch_engine.detectors.factory import create_detector, register_detector
from pitch_engine.detectors.mock import MockFieldDetector


def test_color_threshold_detector_detects_black_frame():
    config = FieldDetectorConfig(type="color_threshold", sport=SportType.FOOTBALL)
    detector = ColorThresholdDetector(config)

    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    res = detector.detect(black_frame, frame_index=0, timestamp_seconds=0.0)

    assert res.status == DetectionStatus.NO_PITCH_VISIBLE
    assert res.confidence == 0.0
    assert res.polygon is None


def test_color_threshold_detector_detects_pitch():
    config = FieldDetectorConfig(type="color_threshold", sport=SportType.FOOTBALL, min_area=1000.0)
    detector = ColorThresholdDetector(config)

    # Create synthetic green pitch canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (34, 139, 34)  # BGR Forest green

    res = detector.detect(canvas, frame_index=1, timestamp_seconds=0.033)
    assert res.status == DetectionStatus.VALID
    assert res.confidence > 0.5
    assert res.polygon is not None
    assert len(res.polygon) >= 3


def test_detector_seam_swappability():
    # Define custom detector plugin implementing the seam
    class CustomRugbyDetector(BaseFieldDetector):
        def detect(self, frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> FieldDetection:
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.VALID,
                confidence=0.99,
                area=50000.0,
                intersection_area=50000.0,
                metadata={"custom": "rugby_plugin"}
            )

    # Register in factory
    register_detector("custom_rugby", CustomRugbyDetector)

    config = FieldDetectorConfig(type="custom_rugby", sport=SportType.RUGBY)
    detector = create_detector(config)

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame, 0, 0.0)

    assert res.confidence == 0.99
    assert res.metadata.get("custom") == "rugby_plugin"


def test_unregistered_detector_fails_fast():
    config = FieldDetectorConfig(type="non_existent_detector_xyz")
    with pytest.raises(ConfigurationError) as exc_info:
        create_detector(config)

    assert "Unknown field detector type" in str(exc_info.value)
