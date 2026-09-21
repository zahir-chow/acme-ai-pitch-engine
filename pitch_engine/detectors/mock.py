"""Mock field detector for testing, benchmarking, and CI."""

from typing import List, Optional, Tuple
import numpy as np

from pitch_engine.config.schema import FieldDetectorConfig
from pitch_engine.core.models import DetectionStatus, FieldDetection
from pitch_engine.detectors.base import BaseFieldDetector


class MockFieldDetector(BaseFieldDetector):
    """Deterministic mock detector allowing injection of specific test detection patterns."""

    def __init__(self, config: FieldDetectorConfig):
        self.config = config
        self.default_polygon: List[Tuple[float, float]] = [
            (100.0, 100.0),
            (1180.0, 100.0),
            (1230.0, 620.0),
            (50.0, 620.0),
            (100.0, 100.0),
        ]
        self.mode = config.extra_params.get("mock_mode", "valid")

    def detect(self, frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> FieldDetection:
        if self.mode == "no_pitch":
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.NO_PITCH_VISIBLE,
                confidence=0.0,
                area=0.0,
                intersection_area=0.0,
            )
        elif self.mode == "noise":
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.NOISE,
                confidence=0.1,
                area=200.0,
                intersection_area=200.0,
            )
        elif self.mode == "invalid_geometry":
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.INVALID_GEOMETRY,
                confidence=0.0,
                area=0.0,
                intersection_area=0.0,
            )

        # Default valid mode
        return FieldDetection(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            polygon=self.default_polygon,
            confidence=0.92,
            area=560000.0,
            intersection_area=560000.0,
            status=DetectionStatus.VALID,
        )
