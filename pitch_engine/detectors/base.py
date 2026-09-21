"""Abstract base class establishing the field boundary detection seam."""

from abc import ABC, abstractmethod
import numpy as np

from pitch_engine.core.models import FieldDetection


class BaseFieldDetector(ABC):
    """Abstract interface defining the field-detection seam.

    Allows plugging in different detection implementations (color thresholding,
    SAM segmentation, Hough line models, sport-specific models) without modifying
    the core pipeline engine.
    """

    def __init__(self, config=None):
        self.config = config

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> FieldDetection:
        """Analyzes a video frame and detects the playing field boundary polygon.

        Args:
            frame: BGR numpy image array (H, W, 3).
            frame_index: 0-indexed frame number.
            timestamp_seconds: Video timestamp in seconds.

        Returns:
            FieldDetection containing the detected polygon, status, area, and metrics.
        """
        pass
