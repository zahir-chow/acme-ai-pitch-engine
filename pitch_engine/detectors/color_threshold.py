"""Color-thresholding field detector implementation."""

from typing import Optional, Tuple
import cv2
import numpy as np

from pitch_engine.config.schema import FieldDetectorConfig, SportType
from pitch_engine.core.geometry import (
    calculate_frame_intersection_area,
    polygon_to_coords,
    sanitize_and_simplify_polygon,
)
from pitch_engine.core.models import DetectionStatus, FieldDetection
from pitch_engine.detectors.base import BaseFieldDetector


class ColorThresholdDetector(BaseFieldDetector):
    """Field detector utilizing HSV color thresholding to segment playing surface.

    Suitable for sports played on green grass/turf (football, soccer, rugby).
    """

    def __init__(self, config: FieldDetectorConfig):
        self.config = config
        self.sport = config.sport
        self.min_area = config.min_area
        self.confidence_threshold = config.confidence_threshold

        # Configure color range based on sport
        self.lower_bound, self.upper_bound = self._get_hsv_bounds(config.sport, config.extra_params)

    @staticmethod
    def _get_hsv_bounds(sport: SportType, extra_params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower and upper HSV bounds for the specified sport's playing field."""
        if "lower_hsv" in extra_params and "upper_hsv" in extra_params:
            return np.array(extra_params["lower_hsv"]), np.array(extra_params["upper_hsv"])

        if sport in (SportType.FOOTBALL, SportType.SOCCER, SportType.RUGBY):
            # Standard green grass/turf range
            return np.array([35, 40, 40]), np.array([85, 255, 255])
        elif sport == SportType.BASKETBALL:
            # Hardwood court (yellowish/orange brown)
            return np.array([10, 50, 50]), np.array([25, 255, 255])
        elif sport == SportType.HOCKEY:
            # White / light blue ice
            return np.array([80, 0, 180]), np.array([140, 50, 255])
        else:
            return np.array([35, 40, 40]), np.array([85, 255, 255])

    def detect(self, frame: np.ndarray, frame_index: int, timestamp_seconds: float) -> FieldDetection:
        h, w = frame.shape[:2]

        # Fast check: completely black frame (camera cut / offline feed)
        if frame.max() == 0:
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.NO_PITCH_VISIBLE,
                confidence=0.0,
                area=0.0,
                intersection_area=0.0,
                metadata={"reason": "black_frame"}
            )

        # Convert to HSV and generate mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)

        # Check for pitch presence
        green_pixel_count = cv2.countNonZero(mask)
        total_pixels = h * w
        green_ratio = green_pixel_count / total_pixels

        if green_ratio < 0.05:
            # Less than 5% pitch color -> Close-up on player or crowd
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.NO_PITCH_VISIBLE,
                confidence=0.0,
                area=0.0,
                intersection_area=0.0,
                metadata={"green_ratio": green_ratio, "reason": "insufficient_pitch_pixels"}
            )

        # Find external contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.NO_PITCH_VISIBLE,
                confidence=0.0,
                area=0.0,
                intersection_area=0.0,
                metadata={"reason": "no_contours_found"}
            )

        largest = max(contours, key=cv2.contourArea)
        raw_area = float(cv2.contourArea(largest))

        # Check if contour is too small (momentary detection noise)
        if raw_area < self.min_area:
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.NOISE,
                confidence=0.1,
                area=raw_area,
                intersection_area=0.0,
                metadata={"reason": "below_min_area", "raw_area": raw_area}
            )

        # Sanitize and simplify polygon
        poly = sanitize_and_simplify_polygon(largest, min_points=3, simplify_tolerance=2.0)
        if poly is None or not poly.is_valid:
            return FieldDetection(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                status=DetectionStatus.INVALID_GEOMETRY,
                confidence=0.0,
                area=raw_area,
                intersection_area=0.0,
                metadata={"reason": "invalid_polygon_geometry"}
            )

        poly_area = float(poly.area)
        intersection_area = calculate_frame_intersection_area(poly, w, h)

        # Compute confidence based on plausible field coverage (15% to 100% of frame)
        coverage_ratio = intersection_area / total_pixels
        if 0.15 <= coverage_ratio <= 1.0:
            confidence = min(0.95, 0.70 + (coverage_ratio * 0.25))
        else:
            confidence = max(0.1, coverage_ratio * 2.0)

        status = DetectionStatus.VALID if confidence >= self.confidence_threshold else DetectionStatus.NOISE

        return FieldDetection(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            polygon=polygon_to_coords(poly),
            confidence=round(confidence, 3),
            area=round(poly_area, 2),
            intersection_area=round(intersection_area, 2),
            status=status,
            metadata={
                "coverage_ratio": round(coverage_ratio, 4),
                "vertex_count": len(poly.exterior.coords) if poly else 0
            }
        )
