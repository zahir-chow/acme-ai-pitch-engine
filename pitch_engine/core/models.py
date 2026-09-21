"""Core domain models and value objects for pitch boundary detection and aggregation."""

from enum import Enum
from typing import Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class DetectionStatus(str, Enum):
    """Classification status of a field boundary detection in a given frame."""

    VALID = "valid"
    NO_PITCH_VISIBLE = "no_pitch_visible"  # Broadcast close-ups, audience, or camera cutaways
    NOISE = "noise"                        # Transient detection noise or below min_area
    INVALID_GEOMETRY = "invalid_geometry"  # Malformed polygon (self-intersecting or < 3 points)
    SKIPPED_FRAME = "skipped_frame"        # Non-sampled frame or stationary scene reuse


class FieldDetection(BaseModel):
    """Detection result for an individual inspected frame."""

    frame_index: int = Field(..., ge=0, description="0-indexed frame number in the video")
    timestamp_seconds: float = Field(..., ge=0.0, description="Video timestamp in seconds")
    polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Vertices of the detected boundary polygon [(x1, y1), (x2, y2), ...]"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence score")
    area: float = Field(default=0.0, ge=0.0, description="Pixel area of the detected boundary")
    intersection_area: float = Field(
        default=0.0,
        ge=0.0,
        description="Area of intersection between polygon and outer video frame"
    )
    status: DetectionStatus = Field(
        default=DetectionStatus.VALID,
        description="Classification of the detection"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Detector-specific diagnostics")

    @property
    def is_valid(self) -> bool:
        return self.status == DetectionStatus.VALID and self.polygon is not None and len(self.polygon) >= 3


class CropBox(BaseModel):
    """Recommended camera crop window (x, y, width, height)."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class AggregatedPitchBoundary(BaseModel):
    """Aggregated pitch boundary and crop recommendation for a continuous scene segment."""

    scene_id: int = Field(..., ge=0)
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)
    start_timestamp: float = Field(..., ge=0.0)
    end_timestamp: float = Field(..., ge=0.0)
    samples_inspected: int = Field(..., ge=0)
    valid_detections: int = Field(..., ge=0)
    consensus_polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Temporally smoothed / consensus pitch boundary polygon"
    )
    recommended_crop: Optional[CropBox] = Field(
        default=None,
        description="Derived camera crop rectangle"
    )
    average_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    pitch_visibility_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ratio of inspected frames where a valid pitch was detected"
    )


class PipelineExecutionSummary(BaseModel):
    """Comprehensive execution summary and metrics for an entire video analysis run."""

    job_id: str
    video_path: str
    total_frames_in_video: int = Field(..., ge=0)
    frames_inspected: int = Field(..., ge=0)
    valid_detections_count: int = Field(..., ge=0)
    missed_or_noise_count: int = Field(..., ge=0)
    scene_cuts_detected: int = Field(..., ge=0)
    processing_time_seconds: float = Field(..., ge=0.0)
    processing_fps: float = Field(..., ge=0.0)
    pitch_coverage_ratio: float = Field(..., ge=0.0, le=1.0)
    status: str = "completed"
    scenes: List[AggregatedPitchBoundary] = Field(default_factory=list)
