"""Core models, geometry utilities, and domain exceptions."""

from pitch_engine.core.exceptions import (
    ConfigurationError,
    DetectorExecutionError,
    FrameDecodeError,
    PitchEngineError,
    PlatformReportingError,
    SourceMediaError,
)
from pitch_engine.core.geometry import (
    calculate_frame_intersection_area,
    derive_crop_window,
    get_cached_frame_boundary,
    polygon_to_coords,
    sanitize_and_simplify_polygon,
)
from pitch_engine.core.models import (
    AggregatedPitchBoundary,
    CropBox,
    DetectionStatus,
    FieldDetection,
    PipelineExecutionSummary,
)

__all__ = [
    "PitchEngineError",
    "ConfigurationError",
    "SourceMediaError",
    "FrameDecodeError",
    "DetectorExecutionError",
    "PlatformReportingError",
    "get_cached_frame_boundary",
    "sanitize_and_simplify_polygon",
    "calculate_frame_intersection_area",
    "derive_crop_window",
    "polygon_to_coords",
    "DetectionStatus",
    "FieldDetection",
    "CropBox",
    "AggregatedPitchBoundary",
    "PipelineExecutionSummary",
]
