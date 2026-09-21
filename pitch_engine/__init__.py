"""Pitch Boundary & Crop Engine - Production Video Analytics Library."""

from pitch_engine.config.schema import (
    CropSearchConfig,
    EngineConfig,
    FieldDetectorConfig,
    PlatformReportingConfig,
    SamplingConfig,
    SportType,
)
from pitch_engine.core.exceptions import (
    ConfigurationError,
    DetectorExecutionError,
    FrameDecodeError,
    PitchEngineError,
    PlatformReportingError,
    SourceMediaError,
)
from pitch_engine.core.models import (
    AggregatedPitchBoundary,
    CropBox,
    DetectionStatus,
    FieldDetection,
    PipelineExecutionSummary,
)
from pitch_engine.detectors.base import BaseFieldDetector
from pitch_engine.pipeline.engine import PitchBoundaryEngine

__version__ = "1.0.0"

__all__ = [
    "PitchBoundaryEngine",
    "EngineConfig",
    "FieldDetectorConfig",
    "SamplingConfig",
    "CropSearchConfig",
    "PlatformReportingConfig",
    "SportType",
    "BaseFieldDetector",
    "FieldDetection",
    "DetectionStatus",
    "CropBox",
    "AggregatedPitchBoundary",
    "PipelineExecutionSummary",
    "PitchEngineError",
    "ConfigurationError",
    "SourceMediaError",
    "FrameDecodeError",
    "DetectorExecutionError",
    "PlatformReportingError",
]
