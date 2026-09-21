"""Strictly validated configuration models for the Pitch Boundary Engine using Pydantic v2.

Fails immediately at load time with clear, structured error messages upon encountering
bad, missing, or out-of-range configuration parameters.
"""

from enum import Enum
import json
import os
from typing import Any, Dict, Optional
import uuid
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from pitch_engine.core.exceptions import ConfigurationError


class SportType(str, Enum):
    """Supported sports for pitch boundary detection."""

    FOOTBALL = "football"
    SOCCER = "soccer"
    RUGBY = "rugby"
    BASKETBALL = "basketball"
    HOCKEY = "hockey"


class FieldDetectorConfig(BaseModel):
    """Configuration for the field boundary detector."""

    type: str = Field(
        default="color_threshold",
        description="Detector identifier (e.g. 'color_threshold', 'sam_mask_v1', 'mock')"
    )
    sport: SportType = Field(
        default=SportType.FOOTBALL,
        description="Target sport determining field color space and geometry characteristics"
    )
    min_area: float = Field(
        default=1000.0,
        gt=0.0,
        description="Minimum contour pixel area to be considered a plausible field boundary"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum detector confidence threshold"
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific tuning hyperparameters"
    )

    @field_validator("type")
    @classmethod
    def validate_type_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field detector 'type' cannot be empty or whitespace.")
        return v.strip().lower()


class SamplingConfig(BaseModel):
    """Controls frame sampling efficiency and scene cut detection (Part 2)."""

    sample_fps: Optional[float] = Field(
        default=2.0,
        description="Inspection rate in FPS (e.g., 2.0 fps inspects 2 frames per second regardless of source FPS)"
    )
    sample_interval_frames: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional fixed frame stride (e.g., inspect every Nth frame). Overrides sample_fps if set."
    )
    enable_scene_cut_detection: bool = Field(
        default=True,
        description="Whether to perform lightweight scene cut detection between sampled frames"
    )
    scene_change_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Histogram distance threshold to classify a camera cut / angle change"
    )
    min_scene_frames: int = Field(
        default=5,
        ge=1,
        description="Minimum consecutive frames to form a distinct scene"
    )

    @field_validator("sample_fps")
    @classmethod
    def validate_sample_fps(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("sample_fps must be strictly greater than 0.0")
        return v


class CropSearchConfig(BaseModel):
    """Parameters for deriving camera crop rectangles from detected boundaries."""

    aspect_ratio: str = Field(
        default="16:9",
        description="Target aspect ratio formatted as 'WIDTH:HEIGHT' (e.g., '16:9', '4:3', '1:1')"
    )
    padding_px: int = Field(
        default=20,
        ge=0,
        description="Safe padding pixels applied around the boundary bounding box"
    )

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio_format(cls, v: str) -> str:
        parts = v.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid aspect_ratio '{v}'. Expected format 'W:H' (e.g., '16:9').")
        try:
            w, h = float(parts[0]), float(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError()
        except Exception:
            raise ValueError(f"aspect_ratio values must be positive numbers. Got '{v}'.")
        return v.strip()

    @property
    def aspect_ratio_float(self) -> float:
        parts = self.aspect_ratio.split(":")
        return float(parts[0]) / float(parts[1])


class PlatformReportingConfig(BaseModel):
    """Configuration for communicating with the platform reporting service (mock_api)."""

    enabled: bool = Field(
        default=True,
        description="Whether to report progress and events to the platform service"
    )
    api_base_url: str = Field(
        default="http://localhost:5000",
        description="Base URL of the mock_api or production orchestrator API"
    )
    progress_endpoint: str = Field(default="/api/v1/jobs/progress")
    events_endpoint: str = Field(default="/api/v1/jobs/events")
    heartbeat_interval_frames: int = Field(
        default=30,
        ge=1,
        description="How often to dispatch progress reports (in processed video frames)"
    )
    timeout_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="HTTP network request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts on transient network errors"
    )
    fail_on_reporting_error: bool = Field(
        default=False,
        description="If True, network reporting failures abort the pipeline. If False (recommended), "
                    "reporting errors are logged and decoupled from core video analysis."
    )

    @field_validator("api_base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"api_base_url must start with http:// or https://. Got '{v}'")
        return v


class EngineConfig(BaseModel):
    """Root configuration model for the Automated Pitch Boundary & Crop Engine."""

    job_id: str = Field(
        default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}",
        description="Unique identifier for this processing job"
    )
    video_path: str = Field(
        ...,
        description="Path to the video file to be analyzed"
    )
    target_fps: float = Field(
        default=30.0,
        gt=0.0,
        description="Nominal video FPS"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Global detection confidence threshold"
    )
    field_detector: FieldDetectorConfig = Field(
        default_factory=FieldDetectorConfig,
        description="Field detection algorithm configuration"
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Frame sampling and processing efficiency tuning"
    )
    crop_search: CropSearchConfig = Field(
        default_factory=CropSearchConfig,
        description="Downstream crop window derivation settings"
    )
    reporting: PlatformReportingConfig = Field(
        default_factory=PlatformReportingConfig,
        description="Platform orchestrator reporting settings"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable verbose debug logging and diagnostics"
    )

    @field_validator("video_path")
    @classmethod
    def validate_video_path(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("video_path must be a non-empty string path.")
        return v.strip()

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], validate_file_exists: bool = False) -> "EngineConfig":
        """Constructs and strictly validates an EngineConfig from a dictionary.

        Raises:
            ConfigurationError: If any field fails validation, providing a clean human-readable explanation.
        """
        if not isinstance(raw, dict):
            raise ConfigurationError(
                f"Configuration must be a dictionary/object, received: {type(raw).__name__}"
            )

        try:
            config = cls(**raw)
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = " -> ".join(str(item) for item in err["loc"])
                msg = err["msg"]
                errors.append(f"Field '{loc}': {msg}")
            error_details = "\n  - " + "\n  - ".join(errors)
            raise ConfigurationError(
                f"Configuration validation failed with {len(errors)} error(s):{error_details}",
                details={"errors": e.errors()}
            ) from e

        if validate_file_exists and not os.path.exists(config.video_path):
            raise ConfigurationError(
                f"Configured video file does not exist at: '{config.video_path}'",
                details={"video_path": config.video_path}
            )

        return config

    @classmethod
    def from_file(cls, file_path: str, validate_file_exists: bool = False) -> "EngineConfig":
        """Loads and validates configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Configuration file not found: '{file_path}'")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to read/parse configuration file '{file_path}': {e}") from e

        return cls.from_dict(data, validate_file_exists=validate_file_exists)
