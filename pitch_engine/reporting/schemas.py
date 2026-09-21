"""Validated wire models for communicating with the platform API (mock_api)."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class JobProgressPayload(BaseModel):
    """Payload dispatched to /api/v1/jobs/progress."""

    job_id: str
    progress_percent: float = Field(..., ge=0.0, le=100.0)
    current_frame: int = Field(..., ge=0)
    total_frames: int = Field(..., ge=0)
    elapsed_seconds: float = Field(..., ge=0.0)
    valid_detections: int = Field(..., ge=0)
    status: str = Field(default="processing")


class JobEventPayload(BaseModel):
    """Payload dispatched to /api/v1/jobs/events."""

    job_id: str
    event_type: str = Field(
        ...,
        description="Event category: 'job_started', 'scene_cut', 'job_completed', 'job_failed'"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    payload: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
