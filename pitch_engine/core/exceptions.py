"""Domain exception hierarchy for the Pitch Boundary & Crop Engine.

Distinguishes between fatal errors (which must halt the pipeline immediately)
and recoverable or decoupled errors (which should be logged or handled gracefully).
"""

from typing import Optional


class PitchEngineError(Exception):
    """Base exception for all domain errors within pitch_engine."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details={self.details})"
        return self.message


class ConfigurationError(PitchEngineError):
    """Raised when configuration is missing, malformed, or fails strict validation.

    FATAL: Pipeline must fail immediately at load time.
    """


class SourceMediaError(PitchEngineError):
    """Raised when the video file cannot be opened, is missing, or contains unreadable stream headers.

    FATAL: Pipeline cannot proceed without valid input media.
    """


class FrameDecodeError(PitchEngineError):
    """Raised when an individual frame cannot be decoded or read from the stream.

    NON-FATAL: Recoverable; the pipeline should log and skip the corrupted frame.
    """


class DetectorExecutionError(PitchEngineError):
    """Raised when a field detector encounters an unexpected runtime error."""


class PlatformReportingError(PitchEngineError):
    """Raised when communicating with the platform reporting service (mock_api) fails.

    DECOUPLED: By default, platform reporting errors do not crash the core video analysis pipeline.
    """
