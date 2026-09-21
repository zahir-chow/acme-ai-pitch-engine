"""Factory and registry for field boundary detectors."""

from typing import Dict, Type
from pitch_engine.config.schema import FieldDetectorConfig
from pitch_engine.core.exceptions import ConfigurationError
from pitch_engine.detectors.base import BaseFieldDetector
from pitch_engine.detectors.color_threshold import ColorThresholdDetector
from pitch_engine.detectors.mock import MockFieldDetector

_REGISTRY: Dict[str, Type[BaseFieldDetector]] = {
    "color_threshold": ColorThresholdDetector,
    "sam_mask_v1": ColorThresholdDetector,  # Compatible alias for prototype config
    "mock": MockFieldDetector,
}


def register_detector(name: str, detector_cls: Type[BaseFieldDetector]) -> None:
    """Registers a new field detector implementation into the factory."""
    if not issubclass(detector_cls, BaseFieldDetector):
        raise TypeError(f"Detector class '{detector_cls.__name__}' must inherit from BaseFieldDetector.")
    _REGISTRY[name.strip().lower()] = detector_cls


def create_detector(config: FieldDetectorConfig) -> BaseFieldDetector:
    """Instantiates a field detector based on validated configuration.

    Raises:
        ConfigurationError: If the configured detector type is not registered.
    """
    det_type = config.type.strip().lower()
    if det_type not in _REGISTRY:
        available = ", ".join(f"'{k}'" for k in _REGISTRY.keys())
        raise ConfigurationError(
            f"Unknown field detector type: '{det_type}'. Registered detector types: {available}",
            details={"configured_type": det_type, "available_types": list(_REGISTRY.keys())}
        )

    detector_cls = _REGISTRY[det_type]
    try:
        return detector_cls(config)
    except TypeError:
        return detector_cls()
