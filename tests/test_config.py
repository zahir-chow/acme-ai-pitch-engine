"""Unit tests for configuration validation and fail-fast behavior (Part 1)."""

import pytest
from pitch_engine.config.schema import (
    CropSearchConfig,
    EngineConfig,
    FieldDetectorConfig,
    PlatformReportingConfig,
    SamplingConfig,
    SportType,
)
from pitch_engine.core.exceptions import ConfigurationError


def test_valid_minimal_config():
    raw = {"video_path": "test_video.mp4"}
    config = EngineConfig.from_dict(raw)

    assert config.video_path == "test_video.mp4"
    assert config.target_fps == 30.0
    assert config.confidence_threshold == 0.5
    assert config.field_detector.sport == SportType.FOOTBALL
    assert config.field_detector.type == "color_threshold"
    assert config.sampling.sample_fps == 2.0
    assert config.crop_search.aspect_ratio == "16:9"
    assert config.crop_search.aspect_ratio_float == pytest.approx(16.0 / 9.0)
    assert config.reporting.fail_on_reporting_error is False


def test_missing_video_path_fails_fast():
    raw = {}
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "Field 'video_path'" in str(exc_info.value)


def test_empty_video_path_fails_fast():
    raw = {"video_path": "   "}
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "video_path" in str(exc_info.value)


def test_invalid_target_fps_fails_fast():
    raw = {"video_path": "test.mp4", "target_fps": -5.0}
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "target_fps" in str(exc_info.value)


def test_invalid_confidence_threshold_fails_fast():
    raw = {"video_path": "test.mp4", "confidence_threshold": 1.5}
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "confidence_threshold" in str(exc_info.value)


def test_invalid_aspect_ratio_format_fails_fast():
    raw = {
        "video_path": "test.mp4",
        "crop_search": {"aspect_ratio": "invalid_ratio"}
    }
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "aspect_ratio" in str(exc_info.value)


def test_invalid_sample_fps_fails_fast():
    raw = {
        "video_path": "test.mp4",
        "sampling": {"sample_fps": -2.0}
    }
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "sample_fps" in str(exc_info.value)


def test_invalid_api_url_fails_fast():
    raw = {
        "video_path": "test.mp4",
        "reporting": {"api_base_url": "ftp://invalid-server"}
    }
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(raw)

    assert "api_base_url" in str(exc_info.value)


def test_non_dict_config_fails():
    with pytest.raises(ConfigurationError) as exc_info:
        EngineConfig.from_dict(["not", "a", "dict"])  # type: ignore

    assert "must be a dictionary" in str(exc_info.value)
