"""End-to-end integration test running the engine against synthetic video feed."""

import os
import tempfile
import pytest
from pitch_engine.config.schema import EngineConfig
from pitch_engine.pipeline.engine import PitchBoundaryEngine
from synthetic_generator import generate_synthetic_video


@pytest.fixture(scope="module")
def sample_video():
    """Generates a small synthetic video feed for end-to-end testing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    try:
        generate_synthetic_video(outputPath=tmp_path, numFrames=60, imgWidth=640, imgHeight=360)
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def test_full_pipeline_run(sample_video):
    config_data = {
        "video_path": sample_video,
        "target_fps": 30.0,
        "confidence_threshold": 0.5,
        "field_detector": {
            "type": "color_threshold",
            "sport": "football",
            "min_area": 500.0,
        },
        "sampling": {
            "sample_fps": 3.0,  # 3 fps on 30 fps video -> stride = 10 -> inspects ~6 frames
            "enable_scene_cut_detection": True,
        },
        "crop_search": {
            "aspect_ratio": "16:9",
            "padding_px": 10,
        },
        "reporting": {
            "enabled": False,  # Offline for unit tests
        },
        "debug_mode": True,
    }

    config = EngineConfig.from_dict(config_data)
    engine = PitchBoundaryEngine(config)
    summary = engine.run()

    assert summary.status == "completed"
    assert summary.total_frames_in_video == 60
    assert summary.frames_inspected == 6  # Exact stride 10 on 60 frames
    assert summary.valid_detections_count > 0
    assert len(summary.scenes) >= 1

    scene = summary.scenes[0]
    assert scene.consensus_polygon is not None
    assert scene.recommended_crop is not None
    assert scene.recommended_crop.width > 0
    assert scene.recommended_crop.height > 0
    assert summary.processing_time_seconds > 0
