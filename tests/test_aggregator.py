"""Unit tests for boundary aggregation, noise filtering, and crop window derivation (Part 3)."""

import pytest
from shapely.geometry import Polygon
from pitch_engine.config.schema import CropSearchConfig
from pitch_engine.core.models import DetectionStatus, FieldDetection
from pitch_engine.pipeline.aggregator import BoundaryAggregator


def test_aggregator_filters_noise():
    crop_config = CropSearchConfig(aspect_ratio="16:9", padding_px=20)
    aggregator = BoundaryAggregator(crop_config, canvas_width=1280, canvas_height=720)

    # 1. Record valid detection
    poly_pts = [(100.0, 100.0), (1100.0, 100.0), (1100.0, 600.0), (100.0, 600.0), (100.0, 100.0)]
    valid_det = FieldDetection(
        frame_index=0,
        timestamp_seconds=0.0,
        polygon=poly_pts,
        confidence=0.9,
        area=500000.0,
        intersection_area=500000.0,
        status=DetectionStatus.VALID,
    )
    aggregator.record_detection(valid_det, is_scene_cut=False)

    # 2. Record noisy detection (must not pollute consensus boundary)
    noise_det = FieldDetection(
        frame_index=15,
        timestamp_seconds=0.5,
        polygon=[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0), (10.0, 10.0)],
        confidence=0.1,
        area=100.0,
        intersection_area=100.0,
        status=DetectionStatus.NOISE,
    )
    aggregator.record_detection(noise_det, is_scene_cut=False)

    # 3. Record close-up / missing pitch
    missing_det = FieldDetection(
        frame_index=30,
        timestamp_seconds=1.0,
        status=DetectionStatus.NO_PITCH_VISIBLE,
        confidence=0.0,
    )
    aggregator.record_detection(missing_det, is_scene_cut=False)

    assert aggregator.frames_inspected == 3
    assert aggregator.valid_detections_count == 1
    assert aggregator.missed_or_noise_count == 2

    scenes = aggregator.finalize(last_frame_index=30, last_timestamp=1.0)
    assert len(scenes) == 1
    scene = scenes[0]

    assert scene.valid_detections == 1
    assert scene.samples_inspected == 3
    assert scene.pitch_visibility_ratio == pytest.approx(1.0 / 3.0, abs=0.01)
    assert scene.consensus_polygon is not None
    assert scene.recommended_crop is not None
    assert scene.recommended_crop.width > 0
    assert scene.recommended_crop.height > 0


def test_aggregator_scene_transition():
    crop_config = CropSearchConfig(aspect_ratio="16:9")
    aggregator = BoundaryAggregator(crop_config, canvas_width=1280, canvas_height=720)

    # Scene 1: frame 0 and frame 15
    aggregator.record_detection(
        FieldDetection(frame_index=0, timestamp_seconds=0.0, status=DetectionStatus.VALID,
                       polygon=[(50, 50), (1000, 50), (1000, 500), (50, 500), (50, 50)], confidence=0.8),
        is_scene_cut=False
    )
    aggregator.record_detection(
        FieldDetection(frame_index=15, timestamp_seconds=0.5, status=DetectionStatus.VALID,
                       polygon=[(50, 50), (1000, 50), (1000, 500), (50, 500), (50, 50)], confidence=0.85),
        is_scene_cut=False
    )

    # Scene 2 begins at frame 30 (scene cut detected)
    aggregator.record_detection(
        FieldDetection(frame_index=30, timestamp_seconds=1.0, status=DetectionStatus.VALID,
                       polygon=[(200, 200), (800, 200), (800, 600), (200, 600), (200, 200)], confidence=0.75),
        is_scene_cut=True
    )

    scenes = aggregator.finalize(last_frame_index=30, last_timestamp=1.0)
    assert len(scenes) == 2
    assert scenes[0].scene_id == 0
    assert scenes[1].scene_id == 1
