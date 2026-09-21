"""Unit tests for processing efficiency, frame sampling, and geometry caching (Part 2)."""

import numpy as np
import pytest
from shapely.geometry import Polygon
from pitch_engine.config.schema import SamplingConfig
from pitch_engine.core.geometry import (
    calculate_frame_intersection_area,
    get_cached_frame_boundary,
    sanitize_and_simplify_polygon,
)
from pitch_engine.pipeline.sampler import AdaptiveSampler


def test_sampling_stride_calculation():
    # 30 fps video sampled at 2 fps -> stride = 15
    config = SamplingConfig(sample_fps=2.0)
    sampler = AdaptiveSampler(config, video_fps=30.0)
    assert sampler.stride == 15
    assert sampler.calculate_next_sample_frame(0) == 15
    assert sampler.calculate_next_sample_frame(15) == 30

    # 60 fps video sampled at 5 fps -> stride = 12
    config_60 = SamplingConfig(sample_fps=5.0)
    sampler_60 = AdaptiveSampler(config_60, video_fps=60.0)
    assert sampler_60.stride == 12

    # Fixed frame stride override
    config_interval = SamplingConfig(sample_interval_frames=10)
    sampler_interval = AdaptiveSampler(config_interval, video_fps=30.0)
    assert sampler_interval.stride == 10


def test_scene_cut_detection():
    config = SamplingConfig(enable_scene_cut_detection=True, scene_change_threshold=0.35)
    sampler = AdaptiveSampler(config, video_fps=30.0)

    # Frame 1: green field
    green_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    green_frame[:] = (34, 139, 34)
    is_cut, score = sampler.compute_scene_difference(green_frame)
    assert is_cut is True  # First frame is always the start of a scene

    # Frame 2: nearly identical green field (smooth broadcast shot)
    green_frame_2 = green_frame.copy()
    green_frame_2[0:10, 0:10] = (255, 255, 255)  # Minor difference
    is_cut_2, score_2 = sampler.compute_scene_difference(green_frame_2)
    assert is_cut_2 is False
    assert score_2 < 0.35

    # Frame 3: camera cut to black or crowd
    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    is_cut_3, score_3 = sampler.compute_scene_difference(black_frame)
    assert is_cut_3 is True
    assert score_3 >= 0.35


def test_geometry_caching():
    # Cache must return the exact same object reference for identical dimensions
    poly1 = get_cached_frame_boundary(1280, 720)
    poly2 = get_cached_frame_boundary(1280, 720)
    assert poly1 is poly2
    assert poly1.area == 1280 * 720


def test_polygon_simplification():
    # Construct a high-vertex count polygon (100 vertices along a circle)
    theta = np.linspace(0, 2 * np.pi, 100)
    x = 640 + 200 * np.cos(theta)
    y = 360 + 200 * np.sin(theta)
    pts = np.column_stack([x, y])

    raw_poly = Polygon(pts)
    assert len(raw_poly.exterior.coords) > 50

    simplified = sanitize_and_simplify_polygon(pts, simplify_tolerance=5.0)
    assert simplified is not None
    assert simplified.is_valid
    # Vertex count should be reduced significantly
    assert len(simplified.exterior.coords) < len(raw_poly.exterior.coords)
