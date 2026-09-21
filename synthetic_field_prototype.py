"""SYNTHETIC RESEARCH PROTOTYPE — FIELD BOUNDARY & CROP DERIVATION (Refactored)
-------------------------------------------------------------------------------
This entrypoint now executes the production-grade `pitch_engine` library
while maintaining backwards compatibility for existing prototype invocations.
"""

import os
from pitch_engine.config.schema import EngineConfig
from pitch_engine.pipeline.engine import PitchBoundaryEngine
from synthetic_generator import generate_synthetic_video

CONFIG = {
    "video_path": "synthetic_pitch_feed.mp4",
    "target_fps": 30.0,
    "confidence_threshold": 0.5,
    "field_detector": {
        "type": "color_threshold",
        "sport": "football",
        "min_area": 1000.0,
    },
    "sampling": {
        "sample_fps": 2.0,
        "enable_scene_cut_detection": True,
        "scene_change_threshold": 0.35,
    },
    "crop_search": {
        "aspect_ratio": "16:9",
        "padding_px": 20,
    },
    "reporting": {
        "enabled": False,  # Disabled by default for local prototype script runs
    },
    "debug_mode": True,
}


def run_pipeline():
    video_path = CONFIG["video_path"]
    if not os.path.exists(video_path):
        print(f"Generating synthetic video: {video_path}...")
        generate_synthetic_video(video_path, numFrames=600)

    # Strictly validate configuration using Pydantic model
    engine_config = EngineConfig.from_dict(CONFIG)

    engine = PitchBoundaryEngine(engine_config)
    summary = engine.run()

    print(
        f"Pipeline finished: inspected {summary.frames_inspected} frames across "
        f"{len(summary.scenes)} scene(s). Valid detections: {summary.valid_detections_count}."
    )
    return summary


if __name__ == "__main__":
    run_pipeline()
