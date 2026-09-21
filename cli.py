"""Thin CLI entry point for the Automated Pitch Boundary & Crop Engine (Part 1).

Keeps CLI argument handling, environment variables, and process exit codes separate
from the reusable core library.
"""

import argparse
import json
import os
import sys

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
    PitchEngineError,
    SourceMediaError,
)
from pitch_engine.pipeline.engine import PitchBoundaryEngine
from synthetic_generator import generate_synthetic_video


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrackBox Automated Pitch Boundary & Crop Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--video",
        type=str,
        default=os.environ.get("VIDEO_PATH", "synthetic_pitch_feed.mp4"),
        help="Path to input video file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("CONFIG_PATH"),
        help="Path to JSON configuration file (overrides CLI defaults)",
    )
    parser.add_argument(
        "--mock-api-url",
        type=str,
        default=os.environ.get("MOCK_API_URL", "http://localhost:5000"),
        help="Platform reporting service base URL",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=float(os.environ.get("SAMPLE_FPS", "2.0")),
        help="Processing sample rate in FPS (Part 2 efficiency)",
    )
    parser.add_argument(
        "--sport",
        type=str,
        default=os.environ.get("SPORT", "football"),
        help="Target sport (football, soccer, rugby, etc.)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1000.0,
        help="Minimum boundary contour pixel area",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--disable-reporting",
        action="store_true",
        help="Disable network reporting to mock_api",
    )
    parser.add_argument(
        "--export-summary",
        type=str,
        help="Optional path to export execution summary JSON",
    )
    parser.add_argument(
        "--generate-synthetic-if-missing",
        action="store_true",
        default=True,
        help="Automatically generate synthetic video feed if the input video is missing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode and verbose logging",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EngineConfig:
    """Builds and strictly validates EngineConfig from file or CLI args."""
    if args.config:
        # Load from file if provided
        return EngineConfig.from_file(args.config, validate_file_exists=False)

    # Convert sport string to enum safely or let validator handle error
    sport_val = args.sport.lower()
    try:
        sport_enum = SportType(sport_val)
    except ValueError:
        valid_sports = ", ".join([s.value for s in SportType])
        raise ConfigurationError(
            f"Unsupported sport '{args.sport}'. Supported sports: {valid_sports}"
        )

    # Build config dictionary for validation
    config_data = {
        "video_path": args.video,
        "confidence_threshold": args.confidence_threshold,
        "debug_mode": args.debug,
        "field_detector": {
            "type": "color_threshold",
            "sport": sport_enum.value,
            "min_area": args.min_area,
            "confidence_threshold": args.confidence_threshold,
        },
        "sampling": {
            "sample_fps": args.sample_fps,
            "enable_scene_cut_detection": True,
            "scene_change_threshold": 0.35,
        },
        "crop_search": {
            "aspect_ratio": "16:9",
            "padding_px": 20,
        },
        "reporting": {
            "enabled": not args.disable_reporting,
            "api_base_url": args.mock_api_url,
            "timeout_seconds": 2.0,
            "max_retries": 2,
            "fail_on_reporting_error": False,  # Decoupled network errors
        },
    }

    return EngineConfig.from_dict(config_data)


def main() -> int:
    args = parse_arguments()

    # If synthetic feed requested or default missing, generate offline test feed
    if args.generate_synthetic_if_missing and not os.path.exists(args.video):
        print(f"[CLI] Video file '{args.video}' not found locally. Generating synthetic feed (1800 frames)...")
        generate_synthetic_video(outputPath=args.video, numFrames=1800)
        print(f"[CLI] Synthetic feed successfully written to '{args.video}'.")

    try:
        config = build_config(args)
    except ConfigurationError as e:
        print(f"\n[FATAL CONFIGURATION ERROR]\n{e}\n", file=sys.stderr)
        return 1

    try:
        engine = PitchBoundaryEngine(config)
        summary = engine.run()

        if args.export_summary:
            with open(args.export_summary, "w", encoding="utf-8") as f:
                json.dump(summary.model_dump(), f, indent=2)
            print(f"[CLI] Execution summary exported to: {args.export_summary}")

        return 0

    except SourceMediaError as e:
        print(f"\n[FATAL MEDIA ERROR]\n{e}\n", file=sys.stderr)
        return 2
    except PitchEngineError as e:
        print(f"\n[FATAL PIPELINE ERROR]\n{e}\n", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"\n[UNEXPECTED SYSTEM ERROR]\n{e}\n", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
