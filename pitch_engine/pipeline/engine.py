"""Core pipeline engine orchestrating video ingestion, sampling, detection, and reporting."""

import logging
import os
import time
from typing import Optional
import cv2

from pitch_engine.config.schema import EngineConfig
from pitch_engine.core.exceptions import (
    PitchEngineError,
    SourceMediaError,
)
from pitch_engine.core.models import PipelineExecutionSummary
from pitch_engine.detectors.factory import create_detector
from pitch_engine.logging_utils import configure_logging
from pitch_engine.pipeline.aggregator import BoundaryAggregator
from pitch_engine.pipeline.sampler import AdaptiveSampler
from pitch_engine.reporting.client import PlatformReportingClient
from pitch_engine.reporting.schemas import JobEventPayload, JobProgressPayload


class PitchBoundaryEngine:
    """Production pipeline engine for automated pitch boundary detection and crop derivation."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.logger = configure_logging(
            job_id=config.job_id,
            debug=config.debug_mode
        )
        self.detector = create_detector(config.field_detector)
        self.reporter = PlatformReportingClient(config.reporting)

    def run(self) -> PipelineExecutionSummary:
        """Executes the analysis pipeline on the configured video.

        Returns:
            PipelineExecutionSummary containing metrics, aggregated boundaries, and status.

        Raises:
            SourceMediaError: If the video media cannot be opened or is missing.
            PitchEngineError: If an unrecoverable engine failure occurs.
        """
        start_time = time.time()
        video_path = self.config.video_path

        self.logger.info(f"Initiating Pitch Boundary Engine for video: '{video_path}' [Job: {self.config.job_id}]")

        # Verify video source exists
        if not os.path.exists(video_path):
            err_msg = f"Input video file does not exist at: '{video_path}'"
            self._notify_failure(err_msg)
            raise SourceMediaError(err_msg, details={"video_path": video_path})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            err_msg = f"Failed to open video stream. Invalid or corrupt media: '{video_path}'"
            self._notify_failure(err_msg)
            raise SourceMediaError(err_msg, details={"video_path": video_path})

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or self.config.target_fps
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0 or frame_width <= 0 or frame_height <= 0:
            cap.release()
            err_msg = f"Invalid video stream properties: frames={total_frames}, res=({frame_width}x{frame_height})"
            self._notify_failure(err_msg)
            raise SourceMediaError(err_msg, details={"video_path": video_path})

        self.logger.info(
            f"Video metadata verified: {total_frames} frames, {video_fps:.2f} fps, {frame_width}x{frame_height}"
        )

        # Initialize efficiency components
        sampler = AdaptiveSampler(self.config.sampling, video_fps)
        aggregator = BoundaryAggregator(self.config.crop_search, frame_width, frame_height)

        self.logger.info(
            f"Adaptive sampling initialized with stride={sampler.stride} "
            f"(inspecting ~{total_frames // sampler.stride} keyframes instead of all {total_frames})"
        )

        # Report job started event
        self.reporter.report_event(JobEventPayload(
            job_id=self.config.job_id,
            event_type="job_started",
            payload={
                "video_path": video_path,
                "total_frames": total_frames,
                "video_fps": video_fps,
                "resolution": f"{frame_width}x{frame_height}",
                "sampling_stride": sampler.stride,
            }
        ))

        current_frame_idx = 0
        last_progress_report_frame = 0
        consecutive_decode_failures = 0

        try:
            while current_frame_idx < total_frames:
                # Direct seek or fast-forward to sampled frame
                if sampler.stride > 1 and current_frame_idx > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_decode_failures += 1
                    if consecutive_decode_failures > 10:
                        self.logger.warning(
                            f"Reached end of stream or encountered persistent decode errors at frame {current_frame_idx}"
                        )
                        break
                    current_frame_idx += 1
                    continue

                consecutive_decode_failures = 0
                timestamp_sec = current_frame_idx / video_fps

                # 1. Fast scene change check
                is_cut, cut_metric = sampler.compute_scene_difference(frame)
                if is_cut and current_frame_idx > 0:
                    self.logger.debug(
                        f"Scene cut detected at frame {current_frame_idx} (diff metric={cut_metric:.3f})"
                    )

                # 2. Field boundary detection (pluggable seam)
                detection = self.detector.detect(frame, current_frame_idx, timestamp_sec)

                # 3. Aggregation & noise filtering
                aggregator.record_detection(detection, is_cut)

                # 4. Periodic progress reporting to platform
                if current_frame_idx - last_progress_report_frame >= self.config.reporting.heartbeat_interval_frames:
                    pct = round((current_frame_idx / total_frames) * 100.0, 1)
                    self.reporter.report_progress(JobProgressPayload(
                        job_id=self.config.job_id,
                        progress_percent=pct,
                        current_frame=current_frame_idx,
                        total_frames=total_frames,
                        elapsed_seconds=round(time.time() - start_time, 2),
                        valid_detections=aggregator.valid_detections_count,
                        status="processing"
                    ))
                    last_progress_report_frame = current_frame_idx

                # Advance by stride
                current_frame_idx = sampler.calculate_next_sample_frame(current_frame_idx)

        except Exception as e:
            cap.release()
            err_msg = f"Fatal pipeline execution failure: {e}"
            self.logger.error(err_msg, exc_info=True)
            self._notify_failure(err_msg)
            raise PitchEngineError(err_msg) from e

        finally:
            cap.release()

        # Finalize scenes
        elapsed_sec = max(0.001, time.time() - start_time)
        effective_processing_fps = round(aggregator.frames_inspected / elapsed_sec, 2)
        total_throughput_fps = round(total_frames / elapsed_sec, 2)

        last_timestamp = current_frame_idx / video_fps
        scenes = aggregator.finalize(current_frame_idx, last_timestamp)

        coverage_ratio = (
            round(aggregator.valid_detections_count / aggregator.frames_inspected, 3)
            if aggregator.frames_inspected > 0
            else 0.0
        )

        summary = PipelineExecutionSummary(
            job_id=self.config.job_id,
            video_path=video_path,
            total_frames_in_video=total_frames,
            frames_inspected=aggregator.frames_inspected,
            valid_detections_count=aggregator.valid_detections_count,
            missed_or_noise_count=aggregator.missed_or_noise_count,
            scene_cuts_detected=aggregator.scene_cuts_count,
            processing_time_seconds=round(elapsed_sec, 3),
            processing_fps=effective_processing_fps,
            pitch_coverage_ratio=coverage_ratio,
            status="completed",
            scenes=scenes
        )

        self.logger.info(
            f"Pipeline run completed successfully in {elapsed_sec:.2f}s "
            f"({aggregator.frames_inspected} sampled frames inspected, "
            f"effective {total_throughput_fps:.1f} equivalent source fps)"
        )
        self.logger.info(
            f"Found {aggregator.valid_detections_count} valid boundaries across {len(scenes)} scene(s). "
            f"Noise/cutaway frames dropped: {aggregator.missed_or_noise_count}."
        )

        # Final 100% progress report and completion event
        self.reporter.report_progress(JobProgressPayload(
            job_id=self.config.job_id,
            progress_percent=100.0,
            current_frame=total_frames,
            total_frames=total_frames,
            elapsed_seconds=round(elapsed_sec, 2),
            valid_detections=aggregator.valid_detections_count,
            status="completed"
        ))

        self.reporter.report_event(JobEventPayload(
            job_id=self.config.job_id,
            event_type="job_completed",
            payload={
                "summary": summary.model_dump(),
                "scene_count": len(scenes)
            }
        ))

        self.reporter.close()
        return summary

    def _notify_failure(self, error_message: str) -> None:
        """Attempts to inform the platform service of a fatal pipeline failure."""
        try:
            self.reporter.report_event(JobEventPayload(
                job_id=self.config.job_id,
                event_type="job_failed",
                error_message=error_message,
                payload={"error": error_message}
            ))
        except Exception:
            pass
