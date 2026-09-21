"""Adaptive frame sampler and fast scene change detector.

Ensures pipeline execution time scales with the content that actually needs
inspection rather than the raw file length (Part 2).
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from pitch_engine.config.schema import SamplingConfig


class AdaptiveSampler:
    """Manages frame sampling cadence, fast-forwarding, and lightweight camera cut detection."""

    def __init__(self, config: SamplingConfig, video_fps: float):
        self.config = config
        self.video_fps = max(1.0, video_fps)

        # Compute sampling stride (number of video frames between inspections)
        if config.sample_interval_frames is not None:
            self.stride = max(1, config.sample_interval_frames)
        elif config.sample_fps is not None and config.sample_fps > 0:
            self.stride = max(1, int(round(self.video_fps / config.sample_fps)))
        else:
            self.stride = 1

        self.enable_scene_cuts = config.enable_scene_cut_detection
        self.scene_threshold = config.scene_change_threshold

        self._last_sample_hist: Optional[np.ndarray] = None
        self._last_thumbnail: Optional[np.ndarray] = None

    def calculate_next_sample_frame(self, current_frame_index: int) -> int:
        """Returns the frame index of the next frame to be inspected."""
        return current_frame_index + self.stride

    def compute_scene_difference(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Performs an O(1) downsampled histogram comparison to detect camera cuts.

        Args:
            frame: Full-resolution BGR frame.

        Returns:
            Tuple of (is_scene_cut: bool, difference_metric: float in [0.0, 1.0]).
        """
        if not self.enable_scene_cuts:
            return False, 0.0

        # Downsample to a tiny 64x36 thumbnail for instantaneous computation
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_NEAREST)

        # Compute fast 2D HSV histogram (Hue & Saturation)
        hsv_small = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv_small], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if self._last_sample_hist is None:
            self._last_sample_hist = hist
            self._last_thumbnail = small
            return True, 1.0  # First frame is always the start of a scene

        # Bhattacharyya distance: 0.0 (identical) to 1.0 (completely distinct)
        diff = float(cv2.compareHist(self._last_sample_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        diff = max(0.0, min(1.0, diff))

        is_cut = diff >= self.scene_threshold

        # Update historical state
        self._last_sample_hist = hist
        self._last_thumbnail = small

        return is_cut, round(diff, 4)

    def reset(self) -> None:
        """Resets temporal tracking state."""
        self._last_sample_hist = None
        self._last_thumbnail = None
