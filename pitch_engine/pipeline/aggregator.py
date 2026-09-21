"""Boundary aggregation, noise filtering, and temporal smoothing across video scenes."""

from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon

from pitch_engine.config.schema import CropSearchConfig
from pitch_engine.core.geometry import derive_crop_window, sanitize_and_simplify_polygon
from pitch_engine.core.models import (
    AggregatedPitchBoundary,
    DetectionStatus,
    FieldDetection,
)


class BoundaryAggregator:
    """Aggregates valid frame detections into scene-level pitch boundaries while rejecting noise."""

    def __init__(self, crop_config: CropSearchConfig, canvas_width: int, canvas_height: int):
        self.crop_config = crop_config
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.completed_scenes: List[AggregatedPitchBoundary] = []
        self._current_scene_id = 0
        self._current_scene_detections: List[FieldDetection] = []
        self._scene_start_frame = 0
        self._scene_start_timestamp = 0.0

        # Run-level metric counters
        self.total_frames_seen = 0
        self.frames_inspected = 0
        self.valid_detections_count = 0
        self.missed_or_noise_count = 0
        self.scene_cuts_count = 0

    def record_detection(self, detection: FieldDetection, is_scene_cut: bool) -> None:
        """Records an inspected frame detection.

        Handles scene transitions and separates valid detections from noise/missing boundaries.
        """
        self.frames_inspected += 1

        if is_scene_cut and self._current_scene_detections:
            self._finalize_current_scene()
            self._current_scene_id += 1
            self.scene_cuts_count += 1
            self._scene_start_frame = detection.frame_index
            self._scene_start_timestamp = detection.timestamp_seconds

        if detection.is_valid:
            self.valid_detections_count += 1
        else:
            self.missed_or_noise_count += 1

        self._current_scene_detections.append(detection)

    def finalize(self, last_frame_index: int, last_timestamp: float) -> List[AggregatedPitchBoundary]:
        """Finalizes the last open scene segment and returns all aggregated scene summaries."""
        if self._current_scene_detections:
            self._finalize_current_scene(last_frame_index, last_timestamp)
        return self.completed_scenes

    def _finalize_current_scene(
        self,
        override_end_frame: Optional[int] = None,
        override_end_timestamp: Optional[float] = None
    ) -> None:
        if not self._current_scene_detections:
            return

        valid_detections = [d for d in self._current_scene_detections if d.is_valid]
        total_samples = len(self._current_scene_detections)

        end_frame = (
            override_end_frame
            if override_end_frame is not None
            else self._current_scene_detections[-1].frame_index
        )
        end_timestamp = (
            override_end_timestamp
            if override_end_timestamp is not None
            else self._current_scene_detections[-1].timestamp_seconds
        )

        consensus_coords: Optional[List[Tuple[float, float]]] = None
        consensus_polygon_obj: Optional[Polygon] = None
        avg_confidence = 0.0

        if valid_detections:
            avg_confidence = round(
                float(np.mean([d.confidence for d in valid_detections])), 3
            )

            # Consensus boundary: pick the representative medoid polygon
            # (the detection closest to median area and centroid)
            consensus_polygon_obj = self._compute_medoid_polygon(valid_detections)
            if consensus_polygon_obj and consensus_polygon_obj.is_valid:
                consensus_coords = list(consensus_polygon_obj.exterior.coords)

        # Derive recommended crop window from consensus polygon
        recommended_crop = derive_crop_window(
            poly=consensus_polygon_obj,
            frame_width=self.canvas_width,
            frame_height=self.canvas_height,
            target_aspect_ratio=self.crop_config.aspect_ratio_float,
            padding_px=self.crop_config.padding_px,
        )

        pitch_ratio = round(len(valid_detections) / total_samples, 3) if total_samples > 0 else 0.0

        scene_summary = AggregatedPitchBoundary(
            scene_id=self._current_scene_id,
            start_frame=self._scene_start_frame,
            end_frame=end_frame,
            start_timestamp=self._scene_start_timestamp,
            end_timestamp=end_timestamp,
            samples_inspected=total_samples,
            valid_detections=len(valid_detections),
            consensus_polygon=consensus_coords,
            recommended_crop=recommended_crop,
            average_confidence=avg_confidence,
            pitch_visibility_ratio=pitch_ratio,
        )

        self.completed_scenes.append(scene_summary)
        self._current_scene_detections = []

    def _compute_medoid_polygon(self, valid_detections: List[FieldDetection]) -> Optional[Polygon]:
        """Selects the most representative boundary polygon from a cluster of valid detections."""
        if len(valid_detections) == 1:
            poly_coords = valid_detections[0].polygon
            return sanitize_and_simplify_polygon(np.array(poly_coords)) if poly_coords else None

        polygons: List[Polygon] = []
        for det in valid_detections:
            if det.polygon:
                p = sanitize_and_simplify_polygon(np.array(det.polygon))
                if p and p.is_valid:
                    polygons.append(p)

        if not polygons:
            return None

        # Compare pairwise symmetric difference area to find the medoid
        # Medoid minimizes sum of symmetric difference areas with all other detections
        best_poly = polygons[0]
        min_total_diff = float("inf")

        for i, p1 in enumerate(polygons):
            total_diff = 0.0
            for j, p2 in enumerate(polygons):
                if i != j:
                    try:
                        total_diff += p1.symmetric_difference(p2).area
                    except Exception:
                        total_diff += abs(p1.area - p2.area)
            if total_diff < min_total_diff:
                min_total_diff = total_diff
                best_poly = p1

        return best_poly
