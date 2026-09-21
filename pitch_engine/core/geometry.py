"""Spatial geometry computations, polygon validation, and metric caching.

Eliminates repetitive allocations and accelerates polygon operations.
"""

from functools import lru_cache
from typing import List, Optional, Tuple
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from pitch_engine.core.models import CropBox


@lru_cache(maxsize=16)
def get_cached_frame_boundary(width: int, height: int) -> Polygon:
    """Pre-allocates and caches the frame canvas boundary polygon for given dimensions."""
    return Polygon([(0, 0), (width, 0), (width, height), (0, height)])


def sanitize_and_simplify_polygon(
    pts: np.ndarray,
    min_points: int = 3,
    simplify_tolerance: float = 2.0,
) -> Optional[Polygon]:
    """Validates raw contour points, repairs minor self-intersections, and simplifies vertices.

    Args:
        pts: 2D numpy array of coordinates (N, 2)
        min_points: Minimum required vertices (default 3)
        simplify_tolerance: Douglas-Peucker tolerance for vertex reduction.

    Returns:
        A valid Shapely Polygon or None if geometry is degenerate.
    """
    if pts is None or len(pts) < min_points:
        return None

    try:
        # Flatten if 3D contour from cv2.findContours (N, 1, 2)
        if pts.ndim == 3 and pts.shape[1] == 1:
            pts = pts.reshape(-1, 2)

        poly = Polygon(pts)

        if not poly.is_valid:
            # Attempt repair via buffer(0) or make_valid
            poly = poly.buffer(0)
            if not poly.is_valid:
                poly = make_valid(poly)

        # In case make_valid or buffer returned MultiPolygon, extract largest polygon
        if poly.geom_type == "MultiPolygon":
            polys = list(poly.geoms)
            if not polys:
                return None
            poly = max(polys, key=lambda p: p.area)
        elif poly.geom_type != "Polygon":
            return None

        if poly.is_empty or poly.area <= 0:
            return None

        # Simplify to reduce point count and drastically accelerate intersection calculations
        if simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)

        return poly if (poly.is_valid and not poly.is_empty) else None

    except Exception:
        return None


def calculate_frame_intersection_area(poly: Polygon, frame_width: int, frame_height: int) -> float:
    """Computes area of intersection between the polygon and cached canvas frame bounds."""
    if poly is None or poly.is_empty:
        return 0.0

    canvas_boundary = get_cached_frame_boundary(frame_width, frame_height)
    try:
        return float(poly.intersection(canvas_boundary).area)
    except Exception:
        return 0.0


def derive_crop_window(
    poly: Optional[Polygon],
    frame_width: int,
    frame_height: int,
    target_aspect_ratio: float = 16.0 / 9.0,
    padding_px: int = 20,
) -> Optional[CropBox]:
    """Derives a recommended camera crop window encompassing the pitch boundary.

    Constrains the crop box to the given aspect ratio and canvas boundaries.
    """
    if poly is None or poly.is_empty:
        # Fallback to full frame if no boundary polygon
        return CropBox(x=0, y=0, width=frame_width, height=frame_height)

    min_x, min_y, max_x, max_y = poly.bounds

    # Apply padding
    box_x1 = max(0, int(min_x - padding_px))
    box_y1 = max(0, int(min_y - padding_px))
    box_x2 = min(frame_width, int(max_x + padding_px))
    box_y2 = min(frame_height, int(max_y + padding_px))

    content_w = max(1, box_x2 - box_x1)
    content_h = max(1, box_y2 - box_y1)

    # Adjust to target aspect ratio (W / H)
    current_ratio = content_w / content_h

    if current_ratio < target_aspect_ratio:
        # Content is taller than target aspect ratio -> expand width
        desired_w = min(frame_width, int(content_h * target_aspect_ratio))
        delta_w = desired_w - content_w
        box_x1 = max(0, box_x1 - delta_w // 2)
        box_x2 = min(frame_width, box_x1 + desired_w)
        if box_x2 == frame_width:
            box_x1 = max(0, frame_width - desired_w)
    else:
        # Content is wider than target aspect ratio -> expand height
        desired_h = min(frame_height, int(content_w / target_aspect_ratio))
        delta_h = desired_h - content_h
        box_y1 = max(0, box_y1 - delta_h // 2)
        box_y2 = min(frame_height, box_y1 + desired_h)
        if box_y2 == frame_height:
            box_y1 = max(0, frame_height - desired_h)

    final_w = max(1, box_x2 - box_x1)
    final_h = max(1, box_y2 - box_y1)

    return CropBox(x=box_x1, y=box_y1, width=final_w, height=final_h)


def polygon_to_coords(poly: Optional[Polygon]) -> Optional[List[Tuple[float, float]]]:
    """Converts a Shapely Polygon exterior to a list of (x, y) tuples."""
    if poly is None or poly.is_empty:
        return None
    try:
        return list(poly.exterior.coords)
    except Exception:
        return None
