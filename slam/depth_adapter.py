"""
Adapter to inject refined depth into SLAM / 3DGS pipelines.

Integration targets:
  - ORB-SLAM3: KeyFrame::mDepthMap replacement via RGB-D mode
  - Gaussian Splatting SLAM: depth prior for initialization & densification
  - TSDF / volumetric fusion: confidence-weighted depth integration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DepthFrame:
    """Single depth frame for SLAM consumption."""

    depth: np.ndarray  # HxW metric depth (meters or mm, consistent with calibration)
    confidence: Optional[np.ndarray] = None  # HxW in [0,1]
    timestamp: float = 0.0
    frame_id: int = 0


class SLAMDepthAdapter:
    """Convert fusion network output to SLAM-ready depth maps."""

    def __init__(
        self,
        *,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        hole_fill: bool = True,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hole_fill = hole_fill

    def process(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> DepthFrame:
        d = depth.astype(np.float32).copy()
        valid = (d > self.min_depth) & (d < self.max_depth) & np.isfinite(d)
        d[~valid] = 0.0

        if self.hole_fill and confidence is not None:
            # Simple inpainting: fill invalid from neighbors with high confidence
            from scipy.ndimage import distance_transform_edt

            invalid = ~valid
            if invalid.any() and valid.any():
                _, indices = distance_transform_edt(invalid, return_indices=True)
                d[invalid] = d[tuple(indices[:, invalid])]

        conf = confidence.astype(np.float32) if confidence is not None else np.ones_like(d)
        return DepthFrame(depth=d, confidence=conf)
