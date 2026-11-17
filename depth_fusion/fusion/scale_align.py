"""
Scale alignment: relative monocular depth → metric RGB-D scale.

Methods:
  - Global median (fast, used in current pipeline)
  - Per-frame least squares on valid pixels
  - RANSAC robust regression (handles outliers)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def align_depth_global_median(
    mono: np.ndarray,
    metric: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Scale mono depth to metric using global median ratio."""
    if valid_mask is None:
        valid_mask = (metric > 1e-3) & (metric < 1e4) & np.isfinite(metric)
        valid_mask &= (mono > 1e-6) & np.isfinite(mono)

    m_mono = float(np.median(mono[valid_mask])) if valid_mask.any() else 1.0
    m_met = float(np.median(metric[valid_mask])) if valid_mask.any() else 1.0
    scale = m_met / max(m_mono, 1e-8)
    return mono.astype(np.float32) * scale, scale


def align_depth_per_frame_least_squares(
    mono: np.ndarray,
    metric: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    with_offset: bool = True,
) -> Tuple[np.ndarray, float, float]:
    """Per-frame scale s and optional offset b: metric ≈ s * mono + b."""
    if valid_mask is None:
        valid_mask = (metric > 1e-3) & (metric < 1e4) & np.isfinite(metric)
        valid_mask &= (mono > 1e-6) & np.isfinite(mono)

    x = mono[valid_mask].ravel()
    y = metric[valid_mask].ravel()
    if x.size < 50:
        scaled, s = align_depth_global_median(mono, metric, valid_mask=valid_mask)
        return scaled, s, 0.0

    if with_offset:
        A = np.stack([x, np.ones_like(x)], axis=1)
        sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        s, b = float(sol[0]), float(sol[1])
    else:
        s = float(np.dot(x, y) / (np.dot(x, x) + 1e-8))
        b = 0.0

    return mono.astype(np.float32) * s + b, s, b


def align_depth_ransac(
    mono: np.ndarray,
    metric: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    min_samples: float = 0.3,
    max_points: int = 20000,
) -> Tuple[np.ndarray, float, float]:
    """Robust RANSAC scale+offset alignment.

    Sous-échantillonne à max_points pixels (déterministe) : un ajustement
    robuste à 2 paramètres n'a pas besoin des ~140k pixels d'une frame
    complète, et le fit RANSAC domine sinon le temps de précalcul DA3.
    """
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    if valid_mask is None:
        valid_mask = (metric > 1e-3) & (metric < 1e4) & np.isfinite(metric)
        valid_mask &= (mono > 1e-6) & np.isfinite(mono)

    x = mono[valid_mask].ravel().reshape(-1, 1)
    y = metric[valid_mask].ravel()
    if x.shape[0] < 50:
        return align_depth_per_frame_least_squares(mono, metric, valid_mask=valid_mask)

    if x.shape[0] > max_points:
        idx = np.random.RandomState(42).choice(x.shape[0], max_points, replace=False)
        x, y = x[idx], y[idx]

    model = RANSACRegressor(LinearRegression(), min_samples=min_samples, random_state=42)
    model.fit(x, y)
    coef = np.asarray(model.estimator_.coef_).ravel()
    s = float(coef[0])
    b = float(model.estimator_.intercept_)
    return mono.astype(np.float32) * s + b, s, b
