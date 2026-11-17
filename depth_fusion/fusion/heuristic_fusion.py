"""Fusion heuristique capteur + mono (sans fuite GT)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from depth_fusion.fusion.scale_align import align_depth_ransac


def heuristic_fusion(sensor: np.ndarray, mono: np.ndarray) -> np.ndarray:
    """
    Fusion test-time : aligne mono sur capteur (RANSAC), pondère par validité capteur.
    N'utilise PAS la GT.
    """
    sensor = sensor.astype(np.float32)
    mono = mono.astype(np.float32)
    if sensor.shape != mono.shape:
        from skimage.transform import resize
        mono = resize(mono, sensor.shape, mode="reflect", anti_aliasing=True).astype(np.float32)

    mask_valid = (sensor > 1e-3) & (sensor < 1e4) & np.isfinite(sensor)
    mono_aligned, _, _ = align_depth_ransac(mono, sensor, valid_mask=mask_valid)

    sensor_smooth = gaussian_filter(median_filter(sensor, size=3), sigma=0.8)
    gy, gx = np.gradient(sensor_smooth)
    grad = np.hypot(gx, gy)
    grad_norm = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    w_sensor = np.clip(mask_valid.astype(float) * (1.0 - grad_norm * 2.0), 0.0, 1.0)
    w_sensor = gaussian_filter(w_sensor, sigma=2)

    fused = w_sensor * sensor_smooth + (1.0 - w_sensor) * mono_aligned
    return median_filter(fused, size=3).astype(np.float32)
