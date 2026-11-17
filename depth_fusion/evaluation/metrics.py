"""
Depth evaluation metrics (publication-standard).

References: Eigen et al. (NeurIPS 2014), Uhrig et al. (ICCV 2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DepthMetrics:
    """Aggregated depth metrics (lower is better except delta/SSIM/PSNR)."""

    mae: float = 0.0
    rmse: float = 0.0
    abs_rel: float = 0.0
    sq_rel: float = 0.0
    log10: float = 0.0
    delta1: float = 0.0
    delta2: float = 0.0
    delta3: float = 0.0
    ssim: float = 0.0
    psnr: float = 0.0
    boundary_acc: float = 0.0
    n: int = 0

    def update(self, other: "DepthMetrics") -> None:
        n = self.n + other.n
        if n == 0:
            return
        w_self, w_other = self.n / n, other.n / n
        for key in ("mae", "rmse", "abs_rel", "sq_rel", "log10", "ssim", "psnr", "boundary_acc"):
            setattr(self, key, getattr(self, key) * w_self + getattr(other, key) * w_other)
        for key in ("delta1", "delta2", "delta3"):
            setattr(self, key, getattr(self, key) * w_self + getattr(other, key) * w_other)
        self.n = n

    def to_dict(self) -> Dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "abs_rel": self.abs_rel,
            "sq_rel": self.sq_rel,
            "log10": self.log10,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "delta3": self.delta3,
            "ssim": self.ssim,
            "psnr": self.psnr,
            "boundary_acc": self.boundary_acc,
        }


def _valid_mask(gt: torch.Tensor, min_depth: float, max_depth: float) -> torch.Tensor:
    return (gt > min_depth) & (gt < max_depth) & torch.isfinite(gt)


def compute_depth_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    min_depth: float = 1e-3,
    max_depth: float = 1e4,
    align_median: bool = False,
) -> DepthMetrics:
    """
    Compute standard depth metrics on batched tensors (B,1,H,W) or (B,H,W).

    If align_median=True, applies per-sample scale alignment s* = median(gt)/median(pred)
    before metric computation (standard for monocular depth benchmarks).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    pred = pred.detach().float()
    gt = gt.detach().float()
    mask = _valid_mask(gt, min_depth, max_depth)

    if align_median:
        for b in range(pred.shape[0]):
            m = mask[b]
            if m.sum() < 10:
                continue
            gp = gt[b][m]
            pp = pred[b][m]
            med_p = pp.median().clamp(min=1e-6)
            med_g = gp.median()
            pred[b] = pred[b] * (med_g / med_p)

    diff = (pred - gt).abs()
    mae = (diff * mask).sum() / mask.sum().clamp(min=1)
    rmse = torch.sqrt(((pred - gt) ** 2 * mask).sum() / mask.sum().clamp(min=1))

    gt_m = gt.clamp(min=min_depth)
    pred_m = pred.clamp(min=min_depth)
    abs_rel = ((diff / gt_m) * mask).sum() / mask.sum().clamp(min=1)
    sq_rel = ((((pred - gt) ** 2) / gt_m) * mask).sum() / mask.sum().clamp(min=1)
    log10 = (torch.log10(pred_m + 1e-8) - torch.log10(gt_m + 1e-8)).abs()
    log10 = (log10 * mask).sum() / mask.sum().clamp(min=1)

    ratio = torch.max(pred_m / gt_m, gt_m / pred_m)
    d1 = ((ratio < 1.25) * mask).float().sum() / mask.sum().clamp(min=1)
    d2 = ((ratio < 1.25**2) * mask).float().sum() / mask.sum().clamp(min=1)
    d3 = ((ratio < 1.25**3) * mask).float().sum() / mask.sum().clamp(min=1)

    return DepthMetrics(
        mae=float(mae),
        rmse=float(rmse),
        abs_rel=float(abs_rel),
        sq_rel=float(sq_rel),
        log10=float(log10),
        delta1=float(d1),
        delta2=float(d2),
        delta3=float(d3),
        n=1,
    )


def depth_boundary_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    threshold: float = 0.05,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Boundary F1-style accuracy (Uhrig et al.)."""
    from scipy.ndimage import sobel

    def edges(d: np.ndarray) -> np.ndarray:
        gx = sobel(d, axis=1)
        gy = sobel(d, axis=0)
        g = np.hypot(gx, gy)
        g = g / (g.max() + 1e-8)
        return g > threshold

    ep, eg = edges(pred), edges(gt)
    if mask is not None:
        ep, eg = ep & mask, eg & mask
    tp = (ep & eg).sum()
    fp = (ep & ~eg).sum()
    fn = (~ep & eg).sum()
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))
