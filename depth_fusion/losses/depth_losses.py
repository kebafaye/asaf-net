"""
Multi-term depth fusion losses for research-grade training.

L_total = w_l1 * L1 + w_grad * L_grad + w_ssim * L_ssim + w_smooth * L_smooth
        + w_normal * L_normal + w_scale * L_scale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthLossConfig:
    w_l1: float = 1.0
    w_grad: float = 0.5
    w_ssim: float = 0.2
    w_smooth: float = 0.1
    w_normal: float = 0.1
    w_scale: float = 0.05
    w_confidence: float = 0.0  # optional confidence head supervision


class DepthFusionLoss(nn.Module):
    """Composite loss for depth refinement / fusion."""

    def __init__(self, cfg: Optional[DepthLossConfig] = None):
        super().__init__()
        self.cfg = cfg or DepthLossConfig()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        rgb: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        losses: Dict[str, torch.Tensor] = {}

        mask = (gt > 1e-3) & (gt < 1e4) & torch.isfinite(gt)
        if mask.sum() == 0:
            zero = pred.sum() * 0.0
            return zero, {"total": 0.0}

        losses["l1"] = F.l1_loss(pred[mask], gt[mask])
        losses["grad"] = self._gradient_loss(pred, gt, mask)
        losses["ssim"] = self._ssim_loss(pred, gt, mask)
        losses["smooth"] = self._edge_aware_smooth(pred, rgb) if rgb is not None else pred.sum() * 0.0
        losses["normal"] = self._normal_consistency(pred, gt, mask)
        losses["scale"] = self._scale_consistency(pred, gt, mask)

        total = (
            cfg.w_l1 * losses["l1"]
            + cfg.w_grad * losses["grad"]
            + cfg.w_ssim * losses["ssim"]
            + cfg.w_smooth * losses["smooth"]
            + cfg.w_normal * losses["normal"]
            + cfg.w_scale * losses["scale"]
        )

        if confidence is not None and cfg.w_confidence > 0:
            # Encourage high confidence where error is low
            err = (pred - gt).abs().detach()
            target_conf = torch.exp(-err / (err.median() + 1e-6))
            losses["conf"] = F.mse_loss(confidence, target_conf)
            total = total + cfg.w_confidence * losses["conf"]

        log = {k: float(v.detach()) for k, v in losses.items()}
        log["total"] = float(total.detach())
        return total, log

    @staticmethod
    def _gradient_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        def grad_x(t: torch.Tensor) -> torch.Tensor:
            return t[..., :, 1:] - t[..., :, :-1]

        def grad_y(t: torch.Tensor) -> torch.Tensor:
            return t[..., 1:, :] - t[..., :-1, :]

        m = mask[..., :, 1:] & mask[..., :, :-1]
        gx_p, gx_g = grad_x(pred), grad_x(gt)
        loss_x = (gx_p - gx_g).abs()[m].mean() if m.any() else pred.sum() * 0.0

        m = mask[..., 1:, :] & mask[..., :-1, :]
        gy_p, gy_g = grad_y(pred), grad_y(gt)
        loss_y = (gy_p - gy_g).abs()[m].mean() if m.any() else pred.sum() * 0.0
        return loss_x + loss_y

    @staticmethod
    def _ssim_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Simplified SSIM on depth (1-channel)
        mu_p = F.avg_pool2d(pred, 3, 1, 1)
        mu_g = F.avg_pool2d(gt, 3, 1, 1)
        sigma_p = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_p**2
        sigma_g = F.avg_pool2d(gt * gt, 3, 1, 1) - mu_g**2
        sigma_pg = F.avg_pool2d(pred * gt, 3, 1, 1) - mu_p * mu_g
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)) / (
            (mu_p**2 + mu_g**2 + c1) * (sigma_p + sigma_g + c2) + 1e-8
        )
        return (1.0 - ssim[mask]).mean()

    @staticmethod
    def _edge_aware_smooth(depth: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Edge-aware smoothness: penalize depth gradients except at RGB edges."""
        d_dx = depth[..., :, 1:] - depth[..., :, :-1]
        d_dy = depth[..., 1:, :] - depth[..., :-1, :]
        if rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]
        g_dx = torch.exp(-(gray[..., :, 1:] - gray[..., :, :-1]).abs().mean(1, keepdim=True))
        g_dy = torch.exp(-(gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(1, keepdim=True))
        return (d_dx.abs() * g_dx).mean() + (d_dy.abs() * g_dy).mean()

    @staticmethod
    def _normal_consistency(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        def normals(d: torch.Tensor) -> torch.Tensor:
            dz_dx = d[..., :-1, 1:] - d[..., :-1, :-1]  # (H-1, W-1)
            dz_dy = d[..., 1:, :-1] - d[..., :-1, :-1]
            nx, ny = -dz_dx, -dz_dy
            nz = torch.ones_like(nx)
            return F.normalize(torch.cat([nx, ny, nz], dim=1), dim=1)

        np_ = normals(pred)
        ng = normals(gt)
        m = mask[..., :-1, :-1]
        cos = (np_ * ng).sum(dim=1, keepdim=True).clamp(-1, 1)
        return (1.0 - cos[m]).mean() if m.any() else pred.sum() * 0.0

    @staticmethod
    def _scale_consistency(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Penalize global scale drift between pred and gt (moyenne sur le batch)."""
        errs = []
        for b in range(pred.shape[0]):
            m = mask[b]
            if m.sum() < 10:
                continue
            med_p = pred[b][m].median()
            med_g = gt[b][m].median()
            errs.append((med_p / med_g.clamp(min=1e-6) - 1.0).abs())
        if not errs:
            return pred.sum() * 0.0
        return torch.stack(errs).mean()


# ---------------------------------------------------------------------------
# Perte V2 — supervision métrique (mètres) + incertitude aléatoirique
# ---------------------------------------------------------------------------


@dataclass
class DepthLossV2Config:
    w_l1: float = 1.0
    w_grad: float = 0.5
    w_ssim: float = 0.2
    w_smooth_holes: float = 0.1   # lissage edge-aware appliqué surtout aux trous
    w_normal: float = 0.05
    w_scale: float = 0.05
    w_nll: float = 0.2            # NLL Laplace (Kendall & Gal, 2017)
    max_depth_m: float = 20.0


class DepthFusionLossV2(nn.Module):
    """
    Perte composite V2, calculée en mètres sur les pixels GT valides.

    L = w_l1·L1 + w_grad·L_grad + w_ssim·L_ssIM + w_smooth·L_smooth(trous)
      + w_normal·L_normal + w_scale·L_scale + w_nll·L_NLL

    L_NLL = mean( |D̂-D|/b + log(2b) ) : la tête d'incertitude apprend
    l'erreur résiduelle attendue, sans cible heuristique.
    """

    def __init__(self, cfg: Optional[DepthLossV2Config] = None):
        super().__init__()
        self.cfg = cfg or DepthLossV2Config()

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        rgb: Optional[torch.Tensor] = None,
        laplace_b: Optional[torch.Tensor] = None,
        valid_sensor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.cfg
        losses: Dict[str, torch.Tensor] = {}

        mask = (gt > 1e-3) & (gt < cfg.max_depth_m) & torch.isfinite(gt)
        if mask.sum() == 0:
            zero = pred.sum() * 0.0
            return zero, {"total": 0.0}

        losses["l1"] = F.l1_loss(pred[mask], gt[mask])
        losses["grad"] = DepthFusionLoss._gradient_loss(pred, gt, mask)
        losses["ssim"] = DepthFusionLoss._ssim_loss(pred, gt, mask)
        losses["normal"] = DepthFusionLoss._normal_consistency(pred, gt, mask)
        losses["scale"] = DepthFusionLoss._scale_consistency(pred, gt, mask)

        if rgb is not None:
            smooth = self._edge_aware_smooth_weighted(pred, rgb, valid_sensor)
            losses["smooth"] = smooth
        else:
            losses["smooth"] = pred.sum() * 0.0

        total = (
            cfg.w_l1 * losses["l1"]
            + cfg.w_grad * losses["grad"]
            + cfg.w_ssim * losses["ssim"]
            + cfg.w_smooth_holes * losses["smooth"]
            + cfg.w_normal * losses["normal"]
            + cfg.w_scale * losses["scale"]
        )

        if laplace_b is not None and cfg.w_nll > 0:
            err = (pred - gt).abs()
            b = laplace_b.clamp(min=1e-4)
            nll = (err / b + torch.log(2.0 * b))[mask].mean()
            losses["nll"] = nll
            total = total + cfg.w_nll * nll

        log = {k: float(v.detach()) for k, v in losses.items()}
        log["total"] = float(total.detach())
        return total, log

    @staticmethod
    def _edge_aware_smooth_weighted(
        depth: torch.Tensor,
        rgb: torch.Tensor,
        valid_sensor: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Lissage guidé RGB, pondéré vers les zones sans supervision capteur."""
        d_dx = (depth[..., :, 1:] - depth[..., :, :-1]).abs()
        d_dy = (depth[..., 1:, :] - depth[..., :-1, :]).abs()
        if rgb.shape[1] == 3:
            gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        else:
            gray = rgb[:, :1]
        g_dx = torch.exp(-(gray[..., :, 1:] - gray[..., :, :-1]).abs())
        g_dy = torch.exp(-(gray[..., 1:, :] - gray[..., :-1, :]).abs())

        if valid_sensor is not None:
            hole = 1.0 - valid_sensor.float()
            # poids fort dans les trous (1.0), faible ailleurs (0.1)
            w_x = 0.1 + 0.9 * torch.maximum(hole[..., :, 1:], hole[..., :, :-1])
            w_y = 0.1 + 0.9 * torch.maximum(hole[..., 1:, :], hole[..., :-1, :])
        else:
            w_x = torch.ones_like(d_dx)
            w_y = torch.ones_like(d_dy)

        return (d_dx * g_dx * w_x).mean() + (d_dy * g_dy * w_y).mean()
