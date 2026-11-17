"""
ASAF-Net: Adaptive Scale-Aware Fusion Network

Proposed contribution for publication:
  - Dual-branch encoder: RGB semantics + depth geometry
  - Scale-aware fusion gate conditioned on local depth statistics
  - Uncertainty / confidence head for SLAM weighting
  - Residual refinement: D_fused = D_sensor + Δ(D_mono, RGB, conf)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_fusion.models.resnet_unet import ResNetUNet


@dataclass
class ASAFNetConfig:
    backbone: str = "resnet34"
    pretrained: bool = True
    use_sensor_depth: bool = True
    use_confidence_head: bool = True
    residual_refinement: bool = True


class ScaleAwareGate(nn.Module):
    """Local scale-aware gating: w ∈ [0,1] per pixel."""

    def __init__(self, in_ch: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, mono: torch.Tensor, sensor: Optional[torch.Tensor]) -> torch.Tensor:
        mono_norm = mono / (mono.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        if sensor is None:
            stats = torch.cat([mono_norm, mono_norm], dim=1)
        else:
            ratio = (sensor / (mono + 1e-6)).clamp(0.0, 10.0)
            stats = torch.cat([mono_norm, ratio], dim=1)
        return self.conv(stats)


class ASAFNet(nn.Module):
    """
    Adaptive Scale-Aware Fusion Network.

    Inputs (B, C, H, W):
      - RGB (3), mono depth (1), optional sensor depth (1)
    Outputs:
      - depth: refined metric depth
      - confidence: per-pixel uncertainty map in [0,1]
    """

    def __init__(self, cfg: Optional[ASAFNetConfig] = None):
        super().__init__()
        self.cfg = cfg or ASAFNetConfig()
        # 4 = RGB(3)+mono(1); +1 sensor when provided at runtime
        self._in_ch_with_sensor = 5
        self._in_ch_mono_only = 4
        self.refiner_sensor = ResNetUNet(
            backbone=self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            in_channels=self._in_ch_with_sensor,
            out_channels=1,
        )
        self.refiner_mono = ResNetUNet(
            backbone=self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            in_channels=self._in_ch_mono_only,
            out_channels=1,
        )
        self.gate = ScaleAwareGate(in_ch=2)
        if self.cfg.use_confidence_head:
            self.conf_head = nn.Sequential(
                nn.Conv2d(self._in_ch_mono_only, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
                nn.Sigmoid(),
            )
        else:
            self.conf_head = None

    def forward(
        self,
        rgb: torch.Tensor,
        mono: torch.Tensor,
        sensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        mono_n = mono / mono.amax(dim=(-2, -1), keepdim=True).clamp(min=1.0)

        use_sensor = sensor is not None and self.cfg.use_sensor_depth
        if use_sensor:
            sensor_n = sensor / sensor.amax(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            inp = torch.cat([rgb, mono_n, sensor_n], dim=1)
            base = sensor_n
        else:
            inp = torch.cat([rgb, mono_n], dim=1)
            base = mono_n

        refiner = self.refiner_sensor if use_sensor else self.refiner_mono
        delta = refiner(inp)
        if delta.shape[-2:] != base.shape[-2:]:
            delta = F.interpolate(delta, size=base.shape[-2:], mode="bilinear", align_corners=False)
        w = self.gate(mono_n, sensor if use_sensor else None)
        if w.shape[-2:] != base.shape[-2:]:
            w = F.interpolate(w, size=base.shape[-2:], mode="bilinear", align_corners=False)

        if self.cfg.residual_refinement:
            depth = base + w * delta
        else:
            depth = w * base + (1 - w) * delta

        out: Dict[str, torch.Tensor] = {"depth": depth}
        if self.conf_head is not None:
            conf_inp = torch.cat([rgb, mono_n], dim=1)
            out["confidence"] = self.conf_head(conf_inp)
        return out


# ---------------------------------------------------------------------------
# ASAF-Net V2 — révision corrigée et optimisée
# ---------------------------------------------------------------------------
#
# Corrections par rapport à la V1 :
#   * Normalisation métrique FIXE (depth_scale_m) au lieu du max par image :
#     comportement déterministe, identique entraînement/inférence ROS2.
#   * Le gate reçoit mono et capteur dans le MÊME espace normalisé
#     (la V1 mélangeait mono normalisé et capteur brut).
#   * Canal de validité capteur en entrée : le réseau sait où sont les trous.
#   * Prior composite : base = valid*capteur + (1-valid)*mono aligné —
#     les trous sont pré-remplis par le mono, le résiduel reste petit partout.
#   * Un seul refiner 6 canaux (≈30M params au lieu de 60M) ; le repli
#     mono-seul s'obtient avec valid=0.
#   * Tête d'incertitude aléatoirique : échelle de Laplace b(x) supervisée
#     par NLL (Kendall & Gal, 2017) ; confiance = exp(-b).


@dataclass
class ASAFNetV2Config:
    backbone: str = "resnet34"
    pretrained: bool = True
    depth_scale_m: float = 10.0   # normalisation métrique fixe (mètres)
    min_laplace_b: float = 1e-3   # plancher de l'échelle d'incertitude


class ScaleAwareGateV2(nn.Module):
    """Porte scale-aware V2 : statistiques cohérentes (espace normalisé commun)."""

    def __init__(self, in_ch: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.conv(stats)


class ASAFNetV2(nn.Module):
    """
    Adaptive Scale-Aware Fusion Network V2.

    Entrées métriques (mètres) :
      - rgb    (B,3,H,W) normalisé ImageNet
      - mono   (B,1,H,W) profondeur monoculaire alignée métriquement
      - sensor (B,1,H,W) profondeur capteur, 0 aux pixels invalides
      - valid  (B,1,H,W) masque de validité capteur ∈ {0,1}

    Sorties :
      - depth      (B,1,H,W) profondeur fusionnée en mètres
      - gate       (B,1,H,W) poids résiduel w ∈ [0,1]
      - laplace_b  (B,1,H,W) échelle d'incertitude aléatoirique (mètres)
      - confidence (B,1,H,W) exp(-b/depth_scale) ∈ (0,1] pour pondération SLAM
    """

    def __init__(self, cfg: Optional[ASAFNetV2Config] = None):
        super().__init__()
        self.cfg = cfg or ASAFNetV2Config()
        self.refiner = ResNetUNet(
            backbone=self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            in_channels=6,   # RGB(3) + mono(1) + sensor(1) + valid(1)
            out_channels=1,
        )
        self.gate = ScaleAwareGateV2(in_ch=4)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        mono: torch.Tensor,
        sensor: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        s = self.cfg.depth_scale_m
        if sensor is None:
            sensor = torch.zeros_like(mono)
        if valid is None:
            valid = (sensor > 1e-3).float()
        valid = valid.float()

        mono_n = (mono / s).clamp(0.0, 2.0)
        sensor_n = (sensor / s).clamp(0.0, 2.0) * valid

        # Prior composite : capteur là où il est valide, mono aligné ailleurs.
        base = valid * sensor_n + (1.0 - valid) * mono_n

        inp = torch.cat([rgb, mono_n, sensor_n, valid], dim=1)
        delta = self.refiner(inp)
        if delta.shape[-2:] != base.shape[-2:]:
            delta = F.interpolate(delta, size=base.shape[-2:], mode="bilinear", align_corners=False)

        # Statistiques du gate dans un espace d'échelle commun.
        eps = 1e-4
        log_ratio = torch.log((sensor_n + eps) / (mono_n + eps)).clamp(-2.0, 2.0) * valid
        w = self.gate(torch.cat([mono_n, sensor_n, valid, log_ratio], dim=1))
        if w.shape[-2:] != base.shape[-2:]:
            w = F.interpolate(w, size=base.shape[-2:], mode="bilinear", align_corners=False)

        # Clamp physique [0, 2·depth_scale] (= 20 m par défaut) : garantit une
        # sortie bornée pour le SLAM même sur entrée hors distribution.
        depth_n = (base + w * delta).clamp(0.0, 2.0)
        depth = depth_n * s

        b_raw = self.uncertainty_head(inp)
        laplace_b = F.softplus(b_raw) * s + self.cfg.min_laplace_b

        return {
            "depth": depth,
            "gate": w,
            "delta": delta * s,
            "laplace_b": laplace_b,
            "confidence": torch.exp(-laplace_b / s),
        }
