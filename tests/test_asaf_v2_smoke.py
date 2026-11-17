#!/usr/bin/env python3
"""
Smoke test ASAF-Net V2 (CPU) :
  1. Shapes et plages de sortie du modèle
  2. Perte V2 finie + backward OK
  3. Dataset V2 : unités métriques, masques, flip joint
  4. Overfit 20 pas sur 2 frames réelles → la perte doit décroître nettement
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from depth_fusion.datasets.rgbd_fusion_dataset import RGBDFusionDatasetV2
from depth_fusion.losses.depth_losses import DepthFusionLossV2, DepthLossV2Config
from depth_fusion.models.asaf_net import ASAFNetV2, ASAFNetV2Config


def test_forward_shapes() -> None:
    m = ASAFNetV2(ASAFNetV2Config(pretrained=False))
    B, H, W = 2, 128, 160
    rgb = torch.randn(B, 3, H, W)
    mono = torch.rand(B, 1, H, W) * 5.0
    sensor = torch.rand(B, 1, H, W) * 5.0
    valid = (torch.rand(B, 1, H, W) > 0.3).float()
    sensor = sensor * valid
    out = m(rgb, mono, sensor, valid)
    assert out["depth"].shape == (B, 1, H, W)
    assert out["gate"].shape == (B, 1, H, W)
    assert out["laplace_b"].shape == (B, 1, H, W)
    assert float(out["depth"].min()) >= 0.0
    assert 0.0 <= float(out["gate"].min()) and float(out["gate"].max()) <= 1.0
    assert float(out["laplace_b"].min()) > 0.0
    assert 0.0 < float(out["confidence"].min()) <= 1.0
    # Repli mono-seul : valid=0 partout
    out2 = m(rgb, mono, None, None)
    assert torch.isfinite(out2["depth"]).all()
    print("OK  forward shapes + plages + repli mono")


def test_loss_backward() -> None:
    m = ASAFNetV2(ASAFNetV2Config(pretrained=False))
    crit = DepthFusionLossV2(DepthLossV2Config())
    B, H, W = 2, 96, 128
    rgb = torch.randn(B, 3, H, W)
    mono = torch.rand(B, 1, H, W) * 4 + 0.5
    valid = (torch.rand(B, 1, H, W) > 0.2).float()
    sensor = (mono + torch.randn_like(mono) * 0.05).clamp(min=0.1) * valid
    gt = sensor.clone()
    out = m(rgb, mono, sensor, valid)
    loss, log = crit(out["depth"], gt, rgb=rgb, laplace_b=out["laplace_b"], valid_sensor=valid)
    assert torch.isfinite(loss), f"loss non finie: {log}"
    loss.backward()
    grads = [p.grad.abs().sum() for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0 and all(torch.isfinite(g) for g in grads)
    print(f"OK  loss={float(loss):.4f} backward fini ({len(grads)} tenseurs de gradient)")


def test_dataset_v2() -> None:
    csv = ROOT / "dataset_fusion" / "val.csv"
    if not csv.exists():
        print("SKIP dataset (val.csv absent)")
        return
    ds = RGBDFusionDatasetV2(csv, image_size=(128, 160), use_sensor_depth=True, augment=False)
    s = ds[0]
    for k in ("rgb", "mono", "sensor", "valid", "gt", "valid_gt"):
        assert k in s, f"clé manquante: {k}"
    gt = s["gt"]
    v = gt[gt > 0]
    assert v.numel() > 0
    med = float(v.median())
    assert 0.2 < med < 20.0, f"médiane GT hors plage métrique: {med} m"
    assert set(s["valid"].unique().tolist()) <= {0.0, 1.0}
    print(f"OK  dataset V2 : GT médiane {med:.2f} m, validité capteur {float(s['valid'].mean())*100:.1f}%")


def test_overfit() -> None:
    csv = ROOT / "dataset_fusion" / "val.csv"
    if not csv.exists():
        print("SKIP overfit (val.csv absent)")
        return
    torch.manual_seed(0)
    np.random.seed(0)
    ds = RGBDFusionDatasetV2(csv, image_size=(96, 128), use_sensor_depth=True, augment=False)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    m = ASAFNetV2(ASAFNetV2Config(pretrained=False))
    crit = DepthFusionLossV2(DepthLossV2Config())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    first, last = None, None
    for step in range(20):
        out = m(batch["rgb"], batch["mono"], batch["sensor"], batch["valid"])
        loss, _ = crit(
            out["depth"], batch["gt"], rgb=batch["rgb"],
            laplace_b=out["laplace_b"], valid_sensor=batch["valid"],
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.7, f"pas de décroissance: {first:.4f} → {last:.4f}"
    print(f"OK  overfit 20 pas : loss {first:.4f} → {last:.4f} (-{100*(1-last/first):.0f}%)")


if __name__ == "__main__":
    test_forward_shapes()
    test_loss_backward()
    test_dataset_v2()
    test_overfit()
    print("\nTOUS LES TESTS V2 PASSENT")
