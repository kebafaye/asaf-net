#!/usr/bin/env python3
"""
Évaluation COMPLÉTION métrique sur trous synthétiques (protocole non trivial).

La GT étant la profondeur capteur, l'évaluation classique récompense la copie
de l'entrée. Ici : on masque ~15 % des pixels capteur valides en ENTRÉE
(déterministe, seed 123) et on mesure l'erreur UNIQUEMENT sur ces pixels
retenus — vraies valeurs capteur jamais vues par la méthode.

Méthodes comparées :
  C0_mono      : DA3 aligné (déjà métrique) tel quel
  C1_heuristic : fusion heuristique RANSAC (sensor corrompu + mono)
  C2_asaf_v2   : ASAF-Net V2 (rgb + mono + sensor corrompu + valid)

Sortie : CSV par frame + résumé (mm) dans --output_dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

from depth_fusion.datasets.rgbd_fusion_dataset import RGBDFusionDatasetV2
from depth_fusion.fusion.heuristic_fusion import heuristic_fusion
from depth_fusion.models.asaf_net import ASAFNetV2, ASAFNetV2Config
from depth_fusion.utils.checkpoint import load_checkpoint

HOLES_SEED = 123
HOLES_FRAC = 0.15
SIZE = 384


def to_meters(d: np.ndarray) -> np.ndarray:
    d = d.astype(np.float32)
    v = d[np.isfinite(d) & (d > 0)]
    if v.size and float(np.median(v)) > 100.0:
        d = d / 1000.0
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def nearest(d: np.ndarray, size: int = SIZE) -> np.ndarray:
    return np.array(Image.fromarray(d).resize((size, size), Image.NEAREST), dtype=np.float32)


def metrics_on_mask(pred_m: np.ndarray, gt_m: np.ndarray, mask: np.ndarray) -> dict:
    """Métriques (mm) restreintes à mask (bool)."""
    m = mask & (gt_m > 0.05) & (gt_m < 20.0) & np.isfinite(pred_m)
    if m.sum() < 10:
        return {}
    p, g = pred_m[m] * 1000.0, gt_m[m] * 1000.0
    err = np.abs(p - g)
    ratio = np.maximum(p / np.maximum(g, 1e-6), g / np.maximum(p, 1e-6))
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt(((p - g) ** 2).mean())),
        "abs_rel": float((err / np.maximum(g, 1e-6)).mean()),
        "delta1": float((ratio < 1.25).mean()),
        "delta2": float((ratio < 1.25**2).mean()),
        "delta3": float((ratio < 1.25**3).mean()),
        "n_px": int(m.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "dataset_fusion/vigs/val.csv"))
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints/experiments/asaf_net_v2_vigs/best_model.pth"))
    p.add_argument("--output_dir", default=str(ROOT / "evaluation_results/v2/completion_val"))
    p.add_argument("--limit", type=int, default=0, help="Max frames (0 = toutes)")
    p.add_argument(
        "--size",
        type=int,
        default=SIZE,
        help="Résolution d'évaluation (doit correspondre à celle d'entraînement du checkpoint)",
    )
    args = p.parse_args()
    size = int(args.size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ASAFNetV2(ASAFNetV2Config(pretrained=False)).to(device)
    load_checkpoint(args.checkpoint, model, device=str(device))
    model.eval()

    tr = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.iloc[: args.limit]

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="completion"):
        rgb_pil = Image.open(row["rgb"]).convert("RGB")
        mono = nearest(to_meters(np.load(row["depth_any"])), size)
        gt = nearest(to_meters(np.load(row["depth_gt"])), size)
        sensor_full = nearest(to_meters(np.load(row["depth_sensor"])), size)

        valid_full = ((sensor_full > 0.05) & (sensor_full < 20.0)).astype(np.float32)
        rng = np.random.RandomState((HOLES_SEED + idx) % (2**31))
        drop = RGBDFusionDatasetV2.make_drop_mask(size, size, rng, HOLES_FRAC)
        drop = drop & (valid_full > 0)
        valid = valid_full * (~drop)
        sensor = sensor_full * valid

        # C0 : mono aligné tel quel
        m0 = metrics_on_mask(mono, gt, drop)

        # C1 : fusion heuristique (mm attendus par le module → passer en mm)
        fused = heuristic_fusion(sensor * 1000.0, mono * 1000.0) / 1000.0
        m1 = metrics_on_mask(fused, gt, drop)

        # C2 : ASAF V2
        rgb_t = tr(rgb_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(
                rgb_t,
                torch.from_numpy(mono)[None, None].to(device),
                torch.from_numpy(sensor)[None, None].to(device),
                torch.from_numpy(valid)[None, None].to(device),
            )["depth"].squeeze().cpu().numpy()
        m2 = metrics_on_mask(pred, gt, drop)

        for name, m in [("C0_mono", m0), ("C1_heuristic", m1), ("C2_asaf_v2", m2)]:
            if m:
                rows.append({"frame": Path(row["rgb"]).name, "method": name, **m})

    per = pd.DataFrame(rows)
    per.to_csv(out / "completion_per_frame.csv", index=False)
    cols = ["mae", "rmse", "abs_rel", "delta1", "delta2", "delta3"]
    summary = per.groupby("method")[cols].mean()
    summary.to_csv(out / "completion_summary_mm.csv")
    print("\n=== Complétion métrique sur trous synthétiques (mm) ===")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
