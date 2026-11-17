#!/usr/bin/env python3
"""
Benchmark 5 baselines sur val.csv — métriques mm + tableau LaTeX.

B0: Capteur RGB-D brut
B1: DA3 scaled
B2: Fusion heuristique RANSAC (sans GT)
B3: ResNet-UNet 4ch
B4: ASAF-Net
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth_fusion.evaluation.metrics import compute_depth_metrics, depth_boundary_accuracy
from depth_fusion.fusion.heuristic_fusion import heuristic_fusion
from depth_fusion.models.asaf_net import ASAFNet, ASAFNetConfig, ASAFNetV2, ASAFNetV2Config
from depth_fusion.models.resnet_unet import ResNetUNet
from depth_fusion.utils.checkpoint import load_checkpoint


def resize_depth(d: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(d.astype(np.float32)).resize(size, Image.BILINEAR), dtype=np.float32)


def metrics_mm(pred: np.ndarray, gt: np.ndarray, *, align: bool = True) -> Dict[str, float]:
    """Métriques en mm avec masque valide capteur."""
    mask = (gt > 1e-3) & (gt < 1e4) & np.isfinite(gt) & np.isfinite(pred)
    pred = pred.astype(np.float32).copy()
    gt = gt.astype(np.float32)

    if align and mask.sum() > 50:
        scale = float(np.median(gt[mask]) / max(np.median(pred[mask]), 1e-6))
        pred = pred * scale

    pt = torch.from_numpy(pred)[None, None]
    gt_t = torch.from_numpy(gt)[None, None]
    m = compute_depth_metrics(pt, gt_t, min_depth=1.0, max_depth=10000.0, align_median=False)
    ba = depth_boundary_accuracy(pred, gt, mask=mask)
    return {**m.to_dict(), "boundary_acc": ba}


class ModelWrapper:
    def __init__(self, name: str, forward_fn: Callable):
        self.name = name
        self.forward_fn = forward_fn

    def predict(self, rgb: np.ndarray, mono: np.ndarray, sensor: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        return self.forward_fn(rgb, mono, sensor, size)


def build_models(
    device: torch.device,
    unet_ckpt: str,
    asaf_ckpt: str,
    method_ids: list[str],
    asaf_v2_ckpt: str = "",
    asaf_size: int = 384,
) -> Dict[str, Optional[ModelWrapper]]:
    tr = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    unet = None
    if "B3_unet" in method_ids and Path(unet_ckpt).is_file():
        unet = ResNetUNet(pretrained=False, in_channels=4).to(device)
        load_checkpoint(unet_ckpt, unet, device=str(device))
        unet.eval()

    asaf = None
    if "B4_asaf" in method_ids:
        asaf = ASAFNet(ASAFNetConfig(use_sensor_depth=True)).to(device)
        load_checkpoint(asaf_ckpt, asaf, device=str(device))
        asaf.eval()

    def unet_fwd(rgb, mono, sensor, size):
        if unet is None:
            raise RuntimeError("UNet non chargé")
        rgb_t = tr(Image.fromarray(rgb)).unsqueeze(0).to(device)
        mono_r = resize_depth(mono, (256, 256))
        scale = max(float(mono_r.max()), 1.0)
        mono_t = torch.from_numpy(mono_r / scale)[None, None].to(device)
        with torch.no_grad():
            pred = unet(torch.cat([rgb_t, mono_t], dim=1))
        pred_np = pred.squeeze().cpu().numpy() * scale
        return resize_depth(pred_np, size)

    asaf_v2 = None
    if "B5_asaf_v2" in method_ids and asaf_v2_ckpt and Path(asaf_v2_ckpt).is_file():
        asaf_v2 = ASAFNetV2(ASAFNetV2Config(pretrained=False)).to(device)
        load_checkpoint(asaf_v2_ckpt, asaf_v2, device=str(device))
        asaf_v2.eval()

    tr_384 = T.Compose([
        T.Resize((asaf_size, asaf_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _mm_to_m(d: np.ndarray) -> np.ndarray:
        v = d[np.isfinite(d) & (d > 0)]
        if v.size and float(np.median(v)) > 100.0:
            return d / 1000.0
        return d

    def asaf_v2_fwd(rgb, mono, sensor, size):
        """V2 : entrées métriques (m), resize NEAREST profondeurs, sortie mm.

        Aucun réalignement médian post-hoc : la sortie est métrique par
        construction (align=False dans le protocole)."""
        if asaf_v2 is None:
            raise RuntimeError("ASAF V2 non chargé")
        rgb_t = tr_384(Image.fromarray(rgb)).unsqueeze(0).to(device)
        mono_m = _mm_to_m(mono)
        sensor_m = _mm_to_m(sensor)

        def _nearest(d: np.ndarray) -> np.ndarray:
            return np.array(
                Image.fromarray(d.astype(np.float32)).resize((asaf_size, asaf_size), Image.NEAREST),
                dtype=np.float32,
            )

        mono_r = _nearest(mono_m)
        sensor_r = _nearest(sensor_m)
        valid = ((sensor_r > 0.05) & (sensor_r < 20.0)).astype(np.float32)
        mono_t = torch.from_numpy(mono_r)[None, None].to(device)
        sensor_t = torch.from_numpy(sensor_r * valid)[None, None].to(device)
        valid_t = torch.from_numpy(valid)[None, None].to(device)
        with torch.no_grad():
            out = asaf_v2(rgb_t, mono_t, sensor_t, valid_t)
        pred_mm = out["depth"].squeeze().cpu().numpy() * 1000.0
        return resize_depth(pred_mm, size)

    def asaf_fwd(rgb, mono, sensor, size):
        if asaf is None:
            raise RuntimeError("ASAF non chargé")
        rgb_t = tr(Image.fromarray(rgb)).unsqueeze(0).to(device)
        mono_r = resize_depth(mono, (256, 256))
        sensor_r = resize_depth(sensor, (256, 256))
        scale = max(float(mono_r.max()), 1.0)
        mono_t = torch.from_numpy(mono_r / scale)[None, None].to(device)
        sensor_t = torch.from_numpy(sensor_r / scale)[None, None].to(device)
        with torch.no_grad():
            out = asaf(rgb_t, mono_t, sensor_t)
            pred = out["depth"]
        pred_np = pred.squeeze().cpu().numpy() * scale
        return resize_depth(pred_np, size)

    return {
        "B0_sensor": None,
        "B1_da3": None,
        "B2_ransac": None,
        "B3_unet": ModelWrapper("B3_unet", unet_fwd) if unet is not None else None,
        "B4_asaf": ModelWrapper("B4_asaf", asaf_fwd) if asaf is not None else None,
        "B5_asaf_v2": ModelWrapper("B5_asaf_v2", asaf_v2_fwd) if asaf_v2 is not None else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--val_csv", default=str(ROOT / "dataset_fusion" / "vigs" / "val.csv"))
    p.add_argument("--output_dir", default=str(ROOT / "evaluation_results" / "benchmark"))
    p.add_argument("--unet_ckpt", default=str(ROOT / "checkpoints" / "best_model_4ch.pth"))
    p.add_argument(
        "--asaf_ckpt",
        default=str(ROOT / "checkpoints/experiments/asaf_net_vigs_multiseq/best_model.pth"),
    )
    p.add_argument(
        "--asaf_v2_ckpt",
        default=str(ROOT / "checkpoints/experiments/asaf_net_v2_vigs/best_model.pth"),
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Sous-ensemble: B0_sensor B1_da3 B2_ransac B3_unet B4_asaf B5_asaf_v2",
    )
    p.add_argument(
        "--asaf_size",
        type=int,
        default=384,
        help="Résolution d'inférence ASAF (doit correspondre à l'entraînement du checkpoint)",
    )
    args = p.parse_args()
    asaf_size = int(args.asaf_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    method_ids = args.methods or ["B0_sensor", "B1_da3", "B2_ransac", "B3_unet", "B4_asaf", "B5_asaf_v2"]
    if not Path(args.unet_ckpt).is_file() and "B3_unet" in method_ids:
        print(f"WARN: UNet absent ({args.unet_ckpt}) — skip B3_unet")
        method_ids = [m for m in method_ids if m != "B3_unet"]
    if "B4_asaf" in method_ids and not Path(args.asaf_ckpt).is_file():
        print(f"WARN: ASAF V1 absent ({args.asaf_ckpt}) — skip B4_asaf")
        method_ids = [m for m in method_ids if m != "B4_asaf"]
    if "B5_asaf_v2" in method_ids and not Path(args.asaf_v2_ckpt).is_file():
        print(f"WARN: ASAF V2 absent ({args.asaf_v2_ckpt}) — skip B5_asaf_v2")
        method_ids = [m for m in method_ids if m != "B5_asaf_v2"]

    models = build_models(
        device, args.unet_ckpt, args.asaf_ckpt, method_ids, args.asaf_v2_ckpt, asaf_size
    )
    df_val = pd.read_csv(args.val_csv)

    all_rows = []
    for method_id in method_ids:
        per_image = []
        for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc=method_id):
            rgb = np.array(Image.open(row["rgb"]).convert("RGB"))
            H, W = rgb.shape[:2]
            gt = np.load(row["depth_gt"]).astype(np.float32)
            mono = np.load(row["depth_any"]).astype(np.float32)
            sensor = np.load(row["depth_sensor"]).astype(np.float32)
            gt_r = resize_depth(gt, (W, H))

            if method_id == "B0_sensor":
                pred = resize_depth(sensor, (W, H))
                align = False
            elif method_id == "B1_da3":
                pred = resize_depth(mono, (W, H))
                align = True
            elif method_id == "B2_ransac":
                pred = heuristic_fusion(
                    resize_depth(sensor, (W, H)),
                    resize_depth(mono, (W, H)),
                )
                align = False
            else:
                wrapper = models.get(method_id)
                if wrapper is None:
                    raise RuntimeError(f"Modèle non chargé pour {method_id}")
                pred = wrapper.predict(rgb, mono, sensor, (W, H))
                # V2 est métrique par construction : AUCUN réalignement
                # médian post-hoc sur la GT (revendication plus forte).
                align = method_id != "B5_asaf_v2"

            m = metrics_mm(pred, gt_r, align=align)
            m["file"] = Path(row["rgb"]).name
            m["method"] = method_id
            per_image.append(m)

        pdf = pd.DataFrame(per_image)
        pdf.to_csv(out / f"{method_id}_per_image.csv", index=False)
        mean = pdf[["mae", "rmse", "abs_rel", "sq_rel", "delta1", "delta2", "delta3", "boundary_acc"]].mean()
        mean["method"] = method_id
        all_rows.append(mean)

    summary = pd.DataFrame(all_rows).set_index("method")
    summary.to_csv(out / "benchmark_summary_mm.csv")
    split_label = "test" if "test" in Path(args.val_csv).name else "validation"
    print(f"\n=== Benchmark (mm, {split_label}) ===")
    print(summary.round(4).to_string())

    latex = summary[["mae", "rmse", "abs_rel", "delta1"]].round(4)
    latex_path = out / "benchmark_table.tex"
    header = "method & mae & rmse & abs\\_rel & $\\delta_1$ \\\\\n\\hline\n"
    body = "\n".join(
        f"{idx} & {row['mae']:.4f} & {row['rmse']:.4f} & {row['abs_rel']:.4f} & {row['delta1']:.4f} \\\\"
        for idx, row in latex.iterrows()
    )
    latex_path.write_text(
        "% Auto-generated benchmark table\n\\begin{tabular}{lrrrr}\n\\hline\n"
        + header
        + body
        + "\n\\hline\n\\end{tabular}\n",
        encoding="utf-8",
    )
    print(f"\nSauvegardé: {out / 'benchmark_summary_mm.csv'}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
