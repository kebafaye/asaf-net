#!/usr/bin/env python3
"""Démo ASAF-Net V2 : complète la profondeur d'une trame RGB-D, sur processeur.

Trois façons de l'utiliser :

  # 1. Vérifier l'installation, sans données ni modèle entraîné
  python scripts/demo_asaf.py --selftest

  # 2. Une trame du corpus VIGS-Fusion déjà extrait et précalculé
  python scripts/demo_asaf.py --checkpoint checkpoints/asaf_net_v2_vigs.pth \\
      --sequence-dir data/vigs_fusion/Line-Rot/extracted --index 1500

  # 3. Vos propres fichiers
  python scripts/demo_asaf.py --checkpoint checkpoints/asaf_net_v2_vigs.pth \\
      --rgb image.png --sensor depth_sensor.npy --mono depth_mono_aligned.npy

Entrées :
  --rgb     image couleur (PNG/JPG)
  --sensor  profondeur du capteur (.npy ou PNG 16 bits), mètres ou millimètres
            (détection automatique), 0 = pixel non mesuré
  --mono    profondeur monoculaire DÉJÀ alignée en mètres sur le capteur
            (sortie de scripts/precompute_da3_sequences.py)

Sorties dans --out : depth_fused.npy (mètres, résolution d'origine),
confidence.npy, et demo_asaf.png (RGB | capteur | fusion | confiance).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from depth_fusion.models.asaf_net import ASAFNetV2, ASAFNetV2Config
from depth_fusion.utils.checkpoint import load_checkpoint

SIZE = 448  # résolution d'entraînement
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_depth(path: Path) -> np.ndarray:
    """Charge une profondeur (.npy ou PNG 16 bits) et la renvoie en mètres."""
    d = np.load(path) if path.suffix == ".npy" else np.array(Image.open(path))
    d = np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    v = d[d > 0]
    if v.size and float(np.median(v)) > 100.0:  # millimètres -> mètres
        d = d / 1000.0
    return d


def resize_nearest(d: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(d).resize((size, size), Image.NEAREST), dtype=np.float32)


def prepare(rgb: Image.Image, mono: np.ndarray, sensor: np.ndarray, size: int):
    x = np.asarray(rgb.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    rgb_t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    mono_r = resize_nearest(mono, size)
    sensor_r = resize_nearest(sensor, size)
    valid = ((sensor_r > 0.05) & (sensor_r < 20.0)).astype(np.float32)
    sensor_r = sensor_r * valid
    t = lambda a: torch.from_numpy(a)[None, None]
    return rgb_t, t(mono_r), t(sensor_r), t(valid)


def build_model(checkpoint: str | None, device: str) -> ASAFNetV2:
    model = ASAFNetV2(ASAFNetV2Config(pretrained=False)).to(device)
    if checkpoint:
        load_checkpoint(checkpoint, model, device=device)
    return model.eval()


def run(model, rgb, mono, sensor, device):
    h, w = sensor.shape
    inputs = [x.to(device) for x in prepare(rgb, mono, sensor, SIZE)]
    with torch.inference_mode():
        t0 = time.perf_counter()
        out = model(*inputs)
        dt = (time.perf_counter() - t0) * 1000.0
    up = lambda x: F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return up(out["depth"]), up(out["confidence"]), dt


def save_figure(path: Path, rgb, sensor, fused, conf) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(np.percentile(fused[fused > 0], 98)) if (fused > 0).any() else 5.0
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        (np.asarray(rgb.convert("RGB")), "RGB", {}),
        (np.ma.masked_equal(sensor, 0), "Capteur (trous en blanc)", dict(cmap="turbo", vmin=0, vmax=vmax)),
        (fused, "ASAF-Net (fusionnée)", dict(cmap="turbo", vmin=0, vmax=vmax)),
        (conf, "Confiance", dict(cmap="magma", vmin=0, vmax=1)),
    ]
    for a, (img, title, kw) in zip(ax, panels):
        a.imshow(img, **kw)
        a.set_title(title)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="poids ASAF-Net V2 (.pth)")
    p.add_argument("--sequence-dir", type=Path, help="dossier extracted/ d'une séquence VIGS-Fusion")
    p.add_argument("--index", type=int, default=0, help="numéro de trame dans --sequence-dir")
    p.add_argument("--rgb", type=Path)
    p.add_argument("--sensor", type=Path)
    p.add_argument("--mono", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo_asaf")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--selftest", action="store_true", help="entrées synthétiques, sans modèle entraîné")
    a = p.parse_args()

    if a.selftest:
        h, w = 240, 320
        rng = np.random.default_rng(0)
        rgb = Image.fromarray((rng.random((h, w, 3)) * 255).astype(np.uint8))
        mono = np.linspace(0.5, 4.0, h * w, dtype=np.float32).reshape(h, w)
        sensor = mono.copy()
        sensor[:, :48] = 0.0  # bande aveugle simulée
        model = build_model(None, a.device)
        fused, conf, dt = run(model, rgb, mono, sensor, a.device)
        assert fused.shape == (h, w) and np.isfinite(fused).all() and (fused >= 0).all()
        assert ((conf > 0) & (conf <= 1)).all()
        print(f"selftest OK : sortie {fused.shape}, inférence {dt:.0f} ms sur {a.device}")
        return 0

    if not a.checkpoint:
        p.error("--checkpoint est requis (ou --selftest)")
    if a.sequence_dir:
        i = a.index
        rgb_p = a.sequence_dir / "rgb" / f"image_{i:04d}.png"
        sensor_p = a.sequence_dir / "depth_sensor" / f"depth_{i:04d}.npy"
        mono_p = a.sequence_dir / "da3_scaled" / f"depth_{i:04d}.npy"
    elif a.rgb and a.sensor and a.mono:
        rgb_p, sensor_p, mono_p = a.rgb, a.sensor, a.mono
    else:
        p.error("donner --sequence-dir, ou --rgb, --sensor et --mono")
    for f in (rgb_p, sensor_p, mono_p):
        if not f.exists():
            p.error(f"fichier introuvable : {f}")

    rgb = Image.open(rgb_p)
    sensor = load_depth(sensor_p)
    mono = load_depth(mono_p)
    if mono.shape != sensor.shape:
        mono = np.array(Image.fromarray(mono).resize(sensor.shape[::-1], Image.NEAREST), dtype=np.float32)

    model = build_model(a.checkpoint, a.device)
    fused, conf, dt = run(model, rgb, mono, sensor, a.device)

    valid = sensor > 0.05
    holes = ~valid
    mae_mm = float(np.abs(fused[valid] - sensor[valid]).mean() * 1000.0) if valid.any() else float("nan")
    filled = float((fused[holes] > 0).mean() * 100.0) if holes.any() else 100.0

    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "depth_fused.npy", fused.astype(np.float32))
    np.save(a.out / "confidence.npy", conf.astype(np.float32))
    save_figure(a.out / "demo_asaf.png", rgb, sensor, fused, conf)

    print(f"Trame            : {rgb_p.name}  ({sensor.shape[1]}x{sensor.shape[0]})")
    print(f"Pixels sans mesure capteur : {holes.mean() * 100:.1f} %  -> complétés : {filled:.1f} %")
    print(f"Écart au capteur (pixels mesurés) : {mae_mm:.1f} mm")
    print(f"Inférence        : {dt:.0f} ms sur {a.device} (hors modèle monoculaire)")
    print(f"Résultats        : {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
