#!/usr/bin/env python3
"""
Exporte une séquence RGB-D compatible ORB-SLAM3 (TUM).

Modes :
  - sensor : profondeur capteur uint16 mm
  - asaf   : profondeur ASAF-Net (checkpoint métrique) → uint16 mm

Usage :
  .venv-ros312/bin/python slam/export_orbslam_depth.py \\
    --mode asaf \\
    --checkpoint checkpoints/experiments/asaf_net_v2_vigs/best_model.pth \\
    --csv dataset_fusion/val.csv \\
    --frames-csv data/vigs_fusion/Rotation-Only/extracted/frames.csv \\
    --output evaluation_results/orbslam3/Rotation-Only/asaf/sequence
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth_fusion.models.asaf_net import ASAFNetV2, ASAFNetV2Config
from depth_fusion.utils.checkpoint import load_checkpoint

SIZE = 384


def to_meters(d: np.ndarray) -> np.ndarray:
    """Convertit mm→m si la médiane des valeurs valides > 100."""
    d = d.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if valid.any() and float(np.median(d[valid])) > 100.0:
        d = d / 1000.0
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def build_frame_table(frames_csv: Path) -> pd.DataFrame:
    """Table de référence d'UNE séquence : rgb, depth_sensor et timestamp
    proviennent tous de la même ligne, donc de la même séquence.

    Ne JAMAIS indexer par stem : toutes les séquences du dataset VIGS-Fusion
    nomment leurs images image_0000.png, image_0001.png… Les stems se
    recouvrent donc intégralement d'une séquence à l'autre, et une jointure par
    stem accepte silencieusement de marier les images d'une séquence aux
    timestamps (puis au ground truth) d'une autre.
    """
    if not frames_csv.is_file():
        raise SystemExit(f"frames-csv introuvable : {frames_csv}")
    df = pd.read_csv(frames_csv)
    missing = {"rgb", "depth_sensor", "timestamp_ns"} - set(df.columns)
    if missing:
        raise SystemExit(f"{frames_csv} : colonnes manquantes {sorted(missing)}")
    df = df.copy()
    df["rgb_key"] = df["rgb"].map(lambda p: str(Path(str(p)).resolve()))
    df["ts"] = df["timestamp_ns"].astype(float) * 1e-9
    return df


def restrict_to_subset(frames: pd.DataFrame, subset_csv: Path) -> pd.DataFrame:
    """Restreint la séquence aux frames listées dans subset_csv.

    L'appariement se fait sur le chemin absolu résolu, jamais sur le stem.
    Toute frame du subset absente de la séquence est une erreur fatale : c'est
    le signe qu'on mélange deux séquences.
    """
    sub = pd.read_csv(subset_csv)
    if "rgb" not in sub.columns:
        raise SystemExit(f"{subset_csv} : colonne 'rgb' manquante")
    keys = sub["rgb"].map(lambda p: str(Path(str(p)).resolve()))
    known = set(frames["rgb_key"])
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise SystemExit(
            f"{len(unknown)}/{len(keys)} frames de {subset_csv} n'appartiennent pas à "
            f"la séquence décrite par --frames-csv.\n"
            f"  exemple : {unknown[0]}\n"
            "Les images de deux séquences différentes ne doivent jamais être mélangées : "
            "les timestamps et le ground truth seraient ceux d'une autre trajectoire.\n"
            "Utilisez le frames.csv de la séquence dont proviennent réellement ces images."
        )
    out = frames[frames["rgb_key"].isin(set(keys))]
    return out.sort_values("ts").reset_index(drop=True)


def resize_depth_m(d_m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest resize pour cartes de profondeur métriques."""
    return np.array(
        Image.fromarray(d_m.astype(np.float32), mode="F").resize(size, Image.NEAREST),
        dtype=np.float32,
    )


def depth_to_uint16_mm(d_m: np.ndarray) -> np.ndarray:
    mm = np.clip(d_m * 1000.0, 0, 65535)
    return mm.astype(np.uint16)


def main() -> None:
    p = argparse.ArgumentParser(description="Export séquence ORB-SLAM3 RGB-D")
    p.add_argument("--mode", choices=("asaf", "sensor"), required=True)
    p.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints/experiments/asaf_net_v2_vigs/best_model.pth"),
    )
    p.add_argument(
        "--frames-csv",
        required=True,
        help="frames.csv de LA séquence à exporter (source unique : rgb, "
        "depth_sensor et timestamp viennent de la même ligne)",
    )
    p.add_argument(
        "--csv",
        default=None,
        help="optionnel : restreint aux frames de ce split. Doit lister des "
        "images de la séquence donnée par --frames-csv, sinon erreur.",
    )
    p.add_argument(
        "--depth-any-dir",
        default=None,
        help="mode asaf : dossier des a priori monoculaires DA3 (.npy), "
        "nommés depth_XXXX.npy sur le frame_id de la séquence",
    )
    p.add_argument("--output", required=True)
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="index de la 1re frame de la séquence (fenêtre [start, start+max-frames))",
    )
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--size", type=int, default=SIZE)
    args = p.parse_args()

    out = Path(args.output)
    rgb_out = out / "rgb"
    depth_out = out / "depth"
    rgb_out.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    frames_csv = Path(args.frames_csv)
    df = build_frame_table(frames_csv)
    if args.csv:
        df = restrict_to_subset(df, Path(args.csv))
    else:
        df = df.sort_values("ts").reset_index(drop=True)
    if args.start > 0:
        df = df.iloc[args.start :].reset_index(drop=True)
    if args.max_frames > 0:
        df = df.head(args.max_frames)
    if df.empty:
        raise SystemExit("aucune frame à exporter")

    # ORB-SLAM3 lit associations.txt dans l'ordre : timestamps strictement croissants
    ts_all = df["ts"].to_numpy()
    if not np.all(np.diff(ts_all) > 0):
        raise SystemExit("timestamps non strictement croissants dans la séquence")

    depth_any_dir = Path(args.depth_any_dir) if args.depth_any_dir else None
    if args.mode == "asaf" and depth_any_dir is None:
        raise SystemExit(
            "mode asaf : --depth-any-dir requis (a priori monoculaire DA3 de la séquence).\n"
            "Précalcul : scripts/precompute_da3_sequences.py --sequences <NOM>"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    tr = None
    if args.mode == "asaf":
        model = ASAFNetV2(ASAFNetV2Config(pretrained=False)).to(device)
        load_checkpoint(args.checkpoint, model, device=str(device))
        model.eval()
        tr = T.Compose(
            [
                T.Resize((args.size, args.size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        print(f"ASAF-Net sur {device} ← {args.checkpoint}")

    assoc: list[str] = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df), desc=f"Export {args.mode}")):
        rgb_path = Path(row.rgb)
        rgb_pil = Image.open(rgb_path).convert("RGB")
        W, H = rgb_pil.size

        sensor_m = to_meters(np.load(row.depth_sensor))
        if sensor_m.shape[:2] != (H, W):
            sensor_m = resize_depth_m(sensor_m, (W, H))

        if args.mode == "sensor":
            depth_m = sensor_m.copy()
            valid = (depth_m > 0.05) & (depth_m < 20.0)
            depth_m[~valid] = 0.0
        else:
            assert depth_any_dir is not None
            mono_path = depth_any_dir / f"depth_{int(row.frame_id):04d}.npy"
            if not mono_path.is_file():
                raise SystemExit(
                    f"a priori DA3 manquant : {mono_path}\n"
                    f"Précalculez la séquence avant l'export asaf."
                )
            mono_m = to_meters(np.load(mono_path))
            mono_r = resize_depth_m(mono_m, (args.size, args.size))
            sensor_r = resize_depth_m(sensor_m, (args.size, args.size))
            valid_r = ((sensor_r > 0.05) & (sensor_r < 20.0)).astype(np.float32)

            assert model is not None and tr is not None
            with torch.no_grad():
                out_pred = model(
                    tr(rgb_pil).unsqueeze(0).to(device),
                    torch.from_numpy(mono_r)[None, None].to(device),
                    torch.from_numpy(sensor_r)[None, None].to(device),
                    torch.from_numpy(valid_r)[None, None].to(device),
                )
            pred = out_pred["depth"].squeeze().cpu().numpy().astype(np.float32)
            depth_m = resize_depth_m(pred, (W, H))
            # garder uniquement une profondeur physiquement plausible
            bad = ~np.isfinite(depth_m) | (depth_m < 0.05) | (depth_m > 20.0)
            depth_m[bad] = 0.0

        # Timestamp issu de la même ligne que l'image : aucune jointure, donc
        # aucun risque de le prendre à une autre séquence.
        ts = float(row.ts)

        stem6 = f"{i:06d}"
        rgb_rel = f"rgb/{stem6}.png"
        depth_rel = f"depth/{stem6}.png"
        rgb_pil.save(out / rgb_rel)
        Image.fromarray(depth_to_uint16_mm(depth_m)).save(out / depth_rel)
        assoc.append(f"{ts:.6f} {rgb_rel} {ts:.6f} {depth_rel}")

    (out / "associations.txt").write_text("\n".join(assoc) + "\n", encoding="utf-8")
    meta = out / "export_meta.txt"
    # Tracer la provenance : indispensable pour savoir contre quel poses_gt.tum
    # cette séquence peut légitimement être évaluée.
    meta.write_text(
        f"mode={args.mode}\n"
        f"frames={len(assoc)}\n"
        f"frames_csv={frames_csv.resolve()}\n"
        f"sequence={frames_csv.resolve().parents[1].name}\n"
        f"subset_csv={args.csv or '(séquence complète)'}\n"
        f"depth_any_dir={args.depth_any_dir or '-'}\n"
        f"checkpoint={args.checkpoint if args.mode == 'asaf' else '-'}\n"
        f"t_start={ts_all[0]:.6f}\nt_end={ts_all[-1]:.6f}\n",
        encoding="utf-8",
    )
    print(f"Exporté {len(assoc)} frames → {out}")
    print(f"associations.txt : format TUM (ts rgb ts depth)")


if __name__ == "__main__":
    main()
