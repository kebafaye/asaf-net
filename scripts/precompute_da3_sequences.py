#!/usr/bin/env python3
"""Précalcul DA3 + alignement métrique RANSAC pour séquences VIGS-Fusion."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from depth_fusion.fusion.scale_align import align_depth_ransac


def _stub_optional_da3_deps() -> None:
    """DA3 importe des modules optionnels (pycolmap, gsplat, moviepy…) — stub si absents."""
    for name in ("pycolmap", "gsplat", "open3d", "trimesh"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    if "moviepy.editor" not in sys.modules:
        mpy = types.ModuleType("moviepy")
        mpy.editor = types.ModuleType("moviepy.editor")  # type: ignore[attr-defined]
        sys.modules["moviepy"] = mpy
        sys.modules["moviepy.editor"] = mpy.editor


def _load_da3_model(repo_path: Path, model_id: str, device: str):
    src_path = str(repo_path / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    _stub_optional_da3_deps()

    import torch
    from depth_anything_3.api import DepthAnything3

    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = DepthAnything3.from_pretrained(model_id).to(dev)
    model.eval()
    return model, dev


def _infer_depth(model, device, rgb_path: Path, process_res: int) -> np.ndarray:
    import torch

    image = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    with torch.no_grad():
        pred = model.inference(
            [image],
            process_res=process_res,
            process_res_method="upper_bound_resize",
            export_format="mini_npz",
        )
    return pred.depth[0].astype(np.float32)


def _load_npy_uint16_fallback(sensor_path: Path) -> np.ndarray | None:
    """Charge un .npy dont l'en-tête shape ne correspond pas au payload (fichier tronqué)."""
    import struct

    with sensor_path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            return None
        major, minor = f.read(2)
        if major == 1:
            hlen = struct.unpack("<H", f.read(2))[0]
        else:
            hlen = struct.unpack("<I", f.read(4))[0]
        f.read(hlen)
        payload = f.read()
    if not payload:
        return None
    data = np.frombuffer(payload, dtype=np.uint16)
    n = int(data.size)
    for width in (848, 640, 480, 424, 320, 256):
        if n % width == 0:
            height = n // width
            if 90 <= height <= 720:
                return data.reshape(height, width)
    side = int(round(n**0.5))
    if side > 0 and side * side == n:
        return data.reshape(side, side)
    return None


def _load_sensor_mm(sensor_path: Path) -> np.ndarray | None:
    try:
        d = np.load(sensor_path)
    except ValueError:
        d = _load_npy_uint16_fallback(sensor_path)
        if d is None:
            return None
    if d.ndim != 2 or d.size == 0:
        return None
    return d.astype(np.float32)


def _resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return np.array(Image.fromarray(depth).resize((w, h), Image.BILINEAR), dtype=np.float32)


def precompute_sequence(
    seq: str,
    vigs_root: Path,
    model,
    *,
    process_res: int = 504,
    skip_existing: bool = True,
    limit: int = 0,
    start: int = 0,
) -> tuple[int, int]:
    frames_csv = vigs_root / seq / "extracted" / "frames.csv"
    if not frames_csv.exists():
        # Séquence non extraite sur cette machine : on la saute au lieu
        # d'interrompre tout le précalcul. Permet de tourner avec un
        # sous-ensemble des séquences de sequences.yaml (p.ex. seulement
        # train+val) sans que les séquences de test absentes fassent échouer
        # le pipeline complet.
        print(f"  SKIP {seq}: frames.csv absent ({frames_csv})")
        return 0, 0

    out_dir = vigs_root / seq / "extracted" / "da3_scaled"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(frames_csv)
    # Fenêtre [start, start+limit) : permet de cibler une phase précise de la
    # trajectoire (p.ex. la phase rotation de Line-Rot, frames 1200-2400)
    # sans payer le précalcul de la séquence entière.
    if start > 0:
        df = df.iloc[start:]
    if limit > 0:
        df = df.head(limit)

    extracted = vigs_root / seq / "extracted"
    done = 0
    skipped = 0
    bad = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"DA3 {seq}"):
        rgb = extracted / "rgb" / Path(str(row["rgb"])).name
        sensor = extracted / "depth_sensor" / Path(str(row["depth_sensor"])).name
        out_path = out_dir / sensor.name
        if skip_existing and out_path.is_file():
            try:
                arr = np.load(out_path)
                if arr.ndim == 2 and arr.size > 0:
                    skipped += 1
                    continue
            except (OSError, ValueError, EOFError):
                pass
            out_path.unlink(missing_ok=True)
        if not rgb.is_file() or not sensor.is_file():
            bad += 1
            continue

        mono = _infer_depth(model, None, rgb, process_res)
        metric = _load_sensor_mm(sensor)
        if metric is None:
            tqdm.write(f"  WARN skip {sensor.name}: depth capteur illisible")
            bad += 1
            continue
        mono_rs = _resize_depth(mono, (metric.shape[1], metric.shape[0]))
        aligned, _, _ = align_depth_ransac(mono_rs, metric)
        np.save(out_path, aligned.astype(np.float32))
        done += 1
    if bad:
        print(f"  {seq}: {bad} frames ignorées (fichiers manquants ou depth corrompue)")
    return done, skipped


def _default_sequences(vigs_root: Path, split: str) -> list[str]:
    with open(vigs_root / "sequences.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if split == "train":
        return list(cfg["train_seqs"])
    if split == "val":
        return list(cfg["val_seqs"])
    if split == "test":
        return list(cfg.get("test_seqs", []))
    if split == "all":
        return (
            list(cfg["train_seqs"])
            + list(cfg["val_seqs"])
            + list(cfg.get("test_seqs", []))
        )
    raise ValueError(f"split inconnu: {split}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--sequences", nargs="*", default=None)
    p.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    p.add_argument(
        "--repo-path",
        type=Path,
        default=None,
        help="Depth-Anything-3 repo (défaut: memoire_upgrade_da3/Depth-Anything-3)",
    )
    p.add_argument("--model-id", default="depth-anything/DA3MONO-LARGE")
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Max frames par séquence (debug)")
    p.add_argument("--start", type=int, default=0, help="Index de la 1re frame (fenêtre [start, start+limit))")
    args = p.parse_args()

    root = args.root.resolve()
    vigs_root = root / "data" / "vigs_fusion"
    repo_path = args.repo_path or (root / "memoire_upgrade_da3" / "Depth-Anything-3")
    if not repo_path.is_dir():
        raise RuntimeError(f"repo DA3 introuvable: {repo_path}")

    sequences = args.sequences or _default_sequences(vigs_root, args.split)
    print(f"Séquences: {', '.join(sequences)}")
    print(f"Repo DA3: {repo_path}")

    model, device = _load_da3_model(repo_path, args.model_id, args.device)
    print(f"Device: {device}")

    total_done = 0
    total_skip = 0
    for seq in sequences:
        done, skipped = precompute_sequence(
            seq,
            vigs_root,
            model,
            process_res=args.process_res,
            skip_existing=not args.no_skip_existing,
            limit=args.limit,
            start=args.start,
        )
        total_done += done
        total_skip += skipped
        print(f"  {seq}: {done} nouvelles, {skipped} déjà présentes")

    print(f"Terminé — {total_done} frames calculées, {total_skip} ignorées (déjà là)")


if __name__ == "__main__":
    main()
