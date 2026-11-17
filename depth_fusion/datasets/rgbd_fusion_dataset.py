"""
RGB-D fusion dataset with optional sensor depth channel.

CSV columns: rgb, depth_any, depth_gt [, depth_sensor]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class RGBDFusionDataset(Dataset):
    def __init__(
        self,
        csv_file: str | Path,
        transform: Optional[T.Compose] = None,
        *,
        use_sensor_depth: bool = False,
        sensor_col: str = "depth_sensor",
        normalize_mode: str = "mono_max",  # mono_max | global | none
        global_scale: float = 1.0,
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.use_sensor_depth = use_sensor_depth and sensor_col in self.df.columns
        self.sensor_col = sensor_col
        self.normalize_mode = normalize_mode
        self.global_scale = global_scale
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _load_depth(path: str) -> np.ndarray:
        path = str(path)
        if path.endswith(".npy"):
            d = np.load(path)
        else:
            d = np.array(Image.open(path).convert("F"), dtype=np.float32)
        return d.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        rgb = Image.open(row["rgb"]).convert("RGB")
        d_any = self._load_depth(row["depth_any"])
        d_gt = self._load_depth(row["depth_gt"])

        d_sensor = None
        if self.use_sensor_depth:
            d_sensor = self._load_depth(row[self.sensor_col])

        if self.transform:
            rgb_t = self.transform(rgb)
        else:
            rgb_t = T.Compose([
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(rgb)

        H, W = rgb_t.shape[1], rgb_t.shape[2]
        d_any_r = np.array(Image.fromarray(d_any).resize((W, H), Image.BILINEAR), dtype=np.float32)
        d_gt_r = np.array(Image.fromarray(d_gt).resize((W, H), Image.BILINEAR), dtype=np.float32)

        d_any_t = torch.from_numpy(d_any_r)[None]
        d_gt_t = torch.from_numpy(d_gt_r)[None]

        if self.normalize_mode == "mono_max":
            scale = max(d_any_t.max().item(), 1.0)
        elif self.normalize_mode == "global":
            scale = self.global_scale
        else:
            scale = 1.0

        d_any_t = d_any_t / scale
        d_gt_t = d_gt_t / scale

        sample: Dict[str, Any] = {
            "rgb": rgb_t,
            "mono": d_any_t,
            "gt": d_gt_t,
            "scale": scale,
            "input": torch.cat([rgb_t, d_any_t], dim=0),
        }

        if d_sensor is not None:
            d_s_r = np.array(Image.fromarray(d_sensor).resize((W, H), Image.BILINEAR), dtype=np.float32)
            d_s_t = torch.from_numpy(d_s_r)[None] / scale
            sample["sensor"] = d_s_t

        return sample


def build_dataloaders(
    train_csv: str,
    val_csv: str,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    use_sensor_depth: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_tf = T.Compose([
        T.Resize(image_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = RGBDFusionDataset(train_csv, train_tf, use_sensor_depth=use_sensor_depth, image_size=image_size)
    val_set = RGBDFusionDataset(val_csv, val_tf, use_sensor_depth=use_sensor_depth, image_size=image_size)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Dataset V2 — corrections : unités métriques, resize NEAREST des profondeurs,
# masque de validité, augmentations géométriques JOINTES (RGB + profondeurs).
# ---------------------------------------------------------------------------
#
# Bugs V1 corrigés :
#   * RandomHorizontalFlip appliqué au RGB seul → 50 % des paires
#     d'entraînement étaient désalignées. V2 : flip joint et déterministe.
#   * Resize bilinéaire des profondeurs → les trous (0) contaminaient les
#     pixels valides aux frontières. V2 : NEAREST pour toutes les profondeurs.
#   * Normalisation par max mono par image (instable). V2 : sorties en mètres,
#     la normalisation fixe est gérée par le modèle (depth_scale_m).


class RGBDFusionDatasetV2(Dataset):
    """
    CSV attendu : colonnes rgb, depth_any, depth_gt [, depth_sensor].

    Sorties (float32) :
      rgb        (3,H,W) normalisé ImageNet
      mono       (1,H,W) mètres
      sensor     (1,H,W) mètres, 0 aux invalides   (si depth_sensor présent)
      valid      (1,H,W) ∈ {0,1} validité capteur
      gt         (1,H,W) mètres
      valid_gt   (1,H,W) ∈ {0,1} validité GT
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        csv_file: "str | Path",
        *,
        image_size: Tuple[int, int] = (384, 384),
        use_sensor_depth: bool = True,
        sensor_col: str = "depth_sensor",
        augment: bool = False,
        depth_unit: str = "auto",   # auto | mm | m
        min_depth_m: float = 0.05,
        max_depth_m: float = 20.0,
        synthetic_holes: float = 0.0,
        synthetic_holes_max: "float | None" = None,
        curriculum_epochs: int = 0,
        holes_seed: "int | None" = None,
    ):
        """
        synthetic_holes : fraction approximative de pixels capteur VALIDES
        masqués artificiellement dans l'ENTRÉE (la GT les conserve).

        La GT étant la profondeur capteur elle-même, sans cette corruption le
        raffinement résiduel peut recopier l'entrée → tâche triviale (MAE=0).
        Les trous synthétiques forcent une vraie complétion métrique depuis
        RGB + mono, mesurable sur des valeurs capteur réelles retenues
        (protocole standard en depth completion, cf. Uhrig et al. 2017).
        holes_seed : si défini, corruption déterministe par index (val/test).
        """
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size
        self.use_sensor_depth = use_sensor_depth and sensor_col in self.df.columns
        self.sensor_col = sensor_col
        self.augment = augment
        self.depth_unit = depth_unit
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.synthetic_holes_base = float(synthetic_holes)
        self.synthetic_holes = float(synthetic_holes)
        self.synthetic_holes_max = (
            float(synthetic_holes_max) if synthetic_holes_max is not None else float(synthetic_holes)
        )
        self.curriculum_epochs = int(curriculum_epochs)
        self.holes_seed = holes_seed
        self.color_jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.normalize = T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)

    @staticmethod
    def make_drop_mask(
        h: int, w: int, rng: np.random.RandomState, target_frac: float
    ) -> np.ndarray:
        """Masque de trous synthétiques : ellipses + rectangles aléatoires
        jusqu'à ~target_frac de l'image. Retourne bool (h, w)."""
        drop = np.zeros((h, w), dtype=bool)
        yy, xx = np.ogrid[:h, :w]
        for _ in range(40):
            if drop.mean() >= target_frac:
                break
            cy, cx = rng.randint(0, h), rng.randint(0, w)
            ry = max(2, int(rng.uniform(0.03, 0.14) * h))
            rx = max(2, int(rng.uniform(0.03, 0.14) * w))
            if rng.rand() < 0.5:
                drop |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
            else:
                drop[max(0, cy - ry): cy + ry, max(0, cx - rx): cx + rx] = True
        return drop

    def __len__(self) -> int:
        return len(self.df)

    def set_epoch(self, epoch: int) -> None:
        """Curriculum de trous : la fraction de pixels capteur masqués en entrée
        croît linéairement de `synthetic_holes_base` à `synthetic_holes_max`
        sur `curriculum_epochs`. Sans effet si curriculum_epochs <= 0 ou si
        base == max. Le réseau apprend d'abord la complétion facile (peu de
        trous) puis des trous plus larges — meilleure généralisation métrique
        et transition d'entraînement plus stable."""
        if self.curriculum_epochs <= 0 or self.synthetic_holes_max == self.synthetic_holes_base:
            return
        frac = min(1.0, max(0.0, epoch / float(self.curriculum_epochs)))
        self.synthetic_holes = (
            self.synthetic_holes_base
            + frac * (self.synthetic_holes_max - self.synthetic_holes_base)
        )

    def _to_meters(self, d: np.ndarray) -> np.ndarray:
        d = d.astype(np.float32)
        if self.depth_unit == "mm":
            return d / 1000.0
        if self.depth_unit == "m":
            return d
        v = d[np.isfinite(d) & (d > 0)]
        if v.size and float(np.median(v)) > 100.0:  # heuristique : médiane >100 → mm
            return d / 1000.0
        return d

    @staticmethod
    def _load_depth_raw(path: str) -> np.ndarray:
        path = str(path)
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        return np.array(Image.open(path).convert("F"), dtype=np.float32)

    def _load_depth_m(self, path: str, size: Tuple[int, int]) -> np.ndarray:
        d = self._to_meters(self._load_depth_raw(path))
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        W, H = size
        # NEAREST : préserve les trous, aucune contamination des frontières.
        return np.array(
            Image.fromarray(d).resize((W, H), Image.NEAREST), dtype=np.float32
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        H, W = self.image_size

        rgb = Image.open(row["rgb"]).convert("RGB").resize((W, H), Image.BILINEAR)
        mono = self._load_depth_m(row["depth_any"], (W, H))
        gt = self._load_depth_m(row["depth_gt"], (W, H))
        sensor = (
            self._load_depth_m(row[self.sensor_col], (W, H))
            if self.use_sensor_depth
            else np.zeros_like(gt)
        )

        do_flip = self.augment and np.random.rand() < 0.5
        if do_flip:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            mono = mono[:, ::-1].copy()
            gt = gt[:, ::-1].copy()
            sensor = sensor[:, ::-1].copy()

        if self.augment:
            rgb = self.color_jitter(rgb)

        rgb_t = self.normalize(T.functional.to_tensor(rgb))

        def _mask(d: np.ndarray) -> np.ndarray:
            return ((d > self.min_depth_m) & (d < self.max_depth_m)).astype(np.float32)

        valid = _mask(sensor)
        valid_gt = _mask(gt)

        # Trous synthétiques : masqués dans l'ENTRÉE, conservés dans la GT.
        drop = np.zeros_like(valid)
        if self.synthetic_holes > 0:
            seed = None if self.holes_seed is None else (self.holes_seed + idx) % (2**31)
            rng = np.random.RandomState(seed)
            drop_b = self.make_drop_mask(H, W, rng, self.synthetic_holes)
            drop = (drop_b & (valid > 0)).astype(np.float32)
            valid = valid * (1.0 - drop)

        return {
            "rgb": rgb_t,
            "mono": torch.from_numpy(mono)[None],
            "sensor": torch.from_numpy(sensor * valid)[None],
            "valid": torch.from_numpy(valid)[None],
            "gt": torch.from_numpy(gt * valid_gt)[None],
            "valid_gt": torch.from_numpy(valid_gt)[None],
            "drop_mask": torch.from_numpy(drop)[None],
        }


def build_dataloaders_v2(
    train_csv: str,
    val_csv: str,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (384, 384),
    use_sensor_depth: bool = True,
    depth_unit: str = "auto",
    synthetic_holes: float = 0.15,
    synthetic_holes_max: "float | None" = None,
    curriculum_epochs: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    train_set = RGBDFusionDatasetV2(
        train_csv,
        image_size=image_size,
        use_sensor_depth=use_sensor_depth,
        augment=True,
        depth_unit=depth_unit,
        synthetic_holes=synthetic_holes,
        synthetic_holes_max=synthetic_holes_max,
        curriculum_epochs=curriculum_epochs,
        holes_seed=None,  # aléatoire à l'entraînement
    )
    val_set = RGBDFusionDatasetV2(
        val_csv,
        image_size=image_size,
        use_sensor_depth=use_sensor_depth,
        augment=False,
        depth_unit=depth_unit,
        synthetic_holes=synthetic_holes,
        holes_seed=123,  # déterministe en validation
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
