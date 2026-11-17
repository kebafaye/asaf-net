"""Lecture / écriture trajectoires format TUM (evo)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

TUMRow = Tuple[float, float, float, float, float, float, float, float]


def save_tum(path: str | Path, rows: List[TUMRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, tx, ty, tz, qx, qy, qz, qw in rows:
            f.write(f"{ts:.9f} {tx:.9f} {ty:.9f} {tz:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")


def load_tum(path: str | Path) -> np.ndarray:
    """Retourne (N, 8) : ts, tx, ty, tz, qx, qy, qz, qw."""
    path = Path(path)
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])
    if not rows:
        raise ValueError(f"Aucune pose dans {path}")
    return np.asarray(rows, dtype=np.float64)


def pose_msg_to_tum_row(stamp_sec: float, position, orientation) -> TUMRow:
    return (
        stamp_sec,
        float(position.x),
        float(position.y),
        float(position.z),
        float(orientation.x),
        float(orientation.y),
        float(orientation.z),
        float(orientation.w),
    )
