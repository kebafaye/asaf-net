"""Publication-quality depth visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def save_error_heatmap(
    pred: np.ndarray,
    gt: np.ndarray,
    path: str | Path,
    *,
    mask: Optional[np.ndarray] = None,
) -> None:
    err = np.abs(pred - gt)
    if mask is not None:
        err = np.where(mask, err, np.nan)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(err, cmap="hot")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Absolute Error")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_depth_comparison(
    rgb: np.ndarray,
    mono: np.ndarray,
    gt: np.ndarray,
    fused: np.ndarray,
    path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    titles = ["RGB", "Mono Depth", "GT", "Fused"]
    data = [rgb, mono, gt, fused]
    for ax, title, d in zip(axes, titles, data):
        if d.ndim == 3:
            ax.imshow(d)
        else:
            d_n = (d - d.min()) / (d.max() - d.min() + 1e-8)
            ax.imshow(d_n, cmap="inferno")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
