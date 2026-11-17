"""Checkpoint I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    out_dir: str | Path,
    *,
    is_best: bool = False,
    epoch: Optional[int] = None,
    save_every: int = 0,
    filename: str = "checkpoint.pth",
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    torch.save(state, path)
    if save_every and epoch and epoch % save_every == 0:
        torch.save(state, out / f"checkpoint_epoch{epoch}.pth")
    if is_best:
        torch.save(state, out / "best_model.pth")
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Tuple[int, float, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_rmse = float(ckpt.get("best_val_rmse", float("inf")))
    return start_epoch, best_rmse, ckpt
