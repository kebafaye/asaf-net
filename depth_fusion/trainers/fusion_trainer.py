"""Research-grade trainer with AMP, TensorBoard, W&B, multi-loss."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from depth_fusion.evaluation.metrics import DepthMetrics, compute_depth_metrics
from depth_fusion.losses.depth_losses import (
    DepthFusionLoss,
    DepthFusionLossV2,
    DepthLossConfig,
    DepthLossV2Config,
)
from depth_fusion.models.asaf_net import ASAFNet, ASAFNetConfig, ASAFNetV2
from depth_fusion.models.resnet_unet import ResNetUNet
from depth_fusion.utils.checkpoint import load_checkpoint, save_checkpoint
from depth_fusion.utils.logging import setup_logger


class FusionTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        *,
        device: str = "cuda",
        lr: float = 1e-4,
        epochs: int = 60,
        checkpoint_dir: str = "checkpoints",
        log_csv: Optional[str] = None,
        loss_cfg: Optional[DepthLossConfig] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        resume: str = "",
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
        use_ema: bool = False,
        ema_decay: float = 0.999,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_csv = log_csv
        self.logger = setup_logger()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # Warmup linéaire (montée en LR) puis annealing cosinus : stabilise le
        # début d'entraînement (poids pré-entraînés + têtes neuves) et lisse la
        # convergence finale. warmup_epochs=0 ⇒ cosinus pur (comportement V2).
        self.warmup_epochs = max(0, int(warmup_epochs))

        def _lr_factor(epoch: int) -> float:
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            import math
            prog = (epoch - self.warmup_epochs) / max(1, epochs - self.warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, _lr_factor)
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # EMA des poids : moyenne mobile exponentielle des paramètres, évaluée
        # et sauvegardée à la place des poids bruts. Réduit le bruit des
        # dernières époques → checkpoint final plus stable (meilleur pour un
        # usage SLAM downstream). use_ema=False ⇒ comportement V2.
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.ema_params: Dict[str, torch.Tensor] = {}
        self._ema_backup: Dict[str, torch.Tensor] = {}
        if self.use_ema:
            self.ema_params = {
                k: v.detach().clone().float()
                for k, v in self.model.named_parameters()
            }
        self.is_v2 = isinstance(model, ASAFNetV2)
        if self.is_v2:
            v2_cfg = loss_cfg if isinstance(loss_cfg, DepthLossV2Config) else DepthLossV2Config()
            self.criterion = DepthFusionLossV2(v2_cfg)
        else:
            self.criterion = DepthFusionLoss(loss_cfg or DepthLossConfig())
        self.grad_clip_norm = 1.0

        self.best_rmse = float("inf")
        self.start_epoch = 1
        if resume:
            self.start_epoch, self.best_rmse, _ = load_checkpoint(
                resume, self.model, self.optimizer, str(self.device)
            )

        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "tb"))
            except ImportError:
                pass

        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project="depth-fusion", dir=str(self.checkpoint_dir))
                self.wandb = wandb
            except ImportError:
                self.logger.warning("wandb not installed")

        self.epoch_log_path: Optional[Path] = None
        if log_csv:
            self._init_csv(log_csv)
            # Fichier texte lisible à la racine logs/ (sans barres tqdm)
            logs_dir = Path(log_csv).resolve().parents[2] / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            exp_name = Path(log_csv).resolve().parents[1].name
            self.epoch_log_path = logs_dir / f"{exp_name}_epochs.log"
            if not self.epoch_log_path.exists():
                self.epoch_log_path.write_text(
                    "# ASAF-Net training — une ligne par epoch\n"
                    "# epoch | train_loss | val_mae | val_rmse | delta1 | lr\n\n",
                    encoding="utf-8",
                )

    def _init_csv(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            with open(p, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "epoch", "train_loss", "val_mae", "val_rmse", "val_abs_rel",
                        "delta1", "drop_mae", "drop_rmse", "drop_delta1", "lr",
                    ]
                )

    def _log_csv(self, path: str, row: list) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    @torch.no_grad()
    def _ema_update(self) -> None:
        if not self.use_ema:
            return
        d = self.ema_decay
        for k, v in self.model.named_parameters():
            self.ema_params[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def _ema_swap_in(self) -> None:
        """Charge les poids EMA dans le modèle (buffers BN conservés)."""
        if not self.use_ema:
            return
        self._ema_backup = {k: v.detach().clone() for k, v in self.model.named_parameters()}
        for k, v in self.model.named_parameters():
            v.copy_(self.ema_params[k].to(v.dtype))

    @torch.no_grad()
    def _ema_swap_out(self) -> None:
        if not self.use_ema or not self._ema_backup:
            return
        for k, v in self.model.named_parameters():
            v.copy_(self._ema_backup[k])
        self._ema_backup = {}

    def _forward_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if isinstance(self.model, ASAFNetV2):
            sensor = batch.get("sensor")
            valid = batch.get("valid")
            return self.model(
                batch["rgb"].to(self.device),
                batch["mono"].to(self.device),
                sensor.to(self.device) if sensor is not None else None,
                valid.to(self.device) if valid is not None else None,
            )
        if isinstance(self.model, ASAFNet):
            sensor = batch.get("sensor")
            if sensor is not None:
                sensor = sensor.to(self.device)
            out = self.model(
                batch["rgb"].to(self.device),
                batch["mono"].to(self.device),
                sensor,
            )
            return out
        inp = batch["input"].to(self.device)
        return {"depth": self.model(inp)}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")

        n_batches = len(self.train_loader)
        progress_file = self.checkpoint_dir.parent.parent / "logs" / "progress_live.txt"

        for batch_idx, batch in enumerate(pbar):
            gt = batch["gt"].to(self.device)
            out = self._forward_batch(batch)
            pred = out["depth"]
            rgb = batch["rgb"].to(self.device)
            conf = out.get("confidence")

            if pred.shape[-2:] != gt.shape[-2:]:
                gt = torch.nn.functional.interpolate(gt, pred.shape[-2:], mode="bilinear", align_corners=False)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                if self.is_v2:
                    loss, _ = self.criterion(
                        pred,
                        gt,
                        rgb=rgb,
                        laplace_b=out.get("laplace_b"),
                        valid_sensor=batch.get("valid").to(self.device) if batch.get("valid") is not None else None,
                    )
                else:
                    loss, _ = self.criterion(pred, gt, rgb=rgb, confidence=conf)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._ema_update()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

            # Fichier lisible pour l'IDE (mis à jour toutes les 5 batches)
            if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
                pct = 100.0 * (batch_idx + 1) / n_batches
                progress_file.parent.mkdir(parents=True, exist_ok=True)
                progress_file.write_text(
                    f"Entraînement ASAF-Net\n"
                    f"{'=' * 40}\n"
                    f"Epoch     : {epoch} / {self.epochs}\n"
                    f"Batch     : {batch_idx + 1} / {n_batches} ({pct:.1f}%)\n"
                    f"Loss moy. : {avg_loss:.6f}\n"
                    f"Device    : {self.device}\n"
                    f"Dernière mise à jour : epoch {epoch}, batch {batch_idx + 1}\n",
                    encoding="utf-8",
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> "tuple[DepthMetrics, DepthMetrics]":
        """Retourne (métriques globales, métriques sur trous synthétiques).

        Les métriques 'drop' (pixels capteur retenus en entrée mais supervisés)
        mesurent la vraie capacité de complétion métrique — les métriques
        globales sont dominées par les pixels où l'entrée contient déjà la GT.
        """
        self.model.eval()
        agg = DepthMetrics()
        agg_drop = DepthMetrics()

        for batch in self.val_loader:
            gt = batch["gt"].to(self.device)
            out = self._forward_batch(batch)
            pred = out["depth"]
            if pred.shape[-2:] != gt.shape[-2:]:
                gt = torch.nn.functional.interpolate(gt, pred.shape[-2:], mode="bilinear", align_corners=False)

            with autocast(enabled=torch.cuda.is_available()):
                agg.update(compute_depth_metrics(pred, gt, align_median=False))
                drop = batch.get("drop_mask")
                if drop is not None:
                    drop = drop.to(self.device)
                    if drop.sum() > 0:
                        gt_drop = gt * drop  # les zéros sont exclus par min_depth
                        agg_drop.update(compute_depth_metrics(pred, gt_drop, align_median=False))

        return agg, agg_drop

    def run(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            # Curriculum de trous : informe le dataset de l'époque courante
            # (sans effet si le dataset ne gère pas set_epoch / pas de curriculum).
            ds = getattr(self.train_loader, "dataset", None)
            if ds is not None and hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch - 1)

            train_loss = self.train_epoch(epoch)
            # Validation et sélection de checkpoint sur les poids EMA si activé.
            self._ema_swap_in()
            metrics, metrics_drop = self.validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            msg = (
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"MAE={metrics.mae:.4f} RMSE={metrics.rmse:.4f} δ1={metrics.delta1:.4f}"
            )
            if metrics_drop.n > 0:
                msg += (
                    f" | DROP MAE={metrics_drop.mae:.4f} "
                    f"RMSE={metrics_drop.rmse:.4f} δ1={metrics_drop.delta1:.4f}"
                )
            self.logger.info(msg)
            print(msg, flush=True)

            if self.epoch_log_path is not None:
                with open(self.epoch_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{epoch:3d}  {train_loss:.6f}  {metrics.mae:.6f}  "
                        f"{metrics.rmse:.6f}  {metrics.delta1:.4f}  {lr:.8f}\n"
                    )

            if self.log_csv:
                self._log_csv(
                    self.log_csv,
                    [
                        epoch, train_loss, metrics.mae, metrics.rmse, metrics.abs_rel,
                        metrics.delta1, metrics_drop.mae, metrics_drop.rmse,
                        metrics_drop.delta1, lr,
                    ],
                )

            if self.writer:
                self.writer.add_scalar("train/loss", train_loss, epoch)
                for k, v in metrics.to_dict().items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
                if metrics_drop.n > 0:
                    for k, v in metrics_drop.to_dict().items():
                        self.writer.add_scalar(f"val_drop/{k}", v, epoch)

            # Sélection du checkpoint : RMSE sur trous synthétiques si dispo
            # (la RMSE globale est triviale quand GT = capteur en entrée).
            select_rmse = metrics_drop.rmse if metrics_drop.n > 0 else metrics.rmse
            is_best = select_rmse < self.best_rmse
            if is_best:
                self.best_rmse = select_rmse

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_val_rmse": self.best_rmse,
                },
                self.checkpoint_dir,
                is_best=is_best,
                epoch=epoch,
                save_every=5,
            )
            # Restaure les poids bruts pour poursuivre l'entraînement.
            self._ema_swap_out()

        if self.writer:
            self.writer.close()
        self.logger.info("Training complete.")
