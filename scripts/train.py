#!/usr/bin/env python3
"""Hydra entry point for depth fusion training."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # racine du dépôt, utilisable dans les YAML : ${project_root:}
    from omegaconf import OmegaConf as _OC
    _OC.register_new_resolver("project_root", lambda: str(ROOT), replace=True)
except ImportError:  # pragma: no cover
    pass
sys.path.insert(0, str(ROOT))

from omegaconf import DictConfig, OmegaConf

try:
    import hydra
    from hydra import compose, initialize_config_dir
except ImportError:
    hydra = None


def build_model(cfg: DictConfig):
    from depth_fusion.models.asaf_net import ASAFNet, ASAFNetConfig, ASAFNetV2, ASAFNetV2Config
    from depth_fusion.models.resnet_unet import ResNetUNet

    if cfg.model.name == "asaf_net_v2":
        return ASAFNetV2(
            ASAFNetV2Config(
                backbone=cfg.model.get("backbone", "resnet34"),
                pretrained=cfg.model.get("pretrained", True),
                depth_scale_m=cfg.model.get("depth_scale_m", 10.0),
            )
        )
    if cfg.model.name == "asaf_net":
        return ASAFNet(
            ASAFNetConfig(
                backbone=cfg.model.get("backbone", "resnet34"),
                pretrained=cfg.model.get("pretrained", True),
                use_sensor_depth=cfg.model.get("use_sensor_depth", False),
                use_confidence_head=cfg.model.get("use_confidence_head", True),
                residual_refinement=cfg.model.get("residual_refinement", True),
            )
        )
    return ResNetUNet(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        in_channels=cfg.model.get("in_channels", 4),
    )


def run(cfg: DictConfig) -> None:
    from depth_fusion.datasets.rgbd_fusion_dataset import build_dataloaders, build_dataloaders_v2
    from depth_fusion.losses.depth_losses import DepthLossConfig, DepthLossV2Config
    from depth_fusion.trainers.fusion_trainer import FusionTrainer
    from depth_fusion.utils.reproducibility import seed_everything

    seed_everything(cfg.experiment.seed)

    is_v2 = cfg.model.name == "asaf_net_v2"
    if is_v2:
        train_loader, val_loader = build_dataloaders_v2(
            cfg.data.train_csv,
            cfg.data.val_csv,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            image_size=tuple(cfg.data.image_size),
            use_sensor_depth=cfg.data.get("use_sensor_depth", True),
            depth_unit=cfg.data.get("depth_unit", "auto"),
            synthetic_holes=cfg.data.get("synthetic_holes", 0.15),
            synthetic_holes_max=cfg.data.get("synthetic_holes_max", None),
            curriculum_epochs=cfg.data.get("curriculum_epochs", 0),
        )
    else:
        train_loader, val_loader = build_dataloaders(
            cfg.data.train_csv,
            cfg.data.val_csv,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            image_size=tuple(cfg.data.image_size),
            use_sensor_depth=cfg.data.get("use_sensor_depth", False),
        )

    model = build_model(cfg)
    if is_v2:
        v2_fields = set(DepthLossV2Config.__dataclass_fields__)
        loss_cfg = DepthLossV2Config(**{k: v for k, v in cfg.loss.items() if k in v2_fields})
    else:
        loss_cfg = DepthLossConfig(**{k: v for k, v in cfg.loss.items() if hasattr(DepthLossConfig, k) or k.startswith("w_")})

    trainer = FusionTrainer(
        model,
        train_loader,
        val_loader,
        lr=cfg.train.lr,
        epochs=cfg.train.epochs,
        checkpoint_dir=cfg.train.checkpoint_dir,
        log_csv=cfg.train.get("log_csv"),
        loss_cfg=loss_cfg,
        use_wandb=cfg.train.get("use_wandb", False),
        use_tensorboard=cfg.train.get("use_tensorboard", True),
        resume=cfg.train.get("resume", ""),
        weight_decay=cfg.train.get("weight_decay", 1e-4),
        warmup_epochs=cfg.train.get("warmup_epochs", 0),
        use_ema=cfg.train.get("use_ema", False),
        ema_decay=cfg.train.get("ema_decay", 0.999),
    )
    trainer.run()


def main() -> None:
    config_dir = str(ROOT / "configs")
    if hydra is not None:

        @hydra.main(version_base=None, config_path="../configs", config_name="train_asaf")
        def hydra_main(cfg: DictConfig) -> None:
            run(cfg)

        hydra_main()
    else:
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = compose(config_name="train_baseline")
        run(cfg)


if __name__ == "__main__":
    main()
