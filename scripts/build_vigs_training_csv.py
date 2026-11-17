#!/usr/bin/env python3
"""Build train/val CSVs for ASAF from VIGS-Fusion extracted sequences (mémoire protocol)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _npy_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 16:
        return False
    try:
        d = np.load(path)
        return d.ndim == 2 and d.size > 0
    except (OSError, ValueError, EOFError):
        return False


def _load_protocol(sequences_yaml: Path) -> tuple[list[str], list[str], list[str]]:
    with open(sequences_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return list(cfg["train_seqs"]), list(cfg["val_seqs"]), list(cfg.get("test_seqs", []))


def _rows_for_sequence(
    seq: str, vigs_root: Path, *, require_da3: bool = False
) -> list[dict[str, str]]:
    frames_csv = vigs_root / seq / "extracted" / "frames.csv"
    if not frames_csv.exists():
        raise FileNotFoundError(f"frames.csv manquant: {frames_csv}")

    extracted = vigs_root / seq / "extracted"
    da3_dir = extracted / "da3_scaled"
    df = pd.read_csv(frames_csv)
    rows: list[dict[str, str]] = []

    for _, row in df.iterrows():
        rgb = extracted / "rgb" / Path(str(row["rgb"])).name
        sensor = extracted / "depth_sensor" / Path(str(row["depth_sensor"])).name
        depth_any = da3_dir / sensor.name
        if not rgb.is_file():
            continue
        if require_da3:
            if not _npy_ok(depth_any):
                continue
        elif not sensor.is_file():
            continue

        gt_path = sensor if sensor.is_file() else depth_any
        rows.append(
            {
                "rgb": str(rgb.resolve()),
                "depth_any": str(depth_any.resolve()),
                "depth_gt": str(gt_path.resolve()),
                "depth_sensor": str(sensor.resolve()),
                "sequence": seq,
            }
        )
    return rows


def _build_df(
    seqs: list[str],
    vigs_root: Path,
    *,
    require_da3: bool,
) -> pd.DataFrame:
    cols = ["rgb", "depth_any", "depth_gt", "depth_sensor", "sequence"]
    rows: list[dict[str, str]] = []
    for seq in seqs:
        try:
            rows.extend(_rows_for_sequence(seq, vigs_root, require_da3=require_da3))
        except FileNotFoundError as e:
            print(f"  SKIP {seq}: {e}")
    df = pd.DataFrame(rows, columns=cols)
    if require_da3 and len(df):
        df = df[df["depth_any"].map(lambda p: _npy_ok(Path(p)))]
    return df


def build_csvs(
    project_root: Path,
    *,
    out_dir: Path | None = None,
    require_da3: bool = False,
    include_test: bool = False,
) -> tuple[Path, Path, Path | None]:
    vigs_root = project_root / "data" / "vigs_fusion"
    sequences_yaml = vigs_root / "sequences.yaml"
    train_seqs, val_seqs, test_seqs = _load_protocol(sequences_yaml)
    out_dir = out_dir or (project_root / "dataset_fusion" / "vigs")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _build_df(train_seqs, vigs_root, require_da3=require_da3)
    val_df = _build_df(val_seqs, vigs_root, require_da3=require_da3)
    test_df = _build_df(test_seqs, vigs_root, require_da3=require_da3) if include_test else None

    n_sensor_train = (
        int(train_df["depth_sensor"].map(lambda p: Path(p).is_file()).sum()) if len(train_df) else 0
    )
    n_sensor_val = (
        int(val_df["depth_sensor"].map(lambda p: Path(p).is_file()).sum()) if len(val_df) else 0
    )
    if require_da3 and (n_sensor_train == 0 or n_sensor_val == 0):
        print(
            "WARN: depth_sensor absent (supprimé après DA3?) — "
            "entraîner avec data.use_sensor_depth=false model.use_sensor_depth=false"
        )

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Train ({len(train_df)} frames): {', '.join(train_seqs)} -> {train_path}")
    print(f"Val   ({len(val_df)} frames): {', '.join(val_seqs)} -> {val_path}")
    test_path = None
    if include_test and test_df is not None:
        test_path = out_dir / "test.csv"
        test_df.to_csv(test_path, index=False)
        print(f"Test  ({len(test_df)} frames): {', '.join(test_seqs)} -> {test_path}")
    if not require_da3:
        n_da3_train = sum(Path(p).is_file() for p in train_df["depth_any"])
        n_da3_val = sum(Path(p).is_file() for p in val_df["depth_any"])
        print(f"DA3 précalculées: train {n_da3_train}/{len(train_df)}, val {n_da3_val}/{len(val_df)}")
    return train_path, val_path, test_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument(
        "--require-da3",
        action="store_true",
        help="Ne garder que les frames avec da3_scaled déjà présent",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Génère aussi dataset_fusion/vigs/test.csv (séquences test_seqs)",
    )
    args = p.parse_args()
    build_csvs(
        args.root,
        out_dir=args.out_dir,
        require_da3=args.require_da3,
        include_test=args.include_test,
    )


if __name__ == "__main__":
    main()
