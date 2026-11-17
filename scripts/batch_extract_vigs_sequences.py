#!/usr/bin/env python3
"""
Extraction batch des séquences VIGS-Fusion (CI0K9G) depuis rosbag2.

Usage :
  python scripts/batch_extract_vigs_sequences.py --sequence Line --bag data/Line/Line
  python scripts/batch_extract_vigs_sequences.py --split test
  python scripts/batch_extract_vigs_sequences.py --all-downloaded

Nécessite : pip install rosbags opencv-python-headless pandas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from depth_fusion.vigs.bag_extract import extract_bag

VIGS_ROOT = ROOT / "data" / "vigs_fusion"
SEQUENCES_YAML = VIGS_ROOT / "sequences.yaml"


def load_config() -> dict:
    with SEQUENCES_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_bag_dir(seq_dir: Path) -> Path | None:
    """Cherche metadata.yaml sous seq_dir (gère Line/Line après unzip)."""
    if (seq_dir / "metadata.yaml").exists():
        return seq_dir
    for child in sorted(seq_dir.iterdir()):
        if child.is_dir() and (child / "metadata.yaml").exists():
            return child
    return None


def resolve_sequence_paths(name: str) -> tuple[Path, Path]:
    """Retourne (bag_dir, output_extracted_dir)."""
    seq_root = VIGS_ROOT / name
    # Fallback : Line historique dans data/Line/
    if name == "Line" and not seq_root.exists():
        legacy = ROOT / "data" / "Line" / "Line"
        if legacy.exists():
            return legacy, VIGS_ROOT / "Line" / "extracted"

    bag = find_bag_dir(seq_root)
    if bag is None and name == "Line":
        legacy = ROOT / "data" / "Line" / "Line"
        if legacy.is_dir() and (legacy / "metadata.yaml").is_file() and any(legacy.glob("*.mcap")):
            return legacy, VIGS_ROOT / "Line" / "extracted"

    if bag is None:
        # sous-dossier après unzip du zip (ex. Line/Line/)
        for sub in seq_root.glob("**/metadata.yaml"):
            return sub.parent, VIGS_ROOT / name / "extracted"
        raise FileNotFoundError(
            f"Bag introuvable pour {name}. Téléchargez {name}.zip depuis CI0K9G "
            f"et extrayez dans {seq_root}"
        )
    return bag, VIGS_ROOT / name / "extracted"


def list_extracted_sequences() -> list[str]:
    """Toutes les séquences avec frames.csv déjà extrait."""
    seqs: list[str] = []
    if not VIGS_ROOT.exists():
        return seqs
    for d in sorted(VIGS_ROOT.iterdir()):
        if d.is_dir() and (d / "extracted" / "frames.csv").exists():
            seqs.append(d.name)
    return seqs


def build_master_csv(seq_names: list[str], out_csv: Path) -> None:
    import pandas as pd

    rows = []
    for name in seq_names:
        frames_csv = VIGS_ROOT / name / "extracted" / "frames.csv"
        if not frames_csv.exists():
            continue
        df = pd.read_csv(frames_csv)
        df["sequence"] = name
        rows.append(df)
    if not rows:
        print("Aucun frames.csv pour build_master_csv")
        return
    master = pd.concat(rows, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_csv, index=False)
    print(f"Master CSV : {out_csv} ({len(master)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description="Extraction batch VIGS-Fusion CI0K9G")
    p.add_argument("--sequence", action="append", help="Nom séquence (ex. Line)")
    p.add_argument("--bag", type=Path, help="Chemin bag (avec --sequence unique)")
    p.add_argument("--split", choices=["train", "val", "test", "extra", "all"], help="Split du protocole")
    p.add_argument("--all-downloaded", action="store_true", help="Toutes les séquences avec bag présent")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--sync-ms", type=float, default=50.0)
    p.add_argument("--force", action="store_true", help="Ré-extraire même si frames.csv existe")
    p.add_argument("--master-csv", type=Path, default=VIGS_ROOT / "master_frames.csv")
    args = p.parse_args()

    cfg = load_config()
    names: list[str] = []

    if args.sequence:
        names.extend(args.sequence)
    elif args.split:
        key = {
            "train": "train_seqs",
            "val": "val_seqs",
            "test": "test_seqs",
            "extra": "extra_seqs",
            "all": None,
        }[args.split]
        if key is None:
            names = (
                cfg.get("train_seqs", [])
                + cfg.get("val_seqs", [])
                + cfg.get("test_seqs", [])
                + cfg.get("extra_seqs", [])
            )
        else:
            names = list(cfg.get(key, []))
    elif args.all_downloaded:
        if VIGS_ROOT.exists():
            for d in sorted(VIGS_ROOT.iterdir()):
                if d.is_dir() and d.name not in ("__pycache__",):
                    if find_bag_dir(d) or any(d.glob("**/metadata.yaml")):
                        names.append(d.name)
        legacy_line = ROOT / "data" / "Line" / "Line"
        if legacy_line.exists() and "Line" not in names:
            names.append("Line")
    else:
        p.error("Spécifiez --sequence, --split ou --all-downloaded")

    names = list(dict.fromkeys(names))
    print(f"Séquences : {names}")

    ok = []
    for name in names:
        try:
            if args.bag and len(names) == 1:
                bag_path = args.bag
                out_dir = VIGS_ROOT / name / "extracted"
            else:
                bag_path, out_dir = resolve_sequence_paths(name)
            print(f"\n=== {name} ===")
            print(f"  bag   : {bag_path}")
            print(f"  out   : {out_dir}")
            stats = extract_bag(
                bag_path,
                out_dir,
                max_frames=args.max_frames,
                sync_tolerance_ms=args.sync_ms,
                skip_existing=not args.force,
            )
            print(f"  stats : {stats}")
            ok.append(name)
        except FileNotFoundError as e:
            print(f"  SKIP  : {e}")
        except Exception as e:
            print(f"  ERREUR: {e}")
            raise

    if ok:
        build_master_csv(list_extracted_sequences(), args.master_csv)


if __name__ == "__main__":
    main()
