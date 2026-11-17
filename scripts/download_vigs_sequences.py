#!/usr/bin/env python3
"""
Télécharge les séquences VIGS-Fusion depuis Recherche Data Gouv (CI0K9G).

Usage :
  python scripts/download_vigs_sequences.py --split test
  python scripts/download_vigs_sequences.py --sequence Square-Shape Rotation-Only
  python scripts/download_vigs_sequences.py --split train --extract

Nécessite : curl ou wget, unzip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VIGS_ROOT = ROOT / "data" / "vigs_fusion"
SEQUENCES_YAML = VIGS_ROOT / "sequences.yaml"
BASE_URL = "https://entrepot.recherche.data.gouv.fr/api/access/datafile"


def load_config() -> dict:
    with SEQUENCES_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_file(file_id: int, dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists() and dest_zip.stat().st_size > 1_000_000:
        print(f"  Déjà présent : {dest_zip}")
        return
    url = f"{BASE_URL}/{file_id}"
    print(f"  Téléchargement {url} → {dest_zip}")
    if shutil.which("curl"):
        subprocess.run(
            ["curl", "-L", "--fail", "--progress-bar", "-o", str(dest_zip), url],
            check=True,
        )
    elif shutil.which("wget"):
        subprocess.run(["wget", "-O", str(dest_zip), url], check=True)
    else:
        raise RuntimeError("Installez curl ou wget")


def unzip_to(seq_dir: Path, zip_path: Path) -> None:
    print(f"  Extraction {zip_path.name} → {seq_dir}")
    seq_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(seq_dir)


def run_extract(sequence: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "batch_extract_vigs_sequences.py"),
            "--sequence",
            sequence,
            "--force",
        ],
        check=True,
        cwd=str(ROOT),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", nargs="+", help="Noms de séquences")
    p.add_argument("--split", choices=["train", "val", "test", "extra", "all"])
    p.add_argument("--extract", action="store_true", help="Lancer batch_extract après unzip")
    p.add_argument("--keep-zip", action="store_true")
    p.add_argument("--skip-existing-bag", action="store_true", default=True)
    args = p.parse_args()

    cfg = load_config()
    file_ids: dict = cfg.get("download_file_ids", {})
    names: list[str] = []

    if args.sequence:
        names = args.sequence
    elif args.split:
        key = {
            "train": "train_seqs",
            "val": "val_seqs",
            "test": "test_seqs",
            "extra": "extra_seqs",
            "all": None,
        }[args.split]
        if key is None:
            names = list(file_ids.keys())
        else:
            names = list(cfg.get(key, []))
    else:
        p.error("--sequence ou --split requis")

    names = list(dict.fromkeys(names))
    print(f"Séquences à télécharger : {names}")

    for name in names:
        legacy_line = ROOT / "data" / "Line" / "Line"
        if name == "Line":
            has_line_bag = (
                (VIGS_ROOT / "Line").exists()
                and any((VIGS_ROOT / "Line").glob("**/*.mcap"))
            ) or (
                legacy_line.is_dir()
                and (legacy_line / "metadata.yaml").is_file()
                and any(legacy_line.glob("*.mcap"))
            )
            if args.skip_existing_bag and has_line_bag:
                print(f"\n=== {name} === SKIP (bag Line déjà présent)")
                if args.extract and not (VIGS_ROOT / "Line" / "extracted" / "frames.csv").exists():
                    run_extract(name)
                continue

        fid = file_ids.get(name)
        if fid is None:
            print(f"\n=== {name} === SKIP (id inconnu dans sequences.yaml)")
            continue

        seq_dir = VIGS_ROOT / name
        zip_name = cfg.get("download_zips", {}).get(name, f"{name}.zip")
        zip_path = seq_dir / zip_name

        print(f"\n=== {name} ===")
        has_bag = any(seq_dir.glob("**/metadata.yaml")) and any(seq_dir.glob("**/*.mcap"))
        if args.skip_existing_bag and has_bag:
            print("  Bag déjà extrait (metadata + .mcap présents)")
        else:
            if args.skip_existing_bag and any(seq_dir.glob("**/metadata.yaml")):
                print("  metadata.yaml sans .mcap — retéléchargement")
            download_file(int(fid), zip_path)
            unzip_to(seq_dir, zip_path)
            if not args.keep_zip:
                zip_path.unlink(missing_ok=True)
                print(f"  Zip supprimé : {zip_path.name}")

        if args.extract:
            run_extract(name)

    print("\nTerminé.")


if __name__ == "__main__":
    main()
