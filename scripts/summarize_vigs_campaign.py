#!/usr/bin/env python3
"""
Agrège la campagne VIGS-Fusion en un tableau exploitable dans le mémoire.

Pour chaque trajectoire produite (diagnostic de cadence et A/B capteur/ASAF) :
  * détecte l'éventuelle divergence et sa date,
  * calcule l'ATE/RPE sur la séquence COMPLÈTE,
  * calcule l'ATE/RPE sur le SEGMENT SUIVI, dans une fenêtre commune.

Pourquoi les deux : sur Line-Rot, VIGS-Fusion perd le suivi en rotation pure et
l'ATE agrégé explose (72 m pour 9 m de trajectoire réelle), ce qui noie toute
différence entre sources de profondeur. Restreint au segment effectivement
suivi, le même run donne 0,17 m — une échelle où une différence est mesurable.
Publier l'un sans l'autre serait trompeur dans un sens ou dans l'autre.

Usage :
  python scripts/summarize_vigs_campaign.py --seq Line-Rot
  python scripts/summarize_vigs_campaign.py --seq Line-Rot --window 55
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_vigs_trajectory import analyze  # noqa: E402
from eval_slam_evo import run_evo  # noqa: E402

# Marge retranchée à l'instant de divergence pour définir la fenêtre commune.
#
# Le détecteur de analyze_vigs_trajectory.py se déclenche lorsque le rayon
# dépasse 3x l'étendue de la GT — un critère volontairement conservateur, qui
# ne se déclenche donc qu'APRÈS le début réel du décrochage. Mesuré sur
# Line-Rot : rupture visible à t≈57 s, seuil franchi seulement à t=65,1 s.
# Une marge trop courte laisse donc des secondes déjà contaminées dans la
# fenêtre (ATE segment 0,481 m à 60 s contre 0,175 m à 55 s).
ONSET_MARGIN_S = 10.0

# Avertissement méthodologique : la fenêtre déduite des résultats sert au
# DIAGNOSTIC. Pour une comparaison publiée, préférer --window fixé a priori
# d'après le profil de mouvement de la GT, afin qu'on ne puisse pas soupçonner
# un réglage choisi après coup pour favoriser un bras.
PREREG_NOTE = (
    "Fenêtre déduite des résultats : convient au diagnostic, mais pour le "
    "mémoire préférer --window fixé a priori (profil de mouvement de la GT)."
)


def arm_of(label: str) -> str | None:
    """Bras auquel appartient un run : 'sensor', 'asaf', ou None."""
    base = label.split("_")[0]
    return base if base in ("sensor", "asaf") else None


def discover(project: Path, seq: str) -> list[tuple[str, Path]]:
    """Trouve les trajectoires de la campagne, étiquetées par leur origine.

    Les exécutions répétées d'un même bras portent un suffixe (RUN_TAG) :
    sensor_a, asaf_a, sensor_b… Il faut toutes les collecter, car la
    dispersion INTRA-bras est ce qui permet de juger si l'écart entre bras
    est significatif.
    """
    runs: list[tuple[str, Path]] = []
    ab = project / "evaluation_results" / "vigs_slam" / seq
    if ab.is_dir():
        for d in sorted(ab.iterdir()):
            if d.is_dir() and arm_of(d.name) and (d / "trajectory.tum").is_file():
                runs.append((d.name, d / "trajectory.tum"))
    diag = project / "evaluation_results" / "vigs_diag" / seq
    if diag.is_dir():
        for d in sorted(diag.glob("sensor_r*")):
            t = d / "trajectory.tum"
            if t.is_file():
                runs.append((f"diag_{d.name}", t))
    return runs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", default="Line-Rot")
    p.add_argument("--project", type=Path, default=ROOT)
    p.add_argument(
        "--window", type=float, default=None,
        help="fin de la fenêtre de segment, en secondes. Par défaut : déduite "
             "du décrochage le PLUS PRÉCOCE observé, marge comprise.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    seq = args.seq
    project = args.project
    gt = project / "data" / "vigs_fusion" / seq / "extracted" / "poses_gt.tum"
    if not gt.is_file():
        print(f"ERREUR : GT absente : {gt}")
        return 1

    runs = discover(project, seq)
    if not runs:
        print(f"ERREUR : aucune trajectoire trouvée pour {seq}")
        return 1
    print(f"{len(runs)} trajectoire(s) : {', '.join(r[0] for r in runs)}\n")

    # --- 1. diagnostic de chaque run ----------------------------------------
    diags = {}
    for label, tum in runs:
        try:
            diags[label] = analyze(gt, tum, label)
        except Exception as e:
            print(f"  {label} : diagnostic impossible ({e})")

    # --- 2. fenêtre COMMUNE --------------------------------------------------
    # Impérativement identique pour tous les bras : utiliser le décrochage
    # propre à chaque run avantagerait mécaniquement celui qui tient le plus
    # longtemps, puisqu'il serait évalué sur une portion plus favorable.
    if args.window is not None:
        window = args.window
        origin = "imposée en ligne de commande"
    else:
        onsets = [
            d["divergence_onset_s"] for d in diags.values()
            if d.get("diverged") and d.get("divergence_onset_s")
        ]
        if onsets:
            window = max(5.0, min(onsets) - ONSET_MARGIN_S)
            origin = (f"décrochage le plus précoce {min(onsets):.1f} s "
                      f"− {ONSET_MARGIN_S:.0f} s de marge")
        else:
            window = None
            origin = "aucune divergence : segment inutile"
    print(f"Fenêtre de segment : "
          f"{'aucune' if window is None else f'0–{window:.1f} s'}  ({origin})\n")

    # --- 3. métriques --------------------------------------------------------
    out_dir = args.output_dir or (project / "evaluation_results" / "vigs_summary" / seq)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, tum in runs:
        d = diags.get(label, {})
        row = {
            "run": label,
            "n_poses": d.get("n_poses_est"),
            "diverged": d.get("diverged"),
            "onset_s": d.get("divergence_onset_s"),
            "path_len_m": d.get("est_path_length_m"),
            "hz_early": d.get("hz_early"),
            "hz_late": d.get("hz_late"),
        }
        for scope, t_end in (("full", None), ("seg", window)):
            if scope == "seg" and window is None:
                row["ate_seg_m"] = row["rpe_seg_m"] = None
                continue
            try:
                s = run_evo(
                    gt, tum, out_dir / f"{label}_{scope}", f"{label}_{scope}",
                    t_end=t_end,
                )
                row[f"ate_{scope}_m"] = round(s["ate_rmse_m"], 4)
                row[f"rpe_{scope}_m"] = round(s["rpe_trans_rmse_m"], 5)
            except Exception as e:
                print(f"  {label}/{scope} : évaluation impossible ({e})")
                row[f"ate_{scope}_m"] = row[f"rpe_{scope}_m"] = None
        rows.append(row)

    # --- 4. sorties ----------------------------------------------------------
    csv_path = out_dir / "campaign_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def fmt(v, nd=3):
        return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    hdr = f"{'run':22s} {'poses':>6s} {'div.':>5s} {'onset':>7s} " \
          f"{'ATE complet':>12s} {'ATE segment':>12s} {'RPE segment':>12s}"
    lines = [f"Campagne VIGS-Fusion — {seq}", "=" * len(hdr), hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['run']:22s} {fmt(r['n_poses']):>6s} "
            f"{('oui' if r['diverged'] else 'non'):>5s} "
            f"{fmt(r['onset_s'], 1):>7s} "
            f"{fmt(r.get('ate_full_m')):>12s} "
            f"{fmt(r.get('ate_seg_m')):>12s} "
            f"{fmt(r.get('rpe_seg_m'), 5):>12s}"
        )
    lines += ["", f"Fenêtre de segment : "
                  f"{'aucune' if window is None else f'0–{window:.1f} s'} ({origin})",
              "ATE/RPE en mètres (RMSE). Le segment est identique pour tous les bras."]
    if args.window is None and window is not None:
        lines += ["", "AVERTISSEMENT : " + PREREG_NOTE]

    # --- comparaison des bras, rapportée à la dispersion intra-bras ---------
    # Le point décisif du mémoire. Un écart entre bras n'a de sens que s'il
    # dépasse la variabilité observée entre deux exécutions du MÊME bras :
    # mesurée sur Line-Rot, celle-ci atteint un facteur 3 selon la cadence de
    # rejeu, ce qui peut écraser l'effet recherché.
    arms: dict[str, list[float]] = {}
    for r in rows:
        a = arm_of(r["run"])
        v = r.get("ate_full_m")
        if a and v is not None:
            arms.setdefault(a, []).append(float(v))

    if arms:
        lines += ["", "-" * 62, "Comparaison des bras (ATE séquence complète)", "-" * 62]
        stats = {}
        for a, vals in sorted(arms.items()):
            mean = sum(vals) / len(vals)
            spread = max(vals) - min(vals) if len(vals) > 1 else None
            stats[a] = (mean, spread, len(vals))
            sp = "—" if spread is None else f"{spread:.4f}"
            lines.append(
                f"  {a:8s} n={len(vals)}  valeurs={[round(v, 4) for v in vals]}  "
                f"moyenne={mean:.4f} m  étendue={sp} m"
            )
        if len(stats) == 2 and "sensor" in stats and "asaf" in stats:
            diff = stats["sensor"][0] - stats["asaf"][0]
            spreads = [s for _, s, _ in stats.values() if s is not None]
            worst = max(spreads) if spreads else None
            lines += ["", f"  écart capteur − ASAF : {diff:+.4f} m"]
            if worst is None:
                lines.append(
                    "  Une seule exécution par bras : la dispersion est inconnue, "
                    "aucun écart ne peut être déclaré significatif."
                )
            elif abs(diff) <= worst:
                lines.append(
                    f"  Cet écart ({abs(diff):.4f} m) NE DÉPASSE PAS la dispersion "
                    f"intra-bras ({worst:.4f} m) :"
                )
                lines.append(
                    "  aucun effet mesurable de la source de profondeur sur cette séquence."
                )
            else:
                lines.append(
                    f"  Cet écart ({abs(diff):.4f} m) dépasse la dispersion intra-bras "
                    f"({worst:.4f} m)."
                )
                lines.append(
                    "  Effet plausible, mais 2 exécutions par bras ne constituent pas "
                    "un test statistique — à confirmer sur d'autres séquences."
                )
    report = "\n".join(lines)
    print(report)
    (out_dir / "campaign_summary.txt").write_text(report + "\n")
    print(f"\nÉcrit : {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
