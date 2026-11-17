#!/usr/bin/env python3
"""
Diagnostic d'une trajectoire VIGS-Fusion : détecte et localise une divergence.

eval_slam_evo.py donne l'ATE, mais un ATE de 72 m ne dit pas SI la divergence
est immédiate (problème de configuration : TF extrinsèque, topic IMU, échelle)
ou tardive (dérive progressive : profondeur bruitée, manque de calcul). Ce
script tranche, et mesure le débit de poses — l'effondrement du débit est la
signature d'un nœud qui ne tient pas le temps réel et bascule sur la seule
préintégration IMU.

Usage :
  python scripts/analyze_vigs_trajectory.py \\
    --ref data/vigs_fusion/Line-Rot/extracted/poses_gt.tum \\
    --est evaluation_results/vigs_slam/Line-Rot/sensor/trajectory.tum \\
    --label sensor_r0.3 --output-dir evaluation_results/vigs_slam/Line-Rot/sensor
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Seuil de divergence : multiple de l'étendue spatiale de la GT au-delà duquel
# la trajectoire estimée ne décrit plus la même scène. 3x laisse passer une
# dérive honnête tout en attrapant sans ambiguïté un décrochage.
DIVERGENCE_FACTOR = 3.0

# Fenêtre d'agrégation du débit de poses, en secondes.
BIN_SECONDS = 5.0


def load_tum(path: Path) -> np.ndarray:
    """Charge un fichier TUM (timestamp tx ty tz qx qy qz qw)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError(f"{path} : format TUM attendu (8 colonnes), lu {data.shape}")
    return data[np.argsort(data[:, 0])]


def path_length(xyz: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())


def speeds(t: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    dt = np.diff(t)
    # Deux poses au même timestamp donneraient une vitesse infinie : on les
    # écarte plutôt que de les plafonner, pour ne pas inventer de valeur.
    valid = dt > 1e-6
    if not valid.any():
        return np.zeros(0)
    return np.linalg.norm(np.diff(xyz, axis=0), axis=1)[valid] / dt[valid]


def analyze(ref_path: Path, est_path: Path, label: str) -> dict:
    ref, est = load_tum(ref_path), load_tum(est_path)
    t_ref, xyz_ref = ref[:, 0], ref[:, 1:4]
    t_est, xyz_est = est[:, 0], est[:, 1:4]

    # Étendue de la GT : rayon maximal autour du point de départ.
    gt_extent = float(np.linalg.norm(xyz_ref - xyz_ref[0], axis=1).max())
    threshold = DIVERGENCE_FACTOR * gt_extent

    t0 = t_est[0]
    rel = t_est - t0
    radius = np.linalg.norm(xyz_est - xyz_est[0], axis=1)

    over = np.flatnonzero(radius > threshold)
    diverged = over.size > 0
    onset = float(rel[over[0]]) if diverged else None

    v_est, v_ref = speeds(t_est, xyz_est), speeds(t_ref, xyz_ref)

    # Débit de poses par fenêtre : révèle un nœud qui décroche du temps réel.
    # La GT est ramenée sur la MÊME origine temporelle que l'estimée, afin de
    # comparer les deux rayons tranche par tranche. C'est ce qui distingue une
    # dérive naissante d'un mouvement réel : sans la colonne GT en regard, un
    # rayon estimé qui croît peut tout aussi bien être la trajectoire réelle.
    rel_ref = t_ref - t0
    radius_ref = np.linalg.norm(xyz_ref - xyz_ref[0], axis=1)

    n_bins = max(1, int(np.ceil(rel[-1] / BIN_SECONDS)))
    bins = []
    for k in range(n_bins):
        lo, hi = k * BIN_SECONDS, (k + 1) * BIN_SECONDS
        m = (rel >= lo) & (rel < hi)
        m_ref = (rel_ref >= lo) & (rel_ref < hi)
        r_ref = round(float(np.median(radius_ref[m_ref])), 3) if m_ref.any() else None
        if not m.any():
            bins.append(
                {
                    "t_start": lo,
                    "n_poses": 0,
                    "hz": 0.0,
                    "radius_median": None,
                    "radius_ref_median": r_ref,
                    "radius_delta": None,
                }
            )
            continue
        r_est = round(float(np.median(radius[m])), 3)
        bins.append(
            {
                "t_start": lo,
                "n_poses": int(m.sum()),
                "hz": round(float(m.sum() / BIN_SECONDS), 2),
                "radius_median": r_est,
                "radius_ref_median": r_ref,
                "radius_delta": None if r_ref is None else round(r_est - r_ref, 3),
            }
        )

    # Effondrement du débit : on compare la seconde moitié au régime établi de
    # la première. Un ratio faible signale une saturation progressive (carte
    # gaussienne qui grossit, GPU sous-dimensionné) plutôt qu'un plantage net.
    hz = np.array([b["hz"] for b in bins if b["n_poses"] > 0])
    if hz.size >= 4:
        half = hz.size // 2
        hz_early = float(np.median(hz[:half]))
        hz_late = float(np.median(hz[half:]))
        throughput_ratio = round(hz_late / hz_early, 3) if hz_early > 0 else None
    else:
        hz_early = hz_late = throughput_ratio = None

    return {
        "label": label,
        "reference": str(ref_path),
        "estimate": str(est_path),
        "n_poses_ref": int(len(ref)),
        "n_poses_est": int(len(est)),
        "duration_ref_s": round(float(t_ref[-1] - t_ref[0]), 2),
        "duration_est_s": round(float(rel[-1]), 2),
        "gt_extent_m": round(gt_extent, 3),
        "gt_path_length_m": round(path_length(xyz_ref), 3),
        "est_path_length_m": round(path_length(xyz_est), 3),
        "est_max_radius_m": round(float(radius.max()), 3),
        "divergence_threshold_m": round(threshold, 3),
        "diverged": bool(diverged),
        "divergence_onset_s": round(onset, 2) if onset is not None else None,
        "speed_ref_median_ms": round(float(np.median(v_ref)), 4) if v_ref.size else None,
        "speed_est_median_ms": round(float(np.median(v_est)), 4) if v_est.size else None,
        "speed_est_max_ms": round(float(v_est.max()), 4) if v_est.size else None,
        "hz_early": hz_early,
        "hz_late": hz_late,
        "throughput_ratio": throughput_ratio,
        "bins": bins,
    }


def report(res: dict) -> str:
    lines = [
        "=" * 62,
        f" Diagnostic trajectoire — {res['label']}",
        "=" * 62,
        f"poses          : {res['n_poses_est']} estimées / {res['n_poses_ref']} GT",
        f"durée          : {res['duration_est_s']} s (GT {res['duration_ref_s']} s)",
        f"parcours       : {res['est_path_length_m']} m (GT {res['gt_path_length_m']} m)",
        f"rayon max      : {res['est_max_radius_m']} m (étendue GT {res['gt_extent_m']} m)",
        f"vitesse méd.   : {res['speed_est_median_ms']} m/s (GT {res['speed_ref_median_ms']} m/s)"
        f" | max {res['speed_est_max_ms']} m/s",
    ]
    if res["throughput_ratio"] is not None:
        lines.append(
            f"débit poses    : {res['hz_early']} Hz -> {res['hz_late']} Hz "
            f"(ratio {res['throughput_ratio']})"
        )
        if res["throughput_ratio"] < 0.6:
            lines.append(
                "  ATTENTION : le débit s'effondre — le nœud ne tient pas la cadence."
            )
    if res["diverged"]:
        lines += [
            "",
            f"DIVERGENCE détectée à t = {res['divergence_onset_s']} s "
            f"(seuil {res['divergence_threshold_m']} m)",
        ]
        if res["divergence_onset_s"] is not None and res["divergence_onset_s"] < 5.0:
            lines.append(
                "  -> décrochage immédiat : suspecter la configuration "
                "(TF extrinsèque IMU->caméra, topic IMU, échelle de profondeur)."
            )
        else:
            lines.append(
                "  -> décrochage tardif : suspecter la dérive "
                "(qualité de profondeur, manque de calcul, cadence trop élevée)."
            )
    else:
        lines += ["", "Pas de divergence détectée."]

    lines += ["", "  t(s)   poses    Hz    rayon EST   rayon GT     écart"]
    for b in res["bins"]:
        est_r = "  —  " if b["radius_median"] is None else f"{b['radius_median']:8.3f}"
        ref_r = "  —  " if b["radius_ref_median"] is None else f"{b['radius_ref_median']:8.3f}"
        dlt = "  —  " if b["radius_delta"] is None else f"{b['radius_delta']:+8.3f}"
        lines.append(
            f"  {b['t_start']:5.0f}  {b['n_poses']:5d}  {b['hz']:5.1f}  {est_r}  {ref_r}  {dlt}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", required=True, type=Path, help="GT au format TUM")
    p.add_argument("--est", required=True, type=Path, help="trajectoire estimée (TUM)")
    p.add_argument("--label", default="run")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()

    if not args.est.exists():
        print(f"ERREUR : trajectoire absente : {args.est}")
        return 1

    res = analyze(args.ref, args.est, args.label)
    text = report(res)
    print(text)

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "diagnostic.json").write_text(json.dumps(res, indent=2))
        (args.output_dir / "diagnostic.txt").write_text(text + "\n")
        print(f"\nÉcrit : {args.output_dir}/diagnostic.{{json,txt}}")

    # Code de sortie non nul si la trajectoire a divergé : permet au script de
    # campagne d'enchaîner ou de s'arrêter selon le résultat.
    return 2 if res["diverged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
