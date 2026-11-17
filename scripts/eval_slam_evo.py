#!/usr/bin/env python3
"""
Évaluation trajectoire SLAM vs GT Darknav (format TUM + evo).

Usage :
  python scripts/eval_slam_evo.py \\
    --ref data/vigs_fusion/Line/extracted/poses_gt.tum \\
    --est evaluation_results/slam/Line/sensor/trajectory.tum \\
    --label sensor --output-dir evaluation_results/slam/Line/sensor

Batch (CSV colonnes: ref,est,label,output_dir) :
  python scripts/eval_slam_evo.py --batch evaluation_results/slam/manifest.csv

Export GT depuis bag (si pas encore extrait) :
  python scripts/batch_extract_vigs_sequences.py --sequence Line --bag data/Line/Line
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_evo(
    ref_tum: Path,
    est_tum: Path,
    output_dir: Path,
    label: str,
    *,
    align: bool = True,
    correct_scale: bool = False,
    t_start: float | None = None,
    t_end: float | None = None,
) -> dict:
    try:
        from evo.core import sync
        from evo.core.metrics import PoseRelation, Unit
        from evo.core.trajectory import PoseTrajectory3D
        from evo.tools import file_interface
        from evo.main_ape import ape
        from evo.main_rpe import rpe
    except ImportError as e:
        raise RuntimeError("Installez evo : pip install evo") from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_ref = file_interface.read_tum_trajectory_file(str(ref_tum))
    traj_est = file_interface.read_tum_trajectory_file(str(est_tum))

    # --- fenêtre temporelle optionnelle --------------------------------------
    # Exprimée en secondes RELATIVES au début de la trajectoire estimée, ce qui
    # évite de manipuler des timestamps absolus en epoch.
    #
    # Le découpage a lieu AVANT l'alignement : c'est essentiel. Sur une
    # trajectoire qui diverge, aligner d'abord sur l'intégralité puis restreindre
    # donnerait un ajustement d'Umeyama dominé par la partie aberrante, et un ATE
    # de segment ininterprétable. En découpant d'abord, l'alignement ne porte que
    # sur la portion réellement suivie.
    window = None
    if t_start is not None or t_end is not None:
        t0 = float(traj_est.timestamps[0])
        a = t0 + (t_start if t_start is not None else 0.0)
        b = t0 + t_end if t_end is not None else float(traj_est.timestamps[-1])
        traj_ref.reduce_to_time_range(a, b)
        traj_est.reduce_to_time_range(a, b)
        window = (t_start, t_end)
        if len(traj_est.timestamps) < 10:
            raise RuntimeError(
                f"fenêtre [{t_start}, {t_end}] s : seulement "
                f"{len(traj_est.timestamps)} poses estimées — trop peu pour évaluer"
            )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    if align:
        traj_est.align(traj_ref, correct_scale=correct_scale)

    ape_metric = ape(traj_ref, traj_est, pose_relation=PoseRelation.translation_part, align=False)
    rpe_trans = rpe(
        traj_ref, traj_est, pose_relation=PoseRelation.translation_part, delta=1.0, delta_unit=Unit.frames
    )
    rpe_rot = rpe(
        traj_ref, traj_est, pose_relation=PoseRelation.rotation_part, delta=1.0, delta_unit=Unit.frames
    )

    def _stat(result, key: str) -> float:
        return float(result.stats[key])

    stats = {
        "label": label,
        "ref": str(ref_tum),
        "est": str(est_tum),
        "n_poses": len(traj_est.timestamps),
        # Sans ces colonnes, un ATE de segment serait indiscernable d'un ATE
        # sur séquence complète une fois les CSV agrégés.
        "t_start_s": "" if window is None else window[0],
        "t_end_s": "" if window is None else window[1],
        "ate_rmse_m": _stat(ape_metric, "rmse"),
        "ate_mean_m": _stat(ape_metric, "mean"),
        "ate_median_m": _stat(ape_metric, "median"),
        "ate_std_m": _stat(ape_metric, "std"),
        "rpe_trans_rmse_m": _stat(rpe_trans, "rmse"),
        "rpe_rot_rmse_rad": _stat(rpe_rot, "rmse"),
    }

    summary_path = output_dir / "evo_summary.csv"
    pd.DataFrame([stats]).to_csv(summary_path, index=False)

    # Erreurs par frame (pour courbes ATE temporelles)
    try:
        ape_errors = np.asarray(ape_metric.error, dtype=np.float64)
        pd.DataFrame(
            {
                "frame_idx": np.arange(len(ape_errors)),
                "ate_m": ape_errors,
            }
        ).to_csv(output_dir / "ate_per_frame.csv", index=False)
    except Exception:
        pass

    try:
        from evo.tools import plot
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        plot.traj(ax, plot.PlotMode.xyz, traj_ref, "--", "GT Darknav", "gray")
        plot.traj(ax, plot.PlotMode.xyz, traj_est, "-", label, "blue")
        ax.legend()
        fig.savefig(output_dir / "trajectory_3d.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # XY (vue dessus)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        plot.traj(ax, plot.PlotMode.xy, traj_ref, "--", "GT Darknav", "gray")
        plot.traj(ax, plot.PlotMode.xy, traj_est, "-", label, "#2563eb")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(output_dir / "trajectory_xy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ATE au fil du temps
        if hasattr(ape_metric, "error") and len(ape_metric.error) > 0:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            ax.plot(np.asarray(ape_metric.error), lw=1.2, color="#dc2626")
            ax.axhline(stats["ate_rmse_m"], color="#64748b", ls="--", lw=1, label=f"RMSE={stats['ate_rmse_m']:.3f} m")
            ax.set_xlabel("Frame associée")
            ax.set_ylabel("ATE translation (m)")
            ax.set_title(f"Erreur absolue — {label}")
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            fig.savefig(output_dir / "ate_over_time.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception:
        pass

    print(f"\n=== {label} ===")
    print(f"ATE RMSE : {stats['ate_rmse_m']:.4f} m")
    print(f"RPE trans RMSE : {stats['rpe_trans_rmse_m']:.4f} m")
    print(f"Résultats : {summary_path}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", type=Path, help="Trajectoire GT (TUM)")
    p.add_argument("--est", type=Path, help="Trajectoire estimée (TUM)")
    p.add_argument("--label", default="estimate")
    p.add_argument("--output-dir", type=Path, default=Path("evaluation_results/slam/evo_run"))
    p.add_argument("--batch", type=Path, help="CSV: ref,est,label,output_dir")
    p.add_argument("--no-align", action="store_true")
    p.add_argument("--correct-scale", action="store_true")
    p.add_argument(
        "--t-start", type=float, default=None,
        help="début de fenêtre, en secondes depuis le début de la trajectoire estimée",
    )
    p.add_argument(
        "--t-end", type=float, default=None,
        help="fin de fenêtre, en secondes depuis le début de la trajectoire estimée "
             "(ex. --t-end 55 pour n'évaluer que le segment suivi avant divergence)",
    )
    args = p.parse_args()

    if args.batch:
        df = pd.read_csv(args.batch)
        rows = []
        for _, row in df.iterrows():
            stats = run_evo(
                Path(row["ref"]),
                Path(row["est"]),
                Path(row["output_dir"]),
                str(row.get("label", "run")),
                align=not args.no_align,
                correct_scale=args.correct_scale,
            )
            rows.append(stats)
        out = Path(args.batch).parent / "evo_batch_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nBatch summary : {out}")
        return

    if not args.ref or not args.est:
        p.error("--ref et --est requis (ou --batch)")

    run_evo(
        args.ref,
        args.est,
        args.output_dir,
        args.label,
        align=not args.no_align,
        correct_scale=args.correct_scale,
        t_start=args.t_start,
        t_end=args.t_end,
    )


if __name__ == "__main__":
    main()
