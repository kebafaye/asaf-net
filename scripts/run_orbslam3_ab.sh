#!/usr/bin/env bash
# A/B ORB-SLAM3 capteur vs ASAF sur une fenêtre d'une séquence VIGS-Fusion.
#
# Garantit que les deux modes portent sur EXACTEMENT les mêmes frames, avec les
# timestamps et le ground truth de la séquence dont proviennent réellement les
# images (cf. bug documenté dans slam/ORBS_SLAM3.md).
#
# Usage :
#   SEQ=Line-Rot START=1200 COUNT=1200 TAG=Line-Rot-rot bash scripts/run_orbslam3_ab.sh
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT}"
PY="${PROJECT}/.venv-ros312/bin/python"

SEQ="${SEQ:?SEQ requis (ex: Line-Rot)}"
START="${START:-0}"
COUNT="${COUNT:-0}"
TAG="${TAG:-${SEQ}}"
CKPT="${CKPT:-${PROJECT}/checkpoints/experiments/asaf_net_v2_vigs/best_model.pth}"
SETTINGS="${ORB_SETTINGS:-${PROJECT}/config/orbslam3_vigs_d456.yaml}"

EXTRACTED="${PROJECT}/data/vigs_fusion/${SEQ}/extracted"
FRAMES="${EXTRACTED}/frames.csv"
GT="${EXTRACTED}/poses_gt.tum"
DA3="${EXTRACTED}/da3_scaled"
ROOT_OUT="${PROJECT}/evaluation_results/orbslam3/${TAG}"

for f in "${FRAMES}" "${GT}"; do
  [[ -f "$f" ]] || { echo "ERREUR: manquant $f"; exit 1; }
done

WINDOW=(--start "${START}")
[[ "${COUNT}" -gt 0 ]] && WINDOW+=(--max-frames "${COUNT}")

# Vérifie que DA3 couvre bien la fenêtre avant de lancer quoi que ce soit
END=$(( START + (COUNT > 0 ? COUNT : 0) ))
if [[ "${COUNT}" -gt 0 ]]; then
  miss=0
  for ((i=START; i<END; i++)); do
    printf -v n "depth_%04d.npy" "$i"
    [[ -f "${DA3}/${n}" ]] || miss=$((miss+1))
  done
  if [[ "${miss}" -gt 0 ]]; then
    echo "ERREUR: ${miss}/${COUNT} a priori DA3 manquants dans ${DA3}"
    echo "  → ${PY} scripts/precompute_da3_sequences.py --sequences ${SEQ} --start ${START} --limit ${COUNT}"
    exit 1
  fi
fi

for MODE in sensor asaf; do
  OUT="${ROOT_OUT}/${MODE}"
  SEQDIR="${OUT}/sequence"
  if [[ -f "${SEQDIR}/export_meta.txt" ]]; then
    echo "[${MODE}] export déjà présent, réutilisé"
  else
    echo "[${MODE}] export…"
    EXTRA=()
    [[ "${MODE}" == "asaf" ]] && EXTRA=(--checkpoint "${CKPT}" --depth-any-dir "${DA3}")
    "${PY}" slam/export_orbslam_depth.py --mode "${MODE}" \
      --frames-csv "${FRAMES}" "${WINDOW[@]}" "${EXTRA[@]}" \
      --output "${SEQDIR}"
  fi

  if [[ -f "${OUT}/trajectory.tum" ]]; then
    echo "[${MODE}] trajectoire déjà présente, réutilisée"
  else
    echo "[${MODE}] ORB-SLAM3…"
    ORB_SETTINGS="${SETTINGS}" SEQUENCE_DIR="${SEQDIR}" OUT_DIR="${OUT}" \
      bash scripts/run_orbslam3_sequence.sh
  fi

  "${PY}" scripts/eval_slam_evo.py --ref "${GT}" --est "${OUT}/trajectory.tum" \
    --label "orbslam3_${TAG}_${MODE}" --output-dir "${OUT}" 2>&1 | grep -E "^(===|ATE|RPE)" || true
done

# Contrôle d'intégrité + synthèse
"${PY}" - "${ROOT_OUT}" "${GT}" <<'PY'
import sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd

root, gt_path = Path(sys.argv[1]), Path(sys.argv[2])

def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()

# Les RGB doivent être identiques entre les deux modes (seule la profondeur change),
# sinon la comparaison n'est pas contrôlée.
for name in ("000000.png", "000100.png"):
    a = root/"sensor"/"sequence"/"rgb"/name
    b = root/"asaf"/"sequence"/"rgb"/name
    if a.is_file() and b.is_file() and md5(a) != md5(b):
        raise SystemExit(f"INCOHÉRENCE: rgb/{name} diffère entre sensor et asaf")
    da, db = root/"sensor"/"sequence"/"depth"/name, root/"asaf"/"sequence"/"depth"/name
    if da.is_file() and db.is_file() and md5(da) == md5(db):
        raise SystemExit(f"SUSPECT: depth/{name} identique entre sensor et asaf")

gt = np.loadtxt(gt_path, comments="#")
_, i = np.unique(gt[:, 0], return_index=True); gt = gt[np.sort(i)]

rows = []
for mode in ("sensor", "asaf"):
    s = root/mode/"evo_summary.csv"
    if not s.exists(): continue
    df = pd.read_csv(s); df.insert(0, "mode", mode)
    est = np.loadtxt(root/mode/"trajectory.tum")
    n_seq = sum(1 for l in (root/mode/"sequence"/"associations.txt").read_text().splitlines() if l.strip())
    w = gt[(gt[:, 0] >= est[0, 0]) & (gt[:, 0] <= est[-1, 0])]
    path = float(np.linalg.norm(np.diff(w[:, 1:4], axis=0), axis=1).sum())
    df["tracking_rate"] = len(est)/max(n_seq, 1)
    df["path_len_m"] = path
    df["ate_pct_of_path"] = 100.0*df["ate_rmse_m"]/max(path, 1e-9)
    rows.append(df)

if not rows: raise SystemExit("aucune métrique")
out = pd.concat(rows, ignore_index=True)
cols = ["mode", "ate_rmse_m", "ate_pct_of_path", "rpe_trans_rmse_m", "rpe_rot_rmse_rad",
        "tracking_rate", "path_len_m"]
out.to_csv(root/"summary.csv", index=False)
print()
print(out[[c for c in cols if c in out.columns]].to_string(index=False))
print(f"\n→ {root/'summary.csv'}")
PY
