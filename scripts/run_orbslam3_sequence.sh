#!/usr/bin/env bash
# Lance ORB-SLAM3 RGB-D sur une séquence exportée (sensor ou asaf).
#
# Usage :
#   SEQUENCE_DIR=evaluation_results/orbslam3/Rotation-Only/sensor/sequence \
#   OUT_DIR=evaluation_results/orbslam3/Rotation-Only/sensor \
#   bash scripts/run_orbslam3_sequence.sh
set -euo pipefail

set +x
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
ORB_ROOT="${ORB_SLAM3_ROOT:-${PROJECT}/external/ORB_SLAM3}"
PANGO_PREFIX="${PROJECT}/external/pangolin_install"
VOCAB="${ORB_VOCAB:-${ORB_ROOT}/Vocabulary/ORBvoc.txt}"
SETTINGS="${ORB_SETTINGS:-${PROJECT}/config/orbslam3_realsense_d435i.yaml}"
SEQ_DIR="${SEQUENCE_DIR:?SEQUENCE_DIR requis}"
OUT_DIR="${OUT_DIR:-$(dirname "${SEQ_DIR}")}"
# Le script fait 'cd ORB_ROOT' plus bas (ORB-SLAM3 ecrit sa trajectoire dans le CWD),
# donc les chemins doivent etre absolus sinon le log et les copies partent a cote.
mkdir -p "${OUT_DIR}"
SEQ_DIR="$(cd "${SEQ_DIR}" && pwd)"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
ASSOC="${SEQ_DIR}/associations.txt"
BIN="${ORB_ROOT}/Examples/RGB-D/rgbd_tum"

# Pangolin installé sans sudo dans le prefix local
export LD_LIBRARY_PATH="${PANGO_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUT_DIR}"
LOG="${OUT_DIR}/orbslam3_run.log"

[[ -x "${BIN}" ]] || { echo "ERREUR: binaire manquant ${BIN}"; exit 1; }
[[ -f "${VOCAB}" ]] || { echo "ERREUR: vocabulaire manquant ${VOCAB}"; exit 1; }
[[ -f "${SETTINGS}" ]] || { echo "ERREUR: yaml manquant ${SETTINGS}"; exit 1; }
[[ -f "${ASSOC}" ]] || { echo "ERREUR: associations manquant ${ASSOC}"; exit 1; }

echo "[$(date -u +'%F %T UTC')] ORB-SLAM3 RGB-D"
echo "  seq=${SEQ_DIR}"
echo "  out=${OUT_DIR}"
echo "  settings=${SETTINGS}"

# Viewer OpenGL : Xvfb si pas de DISPLAY
XVFB_PREFIX=()
if [[ -z "${DISPLAY:-}" ]]; then
  if command -v xvfb-run >/dev/null 2>&1; then
    XVFB_PREFIX=(xvfb-run -a -s "-screen 0 1024x768x24")
  else
    echo "WARN: pas de DISPLAY ni xvfb-run — tentative headless"
  fi
fi

cd "${ORB_ROOT}"
# ORB-SLAM3 écrit CameraTrajectory.txt / KeyFrameTrajectory.txt dans le CWD
rm -f CameraTrajectory.txt KeyFrameTrajectory.txt

set +e
"${XVFB_PREFIX[@]}" "${BIN}" "${VOCAB}" "${SETTINGS}" "${SEQ_DIR}" "${ASSOC}" \
  > "${LOG}" 2>&1
RC=$?
set -e

if [[ -f CameraTrajectory.txt ]]; then
  cp CameraTrajectory.txt "${OUT_DIR}/trajectory.tum"
  echo "OK CameraTrajectory → ${OUT_DIR}/trajectory.tum ($(wc -l < "${OUT_DIR}/trajectory.tum") poses)"
elif [[ -f KeyFrameTrajectory.txt ]]; then
  cp KeyFrameTrajectory.txt "${OUT_DIR}/trajectory.tum"
  echo "OK KeyFrameTrajectory → ${OUT_DIR}/trajectory.tum ($(wc -l < "${OUT_DIR}/trajectory.tum") poses)"
else
  echo "ERREUR: aucune trajectoire écrite (rc=${RC}). Voir ${LOG}"
  tail -40 "${LOG}" || true
  exit 1
fi

[[ -f KeyFrameTrajectory.txt ]] && cp KeyFrameTrajectory.txt "${OUT_DIR}/keyframes.tum"
exit 0
