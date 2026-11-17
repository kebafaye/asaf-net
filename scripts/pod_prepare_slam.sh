#!/usr/bin/env bash
# ============================================================================
# Préparation de la validation SLAM VIGS-Fusion — à lancer APRÈS pod_build_vigs.sh
#
#   bash scripts/pod_prepare_slam.sh
#
# 1. Compile depth_fusion_ros (asaf_node, offline_da3_node)
# 2. Télécharge + extrait la séquence de TEST (jamais vue à l'entraînement)
# 3. Précalcule les profondeurs DA3 alignées sur cette séquence
# 4. Inventorie les topics du bag (nécessaire pour configurer les runs A/B)
#
# Variables :
#   SEQ=Line-Rot          séquence de test (défaut)
#   SKIP_DL=1             ne pas retélécharger
#   SKIP_DA3=1            ne pas recalculer DA3
# ============================================================================
set -eo pipefail

PROJECT="${DEPTH_FUSION_PROJECT:-/workspace/depth_fusion_project}"
WS="${ROS2_WS:-/workspace/ros2_ws}"
SEQ="${SEQ:-Line-Rot}"
# Distro ROS déduite de ce qui est réellement installé : selon l'image du pod
# c'est Humble (Ubuntu 22.04) ou Jazzy (24.04) — cf. pod_build_vigs.sh.
ROS_DISTRO="${ROS_DISTRO:-$(ls -1 /opt/ros 2>/dev/null | head -1)}"
[[ -n "${ROS_DISTRO}" ]] || { echo "ERREUR : aucun ROS 2 dans /opt/ros"; exit 1; }

export DEPTH_FUSION_PROJECT="${PROJECT}"
export DEPTH_FUSION_ROOT="${PROJECT}"
cd "${PROJECT}"

log() { echo "[$(date -u +'%F %T UTC')] $*"; }

log "============================================================"
log " Préparation SLAM VIGS-Fusion — séquence de test : ${SEQ}"
log "============================================================"
nvidia-smi --query-gpu=name --format=csv,noheader || true
df -h /workspace | tail -1

source /opt/ros/${ROS_DISTRO}/setup.bash

# --------------------------------------------------------------------------
# 1. depth_fusion_ros (nœuds ASAF / DA3 offline)
# --------------------------------------------------------------------------
log "=== [1/4] Compilation de depth_fusion_ros ==="
if [[ ! -d "${WS}/src/depth_fusion_ros" && -d "${PROJECT}/ros/depth_fusion_ros" ]]; then
  mkdir -p "${WS}/src"
  cp -r "${PROJECT}/ros/depth_fusion_ros" "${WS}/src/depth_fusion_ros"
  log "  depth_fusion_ros copié depuis le dépôt (${PROJECT}/ros)"
fi
if [[ ! -d "${WS}/src/depth_fusion_ros" ]]; then
  log "ERREUR : ${WS}/src/depth_fusion_ros absent."
  log "Depuis ton PC :  rsync -az -e 'ssh -p PORT -i CLE' <racine du dépôt>/ros/depth_fusion_ros/ root@IP:${WS}/src/depth_fusion_ros/"
  exit 1
fi
cd "${WS}"
colcon build --symlink-install --packages-select depth_fusion_ros 2>&1 | tail -12
source "${WS}/install/setup.bash"
log "  exécutables disponibles :"
ros2 pkg executables depth_fusion_ros 2>&1 | head -10

# --------------------------------------------------------------------------
# 2. Séquence de test (bag + frames)
# --------------------------------------------------------------------------
cd "${PROJECT}"
SEQ_DIR="${PROJECT}/data/vigs_fusion/${SEQ}"
if [[ "${SKIP_DL}" != "1" ]]; then
  log "=== [2/4] Téléchargement + extraction de ${SEQ} ==="
  if [[ -f "${SEQ_DIR}/extracted/frames.csv" ]]; then
    log "  déjà extraite : $(tail -n +2 "${SEQ_DIR}/extracted/frames.csv" | wc -l) frames"
  else
    python3 scripts/download_vigs_sequences.py --sequence "${SEQ}" --extract
  fi
else
  log "=== [2/4] SKIP_DL=1 — téléchargement ignoré ==="
fi

if [[ ! -f "${SEQ_DIR}/extracted/frames.csv" ]]; then
  log "ERREUR : extraction de ${SEQ} échouée (frames.csv absent)"
  exit 1
fi
NFRAMES=$(tail -n +2 "${SEQ_DIR}/extracted/frames.csv" | wc -l)
log "  ${SEQ} : ${NFRAMES} frames"

# --------------------------------------------------------------------------
# 3. DA3 sur la séquence de test
# --------------------------------------------------------------------------
if [[ "${SKIP_DA3}" != "1" ]]; then
  log "=== [3/4] Précalcul DA3 sur ${SEQ} (~$((NFRAMES / 120)) min) ==="
  # Sortie NON filtrée : un « | tail -5 » bufferise tout jusqu'à la fin du
  # processus, laissant 29 minutes sans le moindre signe de vie — impossible
  # alors de distinguer un calcul en cours d'un téléchargement bloqué.
  # La barre tqdm doit rester visible.
  python3 scripts/precompute_da3_sequences.py \
    --sequences "${SEQ}" \
    --repo-path "${PROJECT}/memoire_upgrade_da3/Depth-Anything-3" \
    --device cuda 2>&1
else
  log "=== [3/4] SKIP_DA3=1 — précalcul ignoré ==="
fi
NDA3=$(ls "${SEQ_DIR}/extracted/da3_scaled" 2>/dev/null | wc -l)
log "  DA3 calculées : ${NDA3} / ${NFRAMES}"

# --------------------------------------------------------------------------
# 4. Inventaire du bag (pour configurer les runs A/B)
# --------------------------------------------------------------------------
log "=== [4/4] Topics du bag ${SEQ} ==="
BAG_DIR="$(find "${SEQ_DIR}" -maxdepth 3 -name 'metadata.yaml' -printf '%h\n' 2>/dev/null | head -1)"
if [[ -n "${BAG_DIR}" ]]; then
  log "  bag : ${BAG_DIR}"
  ros2 bag info "${BAG_DIR}" 2>&1 | head -30
else
  log "  AVERTISSEMENT : aucun bag trouvé sous ${SEQ_DIR}"
  log "  (les runs devront rejouer les frames extraites au lieu du bag)"
  find "${SEQ_DIR}" -maxdepth 2 -type d | head
fi

log ""
log "============================================================"
log " PRÉPARATION TERMINÉE"
log "============================================================"
log "Séquence   : ${SEQ} (${NFRAMES} frames, ${NDA3} DA3)"
log "GT poses   : ${SEQ_DIR}/extracted/poses_gt.tum"
log "Checkpoint : ${PROJECT}/checkpoints/experiments/asaf_net_v2_vigs/best_model.pth"
log ""
log "Étape suivante : runs A/B VIGS-Fusion (capteur vs ASAF)."
