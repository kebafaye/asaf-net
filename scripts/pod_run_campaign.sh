#!/usr/bin/env bash
# ============================================================================
# Campagne VIGS-Fusion complète — à lancer APRÈS pod_build_vigs.sh
#
#   bash scripts/pod_run_campaign.sh 2>&1 | tee -a /workspace/campaign.log
#
# Enchaîne, avec reprise sur marqueurs :
#   1. dépendances Python (dans le python de ROS)
#   2. depth_fusion_ros + téléchargement séquence + précalcul DA3
#   3. diagnostic de divergence (capteur, 0.3x puis 0.1x)
#   4. A/B capteur vs ASAF
#
# Chaque phase pose un marqueur : relancer reprend où ça s'est arrêté.
#   FORCE_PHASE=3 bash scripts/pod_run_campaign.sh   # refaire à partir de 3
#
# Variables : SEQ=Line-Rot  RATE=0.3  RATES="0.3 0.1"
# ============================================================================
set -eo pipefail

PROJECT="${DEPTH_FUSION_PROJECT:-/workspace/depth_fusion_project}"
WS="${ROS2_WS:-/workspace/ros2_ws}"
SEQ="${SEQ:-Line-Rot}"
STAMP_DIR="${STAMP_DIR:-/workspace/.campaign_stamps}"
mkdir -p "${STAMP_DIR}"
cd "${PROJECT}"

# Le téléchargement des poids DA3 depuis le CDN HuggingFace s'est figé une fois
# à mi-parcours depuis ce datacenter : 1 Kio/s, processus endormi, aucun
# timeout déclenché. La reprise a suffi (le cache conserve les .incomplete),
# mais sans borne de timeout la phase 2 peut s'enliser indéfiniment sans que
# rien ne le signale.
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
# NB : HF_HUB_ENABLE_HF_TRANSFER est déprécié (huggingface_hub est passé à Xet)
# et n'a plus d'effet. Xet est actif par défaut ; HF_XET_HIGH_PERFORMANCE=1
# serait l'équivalent moderne si le débit redevenait un problème.

log() { echo "[$(date -u +'%F %T UTC')] $*"; }
need() {
  if [[ -n "${FORCE_PHASE}" && "$1" -ge "${FORCE_PHASE}" ]]; then return 0; fi
  [[ ! -f "${STAMP_DIR}/phase$1" ]]
}
mark() { touch "${STAMP_DIR}/phase$1"; }

log "############################################################"
log " CAMPAGNE VIGS-Fusion — séquence ${SEQ}"
log "############################################################"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- phase 0 : build ---------------------------------------------------------
# Intégré ici pour qu'une seule commande suffise du pod nu aux résultats. Le
# script de build a ses propres marqueurs dans BUILD_ROOT : s'il a déjà tourné,
# il ne refait que ce qui manque.
#
# BUILD_ROOT sur le DISQUE CONTENEUR et non sur /workspace : ce dernier est un
# montage réseau MooseFS, calamiteux pour les dizaines de milliers de petites
# écritures d'une compilation OpenCV. Bénéfice annexe : les marqueurs d'étapes
# partagent alors la durée de vie des installations système qu'ils décrivent
# (/usr/local, /opt/ros), au lieu de leur survivre et de faire sauter des
# étapes dont les produits ont disparu.
if [[ ! -f "${WS}/install/vigs_fusion/share/vigs_fusion/package.xml" ]]; then
  log ""
  log "=== [phase 0/4] build VIGS-Fusion (~20 min) ==="
  CUDA_ARCH="${CUDA_ARCH:-8.9}" BUILD_ROOT="${BUILD_ROOT:-/build}" \
    bash scripts/pod_build_vigs.sh
fi
if [[ ! -f "${WS}/install/vigs_fusion/share/vigs_fusion/package.xml" ]]; then
  log "ERREUR : vigs_fusion toujours absent après le build — voir la sortie ci-dessus."
  exit 1
fi
log "vigs_fusion présent : OK"

# --------------------------------------------------------------------------
if need 1; then
  log ""
  log "=== [phase 1/4] dépendances Python ==="
  bash scripts/pod_bootstrap_pydeps.sh
  mark 1
else
  log "=== [phase 1/4] dépendances Python — déjà fait ==="
fi

# --------------------------------------------------------------------------
if need 2; then
  log ""
  log "=== [phase 2/4] séquence ${SEQ} + DA3 (le plus long, ~40 min) ==="
  SEQ="${SEQ}" bash scripts/pod_prepare_slam.sh
  mark 2
else
  log "=== [phase 2/4] séquence + DA3 — déjà fait ==="
fi

# La couverture DA3 conditionne la validité du run ASAF : on la contrôle ici
# aussi, car pod_prepare_slam.sh n'échoue pas si le précalcul est partiel.
SEQ_DIR="${PROJECT}/data/vigs_fusion/${SEQ}/extracted"
if [[ -f "${SEQ_DIR}/frames.csv" ]]; then
  NF=$(( $(wc -l < "${SEQ_DIR}/frames.csv") - 1 ))
  ND=$(ls "${SEQ_DIR}/da3_scaled" 2>/dev/null | wc -l)
  log "couverture DA3 : ${ND}/${NF}"
  if [[ "${ND}" -lt "${NF}" ]]; then
    log "ERREUR : DA3 incomplètes — le run ASAF serait silencieusement faussé."
    log "  Relancer :  SKIP_DL=1 SEQ=${SEQ} bash scripts/pod_prepare_slam.sh"
    exit 1
  fi
fi

# --------------------------------------------------------------------------
if need 3; then
  log ""
  log "=== [phase 3/4] diagnostic de divergence (capteur, 0.3x et 0.1x) ==="
  SEQ="${SEQ}" bash scripts/pod_diag_vigs.sh
  mark 3
else
  log "=== [phase 3/4] diagnostic — déjà fait ==="
fi

# --------------------------------------------------------------------------
if need 4; then
  log ""
  log "=== [phase 4/4] A/B capteur vs ASAF ==="
  SEQ="${SEQ}" bash scripts/pod_run_vigs_ab.sh
  mark 4
else
  log "=== [phase 4/4] A/B — déjà fait ==="
fi

log ""
log "############################################################"
log " CAMPAGNE TERMINÉE — ${SEQ}"
log "############################################################"
log "diagnostic : ${PROJECT}/evaluation_results/vigs_diag/${SEQ}/"
log "A/B        : ${PROJECT}/evaluation_results/vigs_slam/${SEQ}/"
