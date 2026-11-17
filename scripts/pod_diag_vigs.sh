#!/usr/bin/env bash
# ============================================================================
# Diagnostic de divergence VIGS-Fusion — profondeur CAPTEUR à deux cadences
#
#   bash scripts/pod_diag_vigs.sh
#
# Question tranchée par ce script :
#   la divergence observée sur Line-Rot à t≈57 s est-elle due au MANQUE DE
#   CALCUL (le nœud ne tient pas la cadence) ou à la GÉOMÉTRIE du mouvement
#   (rotation pure, sans parallaxe) ?
#
# Protocole : le MÊME bag, en profondeur capteur, rejoué à 0.3x puis 0.1x.
# Baisser la cadence triple le budget de calcul par image sans rien changer
# d'autre. Donc :
#   - divergence à 0.3x mais PAS à 0.1x  -> manque de calcul, corrigeable
#   - divergence aux DEUX cadences       -> géométrique, la cadence n'y peut
#                                            rien : il faudra rapporter l'ATE
#                                            sur le segment suivi.
#
# On reste en mode capteur : introduire ASAF ici ajouterait une variable et
# rendrait le résultat ininterprétable.
#
# Variables : SEQ=Line-Rot  RATES="0.3 0.1"
# ============================================================================
set -eo pipefail

PROJECT="${DEPTH_FUSION_PROJECT:-/workspace/depth_fusion_project}"
WS="${ROS2_WS:-/workspace/ros2_ws}"
SEQ="${SEQ:-Line-Rot}"
RATES="${RATES:-0.3 0.1}"
ROS_DISTRO="${ROS_DISTRO:-$(ls -1 /opt/ros 2>/dev/null | head -1)}"
[[ -n "${ROS_DISTRO}" ]] || { echo "ERREUR : aucun ROS 2 dans /opt/ros"; exit 1; }

SEQ_DIR="${PROJECT}/data/vigs_fusion/${SEQ}/extracted"
GT="${SEQ_DIR}/poses_gt.tum"
OUT_ROOT="${PROJECT}/evaluation_results/vigs_diag/${SEQ}"

log() { echo "[$(date -u +'%F %T UTC')] $*"; }

source /opt/ros/${ROS_DISTRO}/setup.bash
source "${WS}/install/setup.bash"
export DEPTH_FUSION_ROOT="${PROJECT}"

BAG_DIR="$(find "${PROJECT}/data/vigs_fusion/${SEQ}" -maxdepth 3 -name 'metadata.yaml' -printf '%h\n' 2>/dev/null | head -1)"
[[ -n "${BAG_DIR}" ]] || { log "ERREUR : bag introuvable pour ${SEQ}"; exit 1; }
[[ -f "${GT}"     ]] || { log "ERREUR : GT absente : ${GT}"; exit 1; }

# --- contrôle préalable : transport compressé -------------------------------
# Le bag ne contient la couleur QUE sous /camera/color/image_raw/compressed, et
# VIGS s'y abonne via image_transport en mode `compressed`. Si le plugin n'est
# pas découvert au runtime, l'abonnement reste MUET : aucune erreur n'est levée,
# mais rgbImg reste vide, initAndCopyImgs dimensionne la pyramide à 0x0, et le
# premier lancement de kernel échoue sur « invalid configuration argument »
# précédé de « nb Gaussians : 0 » — un message qui accuse faussement CUDA.
# Contrôle bon marché, gardé par précaution — mais ce n'est PAS la cause du
# crash « nb Gaussians : 0 » rencontré le 2026-08-26 : celui-ci venait du type
# de la profondeur (16UC1 au lieu de 32FC1), corrigé par depth_to_metric_node
# plus bas.
TRANSPORTS="$(ros2 run image_transport list_transports 2>&1 || true)"
if ! grep -q 'image_transport/compressed' <<< "${TRANSPORTS}"; then
  log "ERREUR : le transport « compressed » n'est pas disponible pour image_transport."
  log "  VIGS ne recevrait jamais d'image couleur, et échouerait sur un message CUDA trompeur."
  log "  Transports découverts :"
  sed 's/^/    /' <<< "${TRANSPORTS}"
  exit 1
fi
log "transport compressed : disponible"

log "============================================================"
log " Diagnostic VIGS — ${SEQ} — profondeur capteur"
log "============================================================"
log "bag      : ${BAG_DIR}"
log "cadences : ${RATES}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

pkill -f '[g]aussian_splatting_slam_node' 2>/dev/null || true
pkill -f '[r]ecord_odom_tum' 2>/dev/null || true
pkill -f '[s]tatic_transform_publisher' 2>/dev/null || true
sleep 3

PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill "${p}" 2>/dev/null || true; done
  sleep 2
  for p in "${PIDS[@]}"; do kill -9 "${p}" 2>/dev/null || true; done
  PIDS=()
}
trap cleanup EXIT

run_rate() {
  local RATE="$1"
  local OUT="${OUT_ROOT}/sensor_r${RATE}"
  mkdir -p "${OUT}"
  log ""
  log "──────────── cadence x${RATE} ────────────"
  PIDS=()

  # use_sim_time est INDISPENSABLE : VIGS publie /odom_gs sans horodatage
  # (l'affectation est dans un bloc commenté, GaussianSplattingSlamNode.cpp:230),
  # et l'enregistreur se rabat alors sur son horloge. Sans horloge simulée il
  # écrirait l'heure murale, inassociable avec la GT du bag.
  python3 "${PROJECT}/scripts/record_odom_tum.py" \
    --topic /odom_gs --output "${OUT}/trajectory.tum" \
    --ros-args -p use_sim_time:=true \
    > "${OUT}/recorder.log" 2>&1 &
  PIDS+=($!)

  # Extrinsèque IMU -> caméra : sans cette TF, VIGS boucle sur
  # « No TRANSFORM YET » et ne publie aucune odométrie.
  ros2 run tf2_ros static_transform_publisher \
    0.0456823 0.0360948 -0.0745264 \
    -0.00910219 0.0119687 -0.0126518 0.999804 \
    xsens_imu_link camera_link \
    --ros-args -p use_sim_time:=true \
    > "${OUT}/static_tf.log" 2>&1 &
  PIDS+=($!)
  sleep 2

  # --- profondeur en 32FC1 mètres -----------------------------------------
  # VIGS charge la profondeur SANS aucune conversion :
  #   pyrDepth[0].create(..., CV_32FC1)  puis  pyrDepth[0].upload(depthImg)
  # or upload() RÉALLOUE la GpuMat au type de la source. Lui donner le 16UC1
  # brut du capteur (millimètres) transforme donc pyrDepth[0] en CV_16UC1, que
  # le code relit ensuite via Texture<float> : des entiers 16 bits réinterprétés
  # en flottants 32 bits. Toutes les profondeurs deviennent aberrantes, aucune
  # gaussienne n'est générée, et le kernel suivant est lancé sur une grille
  # (0+255)/256 = 0 -> « invalid configuration argument » ligne 1075, un message
  # qui accuse CUDA à tort.
  # C'est exactement l'étape qu'assurait cuda_depth_register chez les auteurs,
  # nœud absent du dépôt public.
  ros2 run depth_fusion_ros depth_to_metric_node --ros-args \
    -p input_topic:=/camera/depth/image_rect_raw \
    -p output_topic:=/camera/depth_metric \
    -p use_sim_time:=true \
    > "${OUT}/depth_metric.log" 2>&1 &
  PIDS+=($!)
  sleep 2

  # Échantillonnage GPU pendant le run : si la thèse du manque de calcul
  # était juste, on verrait l'utilisation coller à 100 % avant le décrochage.
  ( while true; do
      echo "$(date +%s.%N),$(nvidia-smi --query-gpu=utilization.gpu,memory.used \
        --format=csv,noheader,nounits)"
      sleep 2
    done > "${OUT}/gpu_trace.csv" ) &
  PIDS+=($!)

  # Paramètres SLAM identiques au launch officiel : seule la cadence varie.
  #
  # xvfb-run est INDISPENSABLE sur un pod sans écran : le constructeur du nœud
  # appelle `gss.startDisplayLoop()` sans condition (GaussianSplattingSlamNode.cpp:79),
  # AVANT même de lire le paramètre `viewer` (ligne 118). Une boucle cv::imshow
  # démarre donc toujours, et sans DISPLAY OpenCV avorte sur
  # « Can't initialize GTK backend ». `viewer:=false` ne désactive que le
  # visualiseur 3D, pas cette boucle.
  # On fournit un serveur X virtuel plutôt que de patcher le dépôt : le SLAM
  # évalué reste le code publié, sans modification.
  xvfb-run -a ros2 run vigs_fusion gaussian_splatting_slam_node --ros-args \
    -p color_transport:=compressed \
    -p pose_iterations:=12 -p update_iterations:=8 \
    -p eta_pose:=1.0e-5 -p eta_update:=0.5e-3 \
    -p gauss_init_size_px:=3 -p nb_pyr_levels:=3 -p downsample:=true \
    -p acc_n:=0.2 -p gyr_n:=0.0002 -p acc_w:=0.02 -p gyr_w:=0.0000002 \
    -p adam_eta:=1.0e-4 -p adam_beta1:=0.9 -p adam_beta2:=0.999 -p adam_epsilon:=1.0e-8 \
    -p covisibility_threshold:=0.85 -p optimizer:=adam \
    -p w_depth:=1.0 -p w_dist:=0.1 \
    -p world_frame_id:=map \
    -p pose_estimation_method:=WarpingMultipleRendering \
    -p viewer:=false -p use_sim_time:=true \
    -r camera_info:=/camera/color/camera_info \
    -r image_color/compressed:=/camera/color/image_raw/compressed \
    -r image_depth:=/camera/depth_metric \
    -r odom:=/odom_gs \
    -r imu_preint:=/filtered_imu_xsens \
    > "${OUT}/vigs_node.log" 2>&1 &
  PIDS+=($!)
  sleep 12

  log "  lecture du bag (x${RATE})..."
  ros2 bag play "${BAG_DIR}" --clock -r "${RATE}" > "${OUT}/bag_play.log" 2>&1 || true
  sleep 8
  cleanup

  local NP
  NP=$(grep -cvE '^#' "${OUT}/trajectory.tum" 2>/dev/null || echo 0)
  log "  poses estimées : ${NP}"
  [[ "${NP}" -lt 10 ]] && { log "  ⚠ trajectoire vide — voir ${OUT}/vigs_node.log"; return 0; }

  # analyze_vigs_trajectory.py sort 2 en cas de divergence : sous `set -e` un
  # appel nu avorterait le script, d'où le `|| true`. C'est le rapport qui
  # nous intéresse, pas le code retour.
  python3 "${PROJECT}/scripts/analyze_vigs_trajectory.py" \
    --ref "${GT}" --est "${OUT}/trajectory.tum" \
    --label "sensor_r${RATE}" --output-dir "${OUT}" || true

  if [[ -s "${OUT}/gpu_trace.csv" ]]; then
    log "  GPU (util %) : $(awk -F, '{s+=$2; if($2>m)m=$2} END {printf "moy %.0f  max %.0f", s/NR, m}' "${OUT}/gpu_trace.csv")"
  fi
}

for R in ${RATES}; do
  run_rate "${R}"
done

log ""
log "============================================================"
log " SYNTHÈSE — ${SEQ}"
log "============================================================"
for R in ${RATES}; do
  D="${OUT_ROOT}/sensor_r${R}/diagnostic.json"
  if [[ -f "${D}" ]]; then
    python3 -c "
import json,sys
d=json.load(open('${D}'))
print(f\"x${R}  diverge={d['diverged']}  onset={d['divergence_onset_s']}s  \"
      f\"parcours={d['est_path_length_m']}m (GT {d['gt_path_length_m']}m)  \"
      f\"debit {d['hz_early']}->{d['hz_late']} Hz\")"
  else
    echo "x${R}  (pas de diagnostic)"
  fi
done
log ""
log "Dossier : ${OUT_ROOT}"
