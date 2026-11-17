#!/usr/bin/env bash
# ============================================================================
# Validation SLAM VIGS-Fusion : A/B profondeur capteur vs ASAF
#
#   bash scripts/pod_run_vigs_ab.sh
#
# Run A (sensor) : image_depth <- /camera/depth/image_rect_raw   (référence)
# Run B (asaf)   : image_depth <- /camera/depth_fused            (contribution)
#
# Chaîne du run B :
#   offline_da3_node  -> /camera/depth_any     (DA3 précalculées)
#   asaf_node         -> /camera/depth_fused   (RGB + DA3 + capteur, 16UC1 mm)
#   gaussian_splatting_slam_node -> /odom_gs
#
# Paramètres SLAM repris à l'identique du launch officiel du dépôt, pour que
# la SEULE variable entre A et B soit la source de profondeur.
#
# Variables : SEQ=Line-Rot  RATE=0.3  MODES="sensor asaf"
# ============================================================================
set -eo pipefail

PROJECT="${DEPTH_FUSION_PROJECT:-/workspace/depth_fusion_project}"
WS="${ROS2_WS:-/workspace/ros2_ws}"
SEQ="${SEQ:-Line-Rot}"
RATE="${RATE:-0.3}"
MODES="${MODES:-sensor asaf}"
# Distro ROS déduite de ce qui est réellement installé : selon l'image du pod
# c'est Humble (Ubuntu 22.04) ou Jazzy (24.04) — cf. pod_build_vigs.sh.
ROS_DISTRO="${ROS_DISTRO:-$(ls -1 /opt/ros 2>/dev/null | head -1)}"
[[ -n "${ROS_DISTRO}" ]] || { echo "ERREUR : aucun ROS 2 dans /opt/ros"; exit 1; }
CKPT="${CKPT:-${PROJECT}/checkpoints/experiments/asaf_net_v2_vigs/best_model.pth}"

SEQ_DIR="${PROJECT}/data/vigs_fusion/${SEQ}/extracted"
GT="${SEQ_DIR}/poses_gt.tum"
DA3_DIR="${SEQ_DIR}/da3_scaled"
OUT_ROOT="${PROJECT}/evaluation_results/vigs_slam/${SEQ}"

log() { echo "[$(date -u +'%F %T UTC')] $*"; }

source /opt/ros/${ROS_DISTRO}/setup.bash
source "${WS}/install/setup.bash"
export DEPTH_FUSION_ROOT="${PROJECT}"

# --- contrôles préalables --------------------------------------------------
BAG_DIR="$(find "${PROJECT}/data/vigs_fusion/${SEQ}" -maxdepth 3 -name 'metadata.yaml' -printf '%h\n' 2>/dev/null | head -1)"
[[ -n "${BAG_DIR}" ]] || { log "ERREUR : bag introuvable pour ${SEQ}"; exit 1; }
[[ -f "${GT}"     ]] || { log "ERREUR : GT absente : ${GT}"; exit 1; }
[[ -f "${CKPT}"   ]] || { log "ERREUR : checkpoint absent : ${CKPT}"; exit 1; }

# Transport compressé : même contrôle que dans pod_diag_vigs.sh. Sans le
# plugin, VIGS ne reçoit aucune couleur et meurt sur « invalid configuration
# argument » (pyramide image dimensionnée à 0x0) — un symptôme qui désigne
# faussement CUDA. On échoue ici, explicitement.
TRANSPORTS="$(ros2 run image_transport list_transports 2>&1 || true)"
if ! grep -q 'image_transport/compressed' <<< "${TRANSPORTS}"; then
  log "ERREUR : transport « compressed » indisponible — VIGS n'aurait aucune image couleur."
  sed 's/^/    /' <<< "${TRANSPORTS}"
  exit 1
fi

log "============================================================"
log " VIGS-Fusion A/B — ${SEQ}"
log "============================================================"
log "bag        : ${BAG_DIR}"
log "GT         : $(wc -l < "${GT}") poses"
log "checkpoint : ${CKPT}"
log "vitesse    : x${RATE}"

# --- couverture DA3 : bloquant pour le run ASAF -----------------------------
# offline_da3_node se contente de REJOUER des profondeurs précalculées. Si
# da3_scaled est incomplet, asaf_node perd son entrée monoculaire sur les
# frames manquantes, ne publie plus rien sur /camera/depth_fused, et VIGS
# poursuit sur la seule préintégration IMU — la trajectoire diverge sans
# qu'aucun nœud ne signale d'erreur. Un run B partiel est indétectable a
# posteriori dans le TUM : on refuse donc de le lancer.
if [[ " ${MODES} " == *" asaf "* ]]; then
  NFRAMES=$(( $(wc -l < "${SEQ_DIR}/frames.csv") - 1 ))
  NDA3=$(ls "${DA3_DIR}" 2>/dev/null | wc -l)
  if [[ "${NDA3}" -lt "${NFRAMES}" ]]; then
    log "ERREUR : DA3 incomplètes — ${NDA3}/${NFRAMES} frames ($(( 100 * NDA3 / NFRAMES )) %)"
    log "  Compléter :  SKIP_DL=1 SEQ=${SEQ} bash scripts/pod_prepare_slam.sh"
    exit 1
  fi
  log "DA3        : ${NDA3}/${NFRAMES} frames (complet)"
fi

# Nettoyage des nœuds résiduels d'un run précédent (orphelins possibles si le
# script a été interrompu). Le motif entre crochets empêche pkill de matcher
# sa PROPRE ligne de commande : '[g]aussian' matche "gaussian" mais pas la
# chaîne littérale "[g]aussian" présente dans notre cmdline.
log "Nettoyage des processus résiduels..."
pkill -f '[g]aussian_splatting_slam_node' 2>/dev/null || true
pkill -f '[a]saf_node' 2>/dev/null || true
pkill -f '[o]ffline_da3_node' 2>/dev/null || true
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

run_mode() {
  local MODE="$1"
  # RUN_TAG permet d'empiler plusieurs exécutions d'un même bras sans les
  # écraser. C'est indispensable ici : la dispersion entre exécutions s'est
  # révélée bien plus grande que l'effet recherché (ATE capteur 0,23 m à 0.3x
  # contre 0,78 m à 0.1x), donc un run unique par bras ne permet aucune
  # conclusion. Il faut pouvoir estimer la variabilité.
  local OUT="${OUT_ROOT}/${MODE}${RUN_TAG:+_${RUN_TAG}}"
  mkdir -p "${OUT}"
  log ""
  log "──────────── RUN ${MODE} ────────────"

  local DEPTH_TOPIC="/camera/depth/image_rect_raw"
  PIDS=()

  # 1. Enregistreur de trajectoire (/odom_gs -> TUM)
  # use_sim_time est INDISPENSABLE, contrairement à ce que ce commentaire
  # affirmait : VIGS publie /odom_gs avec un horodatage NUL (l'affectation se
  # trouve dans un bloc commenté, GaussianSplattingSlamNode.cpp:230). Le nœud se
  # rabat donc sur son horloge, et sans /clock il écrirait l'heure murale —
  # 47 millions de secondes d'écart avec la GT, aucune association possible.
  # Le script accepte désormais --ros-args (parse_known_args).
  python3 "${PROJECT}/scripts/record_odom_tum.py" \
    --topic /odom_gs --output "${OUT}/trajectory.tum" \
    --ros-args -p use_sim_time:=true \
    > "${OUT}/recorder.log" 2>&1 &
  PIDS+=($!)

  # 1bis. Extrinsèque IMU -> caméra.
  # VIGS fusionne IMU et images : sans cette TF il boucle sur
  # "No TRANSFORM YET" et ne publie aucune odométrie. Valeurs reprises du
  # launch de référence du projet (calibration xsens/RealSense).
  ros2 run tf2_ros static_transform_publisher \
    0.0456823 0.0360948 -0.0745264 \
    -0.00910219 0.0119687 -0.0126518 0.999804 \
    xsens_imu_link camera_link \
    --ros-args -p use_sim_time:=true \
    > "${OUT}/static_tf.log" 2>&1 &
  PIDS+=($!)
  sleep 2

  # --- profondeur en 32FC1 mètres, dans les DEUX bras ----------------------
  # VIGS fait `pyrDepth[0].create(..., CV_32FC1)` puis `upload(depthImg)`, et
  # upload() RÉALLOUE au type de la source : lui passer du 16UC1 (millimètres)
  # rend pyrDepth[0] en CV_16UC1, relu ensuite via Texture<float>. Les valeurs
  # sont aberrantes, aucune gaussienne n'est créée, et le kernel suivant part
  # sur une grille nulle -> « invalid configuration argument » ligne 1075.
  # cuda_depth_register assurait cette conversion chez les auteurs ; il n'est
  # pas publié, depth_to_metric_node le remplace.
  #
  # La conversion doit s'appliquer aux DEUX bras, sans quoi la comparaison
  # porterait sur des encodages différents et non sur la source de profondeur.
  if [[ "${MODE}" == "sensor" ]]; then
    ros2 run depth_fusion_ros depth_to_metric_node --ros-args \
      -p input_topic:=/camera/depth/image_rect_raw \
      -p output_topic:=/camera/depth_metric \
      -p use_sim_time:=true \
      > "${OUT}/depth_metric.log" 2>&1 &
    PIDS+=($!)
    DEPTH_TOPIC="/camera/depth_metric"
    sleep 2
  fi

  # 2. Chaîne ASAF si nécessaire
  if [[ "${MODE}" == "asaf" ]]; then
    DEPTH_TOPIC="/camera/depth_fused"

    ros2 run depth_fusion_ros offline_da3_node --ros-args \
      -p da3_dir:="${DA3_DIR}" \
      -p output_topic:=/camera/depth_any \
      -p use_sim_time:=true \
      > "${OUT}/da3_node.log" 2>&1 &
    PIDS+=($!)

    # IMPORTANT : model_version=v2 et image_size=448 — les défauts du nœud
    # (v1 / 256) ne correspondent PAS au modèle ASAF final et produiraient
    # une profondeur silencieusement dégradée.
    ros2 run depth_fusion_ros asaf_node --ros-args \
      -p project_root:="${PROJECT}" \
      -p checkpoint_path:="${CKPT}" \
      -p model_version:=v2 \
      -p image_size:=448 \
      -p use_cuda:=true \
      -p use_compressed_rgb:=true \
      -p mono_topic:=/camera/depth_any \
      -p sensor_topic:=/camera/depth/image_rect_raw \
      -p output_topic:=/camera/depth_fused \
      -p output_encoding:=32FC1 \
      -p use_sim_time:=true \
      > "${OUT}/asaf_node.log" 2>&1 &
    PIDS+=($!)

    log "  chargement du modèle ASAF..."
    sleep 25
  fi

  # 3. VIGS-Fusion — paramètres identiques au launch officiel
  #
  # xvfb-run : le nœud appelle startDisplayLoop() sans condition
  # (GaussianSplattingSlamNode.cpp:79), avant de lire `viewer` (ligne 118).
  # La boucle cv::imshow démarre donc toujours et avorte sans DISPLAY
  # (« Can't initialize GTK backend »). Serveur X virtuel plutôt que patch du
  # dépôt, pour que le SLAM évalué reste le code publié.
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
    -r image_depth:="${DEPTH_TOPIC}" \
    -r odom:=/odom_gs \
    -r imu_preint:=/filtered_imu_xsens \
    > "${OUT}/vigs_node.log" 2>&1 &
  PIDS+=($!)
  sleep 12

  # 3bis. Sonde de flux sur les topics intermédiaires.
  # Le run ASAF du 2026-08-27 s'est terminé avec 0 pose sans qu'aucun nœud ne
  # signale d'erreur : ni offline_da3_node ni asaf_node ne journalisent leurs
  # frames, si bien qu'un maillon muet est indiscernable d'un maillon qui
  # travaille. Cette sonde tranche en interrogeant les topics eux-mêmes.
  if [[ "${MODE}" == "asaf" ]]; then
    ( sleep 25
      for T in /camera/color/image_raw/compressed /camera/depth_any \
               /camera/depth_fused; do
        if timeout 10 ros2 topic echo --once "${T}" > /dev/null 2>&1; then
          echo "OK   ${T}"
        else
          echo "MUET ${T}   <-- maillon rompu"
        fi
      done
    ) > "${OUT}/topic_probe.log" 2>&1 &
    PIDS+=($!)
  fi

  # 4. Rejeu du bag (--clock : horloge simulée)
  log "  lecture du bag (x${RATE})..."
  ros2 bag play "${BAG_DIR}" --clock -r "${RATE}" > "${OUT}/bag_play.log" 2>&1 || true

  log "  fin du bag, arrêt des nœuds"
  sleep 8
  cleanup

  local NP
  NP=$(grep -cvE '^#' "${OUT}/trajectory.tum" 2>/dev/null || echo 0)
  log "  poses estimées : ${NP}"
  if [[ "${NP}" -lt 10 ]]; then
    log "  ⚠ trajectoire vide/insuffisante — voir ${OUT}/vigs_node.log"
    return 0
  fi

  # 5. Métriques ATE / RPE
  # `|| true` volontaire : sous `set -e`, un échec d'évaluation (association
  # de timestamps impossible, par exemple) avortait TOUT le script après le
  # run capteur, et le bras ASAF n'était jamais lancé — sept minutes de rejeu
  # perdues pour un calcul de métrique. Les trajectoires sont déjà sur disque :
  # on signale l'échec et on poursuit, les métriques se recalculent après coup.
  if ! python3 "${PROJECT}/scripts/eval_slam_evo.py" \
      --ref "${GT}" --est "${OUT}/trajectory.tum" \
      --label "vigs_${SEQ}_${MODE}" --output-dir "${OUT}" 2>&1 | tail -12; then
    log "  ⚠ évaluation impossible pour ${MODE} — trajectoire conservée dans ${OUT}"
  fi
}

for M in ${MODES}; do
  run_mode "${M}"
done

# --- synthèse --------------------------------------------------------------
log ""
log "============================================================"
log " RÉSULTATS — ${SEQ}"
log "============================================================"
for M in ${MODES}; do
  echo "--- ${M} ---"
  cat "${OUT_ROOT}/${M}${RUN_TAG:+_${RUN_TAG}}/evo_summary.csv" 2>/dev/null || echo "(absent)"
done
log ""
log "Dossier : ${OUT_ROOT}"
