#!/usr/bin/env bash
# ============================================================================
# Dépendances Python du pod — à lancer APRÈS pod_build_vigs.sh
#
#   bash scripts/pod_bootstrap_pydeps.sh
#
# pod_build_vigs.sh a fait pointer /usr/bin/python3 vers le python de ROS
# (3.10 sur jammy) et n'y a installé que torch. Tout le reste du projet — DA3,
# depth_fusion, evo — doit y être ajouté : les paquets de l'image RunPod sont
# installés pour 3.11 et donc invisibles.
#
# On n'installe PAS requirements.txt de Depth-Anything-3 : il tire open3d,
# xformers, moviepy, fastapi… inutiles pour de l'inférence, longs à compiler,
# et xformers réinstallerait un torch d'une autre version CUDA. On se limite
# aux dépendances réellement importées par l'API d'inférence.
# ============================================================================
set -eo pipefail

PROJECT="${DEPTH_FUSION_PROJECT:-/workspace/depth_fusion_project}"
DA3_REPO="${PROJECT}/memoire_upgrade_da3/Depth-Anything-3"
PY="${PY:-/usr/bin/python3}"

log() { echo "[$(date -u +'%F %T UTC')] $*"; }

log "python cible : $(${PY} --version 2>&1) (${PY})"
"${PY}" -c 'import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available())'

# numpy < 2 est une contrainte DURE : DA3 l'exige dans son requirements, et
# surtout le checkpoint ASAF et tous les résultats antérieurs du mémoire ont
# été produits sous numpy 1.x.
#
# L'épingler au premier `pip install` NE SUFFIT PAS : un paquet installé plus
# tard le remonte en dépendance transitive et écrase silencieusement la
# version. C'est ce qui s'est produit avec opencv-python-headless, dont la
# borne « >=4.8 » laissait passer OpenCV 5.0 — lequel réclame numpy 2.
# Un fichier de contraintes s'applique à TOUTES les résolutions, y compris
# transitives : c'est le seul mécanisme qui tienne sur la durée du script.
CONSTRAINTS="$(mktemp /tmp/pip-constraints-XXXX.txt)"
printf 'numpy>=1.24,<2\n' > "${CONSTRAINTS}"
PIP_INSTALL=("${PY}" -m pip install --quiet -c "${CONSTRAINTS}")

log "=== [1/4] socle scientifique (numpy < 2 imposé par contrainte) ==="
"${PIP_INSTALL[@]}" numpy scipy pandas pillow tqdm pyyaml

log "=== [2/4] extraction VIGS + évaluation SLAM ==="
# opencv borné à la série 4 : la 5.0 impose numpy 2 et entrerait en conflit
# avec la contrainte ci-dessus.
"${PIP_INSTALL[@]}" \
  rosbags "opencv-python-headless>=4.8,<5" "evo>=1.28"

log "=== [3/4] Depth-Anything-3 (inférence seule) ==="
# depth_anything_3.api importe utils.export EN DUR, un dispatcher GLB/COLMAP/
# gaussian-splats dont le précalcul ne se sert jamais. Ses dépendances doivent
# néanmoins être présentes pour que l'import aboutisse — d'où trimesh (glb),
# pycolmap (colmap) et moviepy (gs_video), sans rapport avec l'inférence.
#   * moviepy < 2 : la 2.x a supprimé le module `moviepy.editor` qu'importe
#     gs.py ; une 2.x casserait l'import alors même que le code est inutilisé.
#   * gsplat est volontairement OMIS : DA3 le détecte absent, émet un simple
#     avertissement et continue. L'installer imposerait une compilation de
#     kernels CUDA de plusieurs minutes pour du code mort ici.
#   * addict n'est listé nulle part dans le requirements.txt du dépôt, alors
#     qu'il est importé par model/da3.py — d'où sa présence explicite.
"${PIP_INSTALL[@]}" \
  einops huggingface_hub safetensors imageio omegaconf \
  addict trimesh plyfile matplotlib "moviepy<2" pycolmap \
  hf_transfer
if [[ -d "${DA3_REPO}" ]]; then
  # --no-deps : le requirements du dépôt réinstallerait torch/xformers et
  # ferait sauter l'épinglage numpy fixé ci-dessus.
  "${PY}" -m pip install --quiet --no-deps -e "${DA3_REPO}"
else
  log "  AVERTISSEMENT : ${DA3_REPO} absent"
fi

log "=== [4/4] paquet depth_fusion ==="
# Le paquet est installé --no-deps (son requirements.txt réinstallerait torch
# et ferait sauter l'épinglage numpy), il faut donc fournir explicitement ce
# que son code importe réellement. Relevé par analyse AST des imports :
#   sklearn  -> fusion/scale_align.py, alignement RANSAC mono/capteur, appelé
#               à CHAQUE frame du précalcul DA3 — son absence casse la phase 2
#               après le chargement du modèle, soit très tard.
#   skimage  -> métriques de qualité d'image.
# wandb est volontairement omis : réservé au logging d'entraînement, et déjà
# commenté comme optionnel dans requirements.txt.
"${PIP_INSTALL[@]}" scikit-learn scikit-image
"${PY}" -m pip install --quiet --no-deps -e "${PROJECT}"

log ""
log "=== vérification des imports ==="
"${PY}" - <<'EOF'
import importlib, sys
mods = [
    ("numpy", True), ("scipy", True), ("pandas", True), ("PIL", True),
    ("yaml", True), ("tqdm", True), ("cv2", True), ("rosbags", True),
    ("evo", True), ("torch", True),
    ("depth_anything_3", True), ("depth_fusion", True),
    ("huggingface_hub", True), ("einops", True),
    ("sklearn", True), ("skimage", True), ("addict", True),
]
bad = []
for name, required in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "")
        print(f"  OK   {name:20s} {v}")
    except Exception as e:
        print(f"  FAIL {name:20s} {type(e).__name__}: {e}")
        if required:
            bad.append(name)
# Le vrai test est l'import du SOUS-MODULE api : c'est lui qui tire la chaîne
# utils.export (addict, trimesh, pycolmap, moviepy). Un simple
# `import depth_anything_3` réussit sans eux et masquerait le problème —
# l'échec ne surviendrait qu'après le téléchargement de la séquence.
try:
    import depth_anything_3.api  # noqa: F401
    print("  OK   depth_anything_3.api")
except Exception as e:
    print(f"  FAIL depth_anything_3.api  {type(e).__name__}: {e}")
    bad.append("depth_anything_3.api")

import cv2
if cv2.__version__.startswith("5."):
    print(f"  ERREUR : cv2 {cv2.__version__} — la série 5 exige numpy 2")
    bad.append("cv2<5")
import numpy
if numpy.__version__.startswith("2."):
    print(f"  ERREUR : numpy {numpy.__version__} — la contrainte < 2 a sauté")
    bad.append("numpy<2")
sys.exit(1 if bad else 0)
EOF
log "Dépendances Python prêtes."
