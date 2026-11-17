#!/usr/bin/env bash
# ============================================================================
# Build VIGS-Fusion sur pod RunPod — Ubuntu 22.04/24.04, ROS 2 auto, CUDA 12.x
#
# La distro ROS, la référence vision_opencv, le dépôt CUDA et l'origine de
# Ceres sont déduits de l'Ubuntu du pod (cf. bloc « Distro ROS » plus bas).
# Cible de référence : jammy + Humble + Ceres 2.0.0 + SuiteSparse 5.10.1,
# soit exactement les versions annoncées par le README de VIGS-Fusion.
#
#   bash scripts/pod_build_vigs.sh
#
# Étagé et REPRENABLE : chaque étape pose un marqueur dans ${STAMP_DIR}.
# Relancer le script reprend à la première étape non validée.
#
#   FORCE_STEP=3 bash scripts/pod_build_vigs.sh   # refaire à partir de l'étape 3
#
# ---------------------------------------------------------------------------
# Choix techniques (justifiés) :
#
#  * CUDA 12.6 et non 13 : OpenCV 4.10 est éprouvé avec CUDA 12.x. Le driver
#    580 du pod est rétro-compatible. Le CMakeLists de VIGS impose en dur
#    /usr/local/cuda/bin/nvcc — on crée donc bien ce lien.
#
#  * CUDA_ARCH_BIN=8.9 (RTX 4090 uniquement) : compiler une seule architecture
#    au lieu de toutes divise le temps de build d'OpenCV par ~4.
#
#  * BUILD_LIST restreint : seuls les modules utilisés par VIGS et cv_bridge
#    sont compilés (pas de dnn, ni de python, ni de tests).
#
#  * cv_bridge/image_geometry RECOMPILÉS depuis les sources dans le workspace :
#    Ubuntu 24.04 fournit OpenCV 4.6, or on installe OpenCV 4.10 dans
#    /usr/local. Sans cela, cv_bridge (lié à 4.6) et vigs_fusion (lié à 4.10)
#    cohabitent dans le MÊME processus -> conflit d'ABI et crash au runtime.
#    C'est le piège classique de cette intégration.
# ============================================================================
set -eo pipefail

PROJECT="${DEPTH_FUSION_PROJECT:-/workspace/depth_fusion_project}"
WS="${ROS2_WS:-/workspace/ros2_ws}"
SRC="${WS}/src"
BUILD_ROOT="${BUILD_ROOT:-/workspace/build}"
STAMP_DIR="${BUILD_ROOT}/.stamps"
OPENCV_VERSION="${OPENCV_VERSION:-4.10.0}"
CUDA_PKG="${CUDA_PKG:-cuda-toolkit-12-6}"
CUDA_ARCH="${CUDA_ARCH:-8.9}"          # RTX 4090
# ATTENTION : sur un pod RunPod, `nproc` ET /proc/meminfo renvoient les valeurs
# de l'HÔTE, pas celles allouées au conteneur. Mesuré sur un pod 32 vCPU/125 Go :
# nproc -> 256, MemTotal -> 1007 Go. Se fier à ces valeurs, c'est lancer -j256
# sur 27 cœurs, et chaque job nvcc réclamant ~1,5 Go, l'OOM-killer répond par un
# « Error 255 » muet en fin de make. Les limites réelles sont dans les cgroups.
_cgroup_cpus() {
  local quota period                       # cgroup v2
  if [[ -r /sys/fs/cgroup/cpu.max ]]; then
    read -r quota period < /sys/fs/cgroup/cpu.max
    [[ "${quota}" != "max" ]] && { echo $(( quota / period )); return; }
  fi                                       # cgroup v1
  if [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
    quota="$(< /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
    period="$(< /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
    [[ "${quota}" -gt 0 ]] && { echo $(( quota / period )); return; }
  fi
  nproc                                    # hors conteneur : valeur honnête
}
_cgroup_mem_gb() {
  local b
  for f in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [[ -r "${f}" ]]; then
      b="$(< "${f}")"
      # « max » (v2) ou la sentinelle ~2^63 (v1) signifient : aucune limite.
      [[ "${b}" != "max" && "${b}" -lt 4611686018427387904 ]] && {
        echo $(( b / 1073741824 )); return; }
    fi
  done
  awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo
}
_CORES="$(_cgroup_cpus)"
_MEM_GB="$(_cgroup_mem_gb)"
[[ "${_CORES}" -lt 1 ]] && _CORES=1
_MAX_BY_MEM=$(( _MEM_GB * 2 / 3 ))          # ~1,5 Go par job de compilation
[[ "${_MAX_BY_MEM}" -lt 1 ]] && _MAX_BY_MEM=1
NPROC="${NPROC:-$(( _CORES < _MAX_BY_MEM ? _CORES : _MAX_BY_MEM ))}"
# --------------------------------------------------------------------------
# Distro ROS déduite de l'Ubuntu du pod. Jazzy N'EXISTE PAS pour 22.04 : forcer
# le dépôt « noble » sur jammy installe des paquets liés à une autre glibc.
#
# Et 22.04 n'est pas un pis-aller : le README de VIGS-Fusion spécifie
# SuiteSparse 5.10.1 et Ceres 2.0.0, qui sont EXACTEMENT les versions de jammy
# (1:5.10.1+dfsg-4build1 et 2.0.0+dfsg1-5). Sur noble, SuiteSparse 7.6 oblige à
# désactiver le support creux de Ceres et Ceres 2.2 à recompiler la 2.1 à la
# main — on s'y éloigne de l'environnement des auteurs.
UBUNTU_CODENAME="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
case "${UBUNTU_CODENAME}" in
  jammy)  ROS_DISTRO=humble; VISION_OPENCV_REF_DEFAULT=humble ;;
  noble)  ROS_DISTRO=jazzy;  VISION_OPENCV_REF_DEFAULT=4.1.0  ;;
  *) echo "Ubuntu '${UBUNTU_CODENAME}' non pris en charge (jammy ou noble)"; exit 1 ;;
esac
ROS_DISTRO="${ROS_DISTRO_OVERRIDE:-${ROS_DISTRO}}"
VISION_OPENCV_REF="${VISION_OPENCV_REF:-${VISION_OPENCV_REF_DEFAULT}}"

# Ceres depuis les paquets système quand ils conviennent (jammy : 2.0.0, la
# version du README). Sur noble il faudra compiler 2.1.0 — cf. étape 6.
case "${UBUNTU_CODENAME}" in
  jammy) CERES_FROM_SOURCE="${CERES_FROM_SOURCE:-0}" ;;
  *)     CERES_FROM_SOURCE="${CERES_FROM_SOURCE:-1}" ;;
esac

# Python de ROS : les paquets ROS d'un distro sont compilés pour LE python de
# l'Ubuntu correspondant (3.10 sur jammy). Or l'image RunPod fait pointer
# /usr/bin/python3 vers 3.11 : les entry points colcon (shebang /usr/bin/python3)
# chargeraient alors un interpréteur sans rclpy. On fixe la cible ici et on y
# installera torch à l'étape 3.
case "${ROS_DISTRO}" in
  humble) ROS_PYTHON=/usr/bin/python3.10 ;;
  jazzy)  ROS_PYTHON=/usr/bin/python3.12 ;;
esac

mkdir -p "${STAMP_DIR}" "${SRC}" "${BUILD_ROOT}"
export DEBIAN_FRONTEND=noninteractive

log()  { echo "[$(date -u +'%F %T UTC')] $*"; }
done_step()  { touch "${STAMP_DIR}/step$1"; }
need_step()  {
  if [[ -n "${FORCE_STEP}" && "$1" -ge "${FORCE_STEP}" ]]; then return 0; fi
  [[ ! -f "${STAMP_DIR}/step$1" ]]
}

log "=== VIGS-Fusion build | GPU / disque ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h /workspace | tail -1
echo "  cœurs hôte : ${_CORES} | RAM : ${_MEM_GB} Go | parallélisme retenu : -j${NPROC}"

# --------------------------------------------------------------------------
# 1. Dépendances système
# --------------------------------------------------------------------------
if need_step 1; then
  log "=== [1/6] Dépendances système (apt) ==="
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config wget curl gnupg lsb-release ca-certificates \
    libeigen3-dev libgoogle-glog-dev libgflags-dev libunwind-dev \
    libsuitesparse-dev libceres-dev \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libboost-dev libboost-python-dev python3-numpy \
    python3-pip python3-dev unzip software-properties-common \
    xvfb
  done_step 1
else
  log "=== [1/6] Dépendances système — déjà fait ==="
fi

# --------------------------------------------------------------------------
# 2. CUDA toolkit (nvcc) -> /usr/local/cuda
# --------------------------------------------------------------------------
if need_step 2; then
  log "=== [2/6] CUDA toolkit (${CUDA_PKG}) ==="
  if ! command -v /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
    # Le dépôt CUDA est indexé par version d'Ubuntu : se tromper de codename
    # installe des paquets liés à la mauvaise glibc.
    case "${UBUNTU_CODENAME}" in
      jammy) CUDA_REPO=ubuntu2204 ;;
      noble) CUDA_REPO=ubuntu2404 ;;
    esac
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb" \
      -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -qq
    apt-get install -y --no-install-recommends "${CUDA_PKG}"
    # /usr/local/cuda est imposé par le CMakeLists de vigs_fusion
    if [[ ! -e /usr/local/cuda ]]; then
      ln -sfn "$(ls -d /usr/local/cuda-12.* | sort -V | tail -1)" /usr/local/cuda
    fi
  fi
  /usr/local/cuda/bin/nvcc --version | tail -2
  done_step 2
else
  log "=== [2/6] CUDA toolkit — déjà fait ==="
fi
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDACXX=/usr/local/cuda/bin/nvcc

# --------------------------------------------------------------------------
# 3. ROS 2 (${ROS_DISTRO}) + python de ROS
# --------------------------------------------------------------------------
if need_step 3; then
  log "=== [3/6] ROS 2 ${ROS_DISTRO} (${UBUNTU_CODENAME}) ==="
  if [[ ! -f /opt/ros/${ROS_DISTRO}/setup.bash ]]; then
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${UBUNTU_CODENAME} main" \
      > /etc/apt/sources.list.d/ros2.list
    apt-get update -qq
    apt-get install -y --no-install-recommends \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-image-transport \
      ros-${ROS_DISTRO}-message-filters \
      ros-${ROS_DISTRO}-tf2 ros-${ROS_DISTRO}-tf2-ros \
      ros-${ROS_DISTRO}-tf2-eigen ros-${ROS_DISTRO}-tf2-geometry-msgs \
      ros-${ROS_DISTRO}-rosbag2 ros-${ROS_DISTRO}-rosbag2-storage-mcap \
      ros-${ROS_DISTRO}-sensor-msgs ros-${ROS_DISTRO}-nav-msgs \
      python3-colcon-common-extensions
  fi

  # --- aligner /usr/bin/python3 sur le python de ROS ------------------------
  # L'image RunPod pointe python3 vers 3.11, pour lequel aucun paquet ROS
  # n'existe. Les entry points générés par colcon portent le shebang
  # /usr/bin/python3 : sans ce réalignement, asaf_node démarrerait sur un
  # interpréteur dépourvu de rclpy (et le torch de l'image, installé en 3.11,
  # serait tout aussi invisible). On bascule donc TOUT sur le python de ROS.
  if [[ "$(readlink -f /usr/bin/python3)" != "$(readlink -f "${ROS_PYTHON}")" ]]; then
    log "  python3 : $(python3 --version 2>&1) -> $(${ROS_PYTHON} --version 2>&1)"
    ln -sfn "${ROS_PYTHON}" /usr/bin/python3
  fi
  "${ROS_PYTHON}" -m pip install --quiet --upgrade pip

  # torch pour le python de ROS (celui de l'image ne sert plus à rien ici).
  # cu121 est le wheel officiel compatible avec le driver du pod et suffit
  # amplement à l'inférence ASAF/DA3 — inutile de coller à la version du CUDA
  # toolkit utilisé pour compiler VIGS, les deux sont indépendants.
  if ! "${ROS_PYTHON}" -c 'import torch' 2>/dev/null; then
    log "  installation de torch pour $(${ROS_PYTHON} --version 2>&1)"
    "${ROS_PYTHON}" -m pip install --quiet \
      torch torchvision --index-url https://download.pytorch.org/whl/cu121
  fi
  "${ROS_PYTHON}" -c 'import torch; print("  torch", torch.__version__, "cuda:", torch.cuda.is_available())'
  done_step 3
else
  log "=== [3/6] ROS 2 ${ROS_DISTRO} — déjà fait ==="
fi
source /opt/ros/${ROS_DISTRO}/setup.bash

# --------------------------------------------------------------------------
# 4. OpenCV + contrib AVEC CUDA  (étape la plus longue)
# --------------------------------------------------------------------------
if need_step 4; then
  log "=== [4/6] OpenCV ${OPENCV_VERSION} + contrib avec CUDA (arch ${CUDA_ARCH}) ==="
  cd "${BUILD_ROOT}"
  [[ -d opencv ]] || git clone --depth 1 -b "${OPENCV_VERSION}" https://github.com/opencv/opencv.git
  [[ -d opencv_contrib ]] || git clone --depth 1 -b "${OPENCV_VERSION}" https://github.com/opencv/opencv_contrib.git

  mkdir -p opencv/build && cd opencv/build
  cmake .. \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH="${BUILD_ROOT}/opencv_contrib/modules" \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN="${CUDA_ARCH}" \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=OFF \
    -D OPENCV_DNN_CUDA=OFF \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_TBB=ON \
    -D BUILD_opencv_python3=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_DOCS=OFF \
    -D BUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,calib3d,features2d,flann,video,cudev,cudaarithm,cudaimgproc,cudawarping,cudafilters \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    2>&1 | tail -25

  log "  compilation OpenCV (-j${NPROC}) — compter 20-50 min..."
  make -j"${NPROC}" 2>&1 | tail -15
  make install
  ldconfig
  log "  OpenCV installé : $(pkg-config --modversion opencv4 2>/dev/null || echo '?')"
  done_step 4
else
  log "=== [4/6] OpenCV CUDA — déjà fait ==="
fi

# --------------------------------------------------------------------------
# 5. Sources du workspace ROS
# --------------------------------------------------------------------------
if need_step 5; then
  log "=== [5/6] Sources workspace ROS ==="
  cd "${SRC}"
  # VIGS-Fusion
  [[ -d VIGS-Fusion ]] || git clone --depth 1 https://github.com/AbdoullahNdoye/VIGS-Fusion.git
  # cuda_utils est un paquet imbriqué : l'exposer au workspace
  if [[ -d VIGS-Fusion/include/cuda_utils && ! -e cuda_utils ]]; then
    ln -sfn "${SRC}/VIGS-Fusion/include/cuda_utils" "${SRC}/cuda_utils"
  fi
  # cv_bridge + image_geometry recompilés contre NOTRE OpenCV (anti conflit ABI).
  # La référence dépend du distro : vision_opencv a bien une branche "humble",
  # mais PAS de branche "jazzy" — pour Jazzy c'est la série 4.x, ciblée par tag.
  # Fixée en tête de script (VISION_OPENCV_REF_DEFAULT).
  # nettoyer un éventuel clone partiel d'une tentative précédente
  # (forme "if" et non "[[ ]] && cmd" : sous set -e, une condition fausse
  #  renverrait 1 et ferait avorter le script)
  if [[ -d vision_opencv && ! -d vision_opencv/cv_bridge ]]; then
    rm -rf vision_opencv
  fi
  [[ -d vision_opencv ]] || git clone --depth 1 -b "${VISION_OPENCV_REF}" https://github.com/ros-perception/vision_opencv.git

  # --- alias cv_bridge.hpp -> cv_bridge.h -----------------------------------
  # VIGS-Fusion inclut <cv_bridge/cv_bridge.hpp>, le nom adopté APRÈS Humble
  # (upstream a renommé l'en-tête en .hpp en gardant .h comme alias déprécié).
  # cv_bridge 3.2.x de Humble ne fournit que le .h. On rétablit donc l'alias
  # dans l'autre sens, du côté de cv_bridge, plutôt que de patcher les sources
  # de VIGS-Fusion : l'objet d'étude reste intact et le mémoire peut affirmer
  # que le dépôt amont a été compilé sans modification.
  #
  # C'est la SEULE incompatibilité d'en-tête sur Humble. Sur Jazzy il y en
  # aurait trois : message_filters/cache.h, message_filters/subscriber.h et
  # tf2/exceptions.h y ont été renommés en .hpp, alors que VIGS les inclut
  # encore en .h — argument supplémentaire pour jammy.
  CVB_INC="${SRC}/vision_opencv/cv_bridge/include/cv_bridge"
  if [[ -f "${CVB_INC}/cv_bridge.h" && ! -f "${CVB_INC}/cv_bridge.hpp" ]]; then
    cat > "${CVB_INC}/cv_bridge.hpp" <<'HPP'
// Alias de compatibilité ajouté par scripts/pod_build_vigs.sh.
// cv_bridge 3.2.x (Humble) ne livre que cv_bridge.h ; VIGS-Fusion inclut la
// forme .hpp introduite plus tard en amont. Ce fichier ne fait que rediriger.
#ifndef CV_BRIDGE_CV_BRIDGE_HPP_
#define CV_BRIDGE_CV_BRIDGE_HPP_
#include <cv_bridge/cv_bridge.h>
#endif  // CV_BRIDGE_CV_BRIDGE_HPP_
HPP
    log "  alias cv_bridge.hpp créé (Humble ne fournit que le .h)"
  fi

  # compressed_image_transport : VIGS souscrit avec color_transport=compressed.
  # Le paquet apt serait lié à l'OpenCV 4.6 du système et rouvrirait le conflit
  # d'ABI — on le compile donc ici, contre notre OpenCV 4.10.
  if [[ -d image_transport_plugins && ! -d image_transport_plugins/compressed_image_transport ]]; then
    rm -rf image_transport_plugins
  fi
  [[ -d image_transport_plugins ]] || git clone --depth 1 -b ${ROS_DISTRO} \
    https://github.com/ros-perception/image_transport_plugins.git
  ls -1 "${SRC}"
  done_step 5
else
  log "=== [5/6] Sources workspace — déjà fait ==="
fi

# --------------------------------------------------------------------------
# 6. Build colcon
# --------------------------------------------------------------------------
if need_step 6; then
  log "=== [6/6] colcon build ==="
  # Dépendances de cv_bridge 4.1.0 : il exige Boost.Python3 et les en-têtes
  # numpy pour ses bindings. Installé ici (et non à l'étape 1) car forcer
  # l'étape 1 relancerait la compilation d'OpenCV. apt est idempotent.
  log "  vérification des dépendances de cv_bridge (Boost.Python, numpy)"
  apt-get install -y --no-install-recommends \
    libboost-dev libboost-python-dev python3-numpy libturbojpeg0-dev >/dev/null 2>&1 || \
    apt-get install -y --no-install-recommends libboost-dev libboost-python-dev python3-numpy libturbojpeg0-dev

  # --- Ceres ----------------------------------------------------------------
  # Sur jammy, le paquet système EST la version du README de VIGS-Fusion
  # (2.0.0), qui expose encore ceres::LocalParameterization et s'appuie sur
  # SuiteSparse 5.10.1 — également la version du README. Rien à recompiler :
  # on reste au plus près de l'environnement des auteurs.
  # Sur noble en revanche, Ceres 2.2 a SUPPRIMÉ LocalParameterization, dont
  # VIGS-Fusion hérite dans LocalParameterization.hpp : il faut alors installer
  # 2.1.0 dans /usr/local et l'imposer via Ceres_DIR.
  if [[ "${CERES_FROM_SOURCE}" != "1" ]]; then
    log "  Ceres : paquet système $(dpkg-query -W -f='${Version}' libceres-dev 2>/dev/null)"
    log "  SuiteSparse : $(dpkg-query -W -f='${Version}' libsuitesparse-dev 2>/dev/null)"
  elif [[ ! -f /usr/local/lib/cmake/Ceres/CeresConfig.cmake ]]; then
    log "  compilation de Ceres 2.1.0 (la 2.2 du système est incompatible)"
    cd "${BUILD_ROOT}"
    [[ -d ceres-solver ]] || git clone --depth 1 -b 2.1.0 \
      https://github.com/ceres-solver/ceres-solver.git
    # Cache CMake obsolète d'une tentative précédente : le repartir à neuf,
    # sinon CXSPARSE=ON y persiste malgré les nouvelles options.
    rm -rf ceres-solver/build
    mkdir -p ceres-solver/build && cd ceres-solver/build
    # SUITESPARSE/CXSPARSE désactivés : Ubuntu 24.04 livre SuiteSparse 7.6,
    # dont le support n'est arrivé qu'en Ceres 2.2 (cibles IMPORTED absentes
    # en 2.1). VIGS n'utilise Ceres que pour une pose 6-DoF : les solveurs
    # Eigen/LAPACK suffisent, aucun besoin de factorisation creuse externe.
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DSUITESPARSE=OFF -DCXSPARSE=OFF -DLAPACK=ON \
      -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF \
      2>&1 | tee "${BUILD_ROOT}/ceres_cmake.log" | tail -12
    make -j"${NPROC}" 2>&1 | tee "${BUILD_ROOT}/ceres_make.log" | tail -8
    make install
    ldconfig
  fi

  # --- Neutraliser l'include OpenCV viz -------------------------------------
  # GaussianSplattingSlam.hpp inclut <opencv2/viz.hpp>, mais les SEULS usages
  # (cv::viz::imshow) sont commentés dans GaussianSplattingSlam.cu. Le module
  # viz exige VTK et n'est pas dans notre BUILD_LIST : on retire l'include
  # plutôt que de recompiler OpenCV pour du code mort.
  VIGS_HPP="${SRC}/VIGS-Fusion/include/gaussian_splatting_slam/GaussianSplattingSlam.hpp"
  if [[ -f "${VIGS_HPP}" ]] && grep -q '^#include <opencv2/viz.hpp>' "${VIGS_HPP}"; then
    sed -i 's|^#include <opencv2/viz.hpp>|// #include <opencv2/viz.hpp>  // desactive : module viz absent (VTK), usages commentes en amont|' "${VIGS_HPP}"
    log "  include opencv2/viz.hpp neutralisé"
  fi

  # --- Restaurer l'horodatage de /odom_gs -----------------------------------
  # GaussianSplattingSlamNode.cpp publie odomMsg (ligne ~268) SANS jamais
  # renseigner son header.stamp : la ligne qui le faisait
  # (`odomMsg.header.stamp = last_rgb_msg->header.stamp;`) se trouve dans un
  # bloc commenté juste au-dessus. Les poses sortent donc horodatées à zéro.
  #
  # Conséquence mesurée : l'enregistreur se rabat sur /clock, c'est-à-dire le
  # temps du bag à l'ARRIVÉE de la pose et non celui de l'image. Tout bras plus
  # lent (l'inférence ASAF, par exemple) voit ses poses décalées d'une latence
  # supplémentaire, et beaucoup moins d'entre elles tombent dans la fenêtre
  # d'association d'evo : 465 poses appariées contre 1964 pour le bras capteur.
  # L'ATE se calculait alors sur des portions de trajectoire différentes, ce qui
  # pénalisait mécaniquement le bras le plus coûteux — comparaison invalide.
  #
  # On restaure donc le comportement manifestement voulu par les auteurs. C'est
  # la seule modification de logique apportée au dépôt (l'autre patch, sur
  # opencv2/viz.hpp, ne retire que du code mort) et elle doit être mentionnée
  # comme telle dans toute publication de ces résultats.
  NODE_CPP="${SRC}/VIGS-Fusion/src/GaussianSplattingSlamNode.cpp"
  if [[ -f "${NODE_CPP}" ]] && ! grep -q 'RESTAURE horodatage odom' "${NODE_CPP}"; then
    sed -i 's|^\( *\)odomPub->publish(odomMsg);|\1// RESTAURE horodatage odom : cf. ligne ~230, neutralisee par un bloc commente\n\1if (last_rgb_msg) odomMsg.header.stamp = last_rgb_msg->header.stamp;\n\1odomPub->publish(odomMsg);|' "${NODE_CPP}"
    log "  horodatage de /odom_gs restauré"
  fi

  cd "${WS}"
  # CMAKE_CUDA_ARCHITECTURES est INDISPENSABLE : sans lui, nvcc retombe sur
  # sm_52 et Thrust/CUB compile ses politiques de tuning pour cette seule
  # architecture (visible dans le namespace THRUST_..._520_NS). Au runtime,
  # CUB ne trouve aucune politique pour le GPU réel (sm_89) et remonte un
  # cudaErrorInvalidDevice trompeur sur parallel_for.
  # La compilation sm_89 est plus gourmande : c'est le -j80 (et non le flag)
  # qui provoquait l'Error 255 muette — d'où le parallélisme bridé plus bas.
  CM_ARGS=(-DCMAKE_BUILD_TYPE=Release
           -DOpenCV_DIR=/usr/local/lib/cmake/opencv4
           -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
           -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH//./}")
  # Chemin des en-têtes numpy imposé explicitement. Les bindings Python de
  # cv_bridge incluent <numpy/ndarrayobject.h> ; or dès qu'un `pip install
  # numpy` a eu lieu (ce que fait pod_bootstrap_pydeps.sh pour épingler la
  # série 1.x), les en-têtes migrent de /usr/lib/python3/dist-packages vers
  # /usr/local/lib/python3.10/dist-packages et le paquet apt disparaît.
  # Un rebuild ultérieur échoue alors sur « numpy/ndarrayobject.h: No such
  # file or directory », bien après que tout ait fonctionné une première fois.
  _NUMPY_INC="$(python3 -c 'import numpy; print(numpy.get_include())' 2>/dev/null || true)"
  if [[ -n "${_NUMPY_INC}" ]]; then
    CM_ARGS+=("-DPython3_NumPy_INCLUDE_DIRS=${_NUMPY_INC}")
    log "  en-têtes numpy : ${_NUMPY_INC}"
  fi

  # Ceres_DIR n'est imposé que si l'on a compilé Ceres dans /usr/local. Pointer
  # vers un répertoire inexistant ferait échouer le find_package sur jammy, où
  # c'est le Ceres système (2.0.0) que l'on veut.
  if [[ -f /usr/local/lib/cmake/Ceres/CeresConfig.cmake ]]; then
    CM_ARGS+=(-DCeres_DIR=/usr/local/lib/cmake/Ceres)
  fi

  # Passe 1 : dépendances + cuda_utils
  colcon build --symlink-install \
    --packages-select cv_bridge image_geometry compressed_image_transport cuda_utils \
    --cmake-args "${CM_ARGS[@]}" 2>&1 | tail -20

  # --- Remplacer la libcuda_utils.so PRÉ-LIVRÉE du dépôt --------------------
  # vigs_fusion lie en dur ${PROJECT_SOURCE_DIR}/include/cuda_utils/build/
  # libcuda_utils.so, un binaire committé dans le dépôt et compilé avec une
  # AUTRE version de Thrust. Mélanger deux ABI Thrust dans un même processus
  # provoque des erreurs CUDA incohérentes (cudaErrorInvalidDevice sur un
  # parallel_for). On y installe donc la version que l'on vient de compiler.
  BUILT_LIB="$(find "${WS}/install/cuda_utils" "${WS}/build/cuda_utils" \
    -name 'libcuda_utils.so' 2>/dev/null | head -1)"
  TARGET_LIB="${SRC}/VIGS-Fusion/include/cuda_utils/build/libcuda_utils.so"
  if [[ -n "${BUILT_LIB}" ]]; then
    mkdir -p "$(dirname "${TARGET_LIB}")"
    [[ -f "${TARGET_LIB}.orig" ]] || cp -f "${TARGET_LIB}" "${TARGET_LIB}.orig" 2>/dev/null || true
    cp -f "${BUILT_LIB}" "${TARGET_LIB}"
    log "  libcuda_utils.so remplacée par la version locale (${BUILT_LIB})"
  else
    log "  AVERTISSEMENT : libcuda_utils.so compilée introuvable"
  fi

  # Passe 2 : vigs_fusion, contre la lib fraîchement compilée.
  # Parallélisme bridé : les .cu de VIGS sont volumineux et nvcc consomme
  # plusieurs Go par processus ; -j80 dépasse la limite mémoire du conteneur.
  rm -rf "${WS}/build/vigs_fusion" "${WS}/install/vigs_fusion"
  MAKEFLAGS="-j${VIGS_JOBS:-8}" colcon build --symlink-install \
    --packages-select vigs_fusion \
    --executor sequential \
    --cmake-args "${CM_ARGS[@]}" 2>&1 | tail -30
  done_step 6
else
  log "=== [6/6] colcon build — déjà fait ==="
fi

# --------------------------------------------------------------------------
log ""
log "============================================================"
log " BUILD TERMINÉ"
log "============================================================"
source "${WS}/install/setup.bash" 2>/dev/null || true
echo "Exécutables vigs_fusion :"
find "${WS}/install" -type f -name "*vigs*" -o -type f -name "*gaussian*" 2>/dev/null | head -10
echo ""
echo "Pour utiliser :"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash && source ${WS}/install/setup.bash"
echo "  ros2 pkg executables vigs_fusion"
