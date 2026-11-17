#!/usr/bin/env bash
# Rebuild Pangolin (C++17) + ORB-SLAM3 without sudo.
set +x
set -euo pipefail
PROJECT="${DEPTH_FUSION_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXT="${PROJECT}/external"
PANGO_PREFIX="${EXT}/pangolin_install"
LOG="${EXT}/orbslam3_build.log"
# Parallelisme limite : Optimizer.cc / LoopClosing.cc prennent ~2 Go chacun,
# -j$(nproc) fait tuer cc1plus par l'OOM killer sur une machine a 8 Go.
JOBS="${JOBS:-$(( $(nproc) < 4 ? $(nproc) : 2 ))}"
{
echo "=== REBUILD $(date -u) ==="
echo "Fix: force C++17 for Pangolin (sigslot needs >=C++14); install prefix local (no sudo)"

mkdir -p "$EXT" "$PANGO_PREFIX"
cd "${EXT}/Pangolin"
rm -rf build
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PANGO_PREFIX}" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_FLAGS="-std=c++17" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_PANGOLIN_PYTHON=OFF
make -j"$(nproc)"
make install
echo "PANGOLIN_INSTALLED=$(ls "${PANGO_PREFIX}/lib"/libpangolin* 2>/dev/null | head -1)"

# Prefer local cmake config
PANGO_CMAKE=""
for d in "${PANGO_PREFIX}/lib/cmake/Pangolin" "${PANGO_PREFIX}/lib/cmake/pangolin" \
         "${PANGO_PREFIX}/lib/cmake/Pangolin" "${PANGO_PREFIX}/share/pangolin/cmake"; do
  if [[ -d "$d" ]]; then PANGO_CMAKE="$d"; break; fi
done
echo "Pangolin_DIR=${PANGO_CMAKE}"
ls "${PANGO_PREFIX}/lib" | head -20

cd "${EXT}/ORB_SLAM3"
if [[ -f Vocabulary/ORBvoc.txt.tar.gz && ! -s Vocabulary/ORBvoc.txt ]]; then
  (cd Vocabulary && tar -xf ORBvoc.txt.tar.gz)
fi
# Ensure vocab exists
ls -lh Vocabulary/ORBvoc.txt | head -1

# DBoW2 : le CMake racine attend Thirdparty/DBoW2/lib/libDBoW2.so deja construit
# (g2o est un add_subdirectory, Sophus est header-only : rien a faire pour eux)
cd "${EXT}/ORB_SLAM3/Thirdparty/DBoW2"
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"${JOBS}"
[[ -f "${EXT}/ORB_SLAM3/Thirdparty/DBoW2/lib/libDBoW2.so" ]] || { echo "BUILD_FAIL: libDBoW2.so"; exit 1; }

cd "${EXT}/ORB_SLAM3"
rm -rf build
mkdir build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_PREFIX_PATH="${PANGO_PREFIX}" \
  ${PANGO_CMAKE:+-DPangolin_DIR="${PANGO_CMAKE}"}
make -j"${JOBS}"
cd ..
BIN=Examples/RGB-D/rgbd_tum
if [[ -x "$BIN" ]]; then
  echo "BUILD_OK $(date -u)"
  echo "BIN=$(pwd)/$BIN"
else
  echo "BUILD_FAIL: $BIN missing"
  find . -name 'rgbd_tum' 2>/dev/null || true
  exit 1
fi
} >"$LOG" 2>&1
echo DONE >>"$LOG"
# mirror short status
tail -30 "$LOG" > /tmp/orb_status.txt
cat /tmp/orb_status.txt
