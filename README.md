# ASAF-Net — fusion adaptative de profondeur RGB-D et monoculaire

Code du mémoire de Master 2 Robotique (Université Numérique Cheikh Hamidou Kane, 2025-2026) :
*Amélioration de la profondeur d'une caméra RGB-D à l'aide d'un modèle d'estimation de
profondeur basé sur l'apprentissage profond*.

ASAF-Net (*Adaptive Scale-Aware Fusion*) complète la carte de profondeur d'une caméra RGB-D
en la fusionnant avec la prédiction du modèle monoculaire Depth Anything 3
(`DA3MONO-LARGE`), recalée en mètres sur le capteur. Le réseau raffine une estimation de
départ composite (capteur là où il mesure, monoculaire aligné ailleurs) et produit une
carte de confiance.

## Résultats principaux

| Mesure | Résultat |
|---|---|
| Écart à la mesure du capteur (pixels mesurés) | 8,03 mm, contre 15,07 mm pour une fusion par règles fixes |
| Erreur sur les régions à compléter | −38 % face au monoculaire aligné (144,9 mm) |
| VIGS-Fusion, séquence de test Line-Rot (banc principal) | erreur absolue de trajectoire 0,096 m contre 0,626 m avec le capteur, soit ÷ 6,6 ; dispersion entre exécutions ÷ 11 |
| ORB-SLAM3 (référence complémentaire) | aucune dégradation (écarts < 4 %) |

Corpus : [VIGS-Fusion](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CI0K9G),
doi:10.57745/CI0K9G, version 2.0. Partition par séquence : entraînement Line +
Square-Shape (5 684 trames), validation Rotation-Only (2 653), test Line-Rot +
XYZ-Rot-obstacles (7 637).

## Trois niveaux de test

| Niveau | Ce qui est vérifié | Il faut | Durée |
|---|---|---|---|
| [1. Démo](#1-démo-sur-processeur) | ASAF complète une trame | Le code et les poids ; un processeur suffit | 10 min |
| [2. Métriques de profondeur](#2-métriques-de-profondeur) | Les chiffres de fidélité et de complétion | + le corpus (~26 Go) et le précalcul DA3 ; GPU conseillé | quelques heures |
| [3. Localisation VIGS-Fusion](#3-localisation-avec-vigs-fusion) | L'erreur de trajectoire ÷ 6,6 | GPU type RTX 4090, Ubuntu 22.04, ROS 2 Humble | ~1 h de préparation + campagne |

## Installation

Python ≥ 3.10.

```bash
git clone https://github.com/kebafaye/asaf-net.git && cd asaf-net
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # installe aussi le paquet depth_fusion
pytest -q tests                     # 6 tests, sur processeur
```

Les chemins sont relatifs à la racine du dépôt. Pour placer les données ailleurs :
`export DEPTH_FUSION_ROOT=/chemin/vers/le/depot_ou_les_donnees`.

### Poids du modèle

Les poids entraînés (`asaf_net_v2_vigs.pth`, 120 Mo, époque 65 sur 80) ne sont pas versionnés.
Ils sont publiés dans la [release v1.0](https://github.com/kebafaye/asaf-net/releases/tag/v1.0) :

```bash
curl -L -o checkpoints/asaf_net_v2_vigs.pth \
  https://github.com/kebafaye/asaf-net/releases/download/v1.0/asaf_net_v2_vigs.pth
cd checkpoints && sha256sum -c asaf_net_v2_vigs.pth.sha256 && cd ..
```

## 1. Démo sur processeur

```bash
# Vérifier l'installation, sans données ni poids
python scripts/demo_asaf.py --selftest

# Une trame du corpus (après les étapes de préparation du niveau 2)
python scripts/demo_asaf.py --checkpoint checkpoints/asaf_net_v2_vigs.pth \
    --sequence-dir data/vigs_fusion/Line-Rot/extracted --index 1500

# Vos propres fichiers : image, profondeur capteur, profondeur monoculaire alignée en mètres
python scripts/demo_asaf.py --checkpoint checkpoints/asaf_net_v2_vigs.pth \
    --rgb image.png --sensor depth.npy --mono mono_aligned.npy
```

Sortie dans `outputs/demo_asaf/` : `depth_fused.npy` (mètres), `confidence.npy` et
`demo_asaf.png`. Sur un Intel Core i7-1255U, l'inférence prend ~0,4 à 0,5 s par trame
(hors modèle monoculaire).

## 2. Métriques de profondeur

```bash
# a. Télécharger et extraire les séquences (bags ROS 2 -> images, profondeur, poses)
python scripts/download_vigs_sequences.py --split all --extract

# b. Précalcul Depth Anything 3 + alignement métrique RANSAC (GPU conseillé)
git clone https://github.com/ByteDance-Seed/depth-anything-3 memoire_upgrade_da3/Depth-Anything-3
git -C memoire_upgrade_da3/Depth-Anything-3 checkout 2c21ea8
pip install -r requirements-train.txt
python scripts/precompute_da3_sequences.py --split all

# c. Manifestes d'entraînement et de validation
python scripts/build_vigs_training_csv.py

# d. Évaluation (fidélité au capteur et complétion des trous synthétiques)
python scripts/benchmark_baselines.py --asaf_v2_ckpt checkpoints/asaf_net_v2_vigs.pth \
    --methods B0_sensor B1_da3 B2_ransac B5_asaf_v2 --output_dir evaluation_results/repro/benchmark_val
python scripts/eval_completion_v2.py --checkpoint checkpoints/asaf_net_v2_vigs.pth \
    --output_dir evaluation_results/repro/completion_val
```

Réentraîner (80 époques, environ 2 h 40 sur un GPU de 96 Go) :

```bash
python scripts/train.py --config-name train_asaf_v2
```

## 3. Localisation avec VIGS-Fusion

La campagne compare deux bras sur la séquence de test Line-Rot, rejouée à 0,1× : VIGS-Fusion
alimenté par la profondeur du capteur, puis par la sortie d'`asaf_node`. Deux exécutions par
bras.

Prérequis : GPU NVIDIA (campagne du mémoire conduite sur RTX 4090 ; `CUDA_ARCH` est à
adapter à la carte, 8.6 pour une RTX 3090, 8.0 pour une A100), Ubuntu 22.04, ROS 2 Humble,
CUDA 12.x. Les scripts ont été écrits pour une instance RunPod mais n'en dépendent pas.

```bash
# Sur la machine GPU, depuis la racine du dépôt
CUDA_ARCH=8.9 BUILD_ROOT=/build bash scripts/pod_build_vigs.sh   # VIGS-Fusion 046c6e4 + 4 correctifs
bash scripts/pod_run_campaign.sh                                  # nœud ROS, séquence, DA3, A/B
python scripts/summarize_vigs_campaign.py --seq Line-Rot
```

`pod_build_vigs.sh` applique au dépôt amont quatre correctifs (conversion de la profondeur en mètres, alias d'en-tête `cv_bridge.hpp`, exécution
sans écran, horodatage des poses). Le nœud de fusion ROS 2 est dans
[`ros/depth_fusion_ros`](ros/depth_fusion_ros) (`asaf_node`, `offline_da3_node`).

Le volet ORB-SLAM3 (référence complémentaire) se lance localement avec
`scripts/run_orbslam3_ab.sh` après `scripts/build_orbslam3_local.sh`.

## Organisation du dépôt

```
depth_fusion/          paquet Python : modèles (asaf_net.py), données, pertes, entraînement
configs/               configurations Hydra (train_asaf_v2.yaml = modèle final)
scripts/               préparation, entraînement, évaluation, démo, campagnes SLAM
ros/depth_fusion_ros/  nœuds ROS 2 (asaf_node, offline_da3_node)
slam/                  export de profondeur pour ORB-SLAM3
config/                réglages ORB-SLAM3 de la caméra du corpus
launch/                fichier de lancement ROS 2
tests/                 tests sur processeur (pytest)
```

## Limites connues

- La référence de profondeur est la mesure du capteur lui-même : les métriques mesurent la
  fidélité au capteur, et la complétion est évaluée sur des trous synthétiques.
- Le gain en localisation est établi sur une séquence et deux exécutions par
  configuration ; il dépend de la cadence de rejeu, à garder identique entre les bras.
- La confiance produite n'est pas calibrée.

## Citation

```
K. Faye, « Amélioration de la profondeur d'une caméra RGB-D à l'aide d'un modèle
d'estimation de profondeur basé sur l'apprentissage profond », mémoire de Master 2
Robotique, Université Numérique Cheikh Hamidou Kane, 2026.
```

Le corpus VIGS-Fusion et le système VIGS-Fusion sont dus à Ndoye et al. ; citez leurs
travaux si vous utilisez le corpus.
