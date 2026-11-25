# Shared settings of the experiment scripts. Override any of them from the environment, e.g.
#   DATA_CONFIG=configs/my_paths.yaml OUT_DIR=results bash scripts/02_main_tasks.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON=${PYTHON:-python}
DATA_CONFIG=${DATA_CONFIG:-configs/data_paths.yaml}
FEATURES_DIR=${FEATURES_DIR:-features}
OUT_DIR=${OUT_DIR:-results}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DEJAVU=MetaCLIP2-ViT-bigG-14
TASKS=(detection attribution detection+attribution)
# the best backbone of each family, compared with DejaVu in Sections 5.2 and 5.3
BEST_BACKBONES=(MetaCLIP2-ViT-bigG-14 MetaCLIP2-ViT-H-14 RN50x64 DINOv3-ViT-7B-16)

extract() { $PYTHON extract_features.py --data_config "$DATA_CONFIG" --features_dir "$FEATURES_DIR" "$@"; }
train()   { $PYTHON train.py    --features_dir "$FEATURES_DIR" --out_dir "$OUT_DIR" "$@"; }
evaluate(){ $PYTHON evaluate.py --features_dir "$FEATURES_DIR" --out_dir "$OUT_DIR" "$@"; }
