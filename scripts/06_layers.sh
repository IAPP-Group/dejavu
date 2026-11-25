#!/bin/bash
# Section 5.1, "Intermediate-layer features" (Figure 11): DejaVu trained on the [CLS] token
# of each of the 48 transformer blocks of MetaCLIP2-ViT-bigG/14, evaluated on GenImage test.
# All layers are extracted in a single pass (one file per layer, ~25 GB of RAM for the
# training split).
source "$(dirname "$0")/common.sh"
extract --backbones "$DEJAVU" --splits genimage --all_layers
LAYERS=($(seq -f "block%g" 0 47))
train --backbones "$DEJAVU" --classifiers 3-MLP --tasks "${TASKS[@]}" --layers "${LAYERS[@]}" \
      --eval_sets genimage_test --skip_existing "$@"
