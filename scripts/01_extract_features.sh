#!/bin/bash
# Final-layer features of the 22 backbones (default preprocessing) on GenImage
# train/validation/test and on the validation/test splits of the four OOD benchmarks.
# Needed by scripts 02-05 and 07. DINOv3 requires `huggingface-cli login` (gated weights).
source "$(dirname "$0")/common.sh"
extract --backbones all --splits genimage ood "$@"
