#!/bin/bash
# Section 5.3 (Tables 15-18, Figures 14-17): the 3-MLP detectors of script 02, evaluated
# (no retraining) on GenImage test images JPEG-compressed, blurred, resized, randomly and
# center cropped. The images are processed on the fly during feature extraction.
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ]; then BACKBONES=("$@"); else BACKBONES=("${BEST_BACKBONES[@]}"); fi
declare -A VALUES=(
    [jpeg]="95 90 85 80 75 70 65 60"
    [blur]="1 2 3 4"
    [resize]="0.5 0.7 0.9 1.1 1.3 1.5"
    [crop]="0.9 0.7 0.5 0.3"
    [center_crop]="0.9 0.7 0.5 0.3"
)
for B in "${BACKBONES[@]}"; do
    for OP in jpeg blur resize crop center_crop; do
        for V in ${VALUES[$OP]}; do
            extract --backbones "$B" --splits genimage_test --"$OP" "$V"
            evaluate --backbones "$B" --classifiers 3-MLP --tasks detection --eval_sets genimage_test --"$OP" "$V"
        done
    done
done
$PYTHON summarize.py "$OUT_DIR/metrics/detection.csv" --filter eval_set=genimage_test \
    --output "$OUT_DIR/summary_robustness.csv" > /dev/null
echo "Summary: $OUT_DIR/summary_robustness.csv (column eval_degradation)"
