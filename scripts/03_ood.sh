#!/bin/bash
# Out-of-domain evaluation of the 3-MLP classifiers of script 02 (no retraining):
#  - detection on Synthbuster, AIGIBench, CO-SPY, Community Forensics (Figure 3, Tables 19-20)
#  - attribution on the images of the GenImage generators found in those benchmarks (Figures 5-6)
#  - joint task, the same images + the real images of the benchmarks (Figure 8)
# Validation and test splits; DejaVu's OOD test results are those of Table 10.
source "$(dirname "$0")/common.sh"
evaluate --backbones all --classifiers 3-MLP --tasks "${TASKS[@]}" --eval_sets ood "$@"
for TASK in "${TASKS[@]}"; do
    $PYTHON summarize.py "$OUT_DIR/metrics/$TASK.csv" --output "$OUT_DIR/summary_$TASK.csv" > /dev/null
done
