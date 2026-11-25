#!/bin/bash
# Section 4 (Tables 4-9, Figures 2, 4, 7): the 6 classifiers on the 22 backbones for
# detection, attribution and joint detection+attribution, trained on GenImage train and
# evaluated on GenImage validation (used for all design choices) and test.
# MLPs: 10 runs each. Then the per-run metrics are averaged by summarize.py.
source "$(dirname "$0")/common.sh"
train --backbones all --classifiers all --tasks "${TASKS[@]}" --eval_sets genimage --skip_existing "$@"
for TASK in "${TASKS[@]}"; do
    $PYTHON summarize.py "$OUT_DIR/metrics/$TASK.csv" --output "$OUT_DIR/summary_$TASK.csv" > /dev/null
done
echo "Summaries: $OUT_DIR/summary_<task>.csv"
