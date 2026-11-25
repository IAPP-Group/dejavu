#!/bin/bash
# Section 5.2 (Tables 13-14, 35-42): detectors trained on a single generator (+ all real
# images) and on one representative per cluster (VQDM + Wukong), evaluated on every
# generator of GenImage validation and test. Per-generator results are written to
# $OUT_DIR/metrics/detection_per_generator.csv.
source "$(dirname "$0")/common.sh"
if [ $# -gt 0 ]; then BACKBONES=("$@"); else BACKBONES=("${BEST_BACKBONES[@]}"); fi
for B in "${BACKBONES[@]}"; do
    for GENS in adm biggan glide vqdm midjourney sdv4 sdv5 wukong "vqdm wukong"; do
        # $GENS unquoted on purpose: "vqdm wukong" -> two generators
        train --backbones "$B" --classifiers 3-MLP --tasks detection --train_generators $GENS \
              --eval_sets genimage --skip_existing
    done
done
$PYTHON summarize.py "$OUT_DIR/metrics/detection_per_generator.csv" --filter task=detection \
    --output "$OUT_DIR/summary_cross_model.csv" > /dev/null
echo "Summary: $OUT_DIR/summary_cross_model.csv (filter train_generators != all)"
