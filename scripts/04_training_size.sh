#!/bin/bash
# Section 4.4 (Figure 9, Tables 21-32): 3-MLP trained on 1, 5, 10, 50, 100, 300, 500 and
# 800 (= all) prompts, 10 runs, each run with its own random prompts (seed 42 + run);
# evaluated on GenImage validation.
# Section 4.4.1 (Figure 10, Tables 33-34): the single-prompt detectors evaluated on the
# validation splits of the OOD benchmarks.
source "$(dirname "$0")/common.sh"
train --backbones all --classifiers 3-MLP --tasks "${TASKS[@]}" \
      --num_prompts 1 5 10 50 100 300 500 800 --eval_sets genimage_validation --skip_existing "$@"
evaluate --backbones all --classifiers 3-MLP --tasks detection --num_prompts 1 --eval_sets ood_validation
