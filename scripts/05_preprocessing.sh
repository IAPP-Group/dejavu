#!/bin/bash
# Section 5.1, "Impact of preprocessing" (Table 12): DejaVu with the De-Fake, UnivFD and
# OpenAI CLIP preprocessing instead of MetaCLIP 2's default one, on GenImage test.
# The default-preprocessing results come from script 02.
source "$(dirname "$0")/common.sh"
for PRE in defake univfd openai; do
    extract --backbones "$DEJAVU" --splits genimage --preprocess "$PRE"
    train --backbones "$DEJAVU" --classifiers 3-MLP --tasks "${TASKS[@]}" --preprocess "$PRE" \
          --eval_sets genimage_test --skip_existing "$@"
done
