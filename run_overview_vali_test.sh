#!/bin/bash

# Usage:
#   ./run_valid_vs_test.sh RUN_NAME [MODEL_NAME] [BATCH_SIZE] [DEVICE]
#
# Example:
#   ./run_valid_vs_test.sh grid_search_20251122-065809_lr1e-4_wc0.05_wd0.1 destruction_finetune_siamese 64 cuda

RUN_NAME="20251126-071224"
MODEL_NAME="${2:-destruction_finetune_siamese}"  # default if not provided
BATCH_SIZE="${3:-64}"                            # default if not provided
DEVICE="${4:-cuda}"                              # default: cuda (falls back to cpu if unavailable in py script)


if [ -z "$RUN_NAME" ]; then
    echo "ERROR: You must provide a run name."
    echo "Usage: $0 RUN_NAME [MODEL_NAME] [BATCH_SIZE] [DEVICE]"
    exit 1
fi

python3 run_overview_vali_test.py \
  --run_name "$RUN_NAME" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --cities hostomel irpin livoberezhnyi moschun rubizhne volnovakha #aleppo damascus daraa deirezzor hama homs idlib raqqa #
