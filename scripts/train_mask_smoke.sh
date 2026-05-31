#!/usr/bin/env bash
set -euo pipefail

TOKEN_ROOT="${1:-examples/tiny_tokens}"
OUTPUT_ROOT="${2:-outputs/mask_smoke}"

python -m token_modeling.mask_transformer.train_multiframe_mask_transformer \
  --token-root "$TOKEN_ROOT" \
  --output-dir "$OUTPUT_ROOT" \
  --device cpu \
  --epochs 1 \
  --batch-size 2 \
  --max-train-steps 2 \
  --max-val-steps 1
