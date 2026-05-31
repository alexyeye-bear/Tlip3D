#!/usr/bin/env bash
set -euo pipefail

python -m vq.export_vq_tokens \
  --make-synthetic \
  --output-root "${1:-examples/tiny_tokens}" \
  --num-train 32 \
  --num-test 16 \
  --token-shape 8,8,8 \
  --num-codebook-vectors 128
