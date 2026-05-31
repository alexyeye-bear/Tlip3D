# Experiment Summary

This document summarizes the reusable ideas and aggregate metrics extracted
from internal fMRI experiments. No raw data, checkpoints, subject manifests, or
local machine paths are included.

## VQ Quantizer Ablation

Short 8-epoch runs compared five 8x8x8 tokenizers under a matched protocol
where possible: EMA, SimVQ, FSQ, BFQ, and Residual VQ. Residual VQ had the best
final reconstruction metrics in this short setting, while SimVQ became the main
8x8x8 token source for downstream token modeling.

See `results/vq_ablation_summary.csv` for the de-identified table.

## Multi-Frame Mask Transformer

The Mask Transformer flattens a short temporal segment of VQ indices, randomly
masks token positions, and predicts masked codes in one encoder pass. It is not
a multi-round MaskGIT sampler; validation accuracy measures single-pass masked
token recovery.

Reference internal result on about 100k SimVQ frames:

- Final validation loss: `0.2819`
- Final masked-token accuracy: `0.9324`

## Qwen VQ-LoRA

The Qwen path maps codebook indices to atomic special tokens such as
`<vq_000>` ... `<vq_127>`. Loss is applied only on VQ token positions; structure
tokens like `<f0>` and `</f0>` organize the 4D segment but are ignored by the
training loss.

Reference internal result:

- Teacher-forced validation loss / VQ accuracy: `0.4455 / 0.8818`
- Constrained 40 percent suffix free-run, 8 samples: micro accuracy `0.9628`

The free-run evaluation is intentionally narrow: structure tags are still
teacher-provided and generation is constrained to valid VQ tokens. It tests VQ
code error accumulation, not full free-form sequence formatting.
