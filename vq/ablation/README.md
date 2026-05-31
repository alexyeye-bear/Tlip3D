# 3D VQ Quantizer Ablation

This folder contains a public, path-free extraction of the short 3D VQ
quantizer ablation used before the token-modeling experiments.

The internal runs compared five 8x8x8 tokenizers:

- EMA VQ
- SimVQ
- FSQ
- BFQ
- Residual VQ

An additional SimVQ 4x4x4 run was used later for lower-resolution token export.
Only the reusable quantizer code, experiment protocol, and aggregate metrics are
included here. Internal data paths, subject identifiers, checkpoints, logs, and
run directories are intentionally omitted.

## Public Example

Prepare volumes as `.npy` or `.pt` tensors:

```text
$DATA_ROOT/volumes/
  train/
    sample_000.npy
  test/
    sample_000.npy
```

Then run a small residual-VQ ablation:

```bash
python -m vq.ablation.train \
  --volume-root "$DATA_ROOT/volumes" \
  --output-dir "$OUTPUT_ROOT/vq_ablation_residual" \
  --quantizer residual_vq
```

The demonstration trainer uses a small convolutional autoencoder so the
quantizer ideas are easy to reuse without depending on private fMRI loaders.
