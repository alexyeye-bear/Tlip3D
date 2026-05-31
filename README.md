# Tlip3D

Combine 3D vector quantization, contrastive learning, and transformer-based
generation for volumetric brain tokens.

## Project Structure

```text
Tlip3D/
  vq/
    vqgan3d.py          # 3D VQ-GAN model
    codebook3d_ema.py   # EMA codebook used by VQGAN3D
    encoder3d.py        # 3D encoder
    decoder3d.py        # 3D decoder
    helper_3d.py        # shared 3D blocks
    simplevq.py         # basic VQ-VAE quantizer
    rqvae.py            # residual vector quantizer
    groupvq.py          # grouped vector quantizer
    train_vq.py         # VQ training entry
    ablation/           # public quantizer ablation templates
    export_vq_tokens.py # sharded VQ-index export utilities
  dataloaders/
    custom_loader.py
  token_modeling/
    mask_transformer/   # multi-frame masked token prediction
    qwen_lora/          # VQ-special-token Qwen LoRA
  configs/              # path-free example configs
  docs/                 # data format and experiment summaries
  results/              # aggregate, de-identified result tables
  train_clip.py
  transformer.py
  train_transformer.py
```

The root-level VQ files are kept as compatibility shims. New code should import from the `vq` package.

## VQ Modules

```python
from vq import VQGAN3D, Codebook3D
from vq import SimpleVectorQuantizer, ResidualVectorQuantizer, GroupVectorQuantizer
```

Available quantizers:

- `Codebook3D`: EMA codebook used by the existing 3D VQ-GAN.
- `SimpleVectorQuantizer`: standard VQ-VAE quantizer for channel-first tensors.
- `ResidualVectorQuantizer`: RQ-VAE style residual quantization.
- `GroupVectorQuantizer`: splits channels into groups and quantizes each group independently.

## Training

Train the 3D VQ model from the repository root:

```bash
python -m vq.train_vq
```

Train the transformer stage:

```bash
python transformer.py
```

Train the contrastive learning stage:

```bash
python train_clip.py
```

## Brain Token Pipeline

Recent experiments added a discrete brain-token workflow:

1. Train or choose a 3D VQ tokenizer.
2. Export each 3D frame to integer codebook indices in `sharded_pt_v1`.
3. Build short temporal segments from consecutive frames.
4. Train either a single-pass Mask Transformer or a Qwen LoRA model with atomic
   `<vq_000>` ... `<vq_127>` tokens.

Create a tiny synthetic token export for smoke tests:

```bash
python -m vq.export_vq_tokens \
  --make-synthetic \
  --output-root examples/tiny_tokens \
  --num-train 32 \
  --num-test 16
```

Train the multi-frame Mask Transformer:

```bash
python -m token_modeling.mask_transformer.train_multiframe_mask_transformer \
  --token-root examples/tiny_tokens \
  --output-dir outputs/mask_transformer \
  --device cpu \
  --epochs 1 \
  --max-train-steps 2 \
  --max-val-steps 1
```

The Qwen LoRA path requires `transformers`, `peft`, and access to the selected
base model:

```bash
python -m token_modeling.qwen_lora.train_qwen_vq_lora \
  --token-root "$TOKEN_ROOT" \
  --output-dir "$OUTPUT_ROOT/qwen_vq_lora" \
  --model-name Qwen/Qwen2.5-0.5B
```

## Reference Results

Aggregate results from internal, non-public fMRI data are included only as
de-identified tables:

- `results/vq_ablation_summary.csv`
- `results/token_generation_summary.csv`
- `docs/experiments.md`

No raw fMRI data, subject-level manifests, checkpoints, token shards, logs, or
local machine paths are distributed in this repository.
