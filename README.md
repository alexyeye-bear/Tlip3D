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

### VQ Tokenizer Ablation

| Method | Quantizer | Token grid | Epochs | Batch | Val corr ↑ | Rec loss ↓ | MSE ↓ | PSNR ↑ | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| EMA | `ema` | 8x8x8 | 8 | 32 | 0.9047 | 0.0984 | 0.0442 | 30.4579 | Matched five-method short ablation setting. |
| SimVQ | `simvq` | 8x8x8 | 8 | 32 | 0.9442 | 0.0814 | 0.0292 | 32.2795 | Main 8x8x8 token source for downstream token modeling. |
| FSQ | `fsq` | 8x8x8 | 8 | 16 | 0.9198 | 0.0793 | 0.0365 | 31.0747 | Rerun used because the original 32-batch run produced no validation CSVs. |
| BFQ | `bfq` | 8x8x8 | 8 | 32 | 0.9383 | 0.0772 | 0.0299 | 32.4662 | Matched five-method short ablation setting. |
| Residual VQ | `residual_vq` | 8x8x8 | 8 | 32 | **0.9451** | **0.0758** | **0.0266** | **32.9467** | Best final reconstruction metrics in the matched short ablation. |
| SimVQ-4grid | `simvq4` | 4x4x4 | 8 | 32 | 0.9370 | 0.0811 | 0.0320 | 32.1000 | Additional low-resolution tokenizer run for 4x4x4 token export. |

### Token Generation And Modeling

| Experiment | Token grid | Objective | Train / eval scope | Metric | Result | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Mask Transformer 30ep | 8x8x8 | Masked token prediction | About 100k frames from SimVQ partial tokens | Final validation loss / masked accuracy | **0.2819 / 0.9324** | Strongest current token-modeling result. |
| Qwen VQ-LoRA teacher forcing | 8x8x8 | Causal LM on VQ special tokens | About 100k frames from SimVQ partial tokens | Epoch-2 validation loss / VQ accuracy | 0.4455 / 0.8818 | Uses atomic `<vq_000>`...`<vq_127>` tokens and LoRA with trainable embeddings/head. |
| Qwen VQ-LoRA free-running | 8x8x8 | Constrained autoregressive VQ suffix rollout | 8 test 4-frame samples; keep first 60% VQ tokens and generate last 40% | Micro / macro VQ accuracy | 0.9628 / 0.9628 | Small favorable evaluation; structure tags are teacher-provided and outputs are constrained to 128 VQ tokens. |
| 4x4x4 SimVQ partial precompute | 4x4x4 | VQ token export from epoch-7 SimVQ4 tokenizer | Target 32 train shards and 4 test shards | Status | Running at last internal snapshot | Needed before matched 4x4x4 token-modeling experiments. |

No raw fMRI data, subject-level manifests, checkpoints, token shards, logs, or
local machine paths are distributed in this repository.
