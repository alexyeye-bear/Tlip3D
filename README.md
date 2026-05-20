# Tlip3D

Combine 3D vector quantization and contrastive learning.

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
  dataloaders/
    custom_loader.py
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
