from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class VQAblationConfig:
    """Path-free configuration for short 3D VQ quantizer ablations."""

    quantizer: str = "residual_vq"
    token_grid: str = "8x8x8"
    num_codebook_vectors: int = 128
    latent_dim: int = 64
    beta: float = 0.25
    fsq_levels: int = 8
    rvq_num_quantizers: int = 2

    volume_root: str = "${DATA_ROOT}/volumes"
    mask_path: str | None = "${DATA_ROOT}/masks/brain_mask.npy"
    output_dir: str = "${OUTPUT_ROOT}/vq_ablation"

    epochs: int = 8
    batch_size: int = 32
    learning_rate: float = 2.25e-5
    train_batches_per_epoch: int | None = 32
    val_batches_per_epoch: int | None = 8
    num_workers: int = 4
    seed: int = 1234

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "VQAblationConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def resolve_path(path: str | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser()
