from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class VolumeFolderDataset(Dataset):
    """Load 3D volumes from a public, path-configured folder.

    The original experiments used internal fMRI loaders. This public dataset
    keeps the training code reproducible without embedding site-specific paths:
    put `.npy` or `.pt` tensors under `<root>/<split>/`.
    """

    def __init__(self, root: str | Path, split: str = "train", max_samples: int | None = None):
        self.root = Path(root)
        self.split = split
        split_root = self.root / split
        self.files = sorted(split_root.glob("*.npy")) + sorted(split_root.glob("*.pt"))
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise RuntimeError(f"No .npy or .pt volumes found under {split_root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        path = self.files[idx]
        if path.suffix == ".npy":
            arr = torch.from_numpy(np.load(path)).float()
        else:
            arr = torch.load(path, map_location="cpu").float()
        if arr.ndim == 3:
            arr = arr.unsqueeze(0)
        if arr.ndim != 4:
            raise RuntimeError(f"Expected C,D,H,W or D,H,W tensor, got shape {tuple(arr.shape)} from {path}")
        return arr, {"relative_path": str(path.relative_to(self.root))}
