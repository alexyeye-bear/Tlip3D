from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import VQAblationConfig
from .data import VolumeFolderDataset
from .quantizers import QuantizerConfig, build_quantizer, codebook_usage


class TinyConvVQAutoencoder(nn.Module):
    """Minimal 3D autoencoder used to demonstrate the quantizer ablation API."""

    def __init__(self, cfg: VQAblationConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, cfg.latent_dim, 3, padding=1),
        )
        self.quantizer = build_quantizer(
            cfg.quantizer,
            QuantizerConfig(
                num_codebook_vectors=cfg.num_codebook_vectors,
                latent_dim=cfg.latent_dim,
                beta=cfg.beta,
                fsq_levels=cfg.fsq_levels,
                rvq_num_quantizers=cfg.rvq_num_quantizers,
            ),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(cfg.latent_dim, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        z_q, indices, q_loss = self.quantizer(z)
        return self.decoder(z_q), indices, q_loss


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, cfg: VQAblationConfig, device: str):
    training = optimizer is not None
    model.train(training)
    rows = []
    max_steps = cfg.train_batches_per_epoch if training else cfg.val_batches_per_epoch
    with torch.set_grad_enabled(training):
        for step, (x, _meta) in enumerate(tqdm(loader, dynamic_ncols=True, leave=False)):
            if max_steps is not None and step >= max_steps:
                break
            x = x.to(device)
            recon, indices, q_loss = model(x)
            rec_loss = F.l1_loss(recon, x)
            mse = F.mse_loss(recon, x)
            loss = rec_loss + cfg.beta * q_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            usage = codebook_usage(indices, cfg.num_codebook_vectors)
            rows.append(
                {
                    "loss": float(loss.item()),
                    "rec_loss": float(rec_loss.item()),
                    "mse": float(mse.item()),
                    "q_loss": float(q_loss.item()),
                    **usage,
                }
            )
    if not rows:
        return {}
    return {k: sum(row[k] for row in rows) / len(rows) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-free 3D VQ quantizer ablation example")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--volume-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quantizer", choices=["fsq", "bfq", "residual_vq"], default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    data = json.loads(args.config_json.read_text()) if args.config_json else {}
    cfg = VQAblationConfig.from_mapping(data)
    for key in ["volume_root", "output_dir", "quantizer", "epochs"]:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)

    out_dir = Path(cfg.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_ds = VolumeFolderDataset(cfg.volume_root, "train")
    val_ds = VolumeFolderDataset(cfg.volume_root, "test")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = TinyConvVQAutoencoder(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    history = []
    for epoch in range(cfg.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, cfg, args.device)
        val_metrics = run_epoch(model, val_loader, None, cfg, args.device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
