from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .multi_frame_token_dataset import VQMultiFrameIndexDataset


@dataclass
class MultiFrameMaskConfig:
    token_root: str
    output_dir: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    segment_len: int = 4
    stride: int = 1
    num_codebook_vectors: int = 128
    dim: int = 192
    hidden_dim: int = 768
    n_layers: int = 4
    n_heads: int = 8
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 2
    learning_rate: float = 1e-4
    mask_ratio: float = 0.45
    max_train_steps: int | None = None
    max_val_steps: int | None = None
    max_segments: int | None = None
    max_shards: int | None = None


class MultiFrameMaskTransformer(nn.Module):
    """Single-pass bidirectional masked-token model for flattened VQ segments."""

    def __init__(self, cfg: MultiFrameMaskConfig, num_tokens: int):
        super().__init__()
        self.mask_token_id = cfg.num_codebook_vectors
        self.sos_token_id = cfg.num_codebook_vectors + 1
        vocab = cfg.num_codebook_vectors + 2
        self.tok_emb = nn.Embedding(vocab, cfg.dim)
        self.pos_emb = nn.Parameter(torch.zeros(num_tokens + 1, cfg.dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, vocab)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz = tokens.shape[0]
        sos = torch.full((bsz, 1), self.sos_token_id, dtype=torch.long, device=tokens.device)
        x = torch.cat([sos, tokens], dim=1)
        h = self.tok_emb(x) + self.pos_emb[: x.shape[1]][None]
        h = self.encoder(h)
        return self.head(self.norm(h))[:, 1:]


def random_mask(tokens: torch.Tensor, mask_token_id: int, mask_ratio: float):
    mask = torch.rand(tokens.shape, device=tokens.device) < mask_ratio
    masked = tokens.clone()
    masked[mask] = mask_token_id
    return masked, mask


def evaluate(model: MultiFrameMaskTransformer, loader: DataLoader, cfg: MultiFrameMaskConfig):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for step, (arr, _info) in enumerate(loader):
            if cfg.max_val_steps is not None and step >= cfg.max_val_steps:
                break
            tokens = arr.reshape(arr.shape[0], -1).to(cfg.device)
            masked, mask = random_mask(tokens, model.mask_token_id, cfg.mask_ratio)
            logits = model(masked)
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], tokens[mask])
            pred = logits.argmax(dim=-1)
            acc = (pred[mask] == tokens[mask]).float().mean()
            losses.append(float(loss.item()))
            accs.append(float(acc.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan"), float(np.mean(accs)) if accs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--token-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--segment-len", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--hidden-dim", type=int, default=768)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-train-steps", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--max-segments", type=int, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    args = p.parse_args()
    cfg = MultiFrameMaskConfig(**vars(args))

    out = Path(cfg.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / "checks").mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    train_ds = VQMultiFrameIndexDataset(cfg.token_root, "train", segment_len=cfg.segment_len, stride=cfg.stride, flatten=True, max_segments=cfg.max_segments, max_shards=cfg.max_shards)
    test_ds = VQMultiFrameIndexDataset(cfg.token_root, "test", segment_len=cfg.segment_len, stride=cfg.stride, flatten=True, max_segments=cfg.max_segments, max_shards=cfg.max_shards)
    sample, info = train_ds[0]
    model = MultiFrameMaskTransformer(cfg, int(sample.numel())).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.96), weight_decay=4.5e-2)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    print("run_dir", out, flush=True)
    print("num_train_segments", len(train_ds), "num_test_segments", len(test_ds), "num_tokens", int(sample.numel()), "sample_info", {k: v for k, v in info.items() if k != "frames"}, flush=True)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        for step, (arr, _info) in enumerate(tqdm(train_loader, desc=f"multiframe-mask-epoch{epoch}", dynamic_ncols=True)):
            if cfg.max_train_steps is not None and step >= cfg.max_train_steps:
                break
            tokens = arr.reshape(arr.shape[0], -1).to(cfg.device)
            masked, mask = random_mask(tokens, model.mask_token_id, cfg.mask_ratio)
            logits = model(masked)
            loss = F.cross_entropy(logits[mask], tokens[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_loss, val_acc = evaluate(model, test_loader, cfg)
        row = {"epoch": epoch, "val_loss": val_loss, "val_masked_acc": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), out / "checks" / f"multiframe_mask_epoch_{epoch}.pt")
    (out / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
