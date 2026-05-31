from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerConfig:
    num_codebook_vectors: int = 128
    latent_dim: int = 64
    beta: float = 0.25
    fsq_levels: int = 8
    rvq_num_quantizers: int = 2


class FSQ3D(nn.Module):
    """Finite scalar quantization with straight-through gradients."""

    def __init__(self, cfg: QuantizerConfig):
        super().__init__()
        self.levels = int(cfg.fsq_levels)
        self.n_codes = int(cfg.num_codebook_vectors)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_tanh = torch.tanh(z)
        scaled = (z_tanh + 1.0) * 0.5 * (self.levels - 1)
        bins = torch.round(scaled).clamp(0, self.levels - 1)
        z_q = (bins / (self.levels - 1)) * 2.0 - 1.0
        z_q = z_tanh + (z_q - z_tanh).detach()
        z_q = torch.atanh(z_q.clamp(-0.999, 0.999))
        q_loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        indices = bins.float().mean(dim=1).round().long() % self.n_codes
        return z_q, indices, q_loss


class BFQ3D(nn.Module):
    """Binary finite quantization; a hard low-bit baseline."""

    def __init__(self, cfg: QuantizerConfig):
        super().__init__()
        self.n_codes = int(cfg.num_codebook_vectors)
        self.scale = nn.Parameter(torch.ones(1, int(cfg.latent_dim), 1, 1, 1))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.scale.abs().clamp_min(1e-4)
        normalized = z / scale
        z_sign = torch.where(normalized >= 0, torch.ones_like(normalized), -torch.ones_like(normalized))
        z_q = normalized + (z_sign - normalized).detach()
        z_q = z_q * scale
        q_loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())
        bits = (z_sign > 0).long()
        n_hash = min(7, bits.shape[1])
        weights = (2 ** torch.arange(n_hash, device=z.device)).view(1, n_hash, 1, 1, 1)
        indices = (bits[:, :n_hash] * weights).sum(dim=1).long() % self.n_codes
        return z_q, indices, q_loss


class ResidualVQ3D(nn.Module):
    """Small residual VQ stack for 3D latent tensors."""

    def __init__(self, cfg: QuantizerConfig):
        super().__init__()
        self.n_codes = int(cfg.num_codebook_vectors)
        self.e_dim = int(cfg.latent_dim)
        self.num_quantizers = int(cfg.rvq_num_quantizers)
        self.beta = float(cfg.beta)
        self.codebooks = nn.ModuleList([nn.Embedding(self.n_codes, self.e_dim) for _ in range(self.num_quantizers)])
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / self.n_codes, 1.0 / self.n_codes)

    def _nearest(self, residual: torch.Tensor, emb: nn.Embedding) -> tuple[torch.Tensor, torch.Tensor]:
        flat = residual.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.e_dim)
        dist = flat.pow(2).sum(1, keepdim=True) + emb.weight.pow(2).sum(1) - 2 * flat @ emb.weight.t()
        idx = torch.argmin(dist, dim=1)
        q = emb(idx).view(residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[4], self.e_dim)
        q = q.permute(0, 4, 1, 2, 3).contiguous()
        return q, idx.view(residual.shape[0], residual.shape[2], residual.shape[3], residual.shape[4])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        all_indices = []
        q_loss = z.new_tensor(0.0)
        for emb in self.codebooks:
            q, idx = self._nearest(residual, emb)
            quantized_sum = quantized_sum + q
            residual = residual - q.detach()
            all_indices.append(idx)
            target = residual + q.detach()
            q_loss = q_loss + F.mse_loss(q.detach(), target) + self.beta * F.mse_loss(q, target.detach())
        z_q = z + (quantized_sum - z).detach()
        indices = torch.stack(all_indices, dim=-1)
        return z_q, indices, q_loss / self.num_quantizers


def build_quantizer(name: str, cfg: QuantizerConfig) -> nn.Module:
    name = name.lower()
    if name == "fsq":
        return FSQ3D(cfg)
    if name == "bfq":
        return BFQ3D(cfg)
    if name in {"residual_vq", "rvq", "residual"}:
        return ResidualVQ3D(cfg)
    raise ValueError(f"Unsupported ablation quantizer: {name}")


def codebook_usage(indices: torch.Tensor, num_codes: int) -> dict[str, float]:
    idx = indices.reshape(-1).long() % int(num_codes)
    hist = torch.bincount(idx, minlength=int(num_codes)).float()
    prob = hist / hist.sum().clamp_min(1)
    entropy = -(prob[prob > 0] * prob[prob > 0].log()).sum()
    return {
        "perplexity": float(entropy.exp().item()),
        "dead_code_rate": float((hist == 0).float().mean().item()),
    }
