import torch
import torch.nn as nn

from .simplevq import SimpleVectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    """Residual quantization used by RQ-VAE style models."""

    def __init__(self, num_quantizers, num_embeddings, embedding_dim, beta=0.25, l2_normalize=False):
        super().__init__()
        if num_quantizers < 1:
            raise ValueError("num_quantizers must be >= 1")
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.quantizers = nn.ModuleList(
            [
                SimpleVectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    beta=beta,
                    l2_normalize=l2_normalize,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z):
        quantized = torch.zeros_like(z)
        residual = z
        all_indices = []
        losses = []

        for quantizer in self.quantizers:
            q, indices, loss = quantizer(residual)
            quantized = quantized + q
            residual = residual - q.detach()
            all_indices.append(indices)
            losses.append(loss)

        stacked_indices = torch.stack(all_indices, dim=1)
        total_loss = torch.stack(losses).sum()
        return quantized, stacked_indices, total_loss

    def embed_code(self, indices):
        if indices.size(1) != self.num_quantizers:
            raise ValueError(f"expected {self.num_quantizers} quantizer index groups, got {indices.size(1)}")

        quantized = None
        for i, quantizer in enumerate(self.quantizers):
            q = quantizer.embed_code(indices[:, i])
            quantized = q if quantized is None else quantized + q
        return quantized
