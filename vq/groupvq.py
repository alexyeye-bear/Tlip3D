import torch
import torch.nn as nn

from .simplevq import SimpleVectorQuantizer


class GroupVectorQuantizer(nn.Module):
    """Split channels into groups and quantize each group independently."""

    def __init__(self, num_groups, num_embeddings, embedding_dim, beta=0.25, l2_normalize=False):
        super().__init__()
        if num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if embedding_dim % num_groups != 0:
            raise ValueError("embedding_dim must be divisible by num_groups")

        self.num_groups = num_groups
        self.embedding_dim = embedding_dim
        self.group_dim = embedding_dim // num_groups
        self.quantizers = nn.ModuleList(
            [
                SimpleVectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=self.group_dim,
                    beta=beta,
                    l2_normalize=l2_normalize,
                )
                for _ in range(num_groups)
            ]
        )

    def forward(self, z):
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"expected channel dim {self.embedding_dim}, got {z.size(1)}")

        chunks = torch.chunk(z, self.num_groups, dim=1)
        quantized_chunks = []
        all_indices = []
        losses = []

        for chunk, quantizer in zip(chunks, self.quantizers):
            q, indices, loss = quantizer(chunk)
            quantized_chunks.append(q)
            all_indices.append(indices)
            losses.append(loss)

        quantized = torch.cat(quantized_chunks, dim=1)
        stacked_indices = torch.stack(all_indices, dim=1)
        total_loss = torch.stack(losses).mean()
        return quantized, stacked_indices, total_loss

    def embed_code(self, indices):
        if indices.size(1) != self.num_groups:
            raise ValueError(f"expected {self.num_groups} group index sets, got {indices.size(1)}")

        chunks = [quantizer.embed_code(indices[:, i]) for i, quantizer in enumerate(self.quantizers)]
        return torch.cat(chunks, dim=1)
