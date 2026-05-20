import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVectorQuantizer(nn.Module):
    """Basic VQ-VAE quantizer for channel-first tensors: [B, C, ...]."""

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, l2_normalize=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.l2_normalize = l2_normalize
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def _flatten(self, z):
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"expected channel dim {self.embedding_dim}, got {z.size(1)}")
        z_perm = z.movedim(1, -1).contiguous()
        z_flat = z_perm.view(-1, self.embedding_dim)
        return z_perm, z_flat

    def _nearest_code(self, z_flat):
        emb = self.embedding.weight
        if self.l2_normalize:
            z_flat = F.normalize(z_flat, dim=1)
            emb = F.normalize(emb, dim=1)
            return (z_flat @ emb.t()).argmax(dim=1)

        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + emb.pow(2).sum(dim=1)
            - 2 * z_flat @ emb.t()
        )
        return distances.argmin(dim=1)

    def forward(self, z):
        z_perm, z_flat = self._flatten(z)
        indices = self._nearest_code(z_flat)
        z_q = self.embedding(indices).view_as(z_perm)

        codebook_loss = F.mse_loss(z_q, z_perm.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z_perm)
        loss = codebook_loss + self.beta * commitment_loss

        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.movedim(-1, 1).contiguous()
        indices = indices.view(z_perm.shape[:-1])
        return z_q, indices, loss

    def embed_code(self, indices):
        z_q = self.embedding(indices)
        return z_q.movedim(-1, 1).contiguous()
