"""MCR2 (Maximal Coding Rate Reduction) loss for language modeling.

Implements the rate reduction objective from:
  Yu et al. (2020), "Learning Diverse and Discriminative Representations
  via the Principle of Maximal Coding Rate Reduction"

Adapted for language modeling where "classes" are next-token identities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MCR2Config


class MCR2Loss(nn.Module):
    """Compute MCR2 = -(R(Z) - sum_c pi_c * R(Z_c)).

    We want to maximize rate reduction (expand whole space, compress classes),
    so we return the negative (to minimize via gradient descent).

    Key design choices for language modeling:
    - Classes defined by next-token identity
    - Random projection to lower dimension for efficient log-det
    - Skip singleton classes (can't compute meaningful rate for 1 sample)
    - Cap number of classes per batch to bound compute
    - Subsample tokens for speed (MCR2 on 512 tokens is informative enough)
    """

    def __init__(self, cfg: MCR2Config = None):
        super().__init__()
        if cfg is None:
            cfg = MCR2Config()
        self.eps_sq = cfg.eps ** 2
        self.min_class_size = cfg.min_class_size
        self.max_classes = cfg.max_classes
        self.use_projection = cfg.use_projection
        self.proj_dim = cfg.proj_dim
        self.expansion_only = cfg.expansion_only
        self.selective_expansion = cfg.selective_expansion
        self.compression_only = cfg.compression_only
        self.subsample_n = 512  # max tokens to use per MCR2 computation
        self._proj_matrix = None

    def _get_projection(self, d: int, device: torch.device, dtype: torch.dtype):
        """Get or create a random projection matrix (fixed per instance)."""
        if self._proj_matrix is None or self._proj_matrix.shape[1] != d:
            P = torch.randn(self.proj_dim, d, device=device, dtype=dtype)
            P = P / (P.norm(dim=1, keepdim=True) + 1e-8)
            P = P * (d / self.proj_dim) ** 0.5
            self._proj_matrix = P
        return self._proj_matrix.to(device=device, dtype=dtype)

    def _compute_coding_rate(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute coding rate R(Z) = (d/2) * log det(I + d/(N*eps^2) * Z^T Z).

        Args:
            Z: (N, d) feature matrix
        """
        N, d = Z.shape
        if N < 2:
            return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

        scalar = d / (N * self.eps_sq)
        cov = Z.T @ Z  # (d, d)
        matrix = torch.eye(d, device=Z.device, dtype=Z.dtype) + scalar * cov

        sign, logabsdet = torch.linalg.slogdet(matrix)
        rate = 0.5 * torch.clamp(sign * logabsdet, min=0.0)
        return rate

    def _compute_class_rates_batched(
        self, Z: torch.Tensor, labels: torch.Tensor, valid_labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-class coding rates with minimal Python loops.

        Groups classes by size and batches the covariance computation.
        Falls back to sequential for very large classes.
        """
        N, d = Z.shape
        R_classes = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)

        # Build a label-to-index mapping for valid labels only
        for c in valid_labels:
            mask = labels == c
            Z_c = Z[mask]
            n_c = Z_c.shape[0]
            R_classes = R_classes + (n_c / N) * self._compute_coding_rate(Z_c)

        return R_classes

    def forward(
        self,
        Z: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MCR2 loss.

        Args:
            Z: (N, d) representations (flattened from B*T)
            labels: (N,) next-token ids as class labels

        Returns:
            Scalar loss (negative rate reduction -- minimize this).
        """
        N, d = Z.shape

        # Subsample for speed -- MCR2 on 512 tokens captures the structure
        if N > self.subsample_n:
            idx = torch.randperm(N, device=Z.device)[:self.subsample_n]
            Z = Z[idx]
            labels = labels[idx]
            N = self.subsample_n

        # Project to lower dimension
        if self.use_projection and d > self.proj_dim:
            P = self._get_projection(d, Z.device, Z.dtype)
            Z = Z @ P.T
            d = self.proj_dim

        # Normalize representations
        Z = F.normalize(Z, dim=-1) * (d ** 0.5)

        # Compression-only: skip expansion, just minimize per-class coding rates
        if self.compression_only:
            unique_labels, counts = torch.unique(labels, return_counts=True)
            valid_mask = counts >= self.min_class_size
            if valid_mask.sum() > self.max_classes:
                _, top_idx = counts[valid_mask].topk(self.max_classes)
                valid_labels = unique_labels[valid_mask][top_idx]
            else:
                valid_labels = unique_labels[valid_mask]
            if len(valid_labels) == 0:
                return torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
            R_classes = self._compute_class_rates_batched(Z, labels, valid_labels)
            return R_classes / d

        # Total coding rate (expansion term)
        R_total = self._compute_coding_rate(Z)

        if self.expansion_only:
            return -(R_total / d)

        if self.selective_expansion:
            # Expand class centroids: each unique next-token gets one point.
            # Pushes different-token representations apart without compressing
            # same-token representations together. Frequency-neutral.
            unique_labels, inverse = torch.unique(labels, return_inverse=True)
            K = unique_labels.shape[0]
            # Scatter-mean: sum then divide by count (differentiable)
            centroid_sum = torch.zeros(K, d, device=Z.device, dtype=Z.dtype)
            centroid_sum.scatter_add_(0, inverse.unsqueeze(1).expand(-1, d), Z)
            counts = torch.zeros(K, device=Z.device, dtype=Z.dtype)
            counts.scatter_add_(0, inverse, torch.ones(N, device=Z.device, dtype=Z.dtype))
            C = centroid_sum / counts.unsqueeze(1)  # (K, d)
            R_centroids = self._compute_coding_rate(C)
            return -(R_centroids / d)

        # Per-class coding rates (compression term)
        unique_labels, counts = torch.unique(labels, return_counts=True)
        valid_mask = counts >= self.min_class_size

        if valid_mask.sum() > self.max_classes:
            _, top_idx = counts[valid_mask].topk(self.max_classes)
            valid_labels = unique_labels[valid_mask][top_idx]
        else:
            valid_labels = unique_labels[valid_mask]

        if len(valid_labels) == 0:
            return -(R_total / d)

        R_classes = self._compute_class_rates_batched(Z, labels, valid_labels)

        delta_R = R_total - R_classes
        # Normalize by dimension for scale-independence
        delta_R = delta_R / d

        return -delta_R


class LayerWiseMCR2Loss(nn.Module):
    """Apply MCR2 loss at each transformer layer's output.

    This is the "local objective" in the Artificial Organism framework.
    Each layer gets its own MCR2 signal pushing it toward well-separated
    representations.
    """

    def __init__(self, cfg: MCR2Config = None, layer_indices: list = None):
        super().__init__()
        self.mcr2 = MCR2Loss(cfg)
        self.layer_indices = layer_indices  # None = all layers

    def forward(
        self,
        layer_representations: list,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute averaged MCR2 loss across selected layers.

        Args:
            layer_representations: list of (B, T, d) tensors, one per layer
            targets: (B, T) next-token ids

        Returns:
            Scalar loss (mean of per-layer MCR2 losses).
        """
        labels_flat = targets.reshape(-1)
        total_loss = torch.tensor(0.0, device=targets.device, dtype=torch.float32)

        indices = self.layer_indices if self.layer_indices is not None else range(len(layer_representations))
        count = 0

        for i in indices:
            Z_layer = layer_representations[i]
            Z_flat = Z_layer.reshape(-1, Z_layer.shape[-1])
            total_loss = total_loss + self.mcr2(Z_flat, labels_flat)
            count += 1

        return total_loss / max(count, 1)
