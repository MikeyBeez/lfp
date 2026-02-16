"""Metrics for measuring representation collapse."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(Z: torch.Tensor) -> float:
    """Effective rank via Shannon entropy of normalized singular values.

    ER = exp(-sum(p_i * log(p_i))) where p_i = sigma_i / sum(sigma_j)

    A representation matrix using all dimensions equally has ER = d.
    A collapsed representation has ER close to 1.

    Args:
        Z: (N, d) representation matrix

    Returns:
        Effective rank (scalar between 1 and min(N, d)).
    """
    # Subsample if too large (SVD on huge matrices is slow)
    if Z.shape[0] > 4096:
        idx = torch.randperm(Z.shape[0])[:4096]
        Z = Z[idx]

    Z = Z.float()
    s = torch.linalg.svdvals(Z)
    s = s[s > 1e-10]  # remove numerical zeros
    if len(s) == 0:
        return 1.0

    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return entropy.exp().item()


@torch.no_grad()
def cosine_similarity_stats(Z: torch.Tensor, n_pairs: int = 10000) -> dict:
    """Statistics of pairwise cosine similarities.

    High mean + low std = anisotropy / collapse.
    Uniform distribution around 0 = isotropic (healthy).

    Args:
        Z: (N, d) representation matrix
        n_pairs: number of random pairs to sample

    Returns:
        Dict with 'mean', 'std', 'max', 'min' of cosine similarities.
    """
    Z = Z.float()
    Z = F.normalize(Z, dim=-1)
    N = Z.shape[0]

    # Random pairs
    idx1 = torch.randint(0, N, (n_pairs,))
    idx2 = torch.randint(0, N, (n_pairs,))
    # Avoid self-pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    sims = (Z[idx1] * Z[idx2]).sum(dim=-1)

    return {
        "mean": sims.mean().item(),
        "std": sims.std().item(),
        "max": sims.max().item(),
        "min": sims.min().item(),
    }


@torch.no_grad()
def intrinsic_dimensionality(Z: torch.Tensor, threshold: float = 0.9) -> int:
    """Number of PCA components needed to explain `threshold` fraction of variance.

    Args:
        Z: (N, d) representation matrix
        threshold: variance fraction to capture (default 90%)

    Returns:
        Number of components (1 to d).
    """
    if Z.shape[0] > 4096:
        idx = torch.randperm(Z.shape[0])[:4096]
        Z = Z[idx]

    Z = Z.float()
    Z = Z - Z.mean(dim=0, keepdim=True)

    s = torch.linalg.svdvals(Z)
    var_explained = (s ** 2).cumsum(dim=0) / (s ** 2).sum()

    n_components = (var_explained < threshold).sum().item() + 1
    return min(n_components, Z.shape[1])
