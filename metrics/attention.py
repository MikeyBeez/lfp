"""Metrics for measuring attention health."""

import torch


@torch.no_grad()
def attention_entropy(attn_weights: torch.Tensor) -> dict:
    """Compute entropy of attention distributions.

    Low entropy = attention is concentrated (potential sink).
    High entropy = attention is spread (healthy).

    Args:
        attn_weights: (B, H, T, T) attention weight matrix

    Returns:
        Dict with 'mean_entropy', 'min_entropy', 'per_head_entropy' (H,).
    """
    # Clamp for numerical stability in log
    attn = attn_weights.clamp(min=1e-10)
    # Entropy per query position: H = -sum_j a_j log(a_j)
    entropy = -(attn * attn.log()).sum(dim=-1)  # (B, H, T)

    per_head = entropy.mean(dim=(0, 2))  # average over batch and positions

    return {
        "mean_entropy": entropy.mean().item(),
        "min_entropy": entropy.min().item(),
        "per_head_entropy": per_head.cpu(),  # (H,) tensor
    }


@torch.no_grad()
def sink_ratio(attn_weights: torch.Tensor) -> dict:
    """Measure fraction of attention mass on the first token (position 0).

    High sink ratio = model is dumping attention on BOS/first token.

    Args:
        attn_weights: (B, H, T, T) attention weight matrix

    Returns:
        Dict with 'mean_sink_ratio', 'max_sink_ratio', 'per_head_sink' (H,).
    """
    # Attention mass on position 0 for each query
    sink_mass = attn_weights[:, :, :, 0]  # (B, H, T)
    # Exclude position 0 querying itself (trivially high)
    sink_mass = sink_mass[:, :, 1:]  # (B, H, T-1)

    per_head = sink_mass.mean(dim=(0, 2))  # (H,)

    return {
        "mean_sink_ratio": sink_mass.mean().item(),
        "max_sink_ratio": sink_mass.max().item(),
        "per_head_sink": per_head.cpu(),
    }


@torch.no_grad()
def head_diversity(attn_weights: torch.Tensor) -> float:
    """Mean pairwise L2 distance between attention patterns of different heads.

    Low diversity = heads are redundant.
    High diversity = heads specialize.

    Args:
        attn_weights: (B, H, T, T) attention weight matrix

    Returns:
        Mean pairwise distance between heads (scalar).
    """
    B, H, T, _ = attn_weights.shape
    # Flatten attention patterns per head: (B, H, T*T)
    flat = attn_weights.reshape(B, H, -1)

    # Pairwise distances between heads
    # (B, H, 1, T*T) - (B, 1, H, T*T) -> (B, H, H)
    dists = torch.cdist(flat, flat, p=2)  # (B, H, H)

    # Mean of upper triangle (exclude self-distances)
    mask = torch.triu(torch.ones(H, H, device=dists.device, dtype=torch.bool), diagonal=1)
    mean_dist = dists[:, mask].mean().item()

    return mean_dist
