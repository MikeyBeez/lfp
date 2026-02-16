"""Evaluation: run all representation quality metrics on a model."""

import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from metrics.collapse import effective_rank, cosine_similarity_stats, intrinsic_dimensionality
from metrics.attention import attention_entropy, sink_ratio, head_diversity
from metrics.separation import class_separation_ratio


@torch.no_grad()
def run_all_metrics(
    model,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
    max_batches: int = 10,
) -> dict:
    """Run all evaluation metrics on validation data.

    Args:
        model: SmallTransformer instance (set to eval mode externally)
        val_loader: validation DataLoader
        device: compute device
        amp_dtype: autocast dtype
        max_batches: cap on batches (metrics are expensive)

    Returns:
        Dict of all metric values.
    """
    model.eval()

    # Accumulators
    total_ce = 0.0
    total_tokens = 0
    all_layer_reps = {i: [] for i in range(len(model.blocks))}
    all_attn_weights = {i: [] for i in range(len(model.blocks))}
    all_targets = []

    for batch_idx, (x, y) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            output = model(x, collect_intermediates=True)

        # CE loss for perplexity
        B, T, V = output.logits.shape
        ce = F.cross_entropy(
            output.logits.reshape(B * T, V),
            y.reshape(B * T),
            reduction="sum",
        )
        total_ce += ce.item()
        total_tokens += B * T

        # Collect representations and attention weights
        for i, rep in enumerate(output.layer_representations):
            all_layer_reps[i].append(rep.float().cpu())
        for i, aw in enumerate(output.attention_weights):
            if aw is not None:
                all_attn_weights[i].append(aw.float().cpu())
        all_targets.append(y.cpu())

    # Aggregate
    targets_flat = torch.cat(all_targets).reshape(-1)
    val_ppl = math.exp(min(total_ce / total_tokens, 20))

    metrics = {"val_ppl": val_ppl, "val_ce": total_ce / total_tokens}

    # Per-layer representation metrics
    eff_ranks = []
    cos_sim_means = []
    intrinsic_dims = []
    sep_ratios = []

    n_layers = len(model.blocks)
    for i in range(n_layers):
        if not all_layer_reps[i]:
            continue
        Z = torch.cat(all_layer_reps[i]).reshape(-1, model.cfg.d_model)

        # Subsample for efficiency
        if Z.shape[0] > 8192:
            idx = torch.randperm(Z.shape[0])[:8192]
            Z = Z[idx]
            labels_sub = targets_flat[idx]
        else:
            labels_sub = targets_flat[: Z.shape[0]]

        er = effective_rank(Z)
        eff_ranks.append(er)

        cs = cosine_similarity_stats(Z)
        cos_sim_means.append(cs["mean"])

        idim = intrinsic_dimensionality(Z)
        intrinsic_dims.append(idim)

        sep = class_separation_ratio(Z, labels_sub)
        sep_ratios.append(sep["ratio"])

    metrics["effective_rank_per_layer"] = eff_ranks
    metrics["effective_rank_mean"] = sum(eff_ranks) / len(eff_ranks) if eff_ranks else 0
    metrics["cosine_sim_per_layer"] = cos_sim_means
    metrics["cosine_sim_mean"] = sum(cos_sim_means) / len(cos_sim_means) if cos_sim_means else 0
    metrics["intrinsic_dim_per_layer"] = intrinsic_dims
    metrics["intrinsic_dim_mean"] = sum(intrinsic_dims) / len(intrinsic_dims) if intrinsic_dims else 0
    metrics["separation_ratio_per_layer"] = sep_ratios
    metrics["separation_ratio"] = sum(sep_ratios) / len(sep_ratios) if sep_ratios else 0

    # Attention metrics (average over layers)
    attn_entropies = []
    sink_ratios = []
    diversities = []

    for i in range(n_layers):
        if not all_attn_weights[i]:
            continue
        aw = torch.cat(all_attn_weights[i])

        ae = attention_entropy(aw)
        attn_entropies.append(ae["mean_entropy"])

        sr = sink_ratio(aw)
        sink_ratios.append(sr["mean_sink_ratio"])

        hd = head_diversity(aw)
        diversities.append(hd)

    metrics["attn_entropy_per_layer"] = attn_entropies
    metrics["attn_entropy_mean"] = sum(attn_entropies) / len(attn_entropies) if attn_entropies else 0
    metrics["sink_ratio_per_layer"] = sink_ratios
    metrics["sink_ratio_mean"] = sum(sink_ratios) / len(sink_ratios) if sink_ratios else 0
    metrics["head_diversity_per_layer"] = diversities
    metrics["head_diversity_mean"] = sum(diversities) / len(diversities) if diversities else 0

    return metrics
