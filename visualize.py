"""Visualization of metric trajectories across regimes."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: str) -> list:
    """Load metrics from a run's metrics.jsonl file."""
    path = Path(run_dir) / "metrics.jsonl"
    metrics = []
    with open(path) as f:
        for line in f:
            metrics.append(json.loads(line))
    return metrics


def plot_comparison(
    runs_dir: str = "runs",
    regimes: list = None,
    output_path: str = "comparison.png",
):
    """Plot metric trajectories for all regimes side by side.

    Creates a 3x3 grid:
      Row 1: Task metrics (perplexity, CE loss)
      Row 2: Collapse metrics (effective rank, cosine similarity)
      Row 3: Attention/separation metrics (entropy, sink ratio, separation)
    """
    if regimes is None:
        regimes = ["ce_only", "mcr2_only", "organism"]

    colors = {
        "ce_only": "#2196F3", "ce_baseline": "#2196F3",
        "mcr2_only": "#FF5722",
        "organism": "#4CAF50",
        "expansion_only": "#FF9800",
        "selective_expansion": "#9C27B0",
        "curriculum_2k": "#00BCD4",
        "curriculum_3k": "#E91E63",
        "curriculum_4k": "#795548",
        "alternating_2k": "#607D8B",
        "compression_only": "#3F51B5",
    }
    labels = {
        "ce_only": "CE Only", "ce_baseline": "CE Only",
        "mcr2_only": "MCR\u00b2 Only",
        "organism": "Full MCR\u00b2",
        "expansion_only": "Expansion Only",
        "selective_expansion": "Selective Expansion",
        "curriculum_2k": "Curriculum 2K",
        "curriculum_3k": "Curriculum 3K",
        "curriculum_4k": "Curriculum 4K",
        "alternating_2k": "Alternating 2K",
        "compression_only": "Compression Only",
    }

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("Loss Functions for Principled Representations: Regime Comparison", fontsize=14)

    metric_specs = [
        # Row 0: Task
        ("val_ppl", "Validation Perplexity", True),
        ("val_ce", "Validation CE Loss", False),
        ("separation_ratio", "Separation Ratio", False),
        # Row 1: Collapse
        ("effective_rank_mean", "Effective Rank (mean)", False),
        ("cosine_sim_mean", "Cosine Similarity (mean)", False),
        ("intrinsic_dim_mean", "Intrinsic Dimensionality", False),
        # Row 2: Attention
        ("attn_entropy_mean", "Attention Entropy (mean)", False),
        ("sink_ratio_mean", "Sink Ratio (mean)", False),
        ("head_diversity_mean", "Head Diversity", False),
    ]

    for regime in regimes:
        run_path = Path(runs_dir) / regime
        if not (run_path / "metrics.jsonl").exists():
            print(f"Skipping {regime}: no metrics found")
            continue

        data = load_metrics(str(run_path))
        steps = [d["step"] for d in data]

        for idx, (key, title, use_log) in enumerate(metric_specs):
            row, col = divmod(idx, 3)
            ax = axes[row][col]

            values = [d.get(key, None) for d in data]
            valid = [(s, v) for s, v in zip(steps, values) if v is not None]
            if not valid:
                continue

            s, v = zip(*valid)
            ax.plot(s, v, color=colors.get(regime, "gray"),
                    label=labels.get(regime, regime), linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Step")
            if use_log:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def plot_per_layer(
    run_dir: str,
    output_path: str = None,
):
    """Plot per-layer metrics for a single regime to see layer-by-layer dynamics."""
    data = load_metrics(run_dir)
    regime_name = Path(run_dir).name
    if output_path is None:
        output_path = f"{regime_name}_per_layer.png"

    # Get the last evaluation point
    last = data[-1]
    n_layers = len(last.get("effective_rank_per_layer", []))
    if n_layers == 0:
        print("No per-layer data found")
        return

    layers = list(range(1, n_layers + 1))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Per-Layer Analysis: {regime_name} (step {last['step']})", fontsize=14)

    specs = [
        ("effective_rank_per_layer", "Effective Rank"),
        ("cosine_sim_per_layer", "Cosine Similarity"),
        ("intrinsic_dim_per_layer", "Intrinsic Dim"),
        ("attn_entropy_per_layer", "Attention Entropy"),
        ("sink_ratio_per_layer", "Sink Ratio"),
        ("separation_ratio_per_layer", "Separation Ratio"),
    ]

    for idx, (key, title) in enumerate(specs):
        row, col = divmod(idx, 3)
        ax = axes[row][col]
        values = last.get(key, [])
        if values:
            ax.bar(layers[:len(values)], values, color="#607D8B", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved per-layer plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--regimes", nargs="+", default=["ce_only", "mcr2_only", "organism"])
    parser.add_argument("--output", default="comparison.png")
    args = parser.parse_args()

    plot_comparison(args.runs_dir, args.regimes, args.output)

    for regime in args.regimes:
        run_dir = Path(args.runs_dir) / regime
        if (run_dir / "metrics.jsonl").exists():
            plot_per_layer(str(run_dir))
