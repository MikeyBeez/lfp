"""Metrics for measuring class separation quality in representations."""

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def class_separation_ratio(
    Z: torch.Tensor,
    labels: torch.Tensor,
    max_classes: int = 256,
    min_class_size: int = 2,
) -> dict:
    """Between-class vs within-class distance ratio.

    Higher ratio = better separation. A ratio of 1 means classes are not
    separated at all. Good separation typically shows ratio > 3.

    Args:
        Z: (N, d) representation matrix
        labels: (N,) class labels (next-token ids)
        max_classes: cap on number of classes to evaluate
        min_class_size: minimum samples per class

    Returns:
        Dict with 'ratio', 'between_dist', 'within_dist'.
    """
    Z = Z.float()
    unique, counts = torch.unique(labels, return_counts=True)
    valid = unique[counts >= min_class_size]

    if len(valid) < 2:
        return {"ratio": 0.0, "between_dist": 0.0, "within_dist": 0.0}

    # Take most frequent classes up to cap
    if len(valid) > max_classes:
        valid_counts = counts[counts >= min_class_size]
        _, top_idx = valid_counts.topk(max_classes)
        valid = valid[top_idx]

    # Compute class centroids and within-class distances
    centroids = []
    within_dists = []

    for c in valid:
        mask = labels == c
        Z_c = Z[mask]
        centroid = Z_c.mean(dim=0)
        centroids.append(centroid)
        within_dists.append((Z_c - centroid).norm(dim=-1).mean())

    centroids = torch.stack(centroids)  # (C, d)
    mean_within = torch.stack(within_dists).mean()

    # Between-class: mean pairwise distance between centroids
    dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    C = len(valid)
    mask = torch.triu(torch.ones(C, C, device=Z.device, dtype=torch.bool), diagonal=1)
    mean_between = dists[mask].mean()

    ratio = (mean_between / (mean_within + 1e-8)).item()

    return {
        "ratio": ratio,
        "between_dist": mean_between.item(),
        "within_dist": mean_within.item(),
    }


class LinearProbe(nn.Module):
    """Simple linear probe for evaluating representation quality."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        return self.linear(x)


@torch.no_grad()
def linear_probe_accuracy(
    Z_train: torch.Tensor,
    labels_train: torch.Tensor,
    Z_test: torch.Tensor,
    labels_test: torch.Tensor,
    n_classes: int = 100,
    n_steps: int = 200,
    lr: float = 0.01,
) -> float:
    """Train a linear probe on frozen representations and return test accuracy.

    Uses the top-n_classes most frequent labels for tractability.

    Args:
        Z_train: (N_train, d) training representations
        labels_train: (N_train,) labels
        Z_test: (N_test, d) test representations
        labels_test: (N_test,) labels
        n_classes: number of most-frequent classes to evaluate
        n_steps: SGD steps for probe training
        lr: learning rate

    Returns:
        Top-1 accuracy on the test set (0 to 1).
    """
    device = Z_train.device

    # Find top-n most frequent classes
    all_labels = torch.cat([labels_train, labels_test])
    unique, counts = torch.unique(all_labels, return_counts=True)
    _, top_idx = counts.topk(min(n_classes, len(unique)))
    top_classes = unique[top_idx]

    # Remap labels to 0..n_classes-1
    label_map = {c.item(): i for i, c in enumerate(top_classes)}

    def filter_and_remap(Z, labels):
        mask = torch.zeros(len(labels), dtype=torch.bool, device=device)
        for c in top_classes:
            mask |= labels == c
        Z_f = Z[mask]
        labels_f = labels[mask]
        remapped = torch.tensor(
            [label_map[l.item()] for l in labels_f], device=device
        )
        return Z_f, remapped

    Z_tr, y_tr = filter_and_remap(Z_train, labels_train)
    Z_te, y_te = filter_and_remap(Z_test, labels_test)

    if len(Z_tr) == 0 or len(Z_te) == 0:
        return 0.0

    actual_classes = len(top_classes)

    # Train probe
    probe = LinearProbe(Z_tr.shape[1], actual_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)

    probe.train()
    for _ in range(n_steps):
        # Mini-batch
        idx = torch.randint(0, len(Z_tr), (min(512, len(Z_tr)),))
        logits = probe(Z_tr[idx])
        loss = F.cross_entropy(logits, y_tr[idx])
        optimizer.zero_grad()
        # Need grads for probe parameters only
        with torch.enable_grad():
            logits = probe(Z_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            loss.backward()
        optimizer.step()

    # Evaluate
    probe.eval()
    logits = probe(Z_te)
    preds = logits.argmax(dim=-1)
    acc = (preds == y_te).float().mean().item()

    return acc
