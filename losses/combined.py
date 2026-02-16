"""Combined Artificial Organism loss: CE global + MCR2 local."""

import math
import torch
import torch.nn as nn

from config import MCR2Config
from .cross_entropy import CrossEntropyLoss
from .mcr2 import LayerWiseMCR2Loss


class CombinedOrganismLoss(nn.Module):
    """Artificial Organism loss combining global and local objectives.

    L = alpha * L_CE(logits, targets) + beta * mean_l(L_MCR2(Z_l, targets))

    The global CE loss drives task performance (next-token prediction).
    The local MCR2 loss shapes intermediate representations at each layer.

    Supports three modes:
    - Fixed beta: traditional fixed weighting
    - Learned beta: beta as nn.Parameter trained on total loss (with floor)
    - Dual beta: beta updated via Lagrangian dual ascent driven by MCR2 quality
      (adversarial — beta has its own objective separate from the model)
    """

    def __init__(
        self,
        alpha_ce: float = 1.0,
        beta_mcr2: float = 0.1,
        mcr2_cfg: MCR2Config = None,
        learn_beta: bool = False,
        per_layer_beta: bool = False,
        n_layers: int = 6,
        layer_indices: list = None,
        beta_min: float = 0.001,
    ):
        super().__init__()
        self.alpha_ce = alpha_ce
        self.ce_loss = CrossEntropyLoss()
        self.mcr2_core = LayerWiseMCR2Loss(mcr2_cfg, layer_indices=layer_indices).mcr2
        self.layer_indices = layer_indices

        self.learn_beta = learn_beta
        self.per_layer_beta = per_layer_beta
        # Floor to prevent the optimizer from killing MCR2 entirely
        self.log_beta_min = math.log(beta_min) if learn_beta else None

        if learn_beta and per_layer_beta:
            n = len(layer_indices) if layer_indices else n_layers
            init_log_beta = math.log(beta_mcr2) * torch.ones(n)
            self.log_betas = nn.Parameter(init_log_beta)
        elif learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta_mcr2)))
        else:
            self.beta_mcr2 = beta_mcr2

    def _get_betas(self):
        """Return beta value(s) with floor applied."""
        if self.learn_beta and self.per_layer_beta:
            clamped = torch.clamp(self.log_betas, min=self.log_beta_min)
            return clamped.exp()
        elif self.learn_beta:
            clamped = torch.clamp(self.log_beta, min=self.log_beta_min)
            return clamped.exp()
        else:
            return torch.tensor(self.beta_mcr2)

    def _compute_mcr2(self, layer_representations, labels_flat):
        """Compute mean MCR2 loss across selected layers."""
        indices = self.layer_indices if self.layer_indices is not None else range(len(layer_representations))
        mcr2_raw = torch.tensor(0.0, device=labels_flat.device, dtype=torch.float32)
        count = 0
        for i in indices:
            Z_flat = layer_representations[i].reshape(-1, layer_representations[i].shape[-1])
            mcr2_raw = mcr2_raw + self.mcr2_core(Z_flat, labels_flat)
            count += 1
        return mcr2_raw / max(count, 1)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        layer_representations: list,
        mcr2_scale: float = 1.0,
        beta_override: float = None,
    ) -> dict:
        """Compute combined loss.

        Args:
            logits: (B, T, V) model output logits
            targets: (B, T) target token ids
            layer_representations: list of (B, T, d) per-layer outputs
            mcr2_scale: external multiplier (for curriculum warmup, 0->1)
            beta_override: if provided, use this beta instead of internal
                           (for dual/adversarial mode where beta is external)

        Returns:
            Dict with 'loss', 'ce_loss', 'mcr2_loss', and 'beta'.
        """
        ce = self.ce_loss(logits, targets)
        labels_flat = targets.reshape(-1)

        if beta_override is not None:
            # Dual mode: beta is an external scalar, not a parameter
            mcr2_raw = self._compute_mcr2(layer_representations, labels_flat)
            total = self.alpha_ce * ce + mcr2_scale * beta_override * mcr2_raw
            beta_val = beta_override

        elif self.learn_beta and self.per_layer_beta:
            # Per-layer learned weights (floored)
            betas = self._get_betas()
            indices = self.layer_indices if self.layer_indices is not None else range(len(layer_representations))
            mcr2_total = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
            mcr2_raw = torch.tensor(0.0, device=targets.device, dtype=torch.float32)
            for j, i in enumerate(indices):
                Z_flat = layer_representations[i].reshape(-1, layer_representations[i].shape[-1])
                layer_mcr2 = self.mcr2_core(Z_flat, labels_flat)
                mcr2_raw = mcr2_raw + layer_mcr2
                mcr2_total = mcr2_total + betas[j] * layer_mcr2
            mcr2_raw = mcr2_raw / max(len(list(indices)), 1)
            total = self.alpha_ce * ce + mcr2_scale * mcr2_total
            beta_val = betas.detach().mean().item()

        else:
            # Fixed or learned scalar beta
            mcr2_raw = self._compute_mcr2(layer_representations, labels_flat)
            if self.learn_beta:
                beta = self._get_betas()  # scalar tensor, floored
            else:
                beta = self.beta_mcr2
            total = self.alpha_ce * ce + mcr2_scale * beta * mcr2_raw
            beta_val = beta.detach().item() if isinstance(beta, torch.Tensor) else beta

        return {
            "loss": total,
            "ce_loss": ce.detach(),
            "mcr2_loss": mcr2_raw.detach(),
            "beta": beta_val,
        }
