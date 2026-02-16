"""Experiment configuration for Loss Functions for Principled Representations."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrainingRegime(Enum):
    CE_ONLY = "ce_only"          # Regime A: standard cross-entropy
    MCR2_ONLY = "mcr2_only"      # Regime B: MCR2 separation only
    ORGANISM = "organism"         # Regime C: CE global + MCR2 local


@dataclass
class ModelConfig:
    vocab_size: int = 50257       # GPT-2 BPE tokenizer
    n_layers: int = 6
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1
    bias: bool = False            # no bias in linear layers (GPT-2 style)


@dataclass
class MCR2Config:
    eps: float = 0.5              # precision parameter for coding rate
    min_class_size: int = 2       # skip classes with fewer samples
    max_classes: int = 512        # cap on classes per batch for efficiency
    use_projection: bool = True   # project to lower dim before computing
    proj_dim: int = 128           # projection dimension
    expansion_only: bool = False  # skip compression term (expansion/anti-collapse only)
    selective_expansion: bool = False  # expand centroids only (no compression, frequency-neutral)
    compression_only: bool = False     # skip expansion term (compression/regularization only)


@dataclass
class TrainingConfig:
    regime: TrainingRegime = TrainingRegime.CE_ONLY
    # dataset
    dataset: str = "wikitext-2"   # "wikitext-2" or "wikitext-103"
    # optimization
    batch_size: int = 64
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 50_000
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    # loss weights (for organism regime)
    alpha_ce: float = 1.0         # weight on global CE loss
    beta_mcr2: float = 0.1        # weight on local MCR2 loss (initial value if learned)
    # learned scaling
    learn_beta: bool = False      # make beta_mcr2 a learnable parameter (trained on total loss)
    per_layer_beta: bool = False  # learn separate beta per layer (requires learn_beta)
    beta_min: float = 0.001       # floor for beta (prevents MCR2 shutdown)
    beta_max: float = 1.0         # ceiling for beta
    mcr2_warmup_steps: int = 0    # steps before MCR2 activates (0 = immediate)
    # dual/adversarial scaling (beta driven by representation quality, not total loss)
    dual_beta: bool = False       # use Lagrangian dual ascent for beta
    beta_lr: float = 1e-4         # learning rate for dual beta update
    mcr2_target: float = -1.0     # target MCR2 loss (beta rises when MCR2 > target)
    # MCR2 satisfaction: stop MCR2 once representations are organized enough
    mcr2_rank_target: float = 0.0  # stop when effective_rank_mean >= this (0 = disabled)
    mcr2_sim_target: float = 0.0   # stop when cosine_sim_mean <= this (0 = disabled)
    # curriculum: expansion-only then full MCR2
    curriculum_switch_step: int = 0    # step to switch from expansion-only to full MCR2 (0 = disabled)
    alternating_period: int = 0        # alternate expansion/full MCR2 every N steps (0 = disabled)
    # layer-selective MCR2
    mcr2_layers: Optional[list] = None  # which layers get MCR2 (None = all)
    # logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    # paths
    output_dir: str = "runs"
    # hardware
    use_bf16: bool = True
    compile_model: bool = False   # torch.compile


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    mcr2: MCR2Config = field(default_factory=MCR2Config)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    run_name: Optional[str] = None

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = self.training.regime.value
