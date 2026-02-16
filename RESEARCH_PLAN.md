# Loss Functions for Principled Representations (LFP)

## Research Question

What happens if you replace or augment cross-entropy with a separation-based
(MCR2-style) objective in a small transformer? Does it prevent representation
collapse and attention sinks that cross-entropy allows?

---

## 1. Literature Synthesis

### 1.1 MCR2: Maximal Coding Rate Reduction (Yi Ma et al.)

**Foundational paper:** Yu et al. (2020), "Learning Diverse and Discriminative
Representations via the Principle of Maximal Coding Rate Reduction"
([arXiv:2006.08558](https://arxiv.org/abs/2006.08558))

MCR2 proposes an information-theoretic objective: maximize the coding rate of
the whole dataset while minimizing the coding rate of each individual class.
The coding rate R(Z) measures how many bits are needed to encode representations
Z up to precision epsilon:

    R(Z) = (d/2) * log det(I + (d / (N * eps^2)) * Z^T Z)

The MCR2 objective is: **max R(Z) - sum_c (n_c / N) * R(Z_c)**

Key properties:
- Provably learns diverse (between-class separation) AND discriminative
  (within-class compression) features
- More robust to label corruption than cross-entropy
- Works in supervised, self-supervised, and unsupervised settings
- Has a clean geometric interpretation: push classes onto incoherent subspaces

### 1.2 CRATE: White-Box Transformers from Rate Reduction

**Key paper:** Yu et al. (2023), "White-Box Transformers via Sparse Rate
Reduction" ([arXiv:2311.13110](https://arxiv.org/abs/2311.13110), JMLR 2024)

CRATE derives the transformer architecture *from* the MCR2 objective:
- Multi-head self-attention ≈ gradient descent step on the coding rate
  (compression term)
- MLP block ≈ sparsification step (promotes sparse representations)
- Each layer is literally one optimization step of rate reduction

This is an **architectural** use of MCR2, not a loss function use. CRATE
replaces the standard transformer with a mathematically-derived architecture
where each layer has a known purpose.

**CRATE-LM:** Bai & Ma (2024), "Improving Neuron-level Interpretability with
White-box Language Models" ([arXiv:2410.16443](https://arxiv.org/abs/2410.16443))
- Applied CRATE architecture to language modeling
- Up to 103% improvement in neuron-level interpretability
- But still trained with **standard cross-entropy loss** on next-token prediction
- The architecture embodies rate reduction; the loss does not

**Scaling CRATE:** Yang et al. (NeurIPS 2024), "Scaling White-Box Transformers
for Vision" - Introduced CRATE-alpha with modifications for scaling.

### 1.3 Alternative Loss Functions for Language Models

**Harmonic Loss:** Baek et al. (2025), "Harmonic Loss Trains Interpretable AI
Models" ([arXiv:2502.01628](https://arxiv.org/abs/2502.01628))
- Replaces cross-entropy with L_harmonic = sum_c (1 / ||z - mu_c||^2)
- Scale invariant, has finite convergence points (class centers)
- Tested on GPT-2: more interpretable representations, less grokking
- **Most directly comparable prior work** to what we propose
- Limitation: still a prediction-based objective, not geometric/structural

**SimCTG:** Su et al. (2022), "A Contrastive Framework for Neural Text
Generation" ([arXiv:2202.06417](https://arxiv.org/abs/2202.06417))
- Adds contrastive objective to calibrate token representation space
- Specifically addresses anisotropy in language model representations
- Shows that cross-entropy alone produces degenerate, anisotropic representations
- Uses contrastive loss as auxiliary, not as primary training signal

### 1.4 Representation Collapse and Attention Sinks

**Representation collapse is real and well-documented:**

- Barbero et al. (2024), "Transformers need glasses!" - Proves that distinct
  input sequences can yield arbitrarily close final representations in
  decoder-only transformers. Exacerbated by low-precision formats.
  ([arXiv:2406.04267](https://arxiv.org/abs/2406.04267))

- Gopalani & Hu (2025), "What Happens During the Loss Plateau?" - Shows that
  representation collapse (hidden states becoming nearly parallel) occurs
  during training, accompanied by repetition bias. Slow learning of attention
  maps is the bottleneck. ([arXiv:2506.13688](https://arxiv.org/abs/2506.13688))

- Zhai et al. (2023), "Stabilizing Transformer Training by Preventing Attention
  Entropy Collapse" - Identifies attention entropy collapse (attention
  concentrating on single tokens) as a cause of training instability. Proposes
  sigma-reparam as fix. ([arXiv:2303.06296](https://arxiv.org/abs/2303.06296))

**Anisotropy is inherent to cross-entropy trained transformers:**

- Godey et al. (2024), "Anisotropy Is Inherent to Self-Attention in
  Transformers" - Shows anisotropy appears even with non-cross-entropy
  objectives, suggesting it's partly architectural.
  ([arXiv:2401.12143](https://arxiv.org/abs/2401.12143))

- Yu et al. (2021), "Rare Tokens Degenerate All Tokens" - Identifies that
  gradient dynamics of rare tokens cause representation degeneration for ALL
  tokens. ([arXiv:2109.03127](https://arxiv.org/abs/2109.03127))

**Attention sinks:**

- Ruscio et al. (2025), "What are you sinking?" - Shows attention sinks are
  not artifacts but establish reference frames in representation space.
  ([arXiv:2508.02546](https://arxiv.org/abs/2508.02546))

### 1.5 Local Learning in Deep Networks

- Pathak et al. (2022), "Local Learning on Transformers via Feature
  Reconstruction" - First to apply local learning to transformers. Each module
  trained independently. Uses feature reconstruction as local objective.
  ([arXiv:2212.14215](https://arxiv.org/abs/2212.14215))

- Ma et al. (2024), "AugLocal: Scaling Supervised Local Learning" - Shows that
  local learning with auxiliary networks can match end-to-end backpropagation
  while reducing GPU memory by ~40%.
  ([arXiv:2402.17318](https://arxiv.org/abs/2402.17318))

- Zhang et al. (2022), "Contrastive Deep Supervision" - Uses contrastive
  learning as intermediate layer supervision instead of task-specific loss.

### 1.6 The Gap: What Nobody Has Done

**Critical observation:** The existing work falls into two camps:

1. **CRATE camp:** Uses rate reduction as an *architectural principle* (the
   architecture IS the unrolled optimization), but still trains with
   cross-entropy loss.

2. **Alternative loss camp:** Replaces or augments cross-entropy (harmonic
   loss, contrastive objectives), but doesn't use rate reduction / separation
   objectives.

**Nobody has tried using MCR2 as a training loss function applied to a standard
transformer.** This is the gap we fill.

Furthermore, nobody has combined MCR2-as-loss with the Artificial Organism
framework of local + global objectives, where each layer gets its own
separation objective while the global cross-entropy maintains task performance.

---

## 2. Experimental Design

### 2.1 Model Architecture

**Base model:** Custom small GPT-2-style decoder-only transformer

| Parameter       | Value   |
|----------------|---------|
| Layers         | 6       |
| Attention heads| 8       |
| d_model        | 512     |
| d_ff           | 2048    |
| Context length | 256     |
| Vocab size     | 50257 (GPT-2 tokenizer) |
| Total params   | ~25M    |

This is well within RTX 5070 Ti capacity (16GB VRAM). We use the same
architecture across all three regimes to isolate the effect of the loss
function.

### 2.2 Dataset

**WikiText-103** (raw token version), using GPT-2's BPE tokenizer.
- Train: ~100M tokens
- Validation: ~200K tokens
- Simple enough for small models, complex enough to exhibit real phenomena

For rapid iteration during development, start with **WikiText-2** (~2M tokens).

### 2.3 Three Training Regimes

#### Regime A: Standard Cross-Entropy (Baseline)

Standard next-token prediction:

    L_CE = -sum_t log p(x_{t+1} | x_{1:t})

This is the conventional approach. We expect it to achieve good perplexity
but allow representation collapse and attention sinks.

#### Regime B: MCR2 Separation Objective Only

Replace cross-entropy entirely with a layer-wise MCR2 objective. At each
layer l, compute:

    L_MCR2(l) = -R(Z_l) + sum_c (n_c / N) * R(Z_c,l)

Where:
- Z_l are the representations at layer l
- "Classes" are defined by the next token: representations at position t
  belong to class c = x_{t+1}
- R(Z) = (d/2) * log det(I + (d / (N * eps^2)) * Z^T Z)

The total loss is: **L_B = sum_l L_MCR2(l)**

**Key design choice:** In a batch, many next-tokens will be unique (class
size = 1). We handle this by:
1. Only computing within-class rates for tokens that appear >= 2 times
   as next-tokens in the batch
2. Using larger batch sizes to ensure more class overlap
3. Grouping by token frequency bands as a fallback

This regime tests whether separation alone can drive language learning
without any prediction objective. The hypothesis is that it will produce
excellent representations but may struggle to rank predictions correctly
since it has no explicit generation objective.

#### Regime C: Combined AO Approach (Cross-Entropy Global + MCR2 Local)

The Artificial Organism approach:

    L_C = alpha * L_CE + beta * sum_l L_MCR2(l)

Where:
- L_CE is the standard cross-entropy at the final layer (global objective)
- L_MCR2(l) is applied at each transformer layer (local objectives)
- alpha, beta are weighting coefficients (we sweep beta in {0.01, 0.1, 1.0}
  with alpha = 1.0)

This tests the core hypothesis: local separation objectives at each layer
should prevent representation collapse and attention sinks while the global
cross-entropy maintains task performance.

### 2.4 MCR2 Loss: Practical Computation

The log-det computation is expensive for large d. We use:

1. **Subsampled approximation:** Sample a subset of dimensions if d is large
2. **Stochastic estimation:** Use Hutchinson's trace estimator for log-det:
   log det(A) = tr(log(A)), estimated via random projections
3. **Numerical stability:** Add small epsilon to diagonal before log-det
4. **Class handling for LM:** Within each batch, group representations by
   their target next-token. For the expansion term R(Z), use all
   representations. For compression terms R(Z_c), use groups.

### 2.5 Metrics

#### M1: Representation Collapse Metrics
- **Effective rank:** nuclear_norm(Z) / spectral_norm(Z) - measures how many
  dimensions the representations actually use. Higher = less collapse.
- **Cosine similarity distribution:** Mean and std of pairwise cosine
  similarities. High mean + low std = collapse (anisotropy).
- **Intrinsic dimensionality:** Via PCA - how many components capture 90% of
  variance. Report at each layer.

#### M2: Attention Health Metrics
- **Attention entropy:** H = -sum_j a_j log(a_j) for each head. Low entropy
  = attention sink / degenerate pattern.
- **Attention sink ratio:** Fraction of total attention mass on the [BOS]
  token or first token across all positions.
- **Attention head diversity:** Mean pairwise cosine distance between
  attention patterns of different heads in the same layer. Low = redundant
  heads.

#### M3: Separation Quality Metrics
- **Between-class / within-class ratio:** Compute mean distance between
  class centroids vs mean distance within classes (using next-token as class).
  Higher ratio = better separation.
- **Coding rate reduction:** The actual MCR2 value on held-out data -
  measures the quality of the learned representation structure.
- **Linear probe accuracy:** Train a linear classifier on frozen
  intermediate representations to predict syntactic/semantic properties.

#### M4: Task Performance Metrics
- **Perplexity:** Standard language model perplexity on validation set.
- **Next-token accuracy:** Top-1 and top-5 accuracy.
- **Generation quality:** Sample generations inspected qualitatively.

### 2.6 Training Details

| Parameter       | Value          |
|----------------|----------------|
| Optimizer      | AdamW          |
| Learning rate  | 3e-4           |
| LR schedule    | Cosine decay   |
| Warmup steps   | 1000           |
| Batch size     | 64 (sequences) |
| Gradient accum | 4 (effective batch = 256 sequences) |
| Training steps | 50K            |
| Weight decay   | 0.1            |
| Precision      | bf16 mixed     |

### 2.7 Evaluation Schedule

- Compute all metrics every 1000 steps
- Save checkpoints at steps {5K, 10K, 25K, 50K}
- Plot metric trajectories over training to see dynamics

### 2.8 Ablations

After the main comparison:
1. **Layer-selective MCR2:** Apply MCR2 only to early layers, middle layers,
   or late layers. Where does it help most?
2. **MCR2 weight schedule:** Start with high beta (strong separation pressure)
   and anneal to low beta. Does early separation help?
3. **Class definition variants:** Instead of next-token, use token-type
   (punctuation, stopword, content word) or POS tags as classes.

---

## 3. Code Structure

```
lfp/
├── config.py              # Experiment configs (dataclass-based)
├── data.py                # WikiText loading + tokenization
├── model.py               # Small GPT-2 transformer (shared architecture)
├── losses/
│   ├── __init__.py
│   ├── cross_entropy.py   # Standard CE loss
│   ├── mcr2.py            # MCR2 separation loss
│   └── combined.py        # AO combined loss (CE + MCR2)
├── metrics/
│   ├── __init__.py
│   ├── collapse.py        # Representation collapse metrics
│   ├── attention.py       # Attention health metrics
│   └── separation.py      # Separation quality metrics
├── train.py               # Training loop
├── evaluate.py            # Evaluation + metric computation
└── visualize.py           # Plotting metric trajectories
```

Key design principles:
- Each loss is a standalone module implementing a common interface
- The model exposes intermediate representations (per-layer) for MCR2
- Metrics are computed as standalone functions, not coupled to training
- Config is a single dataclass that specifies the entire experiment

---

## 4. Expected Outcomes and Hypotheses

### H1: Cross-entropy allows collapse; MCR2 prevents it
Regime A (CE only) will show decreasing effective rank and increasing cosine
similarity over training. Regimes B and C will maintain higher effective rank.

### H2: MCR2 alone may not achieve competitive perplexity
Regime B (MCR2 only) will likely produce good representations (high separation,
low collapse) but may not achieve good perplexity since it lacks a prediction
objective. If this works well anyway, it's an extremely strong result.

### H3: Combined approach gets the best of both worlds
Regime C (AO) should match Regime A's perplexity while matching Regime B's
representation quality. This validates the Artificial Organism insight that
local geometric objectives + global task objectives > either alone.

### H4: MCR2 reduces attention sinks
If representations are well-separated at each layer, attention has no need
to concentrate mass on meaningless tokens. Regimes B and C should show higher
attention entropy and lower sink ratios.

### H5: Representation quality predicts downstream performance
Even if Regime C matches Regime A on perplexity, the better representations
should show higher linear probe accuracy and better transfer properties.

---

## 5. Connection to Artificial Organism Framework

This experiment directly instantiates the AO framework:

| AO Concept           | This Experiment                        |
|---------------------|----------------------------------------|
| Organ               | Each transformer layer                 |
| Local objective (L) | MCR2 loss at that layer's output       |
| Global objective (G)| Cross-entropy at final layer           |
| Communication (C)   | Residual connections + attention       |
| Multi-level optim   | L_C = alpha*G + beta*sum(L_i)          |

The key insight being tested: **cross-entropy only cares about the answer;
MCR2 cares about the geometry of the path to that answer.** By combining
both, we get a system that achieves good predictions through well-structured
representations, rather than achieving good predictions despite degenerate
representations.
