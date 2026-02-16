# LFP: Loss Functions for Principled Representations

Testing MCR2 (Maximal Coding Rate Reduction) as an auxiliary loss for training small transformer language models.

**Key finding:** MCR2 improves generalization by 2.8% (val_ppl 306.4 vs 315.2), but not for the reasons the theory claims. Neither expansion nor compression alone helps — each one alone makes things worse. It's the tension between the two conflicting objectives that regularizes.

**Paper:** [MCR2 as Language Model Regularizer: What Works, What Doesn't, and Why the Theory Oversells It](article_draft.md)

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU. Tested on an RTX 5070 Ti (16GB VRAM) with CUDA 13.0.

## Running experiments

All experiments use a small transformer (d_model=192, 4 layers, 4 heads, ~1.77M non-embedding params) trained on WikiText-2 for 10K steps.

### CE baseline

```bash
python3 train.py \
  --regime ce_only \
  --run-name ce_baseline \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 \
  --output-dir checkpoints \
  --save-interval 1000
```

### Full MCR2 (organism)

```bash
python3 train.py \
  --regime organism \
  --run-name organism \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 \
  --beta-mcr2 0.1 \
  --output-dir checkpoints \
  --save-interval 1000
```

### Ablation variants

**Expansion-only** (no compression term):
```bash
python3 train.py \
  --regime organism --run-name expansion_only \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 --beta-mcr2 0.1 \
  --expansion-only \
  --output-dir checkpoints --save-interval 1000
```

**Compression-only** (no expansion term):
```bash
python3 train.py \
  --regime organism --run-name compression_only \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 --beta-mcr2 0.1 \
  --compression-only \
  --output-dir checkpoints --save-interval 1000
```

**Selective expansion** (centroid-based, frequency-neutral):
```bash
python3 train.py \
  --regime organism --run-name selective_expansion \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 --beta-mcr2 0.1 \
  --selective-expansion \
  --output-dir checkpoints --save-interval 1000
```

**Curriculum** (expansion first, then full MCR2):
```bash
python3 train.py \
  --regime organism --run-name curriculum_2k \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 --beta-mcr2 0.1 \
  --expansion-only --curriculum-switch-step 2000 \
  --output-dir checkpoints --save-interval 1000
```

**Alternating** (cycle expansion/full MCR2 every N steps):
```bash
python3 train.py \
  --regime organism --run-name alternating_2k \
  --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
  --max-steps 10000 --beta-mcr2 0.1 \
  --expansion-only --alternating-period 2000 \
  --output-dir checkpoints --save-interval 1000
```

### Resuming a run

```bash
# Auto-detect latest checkpoint:
python3 train.py --regime organism --run-name organism \
  --output-dir checkpoints --resume

# From a specific checkpoint:
python3 train.py --regime organism --run-name organism \
  --output-dir checkpoints --resume checkpoints/organism/checkpoint_5000.pt
```

## Visualization

Generate comparison plots across all runs:

```bash
python3 visualize.py \
  --runs-dir checkpoints \
  --regimes ce_baseline organism expansion_only compression_only \
  --output comparison.png
```

Per-layer plots are generated automatically for each regime.

## Results

| Run | Val PPL | Train PPL |
|-----|---------|-----------|
| **Full MCR2** | **306.4** | 21.1 |
| CE baseline | 315.2 | 19.5 |
| Curriculum 2K | 317.2 | 20.2 |
| Compression only | 317.2 | 20.8 |
| Alternating 2K | 322.1 | 19.8 |
| Selective expansion | 324.2 | 19.5 |
| Expansion only | 335.5 | 19.4 |

## Project structure

```
train.py          # Training loop with all regime/ablation support
config.py         # Experiment configuration dataclasses
model.py          # Decoder-only transformer
data.py           # WikiText loading and tokenization
evaluate.py       # Evaluation metrics
visualize.py      # Comparison and per-layer plots
losses/
  mcr2.py         # MCR2 loss (expansion, compression, and variants)
  combined.py     # Combined CE + MCR2 organism loss
  cross_entropy.py
metrics/
  collapse.py     # Effective rank, cosine similarity, intrinsic dim
  attention.py    # Attention entropy, sink ratio, head diversity
  separation.py   # Between/within class separation ratio
```
