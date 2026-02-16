#!/usr/bin/env bash
# Round 2 experiments: finding the sweet spot between CE and MCR2.
#
# The key experiment is dual/adversarial beta: beta is NOT trained on the total
# loss (which would just push it toward less MCR2). Instead, beta is updated via
# Lagrangian dual ascent — it rises when representations are poor and falls when
# they're good. This creates productive tension between prediction and geometry.
#
# Experiments:
#   1. Dual beta (adversarial) — the main event
#   2. Dual beta + curriculum — let CE anchor first, then introduce the tension
#   3. Dual beta + early-layer only — MCR2 pressure only where it helps most
#   4. Learned scalar beta (baseline comparison) — trained on total loss with floor

set -e

STEPS=${1:-50000}
DATASET=${2:-wikitext-2}

echo "=== LFP Round 2: Adversarial Beta Experiments ==="
echo "Steps: $STEPS, Dataset: $DATASET"
echo

# Experiment 1: Dual beta (adversarial)
# Beta driven by representation quality, not total loss.
# Starts at 0.01, target MCR2 of -1.0 (moderate rate reduction).
# Beta rises when MCR2 > -1.0 (bad reps), falls when MCR2 < -1.0 (good reps).
echo "=== Exp 1: Dual beta (adversarial) ==="
DUAL_CKPT="runs/organism_dual/checkpoint_5000.pt"
RESUME_FLAG=""
if [ -f "$DUAL_CKPT" ]; then
    echo "Resuming from $DUAL_CKPT"
    RESUME_FLAG="--resume $DUAL_CKPT"
fi
python3 train.py \
    --regime organism \
    --dataset "$DATASET" \
    --max-steps "$STEPS" \
    --beta-mcr2 0.01 \
    --dual-beta \
    --beta-lr 1e-4 \
    --mcr2-target -1.0 \
    --beta-min 0.001 \
    --beta-max 1.0 \
    --run-name organism_dual \
    $RESUME_FLAG

echo
# Experiment 2: Dual beta + curriculum
# Pure CE for 10K steps, then introduce the adversarial tension.
echo "=== Exp 2: Dual beta + curriculum ==="
python3 train.py \
    --regime organism \
    --dataset "$DATASET" \
    --max-steps "$STEPS" \
    --beta-mcr2 0.01 \
    --dual-beta \
    --beta-lr 1e-4 \
    --mcr2-target -1.0 \
    --beta-min 0.001 \
    --beta-max 1.0 \
    --mcr2-warmup-steps 10000 \
    --run-name organism_dual_curriculum

echo
# Experiment 3: Dual beta + early layers only
# Adversarial tension only on layers 0,1,2. Later layers stay prediction-aligned.
echo "=== Exp 3: Dual beta + early layers ==="
python3 train.py \
    --regime organism \
    --dataset "$DATASET" \
    --max-steps "$STEPS" \
    --beta-mcr2 0.01 \
    --dual-beta \
    --beta-lr 1e-4 \
    --mcr2-target -1.0 \
    --beta-min 0.001 \
    --beta-max 1.0 \
    --mcr2-layers "0,1,2" \
    --run-name organism_dual_early

echo
# Experiment 4: Learned scalar beta (baseline)
# Beta trained on total loss — will it just minimize MCR2 away?
# This is the control: shows what happens without the adversarial tension.
echo "=== Exp 4: Learned beta (total loss, baseline) ==="
python3 train.py \
    --regime organism \
    --dataset "$DATASET" \
    --max-steps "$STEPS" \
    --beta-mcr2 0.01 \
    --learn-beta \
    --beta-min 0.001 \
    --run-name organism_learned

echo
echo "=== All Round 2 experiments complete ==="
echo "Results in ./runs/{organism_dual,organism_dual_curriculum,organism_dual_early,organism_learned}/"
echo "Key: watch the beta curve in metrics.jsonl — it tells the whole story."
