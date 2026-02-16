#!/usr/bin/env bash
# Run all three experimental regimes sequentially.
# Adjust --max-steps for quick smoke tests vs full runs.

set -e

STEPS=${1:-50000}
DATASET=${2:-wikitext-2}

echo "=== Running LFP experiments ==="
echo "Steps: $STEPS, Dataset: $DATASET"
echo

echo "=== Regime A: Cross-Entropy Only ==="
python3 train.py --regime ce_only --dataset "$DATASET" --max-steps "$STEPS"

echo
echo "=== Regime B: MCR2 Only ==="
python3 train.py --regime mcr2_only --dataset "$DATASET" --max-steps "$STEPS"

echo
echo "=== Regime C: Organism (CE + MCR2) ==="
python3 train.py --regime organism --dataset "$DATASET" --max-steps "$STEPS" --beta-mcr2 0.1

echo
echo "=== All experiments complete ==="
echo "Results in ./runs/{ce_only,mcr2_only,organism}/"
