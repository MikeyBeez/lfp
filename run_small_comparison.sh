#!/usr/bin/env bash
# Clean apples-to-apples comparison: CE-only vs Organism on the small model.
# Small model: d_model=192, n_layers=4, n_heads=4, d_ff=768 (~1.77M non-embedding params)
# Checkpoints every 1000 steps to checkpoints/<run_name>/

set -e

STEPS=10000
DATASET=wikitext-2
OUTDIR=checkpoints

# Shared model/training args
COMMON="--dataset $DATASET --max-steps $STEPS \
    --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
    --batch-size 64 --lr 3e-4 --seed 42 \
    --output-dir $OUTDIR --save-interval 1000"

echo "=== Small Model Comparison ==="
echo "Steps: $STEPS, Dataset: $DATASET, Output: $OUTDIR"
echo

echo "=== 1/2: CE-only baseline ==="
python3 train.py --regime ce_only --run-name ce_baseline $COMMON --resume

echo
echo "=== 2/2: Organism (CE + MCR2) ==="
python3 train.py --regime organism --run-name organism --beta-mcr2 0.1 $COMMON --resume

echo
echo "=== Training complete ==="
echo "Results in ./$OUTDIR/{ce_baseline,organism}/"

echo
echo "=== Generating comparison plot ==="
python3 visualize.py --runs-dir "$OUTDIR" --regimes ce_baseline organism --output comparison_small.png
echo "Done."
