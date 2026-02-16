#!/usr/bin/env bash
# Curriculum experiments: expansion-only early, then full MCR2.
# Tests whether establishing good geometry first, then constraining with
# compression, produces better results than either alone.
#
# Three switch points: 2000, 3000, 4000 steps (out of 10K total).

set -e

STEPS=10000
DATASET=wikitext-2
OUTDIR=checkpoints

COMMON="--dataset $DATASET --max-steps $STEPS \
    --d-model 192 --n-layers 4 --n-heads 4 --d-ff 768 \
    --batch-size 64 --lr 3e-4 --seed 42 \
    --beta-mcr2 0.1 --expansion-only \
    --output-dir $OUTDIR --save-interval 1000"

echo "=== Curriculum Experiments ==="
echo "Steps: $STEPS, Dataset: $DATASET, Output: $OUTDIR"
echo

echo "=== 1/3: Expansion 2K -> Full MCR2 8K ==="
python3 train.py --regime organism --run-name curriculum_2k $COMMON \
    --curriculum-switch-step 2000 --resume

echo
echo "=== 2/3: Expansion 3K -> Full MCR2 7K ==="
python3 train.py --regime organism --run-name curriculum_3k $COMMON \
    --curriculum-switch-step 3000 --resume

echo
echo "=== 3/3: Expansion 4K -> Full MCR2 6K ==="
python3 train.py --regime organism --run-name curriculum_4k $COMMON \
    --curriculum-switch-step 4000 --resume

echo
echo "=== All curriculum runs complete ==="
echo

echo "=== Generating comparison plot ==="
python3 visualize.py \
    --runs-dir "$OUTDIR" \
    --regimes ce_baseline organism expansion_only selective_expansion \
              curriculum_2k curriculum_3k curriculum_4k \
    --output comparison_all.png
echo "Done."
