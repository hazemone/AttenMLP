#!/usr/bin/env bash
# =====================================================================
#  run_modelnet40_seeds.sh  --  multi-seed ModelNet40 runs (Group C1)
# ---------------------------------------------------------------------
#  Runs your AttenMLP model on ModelNet40 across several random seeds
#  so we can report mean +/- std and a 95% CI (Reviewer #1).
#  Optionally also runs the DualMLP baseline over the SAME seeds so we
#  can report a paired-t significance test (AttenMLP vs DualMLP).
#
#  WHERE TO RUN:  inside  classification_ModelNet40/
#  (the folder with main.py). Copy this script there.
#
#  >>> YOU MUST EDIT THE TWO COMMANDS BELOW <<<
#  Put YOUR exact paper training command in attenmlp_cmd(), keeping the
#  $SEED variable where the seed goes, and using batch 16 / lr 0.025 /
#  min_lr 0.005 / 100 epochs (Table I). If you are unsure of the exact
#  args, open  checkpoints/<any-old-run>/args.txt  -- it lists them.
# =====================================================================
set -euo pipefail

# Seeds to run. 5 is ideal; 3 is the minimum that lets us report a CI.
SEEDS=(0 1 2 3 4)

# Set to "true" to ALSO run the DualMLP baseline over the same seeds
# (needed only if you want the paired significance test).
RUN_BASELINE="true"

# ---------------------------------------------------------------------
# EDIT THIS: your AttenMLP training command. Keep $SEED and --msg as-is.
# Replace --model, --num_points_*, --neighbours_*, --num_channel with
# the EXACT values from your paper run.
# ---------------------------------------------------------------------
attenmlp_cmd() {
  local SEED=$1
  python main.py \
    --model AttenMLP \
    --msg seed \
    --seed "$SEED" \
    --dual_net True \
    --last_layer_concat add \
    --num_points_low 1024 --neighbours_low 32 \
    --num_points_high 2048 --neighbours_high 32 \
    --num_channel 32 \
    --batch_size 16 \
    --learning_rate 0.025 --min_lr 0.005 \
    --epoch 100 \
    --workers 8
}

# ---------------------------------------------------------------------
# Baseline: DualMLP, from the original repo's reproduce command.
# (Only used if RUN_BASELINE=true. Adjust if your baseline differs.)
# ---------------------------------------------------------------------
dualmlp_cmd() {
  local SEED=$1
  python main.py \
    --model pointMLP \
    --msg dualbase-seed \
    --seed "$SEED" \
    --dual_net True \
    --last_layer_concat add \
    --num_points_low 1024 --neighbours_low 32 \
    --num_points_high 2048 --neighbours_high 32 \
    --num_channel 32 \
    --batch_size 16 \
    --learning_rate 0.025 --min_lr 0.005 \
    --epoch 100 \
    --workers 8
}

run_one() {
  local label=$1 seed=$2
  shift 2
  echo "=================================================================="
  echo " [$label] seed=$seed   started $(date '+%F %T')"
  echo "=================================================================="
  local t0=$(date +%s)
  "$@"
  local t1=$(date +%s)
  echo " [$label] seed=$seed   done in $(( (t1 - t0) / 60 )) min"
}

echo "### ModelNet40 multi-seed sweep ###"
echo "Seeds: ${SEEDS[*]}   Baseline: $RUN_BASELINE"
echo

for s in "${SEEDS[@]}"; do
  run_one "AttenMLP" "$s" attenmlp_cmd "$s"
done

if [ "$RUN_BASELINE" = "true" ]; then
  for s in "${SEEDS[@]}"; do
    run_one "DualMLP" "$s" dualmlp_cmd "$s"
  done
fi

echo
echo "All runs finished. Now aggregate the results:"
echo
echo "  python aggregate_results.py \\"
echo "      --name \"AttenMLP\" --glob \"checkpoints/AttenMLP-seed-*\" \\"
echo "      --baseline-name \"DualMLP\" --baseline-glob \"checkpoints/pointMLP-dualbase-seed-*\""
echo
echo "(Run dirs are named checkpoints/<model>-<msg>-<seed>, so the AttenMLP"
echo " runs match 'AttenMLP-seed-*' and the baseline 'pointMLP-dualbase-seed-*'.)"
