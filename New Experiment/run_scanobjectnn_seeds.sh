#!/usr/bin/env bash
# =====================================================================
#  run_scanobjectnn_seeds.sh  --  ScanObjectNN runs (Group C2)
# ---------------------------------------------------------------------
#  Validates AttenMLP on the harder, real-world ScanObjectNN benchmark
#  (Reviewer #3 / external review: "evaluate on a more challenging
#  dataset"). Uses the hardest PB_T50_RS variant
#  (*_objectdataset_augmentedrot_scale75.h5), which is the standard
#  hard split and the one the DualMLP repo loads by default.
#
#  WHERE TO RUN:  inside  classification_ScanObjectNN/
#  (the folder with main.py and ScanObjectNN.py). Copy this script there.
#
#  DATA: downloads automatically on first run into  ./data/h5_files/ .
#  If the auto-download URL is slow/unreachable, fetch h5_files.zip
#  manually and unzip it into  classification_ScanObjectNN/data/  so you
#  have  data/h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5 .
#
#  >>> EDIT attenmlp_cmd() with your model flag / architecture args <<<
#  ScanObjectNN has 15 classes; default epochs 200, lr 0.01, wd 1e-4.
# =====================================================================
set -euo pipefail

# 3 seeds is enough for a CI on the second dataset; add more if time allows.
SEEDS=(0 1 2)

# Also run the DualMLP baseline for a paired significance test?
RUN_BASELINE="true"

# ---------------------------------------------------------------------
# EDIT THIS: your AttenMLP command for ScanObjectNN. Keep $SEED.
# Start from the repo's reproduce settings, then swap --model to yours.
# ---------------------------------------------------------------------
attenmlp_cmd() {
  local SEED=$1
  python main.py \
    --model AttenMLP \
    --msg sonn-seed \
    --seed "$SEED" \
    --num_classes 15 \
    --dual_net True \
    --last_layer_concat multiply \
    --num_points_low 128 --neighbours_low 16 \
    --num_points_high 2048 --neighbours_high 32 \
    --num_channel 16 \
    --batch_size 32 \
    --learning_rate 0.01 --min_lr 0.005 \
    --weight_decay 1e-4 \
    --epoch 200 \
    --workers 8
}

dualmlp_cmd() {
  local SEED=$1
  python main.py \
    --model pointMLP \
    --msg sonn-dualbase-seed \
    --seed "$SEED" \
    --num_classes 15 \
    --dual_net True \
    --last_layer_concat multiply \
    --num_points_low 128 --neighbours_low 16 \
    --num_points_high 2048 --neighbours_high 32 \
    --num_channel 16 \
    --batch_size 32 \
    --learning_rate 0.01 --min_lr 0.005 \
    --weight_decay 1e-4 \
    --epoch 200 \
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

echo "### ScanObjectNN multi-seed sweep (hardest PB_T50_RS split) ###"
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
echo "All runs finished. Aggregate with:"
echo
echo "  python aggregate_results.py \\"
echo "      --name \"AttenMLP (ScanObjectNN)\" --glob \"checkpoints/AttenMLP-sonn-seed-*\" \\"
echo "      --baseline-name \"DualMLP\" --baseline-glob \"checkpoints/pointMLP-sonn-dualbase-seed-*\""
