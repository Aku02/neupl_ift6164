#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
#  Cartesian game sweep: run all policy variants, log to wandb, print report.
#
#  Usage:
#    bash run_cartesian_sweep.bash           # GPU 0 (default)
#    CUDA_VISIBLE_DEVICES=1 bash run_cartesian_sweep.bash   # GPU 1
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- shared hyperparameters ------------------------------------------------
SEED=42
OUTER_ITERS=300
ABR_STEPS=15
EPOCH_LEN=10
LR=0.03
ADAPTIVE_N=64
WANDB_PROJECT="neupl-cartesian"

COMMON="--seed $SEED --outer-iters $OUTER_ITERS --abr-steps $ABR_STEPS \
        --epoch-len $EPOCH_LEN --lr $LR --adaptive-n $ADAPTIVE_N \
        --wandb --wandb-project $WANDB_PROJECT"

# ---- results root ----------------------------------------------------------
RESULTS="cartesian_sweep_results"
mkdir -p "$RESULTS"

echo ""
echo "================================================================"
echo "  Cartesian game sweep  (seed=$SEED, iters=$OUTER_ITERS)"
echo "  wandb project: $WANDB_PROJECT"
echo "================================================================"
echo ""

# --------------------------------------------------------------------------
#  1. NeuPL  pure  (JPSRO)
# --------------------------------------------------------------------------
echo ">>> [1/6] neupl_cartesian.py  --policy-model pure  --graph psro_nash"
python neupl_cartesian.py \
    $COMMON \
    --policy-model pure \
    --graph psro_nash \
    --save-dir "$RESULTS/neupl_pure" \
    2>&1 | tee "$RESULTS/neupl_pure.log"
echo ""

# --------------------------------------------------------------------------
#  2. NeuPL  discrete_mixed  (JPSRO)
# --------------------------------------------------------------------------
echo ">>> [2/6] neupl_cartesian.py  --policy-model discrete_mixed  --graph psro_nash"
python neupl_cartesian.py \
    $COMMON \
    --policy-model discrete_mixed \
    --graph psro_nash \
    --save-dir "$RESULTS/neupl_discrete_mixed" \
    2>&1 | tee "$RESULTS/neupl_discrete_mixed.log"
echo ""

# --------------------------------------------------------------------------
#  3. Diffusion  (JPSRO)
# --------------------------------------------------------------------------
echo ">>> [3/6] neupl_cartesian_diffusion.py  --graph psro_nash"
python neupl_cartesian_diffusion.py \
    $COMMON \
    --graph psro_nash \
    --diffusion-steps 10 \
    --diffusion-support-size 7 \
    --diffusion-hidden 64 \
    --diffusion-aux-coef 0.1 \
    --save-dir "$RESULTS/diffusion_jpsro" \
    2>&1 | tee "$RESULTS/diffusion_jpsro.log"
echo ""

# --------------------------------------------------------------------------
#  4. MeanFlow  (JPSRO)
# --------------------------------------------------------------------------
echo ">>> [4/6] meanflow_cartesian.py  --graph psro_nash"
python meanflow_cartesian.py \
    $COMMON \
    --graph psro_nash \
    --flow-steps 10 \
    --support-size 7 \
    --hidden 64 \
    --consistency-coef 0.1 \
    --save-dir "$RESULTS/meanflow_jpsro" \
    2>&1 | tee "$RESULTS/meanflow_jpsro.log"
echo ""

# --------------------------------------------------------------------------
#  5. MeanFlow  (self-play, non-JPSRO)
# --------------------------------------------------------------------------
echo ">>> [5/6] meanflow_cartesian_nash.py  --graph self_play"
python meanflow_cartesian_nash.py \
    $COMMON \
    --graph self_play \
    --flow-steps 10 \
    --support-size 7 \
    --hidden 64 \
    --consistency-coef 0.1 \
    --save-dir "$RESULTS/meanflow_selfplay" \
    2>&1 | tee "$RESULTS/meanflow_selfplay.log"
echo ""

# --------------------------------------------------------------------------
#  6. Diffusion  (self-play, non-JPSRO)
# --------------------------------------------------------------------------
echo ">>> [6/6] neupl_cartesian_diffusion.py  --graph self_play"
python neupl_cartesian_diffusion.py \
    $COMMON \
    --graph self_play \
    --diffusion-steps 10 \
    --diffusion-support-size 7 \
    --diffusion-hidden 64 \
    --diffusion-aux-coef 0.1 \
    --save-dir "$RESULTS/diffusion_selfplay" \
    2>&1 | tee "$RESULTS/diffusion_selfplay.log"
echo ""

# --------------------------------------------------------------------------
#  Coverage comparison plots
# --------------------------------------------------------------------------
echo ">>> Generating comparison plots ..."

python compare_cartesian_meta_densities.py \
    --a "$RESULTS/meanflow_jpsro/run_summary.npz" \
    --b "$RESULTS/diffusion_jpsro/run_summary.npz" \
    --a-label "meanflow (JPSRO)" \
    --b-label "diffusion (JPSRO)" \
    --out "$RESULTS/compare_jpsro_meanflow_vs_diffusion.png"

python compare_cartesian_meta_densities.py \
    --a "$RESULTS/meanflow_selfplay/run_summary.npz" \
    --b "$RESULTS/diffusion_selfplay/run_summary.npz" \
    --a-label "meanflow (self-play)" \
    --b-label "diffusion (self-play)" \
    --out "$RESULTS/compare_selfplay_meanflow_vs_diffusion.png"

python compare_cartesian_meta_densities.py \
    --a "$RESULTS/meanflow_jpsro/run_summary.npz" \
    --b "$RESULTS/meanflow_selfplay/run_summary.npz" \
    --a-label "meanflow (JPSRO)" \
    --b-label "meanflow (self-play)" \
    --out "$RESULTS/compare_meanflow_jpsro_vs_selfplay.png"

echo ""

# --------------------------------------------------------------------------
#  Terminal summary table
# --------------------------------------------------------------------------
echo "================================================================"
echo "                    CARTESIAN SWEEP SUMMARY"
echo "================================================================"
printf "%-28s %6s %6s %6s %6s %6s %6s %6s %6s\n" \
    "Variant" "Expl" "|v-v*|" "L1row" "L1col" "Rmode" "Cmode" "Rprec" "Cprec"
echo "----------------------------------------------------------------"

for dir in neupl_pure neupl_discrete_mixed diffusion_jpsro meanflow_jpsro \
           meanflow_selfplay diffusion_selfplay; do
    npz="$RESULTS/$dir/run_summary.npz"
    if [ -f "$npz" ]; then
        python -c "
import numpy as np, sys
z = np.load('$npz', allow_pickle=True)
def g(k):
    return float(z[k]) if k in z.files else 0.0
print(f\"{'$dir':<28s} {g('exploit/exploitability'):6.4f} {g('nash/nash_value_gap'):6.4f} {g('nash/row_density_l1'):6.3f} {g('nash/col_density_l1'):6.3f} {int(g('modes_row/n_optimal_modes')):6d} {int(g('modes_col/n_optimal_modes')):6d} {g('modes_row/mode_precision'):6.2f} {g('modes_col/mode_precision'):6.2f}\")
"
    fi
done

echo "================================================================"
echo ""
echo "Plots saved to: $RESULTS/"
echo "Logs:  $RESULTS/*.log"
echo "wandb: https://wandb.ai  (project: $WANDB_PROJECT)"
echo ""
echo "Done!"
