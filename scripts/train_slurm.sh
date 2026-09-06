#!/bin/bash
# SLURM batch script for the from-scratch video text spotter.
#
#   sbatch scripts/train_slurm.sh              # stage 1
#   STAGE=2 sbatch scripts/train_slurm.sh      # stage 2, initialised from stage 1
#
# Adjust --partition and --gres to your cluster; ask your admin for the values.
# The script resumes from last.pt automatically, so a wall-clock limit shorter
# than the run is not a problem -- just submit it again.

#SBATCH --job-name=vtspot
#SBATCH --output=logs/vtspot-%j.out
#SBATCH --error=logs/vtspot-%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs

STAGE="${STAGE:-1}"
CONFIG="configs/a4000_stage${STAGE}.yaml"
CKPT="checkpoints/a4000_stage${STAGE}"
PREV="checkpoints/a4000_stage$((STAGE - 1))/last.pt"

# Activate your environment (edit to match your setup).
if [ -d .venv ]; then
    source .venv/bin/activate
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" && conda activate vtspot
fi

echo "host=$(hostname) stage=${STAGE} config=${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARGS=(--config "${CONFIG}" --ckpt-dir "${CKPT}" --workers "${SLURM_CPUS_PER_TASK:-4}")

if [ -f "${CKPT}/last.pt" ]; then
    echo "resuming from ${CKPT}/last.pt"
    ARGS+=(--resume "${CKPT}/last.pt")
elif [ "${STAGE}" != "1" ] && [ -f "${PREV}" ]; then
    echo "initialising from ${PREV}"
    ARGS+=(--init "${PREV}")
fi

srun python tools/train.py "${ARGS[@]}"
