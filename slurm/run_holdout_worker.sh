#!/bin/bash
# Author: Zihan Wang
# <wangzh011031@163.com>
#SBATCH --partition=general-long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# SLURM worker: hold-out 修补小样本蛋白的 fold PCC（保留 OOF_PCC）
# 由 pcode_job.sh submit-holdout-repair 调用

set -euo pipefail

DATASET=${1:?dataset required}
PROTEIN=${2:?protein required}

CONDA_SH="/mnt/home/jiangj33/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="eeg"
PYTHON="/mnt/home/jiangj33/anaconda3/envs/eeg/bin/python"

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/mnt/home/jiangj33/Pcode}"
SCRIPT_DIR="${PROJECT_DIR}/slurm"
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGDIR"

JOB_TAG="${DATASET}_${PROTEIN}_holdout"
exec > >(tee -a "${LOGDIR}/${JOB_TAG}_${SLURM_JOB_ID}.out") 2>&1

echo "========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${SLURM_JOB_NAME}"
echo "Mode:      holdout-repair"
echo "Dataset:   ${DATASET}"
echo "Protein:   ${PROTEIN}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "========================================="

cd "$PROJECT_DIR" || exit 1
source "$CONDA_SH"
conda activate "$CONDA_ENV"

"$PYTHON" run_holdout_repair.py --dataset "$DATASET" --protein "$PROTEIN"
EXIT_CODE=$?

echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
exit $EXIT_CODE
