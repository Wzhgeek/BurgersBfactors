#!/bin/bash
# Author: Zihan Wang
# <wangzh011031@163.com>
#SBATCH --partition=general-long
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00

# 兼容旧用法；推荐使用: bash slurm/pcode_job.sh submit <dataset> <protein>
# Usage: sbatch --job-name=pcode_33small_1Q9B submit_protein.sh 33small 1Q9B

DATASET=$1
PROTEIN=$2

if [ -z "$DATASET" ] || [ -z "$PROTEIN" ]; then
    echo "Usage: sbatch --job-name=<dataset>_<protein> submit_protein.sh <dataset> <protein>"
    exit 1
fi

# Auto-set job name from args if not already set
if [ -z "$SLURM_JOB_NAME" ] || [ "$SLURM_JOB_NAME" = "submit_protein.sh" ]; then
    scontrol update job=$SLURM_JOB_ID name="${DATASET}_${PROTEIN}" 2>/dev/null || true
fi

PROJECT_DIR="${SLURM_SUBMIT_DIR:-/mnt/home/jiangj33/Pcode}"
LOGDIR="${PROJECT_DIR}/slurm/logs"
mkdir -p "$LOGDIR"

exec > >(tee -a "${LOGDIR}/${DATASET}_${PROTEIN}_${SLURM_JOB_ID}.out") 2>&1

echo "========================================="
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Dataset:   ${DATASET}"
echo "Protein:   ${PROTEIN}"
echo "Start:     $(date)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Node:      $(hostname)"
echo "========================================="

cd "$PROJECT_DIR" || exit 1
echo "Working dir: $(pwd)"

CONDA_SH="/mnt/home/jiangj33/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="eeg"
PYTHON="/mnt/home/jiangj33/anaconda3/envs/eeg/bin/python"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
echo "Python: $($PYTHON --version 2>&1)"

"$PYTHON" run.py --dataset "$DATASET" --protein "$PROTEIN"
EXIT_CODE=$?

echo "========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
echo "========================================="

exit $EXIT_CODE
