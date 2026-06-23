#!/bin/bash
# Author: Zihan Wang
# <wangzh011031@163.com>
#SBATCH --partition=general-long
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# SLURM worker: 在计算节点上激活 eeg 环境并运行 run.py
# 由 pcode_job.sh submit 调用，勿直接 sbatch（除非调试）

set -euo pipefail

DATASET=${1:?dataset required}
PROTEIN=${2:?protein required}

# ── Conda（绝对路径）──
CONDA_SH="/mnt/home/jiangj33/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="eeg"
PYTHON="/mnt/home/jiangj33/anaconda3/envs/eeg/bin/python"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGDIR"

JOB_TAG="${DATASET}_${PROTEIN}"
exec > >(tee -a "${LOGDIR}/${JOB_TAG}_${SLURM_JOB_ID}.out") 2>&1

echo "========================================="
echo "Cluster:   msuhpcc (SLURM)"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${SLURM_JOB_NAME}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Dataset:   ${DATASET}"
echo "Protein:   ${PROTEIN}"
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Mem:       ${SLURM_MEM_PER_NODE:-N/A}"
echo "Start:     $(date)"
echo "========================================="

cd "$PROJECT_DIR" || exit 1
echo "Working dir: $(pwd)"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: conda.sh not found: $CONDA_SH"
    exit 1
fi
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

echo "Python:    $($PYTHON --version 2>&1) ($PYTHON)"
echo "========================================="

"$PYTHON" run.py --dataset "$DATASET" --protein "$PROTEIN"
EXIT_CODE=$?

echo "========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
echo "========================================="

exit $EXIT_CODE
