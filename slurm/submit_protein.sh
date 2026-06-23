#!/bin/bash
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# Usage: sbatch --job-name=33small_1Q9B submit_protein.sh 33small 1Q9B
# Or submit via submit_all.sh

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

LOGDIR="logs"
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

# cd to Pcode/ (parent of slurm/)
cd "$(dirname "$0")/.." || exit 1
echo "Working dir: $(pwd)"

python run.py --dataset "$DATASET" --protein "$PROTEIN"
EXIT_CODE=$?

echo "========================================="
echo "Exit code: ${EXIT_CODE}"
echo "End:       $(date)"
echo "========================================="

exit $EXIT_CODE
