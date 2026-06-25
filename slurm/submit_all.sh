#!/bin/bash
# Author: Zihan Wang
# <wangzh011031@163.com>
# Master submission: submits one job per protein across all datasets.
# 推荐使用: bash slurm/pcode_job.sh submit-all [max_concurrent]
# Usage: bash submit_all.sh [max_concurrent]

MAX_JOBS=${1:-10}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project: ${PROJECT_DIR}"
echo "Max concurrent: ${MAX_JOBS}"

DATASETS=("33small" "35large" "36med")
LOGDIR="${SCRIPT_DIR}/logs"; mkdir -p "$LOGDIR"

TOTAL=0; SUBMITTED=0
for ds in "${DATASETS[@]}"; do
    data_dir="${PROJECT_DIR}/code_data/${ds}"
    [ -d "$data_dir" ] || { echo "WARN: no $data_dir"; continue; }
    for f in "$data_dir"/*_ca.xyzb; do
        [ -f "$f" ] || continue
        protein=$(basename "$f" _ca.xyzb)
        TOTAL=$((TOTAL + 1))
    done
done

echo "Total proteins: ${TOTAL}"
[ $TOTAL -eq 0 ] && { echo "ERROR: No proteins found!"; exit 1; }

for ds in "${DATASETS[@]}"; do
    data_dir="${PROJECT_DIR}/code_data/${ds}"
    [ -d "$data_dir" ] || continue
    for f in "$data_dir"/*_ca.xyzb; do
        [ -f "$f" ] || continue
        protein=$(basename "$f" _ca.xyzb)

        # Limit concurrency
        while true; do
            RUNNING=$(squeue -h -u "$USER" -o "%T" 2>/dev/null | grep -c "RUNNING\|PENDING" || echo 0)
            [ "$RUNNING" -lt "$MAX_JOBS" ] && break
            echo "  ${RUNNING} jobs running, waiting..."
            sleep 30
        done

        sbatch --partition="${PCODE_PARTITION:-general-long}" \
               --cpus-per-task="${PCODE_CPUS:-16}" \
               --mem="${PCODE_MEM:-32G}" \
               --time="${PCODE_TIME:-3-00:00:00}" \
               --job-name="pcode_${ds}_${protein}" \
               --output="${LOGDIR}/${ds}_${protein}_%j.out" \
               "${SCRIPT_DIR}/run_worker.sh" "$ds" "$protein"

        SUBMITTED=$((SUBMITTED + 1))
        echo "[$SUBMITTED/$TOTAL] ${ds}/${protein}"
        sleep 1
    done
done

echo "Done: $SUBMITTED jobs submitted. Logs: $LOGDIR"
