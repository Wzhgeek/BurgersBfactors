#!/bin/bash
# Master submission: submits one job per protein across all datasets.
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

        sbatch --job-name="${ds}_${protein}" \
               --output="${LOGDIR}/${ds}_${protein}_%j.out" \
               "${SCRIPT_DIR}/submit_protein.sh" "$ds" "$protein"

        SUBMITTED=$((SUBMITTED + 1))
        echo "[$SUBMITTED/$TOTAL] ${ds}/${protein}"
        sleep 1
    done
done

echo "Done: $SUBMITTED jobs submitted. Logs: $LOGDIR"
