#!/bin/bash
# Author: Zihan Wang
# <wangzh011031@163.com>
#
# Pcode 集群任务管理脚本（MSU HPPC / SLURM）
#
# 用法:
#   bash slurm/pcode_job.sh status                         # 查看集群与用户任务
#   bash slurm/pcode_job.sh submit 33small 1Q9B            # 提交单个蛋白
#   bash slurm/pcode_job.sh submit 33small                  # 提交整个数据集
#   bash slurm/pcode_job.sh submit-all [max_concurrent]     # 提交 config 中全部数据集
#   bash slurm/pcode_job.sh stop 33small                    # 停止该数据集全部 pcode 任务
#   bash slurm/pcode_job.sh stop 33small 1Q9B               # 停止指定任务
#   bash slurm/pcode_job.sh stop --all                      # 停止当前用户全部 pcode 任务
#   bash slurm/pcode_job.sh stop <jobid> [jobid ...]        # 按 JobID 停止
#
# 资源参数（环境变量覆盖默认值）:
#   PCODE_PARTITION=general-long   # CPU 分区: general-short(4h) / general-long(7d) / scavenger(7d)
#   PCODE_CPUS=16                  # 每任务 CPU 核数（run.py n_jobs=-1 会吃满）
#   PCODE_MEM=32G                  # 内存
#   PCODE_TIME=12:00:00            # 最长运行时间
#
# Conda 环境:
#   激活脚本: /mnt/home/jiangj33/anaconda3/etc/profile.d/conda.sh
#   环境名:   eeg
#   Python:   /mnt/home/jiangj33/anaconda3/envs/eeg/bin/python

set -euo pipefail

# ── 默认资源配置（CPU 密集任务）──
PARTITION="${PCODE_PARTITION:-general-long}"
CPUS="${PCODE_CPUS:-16}"
MEM="${PCODE_MEM:-32G}"
TIME="${PCODE_TIME:-12:00:00}"
JOB_PREFIX="pcode"
MAX_CONCURRENT="${PCODE_MAX_JOBS:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WORKER="${SCRIPT_DIR}/run_worker.sh"
LOGDIR="${SCRIPT_DIR}/logs"
DATASETS=(33small 35large 36med)

CONDA_SH="/mnt/home/jiangj33/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="eeg"
PYTHON="/mnt/home/jiangj33/anaconda3/envs/eeg/bin/python"

# ── 工具函数 ──────────────────────────────────────────────────────────────

usage() {
    sed -n '3,26p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

die() { echo "ERROR: $*" >&2; exit 1; }

list_proteins() {
    local ds=$1
    local data_dir="${PROJECT_DIR}/code_data/${ds}"
    [[ -d "$data_dir" ]] || return 0
    for f in "$data_dir"/*_ca.xyzb; do
        [[ -f "$f" ]] || continue
        basename "$f" _ca.xyzb
    done
}

count_running_pcode_jobs() {
    squeue -h -u "$USER" -o "%j" 2>/dev/null | grep -c "^${JOB_PREFIX}_" || true
}

wait_for_slot() {
    local max_jobs=$1
    while true; do
        local running
        running=$(count_running_pcode_jobs)
        [[ "$running" -lt "$max_jobs" ]] && break
        echo "  ${running}/${max_jobs} pcode jobs active, waiting 30s..."
        sleep 30
    done
}

submit_one() {
    local ds=$1 protein=$2
    protein=$(echo "$protein" | tr '[:lower:]' '[:upper:]')
    local job_name="${JOB_PREFIX}_${ds}_${protein}"

    sbatch \
        --partition="$PARTITION" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --job-name="$job_name" \
        --output="${LOGDIR}/${ds}_${protein}_%j.out" \
        "$WORKER" "$ds" "$protein"
}

collect_job_ids() {
    # 参数: [dataset] [protein]
    local ds=${1:-} protein=${2:-}
    squeue -h -u "$USER" -o "%i %j" 2>/dev/null | while read -r jid jname; do
        [[ "$jname" == ${JOB_PREFIX}_* ]] || continue
        if [[ -z "$ds" ]]; then
            echo "$jid"
        elif [[ -z "$protein" && "$jname" == ${JOB_PREFIX}_${ds}_* ]]; then
            echo "$jid"
        elif [[ "$jname" == "${JOB_PREFIX}_${ds}_${protein}" ]]; then
            echo "$jid"
        fi
    done
}

# ── status ────────────────────────────────────────────────────────────────

cmd_status() {
    echo "========== MSU HPPC 集群概览 =========="
    echo "Cluster: msuhpcc"
    echo ""
    echo "--- CPU 分区（推荐 Pcode 使用）---"
    sinfo -p general-short,general-long,scavenger,general-long-bigmem \
        -o "%P %a %D %c %m %C %l" 2>/dev/null || true
    echo ""
    echo "  general-short  : 最长 4 小时，适合烟雾测试"
    echo "  general-long   : 最长 7 天，适合完整蛋白扫描（默认）"
    echo "  scavenger      : 最长 7 天，低优先级抢占式"
    echo "  general-long-bigmem : 大内存节点（>1TB）"
    echo ""
    echo "  申请资源: --cpus-per-task=N --mem=XG --time=HH:MM:SS --partition=NAME"
    echo "  默认配置: partition=${PARTITION} cpus=${CPUS} mem=${MEM} time=${TIME}"
    echo ""
    echo "--- Conda 环境 ---"
    echo "  激活脚本: ${CONDA_SH}"
    echo "  环境名:   ${CONDA_ENV}"
    echo "  Python:   ${PYTHON}"
    if [[ -x "$PYTHON" ]]; then
        echo "  版本:     $($PYTHON --version 2>&1)"
    else
        echo "  状态:     NOT FOUND"
    fi
    echo ""
    echo "--- 当前用户 Pcode 任务 (${USER}) ---"
    local n
    n=$(count_running_pcode_jobs)
    if [[ "$n" -eq 0 ]]; then
        echo "  (无运行中/排队中的 pcode 任务)"
    else
        squeue -u "$USER" -o "%.10i %.14P %.9j %.2t %.10M %.6D %R" \
            | grep -E "JOBID|${JOB_PREFIX}_" || true
    fi
    echo ""
    echo "--- 项目路径 ---"
    echo "  ${PROJECT_DIR}"
}

# ── submit ────────────────────────────────────────────────────────────────

cmd_submit() {
    local ds protein
    ds=${1:?用法: pcode_job.sh submit <dataset> [protein]}
    protein=${2:-}

    mkdir -p "$LOGDIR"
    [[ -x "$WORKER" || -f "$WORKER" ]] || die "worker not found: $WORKER"

    if [[ -n "$protein" ]]; then
        echo "Submit: ${ds}/${protein}  [${PARTITION} ${CPUS}cpu ${MEM} ${TIME}]"
        submit_one "$ds" "$protein"
        return
    fi

    local proteins=()
    while IFS= read -r p; do proteins+=("$p"); done < <(list_proteins "$ds")
    [[ ${#proteins[@]} -gt 0 ]] || die "no proteins in code_data/${ds}"

    echo "Submit dataset ${ds}: ${#proteins[@]} proteins"
    local i=0
    for p in "${proteins[@]}"; do
        wait_for_slot "$MAX_CONCURRENT"
        submit_one "$ds" "$p"
        i=$((i + 1))
        echo "  [${i}/${#proteins[@]}] ${ds}/${p}"
        sleep 1
    done
    echo "Done: ${i} jobs submitted. Logs: ${LOGDIR}"
}

cmd_submit_all() {
    local max_jobs=${1:-$MAX_CONCURRENT}
    MAX_CONCURRENT=$max_jobs
    echo "Submit-all: datasets=${DATASETS[*]} max_concurrent=${max_jobs}"
    for ds in "${DATASETS[@]}"; do
        [[ -d "${PROJECT_DIR}/code_data/${ds}" ]] || { echo "SKIP ${ds}: no data dir"; continue; }
        cmd_submit "$ds"
    done
}

# ── stop ──────────────────────────────────────────────────────────────────

cmd_stop() {
    if [[ $# -eq 0 ]]; then
        usage 1
    fi

    local ids=()

    if [[ "$1" == "--all" ]]; then
        while IFS= read -r jid; do ids+=("$jid"); done < <(collect_job_ids)
    elif [[ "$1" =~ ^[0-9]+$ ]]; then
        ids=("$@")
    else
        local ds=$1 protein=${2:-}
        if [[ -n "$protein" ]]; then
            protein=$(echo "$protein" | tr '[:lower:]' '[:upper:]')
        fi
        while IFS= read -r jid; do ids+=("$jid"); done < <(collect_job_ids "$ds" "$protein")
    fi

    if [[ ${#ids[@]} -eq 0 ]]; then
        echo "No matching pcode jobs to cancel."
        return 0
    fi

    echo "Cancelling ${#ids[@]} job(s): ${ids[*]}"
    scancel "${ids[@]}"
    echo "Done."
}

# ── main ──────────────────────────────────────────────────────────────────

main() {
    local cmd=${1:-}
    shift || true

    case "$cmd" in
        status)     cmd_status ;;
        submit)     cmd_submit "$@" ;;
        submit-all) cmd_submit_all "$@" ;;
        stop)       cmd_stop "$@" ;;
        -h|--help|help|"") usage 0 ;;
        *) die "unknown command: $cmd (try: status | submit | submit-all | stop)" ;;
    esac
}

main "$@"
