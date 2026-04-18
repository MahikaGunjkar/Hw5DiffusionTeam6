#!/bin/bash
# Submit training experiments from configs/ablations/exp??.yaml.
#
# Each experiment is a single YAML file containing both training deltas and
# a ``slurm:`` block with node type / GPU count / wall-clock time. This script
# loops over the requested experiment IDs and fires one ``sbatch`` per row.
#
# Usage:
#   scripts/submit.sh                  # submit EVERY configs/ablations/exp??.yaml
#   scripts/submit.sh 03               # submit only exp03.yaml
#   scripts/submit.sh 03 06 09         # specific list
#   scripts/submit.sh --range 01-05    # range expansion
#   scripts/submit.sh --dry-run 03     # print sbatch command, do not submit
#   scripts/submit.sh --export-extra WANDB_RUN_ID=exp03-20260418,WANDB_RESUME=must 03

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASE_CFG="${REPO_ROOT}/configs/ddpm.yaml"
ABL_DIR="${REPO_ROOT}/configs/ablations"
SLURM_TEMPLATE="${SCRIPT_DIR}/run.slurm"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# Source env_ocean.sh on PSC login node so `python` points to conda env
# (ruamel.yaml parsing in _slurm_field below needs it). No-op on local dev.
if [ -f "${SCRIPT_DIR}/env_ocean.sh" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/env_ocean.sh"
fi

# Python binary: prefer conda env, fall back to python3 for local dev.
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"

DRY_RUN=0
IDS=()
EXTRA_EXPORTS=""

# ── Parse flags ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --range)
            IFS='-' read -r lo hi <<< "$2"
            lo_n=$((10#${lo}))
            hi_n=$((10#${hi}))
            for i in $(seq "${lo_n}" "${hi_n}"); do
                IDS+=("$(printf "%02d" "${i}")")
            done
            shift 2
            ;;
        --export-extra)
            EXTRA_EXPORTS=",${2}"
            shift 2
            ;;
        -h|--help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            IDS+=("$1")
            shift
            ;;
    esac
done

# ── If no ids given, take every yaml file under the ablation dir ──
if [[ ${#IDS[@]} -eq 0 ]]; then
    for f in "${ABL_DIR}"/exp*.yaml; do
        fn=$(basename "${f}" .yaml)      # "exp03"
        IDS+=("${fn#exp}")
    done
fi

# Sort for consistent ordering
IFS=$'\n' IDS=($(sort -u <<<"${IDS[*]}")); unset IFS

echo "Submitting experiments: ${IDS[*]}"
echo ""

# ── Tiny python helper to extract slurm block from yaml ──
_slurm_field() {
    # usage: _slurm_field yaml_path key
    "${PYTHON_BIN}" - <<PY
from ruamel.yaml import YAML
d = YAML().load(open("$1"))
print(d['slurm']['$2'])
PY
}

# ── Loop and submit ──
for exp_id in "${IDS[@]}"; do
    yaml_path="${ABL_DIR}/exp${exp_id}.yaml"
    if [[ ! -f "${yaml_path}" ]]; then
        echo "[WARN] exp ${exp_id}: ${yaml_path} not found, skipping"
        continue
    fi

    # preflight 检查（dry-run 下也执行，提前暴露问题）
    if ! bash "${SCRIPT_DIR}/preflight.sh" "${exp_id}"; then
        echo "[ABORT] exp ${exp_id}: preflight 失败，中止提交"
        exit 1
    fi

    node=$(_slurm_field "${yaml_path}" node)
    gpus=$(_slurm_field "${yaml_path}" gpus)
    time=$(_slurm_field "${yaml_path}" time)

    JOB_NAME="ot_exp${exp_id}"
    STDOUT="${LOG_DIR}/exp${exp_id}_%j.out"
    STDERR="${LOG_DIR}/exp${exp_id}_%j.err"

    # PSC allocation: charge to cis260133p (Diffusion Group 6), NOT user's default cis250019p
    PSC_ACCOUNT="${PSC_ACCOUNT:-cis260133p}"

    CMD=(
        sbatch
        --account="${PSC_ACCOUNT}"
        --job-name="${JOB_NAME}"
        --gpus="${node}:${gpus}"
        --ntasks-per-node="${gpus}"
        --time="${time}"
        --output="${STDOUT}"
        --error="${STDERR}"
        --export="ALL,EXP_ID=${exp_id},NUM_GPUS=${gpus},BASE_CFG=${BASE_CFG},EXP_CFG=${yaml_path}${EXTRA_EXPORTS}"
        "${SLURM_TEMPLATE}"
    )

    printf "[exp %s] %-10s x%s  %s   %s\n" "${exp_id}" "${node}" "${gpus}" "${time}" "$(basename "${yaml_path}")"

    if [[ ${DRY_RUN} -eq 1 ]]; then
        printf "  DRY: %s\n\n" "${CMD[*]}"
    else
        "${CMD[@]}"
        echo ""
    fi
done
