#!/bin/bash
#SBATCH -p GPU
#SBATCH -N 1
#SBATCH --cpus-per-task=5
#SBATCH -J otadamask
# Per-job resources (--gpus, --time, --job-name, --output, --export) are
# injected by scripts/submit.sh based on the experiment's YAML slurm block.
#
# Env vars required (set by submit.sh via --export):
#   EXP_ID    — two-digit experiment id (e.g. "03")
#   NUM_GPUS  — GPUs per node (4 or 8)
#   BASE_CFG  — path to configs/ddpm.yaml
#   EXP_CFG   — path to configs/ablations/expXX.yaml

set -euo pipefail

# ——— PSC ocean 环境变量（必须在 module load 之前 source）———
source "$(dirname "$0")/env_ocean.sh"

# Pre-flight: verify wandb auth BEFORE allocating GPU time (plan 决策 5.1)
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    echo "FATAL: wandb not authenticated. Set WANDB_API_KEY or run 'wandb login' first." >&2
    exit 1
fi

echo "=========================================================="
echo "Exp ID         : ${EXP_ID}"
echo "Num GPUs       : ${NUM_GPUS}"
echo "Base config    : ${BASE_CFG}"
echo "Exp  config    : ${EXP_CFG}"
echo "Slurm job id   : ${SLURM_JOB_ID}"
echo "Slurm node(s)  : ${SLURM_JOB_NODELIST}"
echo "=========================================================="

# CUDA 12.x for H100 (sm_90). PyTorch 2.5.1+cu121 wheel bundles its own
# NCCL so no system nccl module needed — avoids version mismatch.
module load cuda/12.6.1 2>/dev/null || true
export NCCL_DEBUG=WARN
# NVLink SHARP (NVLS) — defensive: auto-enabled on NCCL >= 2.17 + H100 NVSwitch
export NCCL_NVLS_ENABLE=1
# NOTE: NCCL auto-detects NVLink topology on h100-80; no NCCL_P2P_LEVEL override needed
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Edit these paths for your PSC account before first submission
DATA_TAR="${DATA_TAR:-/ocean/projects/cis260133p/zjiang9/imagenet100_128x128.tar.gz}"
EXTRACT_DIR="${EXTRACT_DIR:-$PROJ_ROOT/imagenet100_extracted}"

# Unique master port per job to avoid DDP rendezvous collisions
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# 确保 CWD 在 repo 根目录（sbatch 可能以其他目录启动）
cd "${REPO_DIR:-$(pwd)}"

# Fixed run id enables wandb auto-resume on re-submit (plan 决策 3)
: "${WANDB_RUN_ID:=exp${EXP_ID}-$(date +%Y%m%d)}"
export WANDB_RUN_ID

# Launch — `--config BASE EXP` means EXP overrides BASE field-wise.
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_port=${MASTER_PORT} \
    train.py \
    --config "${BASE_CFG}" "${EXP_CFG}" \
    --tar_path "${DATA_TAR}" \
    --extract_dir "${EXTRACT_DIR}" \
    --run_name "exp${EXP_ID}" \
    --output_dir "${OUTPUT_DIR}/exp${EXP_ID}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_id "${WANDB_RUN_ID}" \
    --wandb_resume "${WANDB_RESUME}" \
    --num_workers 8 \
    --save_every_n_epochs 5
EXIT_CODE=$?

# 失败留痕：分 3 档处理退出码
if [ $EXIT_CODE -eq 42 ]; then
    echo "FATAL: NaN detected — NOT retriable" > "${OUTPUT_DIR}/exp${EXP_ID}_fail.flag"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "FAILED exit=$EXIT_CODE (retriable)" > "${OUTPUT_DIR}/exp${EXP_ID}_fail.flag"
fi

# Fallback: upload offline wandb transactions (rules/ml-training-wandb.md §3)
# Wandb run dirs are named `run-<timestamp>-<run_id>`, so glob-sync the whole WANDB_DIR
if [ -d "${WANDB_DIR}" ]; then
    wandb sync --sync-all "${WANDB_DIR}" 2>&1 || true
fi

echo "=========================================================="
echo "Exp ${EXP_ID} finished with exit code ${EXIT_CODE}"
echo "=========================================================="
exit ${EXIT_CODE}
