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

echo "=========================================================="
echo "Exp ID         : ${EXP_ID}"
echo "Num GPUs       : ${NUM_GPUS}"
echo "Base config    : ${BASE_CFG}"
echo "Exp  config    : ${EXP_CFG}"
echo "Slurm job id   : ${SLURM_JOB_ID}"
echo "Slurm node(s)  : ${SLURM_JOB_NODELIST}"
echo "=========================================================="

module load cuda nccl 2>/dev/null || true
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-5}

# Edit these paths for your PSC account before first submission
DATA_TAR="${DATA_TAR:-/ocean/projects/XXXXXX/shared/imagenet100_128x128.tar.gz}"
EXTRACT_DIR="${EXTRACT_DIR:-$PROJECT/imagenet100_extracted}"

# Unique master port per job to avoid DDP rendezvous collisions
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

# Launch — `--config BASE EXP` means EXP overrides BASE field-wise.
torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=1 \
    --master_port=${MASTER_PORT} \
    train.py \
    --config "${BASE_CFG}" "${EXP_CFG}" \
    --tar_path "${DATA_TAR}" \
    --extract_dir "${EXTRACT_DIR}" \
    --run_name "exp${EXP_ID}"

echo "=========================================================="
echo "Exp ${EXP_ID} finished with exit code $?"
echo "=========================================================="
