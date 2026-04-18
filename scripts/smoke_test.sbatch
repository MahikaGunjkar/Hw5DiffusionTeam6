#!/bin/bash
#SBATCH --job-name=ot_smoke
#SBATCH --partition=GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:45:00
#SBATCH --account=cis260133p
#SBATCH --output=/ocean/projects/cis260133p/zjiang9/Hw5DiffusionTeam6/logs/smoke_%j.out
#SBATCH --error=/ocean/projects/cis260133p/zjiang9/Hw5DiffusionTeam6/logs/smoke_%j.err

# Smoke test: 1 epoch of exp15 (U-Net Flow+OT) on 1×V100-32 to verify the
# end-to-end pipeline — env, conda, data, wandb auth (team entity), model
# build, 1 epoch train, checkpoint save, wandb log_artifact. NOT a real run.

set -euo pipefail

# SLURM copies sbatch script to /var/spool/slurm/d/job<id>/ before exec, so
# $0 is NOT the original path. Use SLURM_SUBMIT_DIR (user's CWD at sbatch time)
# with a hardcoded fallback to the repo on ocean.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/ocean/projects/cis260133p/zjiang9/Hw5DiffusionTeam6}"
source "${REPO_ROOT}/scripts/env_ocean.sh"

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
    echo "FATAL: wandb not authenticated" >&2
    exit 1
fi

module load cuda/12.6.1 2>/dev/null || true
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-5}

DATA_TAR="${DATA_TAR:-/ocean/projects/cis260133p/zjiang9/imagenet100_128x128.tar.gz}"
EXTRACT_DIR="${EXTRACT_DIR:-$PROJ_ROOT/imagenet100_extracted}"
BASE_CFG="$REPO_DIR/configs/ddpm.yaml"
EXP_CFG="$REPO_DIR/configs/ablations/exp15.yaml"
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

cd "$REPO_DIR"

# Unique run_id so smoke runs don't collide with real exp15 runs
SMOKE_RUN_ID="smoke-$(date +%Y%m%d-%H%M%S)"

echo "=========================================================="
echo "SMOKE TEST — 1 epoch, V100-32, exp15 (U-Net Flow+OT)"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : ${SLURM_JOB_NODELIST}"
echo "Wandb run  : ${SMOKE_RUN_ID}"
echo "Entity     : ${WANDB_ENTITY}"
echo "Project    : ${WANDB_PROJECT}"
echo "=========================================================="

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --master_port=${MASTER_PORT} \
    train.py \
    --config "${BASE_CFG}" "${EXP_CFG}" \
    --tar_path "${DATA_TAR}" \
    --extract_dir "${EXTRACT_DIR}" \
    --run_name "smoke" \
    --output_dir "${OUTPUT_DIR}/smoke" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_id "${SMOKE_RUN_ID}" \
    --wandb_resume "never" \
    --num_workers 4 \
    --num_epochs 1 \
    --fid_every_n_epochs 0 \
    --save_every_n_epochs 1
EXIT_CODE=$?

echo "=========================================================="
echo "SMOKE exit=${EXIT_CODE}"
echo "=========================================================="

# Upload offline wandb transactions (belt-and-suspenders)
if [ -d "${WANDB_DIR}" ]; then
    wandb sync --sync-all "${WANDB_DIR}" 2>&1 || true
fi

exit ${EXIT_CODE}
