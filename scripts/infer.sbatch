#!/bin/bash
# scripts/infer.sbatch — single-GPU inference + Kaggle submission CSV
#
# Submit:
#   sbatch --account=cis260133p --gpus=h100-80:1 --job-name=infer_exp15 \
#       --output=logs/infer_exp15_%j.out --error=logs/infer_exp15_%j.err \
#       --export=ALL,EXP_ID=15 scripts/infer.sbatch
#
# Optional overrides via --export=ALL,EXP_ID=15,CKPT=/abs/path,EXTRA_ARGS=...
#
# Output: $PROJ_ROOT/inference/exp${EXP_ID}_${SLURM_JOB_ID}/
#   ├── generated_images/  (5000 png)
#   └── submission.csv     (upload to Kaggle)

#SBATCH -p GPU-shared
#SBATCH -t 01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env_ocean.sh"
cd "${REPO_DIR}"

EXP_ID="${EXP_ID:?must export EXP_ID, e.g. 15}"
CKPT="${CKPT:-${PROJ_ROOT}/outputs/exp${EXP_ID}/exp-0-exp${EXP_ID}/checkpoints/latest.pth}"
DATA_REF="${DATA_REF:-${PROJ_ROOT}/imagenet100_extracted/imagenet100_128x128}"
OUT_DIR="${PROJ_ROOT}/inference/exp${EXP_ID}_${SLURM_JOB_ID}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${OUT_DIR}"

echo "===================================================="
echo "  Inference: exp${EXP_ID}"
echo "  CKPT:      ${CKPT}"
echo "  DATA_REF:  ${DATA_REF}"
echo "  OUT_DIR:   ${OUT_DIR}"
echo "  Extra:     ${EXTRA_ARGS}"
echo "===================================================="

# Sanity: ckpt exists, data exists
[ -f "${CKPT}" ] || { echo "[FATAL] ckpt not found: ${CKPT}" >&2; exit 1; }
[ -d "${DATA_REF}" ] || { echo "[FATAL] data_dir not found: ${DATA_REF}" >&2; exit 1; }

# python -u → unbuffered stdout so the .out log is real-time
python -u inference.py \
    --config "configs/ddpm.yaml" "configs/ablations/exp${EXP_ID}.yaml" \
    --ckpt "${CKPT}" \
    --output_dir "${OUT_DIR}" \
    --data_dir "${DATA_REF}" \
    ${EXTRA_ARGS}

EXIT_CODE=$?

echo ""
echo "===================================================="
echo "  Inference exit: ${EXIT_CODE}"
echo "===================================================="
ls -la "${OUT_DIR}/" || true
if [ -f "${OUT_DIR}/submission.csv" ]; then
    SIZE=$(stat -c%s "${OUT_DIR}/submission.csv")
    echo "[OK] submission.csv ready (${SIZE} bytes) → ${OUT_DIR}/submission.csv"
fi

exit ${EXIT_CODE}
