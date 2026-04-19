#!/bin/bash
# scripts/gen_submission.sbatch — H100 sbatch for generate_submission.py
# Use when inference.py crashed at the CSV step but the 5000 PNGs already
# exist on disk. Saves ~12 min of regenerating images.
#
# Submit:
#   sbatch --account=cis260133p --gpus=h100-80:1 --time=00:30:00 \
#       --output=logs/gencsv_%j.out --error=logs/gencsv_%j.err \
#       --export=ALL,IMG_DIR=/abs/path/to/generated_images,OUT_CSV=/abs/path/to/submission.csv \
#       scripts/gen_submission.sbatch

#SBATCH -p GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5

set -euo pipefail
# $0 / BASH_SOURCE points to /var/spool/slurm/... after SLURM copies the
# script. Use SLURM_SUBMIT_DIR (set by sbatch) to find env_ocean.sh.
REPO_ROOT="${SLURM_SUBMIT_DIR:-/ocean/projects/cis260133p/zjiang9/Hw5DiffusionTeam6}"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/env_ocean.sh"
cd "${REPO_DIR}"

IMG_DIR="${IMG_DIR:?must export IMG_DIR=/abs/path/to/generated_images}"
OUT_CSV="${OUT_CSV:?must export OUT_CSV=/abs/path/to/submission.csv}"

[ -d "${IMG_DIR}" ] || { echo "[FATAL] IMG_DIR not found: ${IMG_DIR}" >&2; exit 1; }
N=$(ls "${IMG_DIR}" | wc -l)
echo "[INFO] images=${N} src=${IMG_DIR} out=${OUT_CSV}"

python -u generate_submission.py --image_dir "${IMG_DIR}" --output "${OUT_CSV}"

echo "[OK] $(ls -la "${OUT_CSV}")"
