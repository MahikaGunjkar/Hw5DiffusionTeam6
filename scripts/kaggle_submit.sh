#!/usr/bin/env bash
# scripts/kaggle_submit.sh — Submit a CSV to the Kaggle InClass competition.
# Competition slug is hardcoded because the InClass comp is unlisted and
# cannot be discovered via `kaggle competitions list` search.
#
# Usage:
#   scripts/kaggle_submit.sh <csv_path> "<message>"
#   scripts/kaggle_submit.sh submission_exp01.csv "exp01 latest.pth, DiT-XL + OT + AdaMask, Apr 23"
#
# Prereqs: kaggle CLI installed + ~/.kaggle/kaggle.json configured.

set -euo pipefail

readonly KAGGLE_COMP="11685s26-diffusion"
readonly KAGGLE_URL="https://www.kaggle.com/competitions/${KAGGLE_COMP}"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <csv_path> \"<message>\"" >&2
    echo "Example: $0 submission_exp01.csv \"exp01 latest.pth, DiT-XL + OT + AdaMask\"" >&2
    exit 1
fi

CSV="$1"
MSG="$2"

[ -f "$CSV" ] || { echo "[FATAL] csv not found: $CSV" >&2; exit 1; }
command -v kaggle >/dev/null || { echo "[FATAL] kaggle CLI not in PATH" >&2; exit 1; }

echo "[INFO] competition: $KAGGLE_URL"
echo "[INFO] csv:         $CSV ($(wc -c < "$CSV") bytes)"
echo "[INFO] message:     $MSG"

kaggle competitions submit -c "$KAGGLE_COMP" -f "$CSV" -m "$MSG"

echo ""
echo "[INFO] recent submissions:"
kaggle competitions submissions -c "$KAGGLE_COMP" 2>&1 | head -8
echo ""
echo "[OK] leaderboard → ${KAGGLE_URL}/leaderboard"
