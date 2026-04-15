#!/bin/bash
# Run inference-only ablation variants against trained checkpoints.
#
# Inference experiments (Exp 10-13, 17-18 in the plan) reuse a trained
# checkpoint with different sampler / CFG / NFE settings. These are cheap
# compared to training, so we run them on a single interactive GPU rather
# than occupying a full Slurm node.
#
# Usage (from the repo root):
#
#   # DiT-B checkpoint sweep (Exp 10, 11, 12, 13)
#   scripts/infer_sweep.sh outputs/exp-XX-exp03/checkpoints/ckpt_final.pt dit_b
#
#   # UNet checkpoint sweep (Exp 17, 18)
#   scripts/infer_sweep.sh outputs/exp-XX-exp15/checkpoints/ckpt_final.pt unet
#   scripts/infer_sweep.sh outputs/exp-XX-exp16/checkpoints/ckpt_final.pt unet

set -euo pipefail

CKPT="${1:?usage: $0 CKPT_PATH MODEL_TYPE}"
MODEL_TYPE="${2:?usage: $0 CKPT_PATH MODEL_TYPE}"

run_infer() {
    local label="$1"
    shift
    echo "=========================================================="
    echo "[${label}] extra args: $*"
    echo "=========================================================="
    python inference.py \
        --config configs/ddpm.yaml \
        --ckpt "${CKPT}" \
        --model_type "${MODEL_TYPE}" \
        --run_name "${label}" \
        "$@"
}

if [[ "${MODEL_TYPE}" == dit* ]]; then
    # Exp 10 — Heun solver
    run_infer "exp10_heun"      --framework flow_matching --flow_solver heun --num_inference_steps_flow 10

    # Exp 11 — extreme 4-step sampling
    run_infer "exp11_nfe4"      --framework flow_matching --flow_solver heun --num_inference_steps_flow 4

    # Exp 12 — weak CFG
    run_infer "exp12_cfg1p5"    --framework flow_matching --flow_solver euler --num_inference_steps_flow 10 --cfg_guidance_scale 1.5

    # Exp 13 — strong CFG
    run_infer "exp13_cfg4p5"    --framework flow_matching --flow_solver euler --num_inference_steps_flow 10 --cfg_guidance_scale 4.5

elif [[ "${MODEL_TYPE}" == "unet" ]]; then
    # Exp 17 / 18 — NFE=10 on RF UNet baselines
    run_infer "unet_rf_nfe10"   --framework flow_matching --flow_solver euler --num_inference_steps_flow 10

else
    echo "[ERR] model_type '${MODEL_TYPE}' not handled. Use dit_* or unet."
    exit 1
fi
