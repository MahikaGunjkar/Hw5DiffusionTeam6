#!/usr/bin/env bash
# env_ocean.sh — PSC ocean 环境变量 14 件套
# 可重复 source，幂等无副作用。
# 用法：source "$(dirname "$0")/env_ocean.sh"

# ——— 项目根（先定义一次，其余全引用）———
export PROJ_ROOT=/ocean/projects/cis260133p/zjiang9
export REPO_DIR=$PROJ_ROOT/Hw5DiffusionTeam6

# ——— Wandb 七件套（缺一个漏一处）———
export WANDB_DIR=$PROJ_ROOT/wandb/run
export WANDB_CACHE_DIR=$PROJ_ROOT/wandb/cache
export WANDB_DATA_DIR=$PROJ_ROOT/wandb/data
export WANDB_ARTIFACT_DIR=$PROJ_ROOT/wandb/artifacts
export WANDB_CONFIG_DIR=$PROJ_ROOT/wandb/config
export XDG_DATA_HOME=$PROJ_ROOT/xdg_data

# ——— HuggingFace / Torch / pip / uv cache ———
export HF_HOME=$PROJ_ROOT/hf_cache
export HF_DATASETS_CACHE=$PROJ_ROOT/hf_cache/datasets
export HF_HUB_CACHE=$PROJ_ROOT/hf_cache/hub          # HF Hub model cache（新版 HF 推荐变量）
export TRANSFORMERS_CACHE=$PROJ_ROOT/hf_cache/transformers
export TORCH_HOME=$PROJ_ROOT/torch_cache
export PIP_CACHE_DIR=$PROJ_ROOT/pip_cache
export UV_CACHE_DIR=$PROJ_ROOT/uv_cache

# ——— Wandb 凭证与团队 identity ———
export WANDB_API_KEY="${WANDB_API_KEY:-}"                    # 从 .env.local 或 wandb login 来
export WANDB_ENTITY="${WANDB_ENTITY:-idl-project-mm}"        # CMU 11-785 team entity
export WANDB_PROJECT="${WANDB_PROJECT:-ddpm}"                # 保持团队现有 dashboard
export WANDB_RESUME="${WANDB_RESUME:-allow}"                 # sbatch --export 可覆盖
# WANDB_RUN_ID 不在这里设默认——由 run.slurm 在知道 EXP_ID 后生成

# ——— 项目输出 ———
export CKPT_DIR=$PROJ_ROOT/checkpoints
export OUTPUT_DIR=$PROJ_ROOT/outputs
export TMPDIR=$PROJ_ROOT/tmp

# ——— Pending Slurm job runtime overrides（2026-04-23）———
# Used only for already-queued jobs that must not be cancelled before deadline.
case "${SLURM_JOB_ID:-}" in
    40151101)
        export WANDB_RESUME=never
        echo "NOTICE: job 40151101 starts exp22 from scratch (WANDB_RESUME=never)."
        ;;
    40151103)
        export EXP_ID=21
        export NUM_GPUS=8
        export EXP_CFG="$REPO_DIR/configs/ablations/exp21_8gpu.yaml"
        export WANDB_RESUME=never
        export WANDB_RUN_ID=exp21-40151103
        echo "NOTICE: job 40151103 keeps Slurm name/log prefix exp23, but train.py will run exp21 on 8 GPUs from scratch."
        ;;
esac

# ——— 幂等 mkdir（不存在才创建，不报错）———
mkdir -p \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" \
    "$WANDB_CONFIG_DIR" "$XDG_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
    "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" "$CKPT_DIR" "$OUTPUT_DIR" "$TMPDIR" 2>/dev/null || true

# ——— Conda env 激活（PSC bridges2）———
# PSC AI/pytorch module uses CUDA 10.1 (too old for H100). We build our own
# env on ocean: python 3.11 + torch 2.5.1 + cu121 + diffusers + wandb + FID.
# Path is pre-populated by scripts/psc_conda_setup.sh (one-time bootstrap).
export PSC_CONDA_ENV="${PSC_CONDA_ENV:-$PROJ_ROOT/envs/hw5diff}"
export PSC_CONDA_BASE="${PSC_CONDA_BASE:-/opt/packages/anaconda3-2024.10-1}"

# Only activate on PSC (has the anaconda3 install). Skip gracefully on local dev.
if [ -f "$PSC_CONDA_BASE/etc/profile.d/conda.sh" ] && [ -d "$PSC_CONDA_ENV" ]; then
    # shellcheck disable=SC1091
    source "$PSC_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$PSC_CONDA_ENV"
fi

# ——— NCCL watchdog hardening (multi-GPU 训练必加) ———
# 一个 rank 抛异常 → 全部 rank 同步 abort，避免某一个 rank 死锁拖垮整节点
# (默认行为是只挂掉一个 rank，其他 rank 在 all_reduce 上无限等待，浪费 GPU 小时)
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
# Watchdog 触发时 dump 最近 NCCL ops 的 ring buffer，方便事后定位是哪个 collective 卡的
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000}"
