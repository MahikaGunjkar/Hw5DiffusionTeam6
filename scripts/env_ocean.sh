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

# ——— 幂等 mkdir（不存在才创建，不报错）———
mkdir -p \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" \
    "$WANDB_CONFIG_DIR" "$XDG_DATA_HOME" "$HF_HOME" "$HF_DATASETS_CACHE" \
    "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR" \
    "$UV_CACHE_DIR" "$CKPT_DIR" "$OUTPUT_DIR" "$TMPDIR" 2>/dev/null || true
