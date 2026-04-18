#!/usr/bin/env bash
# preflight.sh — 提交前健康检查
# 用法：bash scripts/preflight.sh <exp_id>
# 返回值：任一 FAIL 则 exit 1；WARN/SKIP 继续

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ID="${1:-}"

if [[ -z "$EXP_ID" ]]; then
    echo "[FAIL] 缺少参数: bash preflight.sh <exp_id>"
    exit 1
fi

FAIL=0

# 检测是否在 PSC（有 SLURM_CLUSTER_NAME 或 ocean 路径存在）
_on_psc() {
    [[ -n "${SLURM_CLUSTER_NAME:-}" ]] || [[ -d "/ocean/projects/cis260133p" ]]
}

# ── 检查 1: env_ocean.sh 存在且可 source ──
ENV_SCRIPT="$SCRIPT_DIR/env_ocean.sh"
if [[ -f "$ENV_SCRIPT" ]]; then
    if bash -c "source '$ENV_SCRIPT'" 2>/dev/null; then
        echo "[PASS] env_ocean.sh 存在且可正常 source"
    else
        echo "[FAIL] env_ocean.sh source 失败，请检查语法"
        FAIL=1
    fi
else
    echo "[FAIL] env_ocean.sh 不存在: $ENV_SCRIPT"
    FAIL=1
fi

# ── 检查 2: configs/ablations/expXX.yaml 存在 ──
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YAML_PATH="$REPO_ROOT/configs/ablations/exp${EXP_ID}.yaml"
if [[ -f "$YAML_PATH" ]]; then
    echo "[PASS] 实验配置存在: configs/ablations/exp${EXP_ID}.yaml"
else
    echo "[FAIL] 实验配置不存在: $YAML_PATH"
    FAIL=1
fi

# ── 检查 3: PSC 专属检查（非 PSC 环境跳过）──
if _on_psc; then
    # 3a: DATA_TAR 文件存在
    DATA_TAR="${DATA_TAR:-/ocean/projects/cis260133p/zjiang9/imagenet100_128x128.tar.gz}"
    if [[ -f "$DATA_TAR" ]]; then
        echo "[PASS] DATA_TAR 存在: $DATA_TAR"
    else
        echo "[FAIL] DATA_TAR 不存在: $DATA_TAR"
        FAIL=1
    fi

    # 3b: ocean 输出目录可写
    OCEAN_DIR="/ocean/projects/cis260133p/zjiang9"
    if [[ -d "$OCEAN_DIR" ]] && touch "$OCEAN_DIR/.write_test" 2>/dev/null; then
        rm -f "$OCEAN_DIR/.write_test"
        echo "[PASS] ocean 目录可写: $OCEAN_DIR"
    else
        echo "[WARN] ocean 目录不可写或不存在: $OCEAN_DIR (可能权限问题)"
    fi
else
    echo "[SKIP] 非 PSC 环境，跳过 DATA_TAR / ocean 可写性检查"
fi

# ── 检查 4: wandb 可导入 + 凭证就绪 ──
if python3 -c "import wandb" 2>/dev/null; then
    echo "[PASS] wandb Python 包可导入"
    if [ -n "${WANDB_API_KEY:-}" ] || [ -f "$HOME/.netrc" ]; then
        echo "[PASS] wandb 凭证就绪 (env or ~/.netrc)"
    else
        echo "[WARN] 未检测到 WANDB_API_KEY 或 ~/.netrc，运行时会 fail-fast"
    fi
else
    echo "[WARN] wandb 包未装，pip install wandb 后重试"
fi

# ── 汇总 ──
if [[ $FAIL -ne 0 ]]; then
    echo ""
    echo "[PREFLIGHT FAILED] 请修复上述 FAIL 项后重试"
    exit 1
fi

echo ""
echo "[PREFLIGHT PASSED] 所有检查通过，可以提交"
exit 0
