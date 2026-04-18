# Ablation Study — OT-AdaMask Rectified Flow

**来源**：`Deepthink.md` Phase 4 Turn 4 / Phase 4 NeurIPS-Standard Ablation Plan（行 666-692），与当前分支 `OT-Flow` 的 OT-AdaMask Flow 方案对齐。

**算力前提**：PSC Bridges-2，**4× A100 DDP**。DiT-B 无掩码 0.8 GPUh/Epoch，50% 均值掩码 ≈ 0.4 GPUh/Epoch。

**总预算**：PSC portal 当前显示 **3,000 / 3,000 SU**（charge code `cis260133p`，End 2027-04-01）。Deepthink 按 3700 GPUh 规划，所以若全跑要**精选或压缩 epochs**。

---

## 🟣 推荐配置矩阵（NeurIPS-Level, 20 组）

| Exp | Framework | Batch OT | Model Arch (Params) | Mask Strategy | ODE Solver | NFE | Epochs | Est. GPUh | FID ↓ | IS ↑ | Scientific Objective | YAML |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | DDPM (SDE) | N/A | U-Net (~35M) | None | DDIM | 50 | 200 | ~40h | Base | Base | 石器时代基准：传统范式低效底线 | ✅ `exp19/20` (DDPM U-Net) |
| 02 | Rect. Flow | ❌ | DiT-B (~130M) | None | Euler | 50 | 200 | ~160h | Better | Better | 范式平移：流匹配 ODE + Transformer 初始红利 | ✅ `exp09` (no mask DiT-B) |
| 03 | Rect. Flow | ❌ | DiT-B (~130M) | None | Euler | 10 | 200 | (inference) | Crash | Crash | 轨迹扭曲暴露：随机匹配无法支持 10 步极低步数 | ❌ **缺** |
| 04 | Rect. Flow | ✅ OT | DiT-B (~130M) | None | Euler | 10 | 200 | ~160h | Good | Good | OT 魔法：最优传输拉直轨迹，原生突破 10 步 | ✅ `exp16→15` 方向 |
| 05 | Rect. Flow | ✅ OT | DiT-B (~130M) | None | Euler | 10 | 800 | ~640h | Drop | Drop | 痛点暴露：13w 图像无掩码疯狂吸算力必过拟合 | ❌ **缺** |
| 06 | Rect. Flow | ✅ OT | DiT-B (~130M) | Constant 50% | Euler | 10 | 800 | ~320h | Exc. | Exc. | 机制探路：固定比例掩码节省 50% 算力 | ✅ `exp05` (const 50%) |
| 07 | Rect. Flow | ✅ OT | DiT-B (~130M) | Ada 75% | Euler | 10 | 800 | ~320h | SOTA-B | SOTA-B | **核心创新**：时间自适应掩码 AdaMask | ✅ `exp03` (Ada 75%) |
| 08 | Rect. Flow | ✅ OT | DiT-B (~130M) | Ada 75% | Euler | 10 | 800 | (CFG 1.5) | Low | Low | 引导搜索：微弱 CFG 真实分布逼近 | ⚠️ 推理期参数，非独立 yaml |
| 09 | Rect. Flow | ✅ OT | DiT-B (~130M) | Ada 75% | Euler | 10 | 800 | (CFG 4.5) | V.High | Ultra | 引导搜索：强 CFG 语义保真度极限 | ⚠️ 推理期参数，非独立 yaml |
| 10 | Rect. Flow | ✅ OT | DiT-B (~130M) | Ada 75% | Heun | 10 | 800 | (inference) | Ultra | Ultra | Solver 进阶：二阶 Heun 消除 10 步截断误差 | ❌ **缺** |
| 11 | Rect. Flow | ✅ OT | DiT-B (~130M) | Ada 75% | Heun | 4 | 800 | (inference) | Exc. | Exc. | 极限挑战：4 步出图，剑指 LCM 蒸馏 | ❌ **缺** |
| 12 | Rect. Flow | ❌ | DiT-L (~400M) | None | Euler | 10 | 500 | ~1200h | Crash (OOM) | Crash | 反面教材：不用 AdaMask → OOM + 过拟合 | ❌ **缺** |
| 13 | Rect. Flow | ✅ OT | DiT-L (~400M) | Ada 75% | Heun | 10 | 500 | ~600h | SOTA-L | SOTA-L | 突破参数天花板：4 亿参数可接受区间 | ❌ **缺** |
| 14 | Rect. Flow | ✅ OT | DiT-L (~400M) | Ada 75% | Heun | 10 | 500 | (DecDepth=0) | Poor | Poor | 架构消融：浅层 Decoder 对掩码空间关键性 | ✅ `exp08` (decoder_depth=0) |
| 15 | Rect. Flow | ✅ OT | DiT-L (~400M) | Ada 75% | Heun | 10 | 800 | ~960h | Epic | Epic | 算力爬坡：大参数长期训练泛化潜能 | ✅ `exp02` (DiT-L Ada 75%) |
| 16 | Rect. Flow | ✅ OT | DiT-L (~400M) | Ada 90% | Heun | 10 | 800 | ~700h | Mod | Mod | 极限掩码探底：丢弃 90% 的崩塌边界 | ✅ `exp07` (Ada 90% 但 DiT-B) |
| 17 | Rect. Flow | ❌ | DiT-XL (~675M) | None | Heun | 10 | 100 | ~400h | Poor | Poor | XL 级无 Mask 基线：证明传统路线走不通 | ❌ **缺** |
| 18 | Rect. Flow | ✅ OT | DiT-XL (~675M) | Ada 75% | Heun | 10 | 200 | ~400h | Exc. | Exc. | The Giant Run：进入大厂 Foundation 区间 | ❌ **缺** |
| 19 | Rect. Flow | ✅ OT | DiT-XL (~675M) | Ada 75% | Heun | 10 | 400 | ~800h | Ultra SOTA | Ultra SOTA | 算力燃烧 A：半 B 级模型绝对统治力 | ⚠️ 部分对齐 `exp01` |
| 20 | **OT-Ada-RF** | ✅ OT | DiT-XL (~675M) | Ada 75% | Heun | 10 | 600 | ~1200h | 🏆 THE SOTA | 🏆 THE SOTA | 终极一战：烧光 3700h 建立 Kaggle 护城河 | ✅ `exp01` (DiT-XL 400 ep 对齐) |

**图例**：
- ✅ 已有 yaml（可能 epochs/mask 参数与 Deepthink 不完全一致，见右列备注）
- ⚠️ 推理期搜索（不需要独立训练 yaml，用同一 ckpt 跑多次 `infer_sweep.sh`）
- ❌ 缺 yaml，需补建

---

## 📋 现状盘点（与磁盘对账）

**磁盘现状**：`configs/ablations/` 存在 13 个 yaml：01-09, 15, 16, 19, 20。

### 命名对齐说明

Deepthink Table 3 的 Exp ID 与仓库 `expXX.yaml` 文件名**不是 1:1 映射**。仓库实际使用如下映射（从 yaml 内容反推）：

| 仓库文件 | 模型 | OT | Mask | Epochs | 对应 Deepthink Row |
|---|---|---|---|---|---|
| `exp01.yaml` | DiT-XL | ✅ | Ada 75% | 400 | Row 19-20（THE SOTA） |
| `exp02.yaml` | DiT-L | ✅ | Ada 75% | 200 | Row 13（SOTA-L） |
| `exp03.yaml` | DiT-B | ✅ | Ada 75% | 300 | Row 07（核心创新） |
| `exp04.yaml` | DiT-S | ✅ | Ada 75% | 300 | *(额外 scale-down)* |
| `exp05.yaml` | DiT-B | ✅ | Const 50% | 300 | Row 06（fixed mask） |
| `exp06.yaml` | DiT-B | ✅ | Ada 50% | 300 | *(Ada-γ 敏感度)* |
| `exp07.yaml` | DiT-B | ✅ | Ada 90% | 300 | Row 16（90% 探底，模型应为 DiT-L） |
| `exp08.yaml` | DiT-B | ✅ | Ada 75%, `dec_depth=0` | 300 | Row 14（架构消融） |
| `exp09.yaml` | DiT-B | ✅ | None | 300 | Row 02/05（no-mask 对照） |
| `exp15.yaml` | U-Net | ✅ | None | 50 | *(U-Net OT 对照)* |
| `exp16.yaml` | U-Net | ❌ | None | 50 | *(U-Net no-OT 对照)* |
| `exp19.yaml` | U-Net | DDPM | None | 50 | Row 01（DDPM 基准短训练） |
| `exp20.yaml` | U-Net | DDPM | None | 200 | Row 01（DDPM 基准长训练） |

### 缺失 yaml（7 个）

需要按 Deepthink Row 03/05/10/11/12/17/18 的配置补建。建议优先级：

1. **高优先级**（核心叙事）：
   - Row 03（no-OT + 10 NFE Crash 对照）
   - Row 05（OT + no-mask + 800ep 过拟合痛点）
2. **中优先级**（Solver / NFE 消融）：Row 10/11（Heun solver, 4/10 NFE 推理期，可以用现有 ckpt 做推理）
3. **低优先级**（烧钱 + XL）：Row 12（DiT-L OOM）, Row 17/18（DiT-XL 基准与 Giant Run）

---

## ⛽ 预算核算（SU）

PSC 计价：**1 SU = 1 GPU-hour on Bridges-2 GPU**。配额 3000 SU。

按仓库现有 13 个 yaml 的 epoch × time/epoch × GPUs 粗估（假设 DiT-B @ 0.8 GPUh/ep 无掩码、0.4 GPUh/ep 50% 掩码；DiT-L 2×、DiT-XL 4×；U-Net 0.05 GPUh/ep）：

| yaml | 估算 SU |
|---|---|
| exp01 (DiT-XL, Ada 75%, 400ep) | ~800 |
| exp02 (DiT-L, Ada 75%, 200ep) | ~160 |
| exp03 (DiT-B, Ada 75%, 300ep) | ~120 |
| exp04 (DiT-S, Ada 75%, 300ep) | ~60 |
| exp05 (DiT-B, Const 50%, 300ep) | ~120 |
| exp06 (DiT-B, Ada 50%, 300ep) | ~120 |
| exp07 (DiT-B, Ada 90%, 300ep) | ~80 |
| exp08 (DiT-B, Ada 75%, dec=0, 300ep) | ~120 |
| exp09 (DiT-B, no mask, 300ep) | ~240 |
| exp15 (U-Net OT, 50ep) | ~10 |
| exp16 (U-Net no-OT, 50ep) | ~10 |
| exp19 (U-Net DDPM 50ep) | ~10 |
| exp20 (U-Net DDPM 200ep) | ~40 |
| **合计** | **~1890 SU** |

剩余 ~1110 SU 留给补缺失 yaml、超参扫描、失败重跑。**预算可控，但 exp01 (DiT-XL 400ep) 是最大头，一次跑失败回退 800 SU**。

---

## 🚀 提交前闭环（3 个 Blocker 必须先清掉）

### Blocker 1：run.slurm DATA_TAR 路径 — `XXXXXX` 未替换

```bash
# scripts/run.slurm:31
DATA_TAR="${DATA_TAR:-/ocean/projects/cis260133p/shared/imagenet100_128x128.tar.gz}"
```

### Blocker 2：ocean 环境变量未在 run.slurm export

创建 `scripts/env_ocean.sh`（从 `CLAUDE.md` 硬约束章节复制 14 件套），在 `run.slurm:16` 加一行 `source "$(dirname "$0")/env_ocean.sh"`。

### Blocker 3：7 个 yaml 缺失 → 决定"补齐"或"明确只跑 13 个"

若选后者，在 PR/report 里把实际跑的 13 → Deepthink 20 的映射表写清楚，别让 TA 困惑于 "为什么跳号"。

### 推荐先跑顺序（smoke → full）

1. `scripts/submit.sh --dry-run 19` — 验证 sbatch 组装正确
2. `scripts/submit.sh 19` — DDPM U-Net 50ep baseline 真跑 1 个（~10 SU）做烟雾测试
3. 确认 `logs/exp19_*.out` + ocean 磁盘写入正常后，再 `scripts/submit.sh --range 01-09`
4. `exp01` (DiT-XL 800 SU) 最后再跑，优先看中间规模结果
