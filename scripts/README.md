# Training / Ablation Orchestration

Single source of truth for every training experiment in the OT-AdaMask plan:
**one YAML per experiment** under `configs/ablations/`, plus a tiny shell
helper that submits them to Slurm.

## Directory layout

```
configs/
  ddpm.yaml                          # base config (shared defaults)
  ablations/
    exp01.yaml   exp02.yaml   ...    # deltas + slurm resources
scripts/
  run.slurm                          # Slurm template
  submit.sh                          # picks YAML(s), fires sbatch
  infer_sweep.sh                     # inference-only variants (Exp 10-13, 17-18)
```

## How a training experiment is defined

Each `expXX.yaml` contains only what **differs** from `configs/ddpm.yaml`,
plus a `slurm:` block for resource allocation:

```yaml
# configs/ablations/exp03.yaml — DiT-B SOTA
slurm:
  node: v100-32
  gpus: 8
  time: "48:00:00"

framework: flow_matching
model_type: dit_b
use_ot: true
use_ada_mask: true
ada_mask_max: 0.75
use_cfg: true
num_epochs: 300
batch_size: 32
learning_rate: 3.0e-4
```

`train.py` loads two configs (`--config BASE EXP`) and the second one
overrides the first field-by-field. The `slurm:` block is stripped at load
time so it never touches argparse.

## Quick Start

```bash
# 1. Edit DATA_TAR inside scripts/run.slurm to point to your copy of
#    imagenet100_128x128.tar.gz on PSC Ocean.

# 2. Submit everything (all 13 training experiments)
bash scripts/submit.sh

# 3. Pick a subset
bash scripts/submit.sh 03                # just Exp 03
bash scripts/submit.sh 03 06 09          # specific list
bash scripts/submit.sh --range 01-05     # range

# 4. Preview without submitting
bash scripts/submit.sh --dry-run 03
```

## Adding a new experiment

1. `cp configs/ablations/exp03.yaml configs/ablations/exp21.yaml`
2. Edit the deltas + `slurm:` resources
3. `bash scripts/submit.sh 21`

No script changes. No registry to update. That is the whole benefit of the
YAML-per-experiment layout.

## Modifying a shared hyperparameter

Change one line in `configs/ddpm.yaml` — every experiment picks it up on
the next submission (because their YAMLs only override *deltas*).

If you need an exp-specific override, add the field to its YAML.

## Recommended submission order

| Wave | Experiments | Rationale |
|---|---|---|
| 1. Baseline anchor | 19 | Reproduce original DDPM FID |
| 2. SOTA main | 03 | DiT-B + full stack — headline result |
| 3. Mask sweep | 05, 06, 07, 09 | Optimal mask schedule |
| 4. Architecture sweep | 04, 02 | DiT-S + DiT-L for Scaling Law plot |
| 5. UNet-RF controls | 15, 16, 20 | RF / OT gain independent of DiT |
| 6. Ultimate SOTA | 01 | DiT-XL, burns ~1600 GPUh |
| 7. Mechanism ablation | 08 | DecDepth=0 — asymmetric decoder necessity |

## Inference sweeps (Exp 10, 11, 12, 13, 17, 18)

These are *inference variants* of trained checkpoints. They are cheap — run
them on a single interactive GPU once the DiT-B / UNet training completes:

```bash
srun --gpus=v100-32:1 --time=2:00:00 --pty bash
bash scripts/infer_sweep.sh outputs/exp-XX-exp03/checkpoints/checkpoint_epoch_299.pth dit_b
bash scripts/infer_sweep.sh outputs/exp-XX-exp15/checkpoints/checkpoint_epoch_199.pth unet
```

`infer_sweep.sh` will automatically reuse the sibling `config.yaml` saved by
`train.py`, so the sweep inherits the original `use_cfg`, scheduler, latent,
and architecture settings instead of silently rebuilding from the base config.

## Kaggle Submission SOP (per training experiment)

`scripts/infer.sbatch` wraps `python inference.py` (README §6) as a SLURM
batch job. It generates 5,000 images, computes local FID/IS, and writes
`submission.csv` ready for Kaggle upload (`11685s26-diffusion`).

### When training is RUNNING — pre-queue the inference

```bash
# Pre-queue inference: starts only when training job exits 0; auto-cancels on fail
EXP_ID=15  TRAIN_JOB=40044869   # replace with your real ids
sbatch --account=cis260133p --gpus=h100-80:1 --job-name=infer_exp${EXP_ID} \
       --output=logs/infer_exp${EXP_ID}_%j.out \
       --error=logs/infer_exp${EXP_ID}_%j.err \
       --dependency=afterok:${TRAIN_JOB} \
       --export=ALL,EXP_ID=${EXP_ID} scripts/infer.sbatch
```

Batch-queue all four DiT ablations against their training jobs:
```bash
declare -A J=([01]=40035557 [02]=40035558 [03]=40035578 [04]=40035528)
for ID in "${!J[@]}"; do
  sbatch --account=cis260133p --gpus=h100-80:1 --job-name=infer_exp${ID} \
         --output=logs/infer_exp${ID}_%j.out \
         --error=logs/infer_exp${ID}_%j.err \
         --dependency=afterok:${J[$ID]} \
         --export=ALL,EXP_ID=${ID} scripts/infer.sbatch
done
```

### When training is already DONE — submit immediately

```bash
EXP_ID=15
sbatch --account=cis260133p --gpus=h100-80:1 --job-name=infer_exp${EXP_ID} \
       --output=logs/infer_exp${EXP_ID}_%j.out \
       --error=logs/infer_exp${EXP_ID}_%j.err \
       --export=ALL,EXP_ID=${EXP_ID} scripts/infer.sbatch
```

### What infer.sbatch produces

`/ocean/projects/cis260133p/zjiang9/inference/exp${EXP_ID}_${SLURM_JOB_ID}/`
- `generated_images/` — 5,000 PNG (50 per class × 100 classes when `use_cfg=true`)
- `submission.csv` — Inception-v3 mu/sigma stats; upload to Kaggle

Resource: 1× H100 GPU-shared, 1.5h walltime (typical 5-15 min runtime).

### Override hooks (export via `--export=`)

| Var | Default | Purpose |
|---|---|---|
| `EXP_ID` | (required) | Exp number, picks `configs/ablations/expNN.yaml` |
| `CKPT` | `${PROJ_ROOT}/outputs/expNN/exp-0-expNN/checkpoints/latest.pth` | Override ckpt path |
| `DATA_REF` | `${PROJ_ROOT}/imagenet100_extracted/imagenet100_128x128` | FID reference dir |
| `EXTRA_ARGS` | (empty) | Extra CLI flags for `inference.py`, e.g. `--cfg_guidance_scale 4.5` |

### Kaggle upload (from local laptop, after `scp` of submission.csv)

```bash
# Pull from PSC DTN to local
scp -i ~/.ssh/bridges2-key \
  "zjiang9@data.bridges2.psc.edu:/ocean/projects/cis260133p/zjiang9/inference/expNN_<jobid>/submission.csv" \
  ~/Desktop/Hw5DiffusionTeam6/submission_expNN.csv

# Submit (5/day quota — see https://www.kaggle.com/competitions/11685s26-diffusion/submit)
kaggle competitions submit -c 11685s26-diffusion \
  -f ~/Desktop/Hw5DiffusionTeam6/submission_expNN.csv \
  -m "expNN: <model> <framework> <epochs> ep, NFE=10"

# Check leaderboard delta
kaggle competitions submissions 11685s26-diffusion
```

Kaggle CLI requires `~/.kaggle/kaggle.json` (legacy `{"username","key"}`
format). Stored in GCP Secret Manager as `kaggle-json-legacy`:
```bash
gcloud secrets versions access latest --secret=kaggle-json-legacy \
  --project=general-secrets-store > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## NVLink verification

```bash
tail -f logs/exp03_<jobid>.out | grep NCCL
```

Expect `Channel NN : X -> Y via NVLink`. If you see `via PCI` you landed on
a v100-16 or l40s-48 — check `squeue` and resubmit with the correct YAML.

## Budget target

- ~3200 GPUh across the 13 training jobs
- ~500 GPUh buffer for re-runs / crashed jobs
- Ultimate ceiling = 3700 GPUh = 200,000 PSC credits

Track via `sreport cluster UserUtilizationByAccount Start=... User=$USER`.
