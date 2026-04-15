# IDL S26 - Guided Project: Diffusion

# Starter Code Usage

**Training**

```
python train.py --config configs/ddpm.yaml
```

**Inference and Evaluating**

```
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint
```

# 1. Download the data

Please first download the data from here: https://drive.google.com/drive/u/0/folders/1Hr8LU7HHPEad8ALmMo5cvisazsm6zE8Z

After download please unzip the data with

```
tar -xvf imagenet100_128x128.tar.gz
```

# 2.Implementing DDPM from Scratch

This homework will start from implementing DDPM from scratch.

We provide the basic code structure for you and you will be implementing the following modules (by filling all TODOs)):

```
1. pipelines/ddpm.py
2. schedulers/scheduling_ddpm.py
3. train.py
4. configs/ddpm.yaml
```

A very basic U-Net architecture is provided to you, and you will need to improve the architecture for better performance.

# 3. Implementing DDIM

Implement the DDIM from scratch:

```
1. schedulers/scheduling_ddpm.py
2. create a config with ddim by setting use_ddim to True
```

**NOTE: you need to set use_ddim to TRUE**

# 4. Implementing Latent DDPM

Implement the Latent DDPM.

The pre-trained weights of VAE and basic modules are provided. 

Download the pretrained weight here: and put it under a folder named 'pretrained' (create one if it doesn't exist)

You need to implement:

```
1. models/vae.py
2. train.py with vae related stuff
3. pipeline/ddpm.py with vae related stuff
```

**NOTE: you need to set use_vae to TRUE**

# 5. Implementing CFG

CFG lets a single UNet learn both conditional `p(x | y)` and unconditional
`p(x)` score functions, then blends their noise predictions at inference
via a guidance scale `w`:

```
ε̂(x_t, t, y) = ε(x_t, t) + w · (ε(x_t, t, y) - ε(x_t, t))
```

**Implementation**:

1. `models/class_embedder.py` — `ClassEmbedder` uses `nn.Embedding(n_classes + 1, embed_dim)`; index `n_classes` is reserved as the unconditional / null token. During training, each label is swapped to the null token with probability `cond_drop_rate` (default 10%). At inference `pipelines/ddpm.py` builds a separate null-token batch for the unconditional forward pass.
2. `train.py` — already wired: `DataLoader` yields `(images, labels)` and the training step calls `class_embedder(labels)` → UNet's `c=` argument.
3. `pipelines/ddpm.py` — already wired: when `guidance_scale != 1.0`, concatenates conditional and unconditional batches, runs one UNet forward pass over both, and applies the guidance formula above.

**To enable**: set `use_cfg: true` in `configs/ddpm.yaml` or pass `--use_cfg` on the command line.

**Smoke test** (fast CPU sanity check, no training required):

```
python tests/test_cfg_smoke.py
```

The test verifies the Embedding shape, that the null-token index is valid, that dropout only fires in train mode, and that `DDPMPipeline(..., class_embedder=ClassEmbedder(...))` runs a 5-step CFG denoising end-to-end.

# 6. Evaluation

`inference.py` loads a trained checkpoint, generates 5,000 images (50 per class × 100 classes with CFG, or 5,000 unconditional), and computes **FID** and **Inception Score** via `torchmetrics`:

- `torchmetrics.image.fid.FrechetInceptionDistance(feature=2048)` — compares the Inception-v3 pool3 feature distributions of generated vs. reference images. Lower is better.
- `torchmetrics.image.inception.InceptionScore()` — measures quality × diversity from Inception-v3 softmax output. Higher is better.

Reference images are loaded from `--data_dir` as an `ImageFolder`. (Note: this currently reuses the training data directory — if you want an unbiased FID, point `--data_dir` at a held-out validation split.)

Run:

```
python inference.py --config configs/ddpm.yaml --ckpt /path/to/checkpoint_epoch_N.pth
```

The script also writes a Kaggle submission CSV; see Section 7.

# 7. Kaggle Submission

After generating 5,000 images, create your Kaggle submission CSV:

**From saved images on disk:**
```
python generate_submission.py \
    --image_dir /path/to/generated_images \
    --output submission.csv
```

**From your inference script (Python API):**
```python
from generate_submission import generate_submission_from_tensors

# all_images: tensor (5000, 3, H, W) in [-1, 1] or [0, 1]
all_images = torch.cat(all_images, dim=0)
generate_submission_from_tensors(all_images, output_csv="submission.csv")
```

This extracts Inception-v3 pool3 features, computes mean and covariance, and writes the CSV. Upload the CSV to the Kaggle InClass competition page.
