"""Smoke tests for Section 5 (Classifier-Free Guidance).

These are fast CPU tests that verify the CFG plumbing end-to-end
without training. They intentionally use a minimal UNet (ch=32,
the smallest value compatible with GroupNorm(32, ch)) and 5-step
denoising so a full pass takes seconds.

Run one of:
    python tests/test_cfg_smoke.py
    python -m pytest tests/test_cfg_smoke.py -v
"""
import sys
from pathlib import Path

import torch

# Make the repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import ClassEmbedder, UNet  # noqa: E402
from pipelines import DDPMPipeline  # noqa: E402
from schedulers import DDPMScheduler  # noqa: E402


def test_class_embedder_shape_and_uncond_index() -> None:
    ce = ClassEmbedder(embed_dim=128, n_classes=100, cond_drop_rate=0.1)
    ce.train()

    labels = torch.randint(0, 100, (8,))
    assert ce(labels).shape == (8, 128)

    # Index n_classes must be a valid slot (this is what
    # pipelines/ddpm.py uses for the unconditional batch).
    assert ce(torch.tensor([100] * 8)).shape == (8, 128)


def test_class_embedder_full_dropout_swaps_all() -> None:
    torch.manual_seed(0)
    ce = ClassEmbedder(embed_dim=32, n_classes=5, cond_drop_rate=1.0)
    ce.train()

    labels = torch.zeros(64, dtype=torch.long)
    out = ce(labels)

    expected = ce.embedding(torch.full((64,), 5, dtype=torch.long))
    assert torch.allclose(out, expected)


def test_class_embedder_inference_mode_never_drops() -> None:
    ce = ClassEmbedder(embed_dim=32, n_classes=5, cond_drop_rate=1.0)
    ce.train(False)  # inference mode — dropout branch should not fire

    labels = torch.tensor([0, 1, 2, 3, 4])
    out = ce(labels)

    expected = ce.embedding(labels)
    assert torch.allclose(out, expected)


def test_cfg_pipeline_end_to_end_cpu() -> None:
    device = "cpu"
    unet = UNet(
        input_size=16,
        input_ch=3,
        T=1000,
        ch=32,
        ch_mult=[1, 2],
        attn=[1],
        num_res_blocks=1,
        dropout=0.0,
        conditional=True,
        c_dim=32,
    ).to(device)
    ce = ClassEmbedder(embed_dim=32, n_classes=5, cond_drop_rate=0.0).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        num_inference_steps=5,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        variance_type="fixed_small",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    ).to(device)

    pipe = DDPMPipeline(unet=unet, scheduler=scheduler, vae=None, class_embedder=ce)
    imgs = pipe(
        batch_size=2,
        num_inference_steps=5,
        classes=[0, 1],
        guidance_scale=2.0,
        device=device,
    )

    assert len(imgs) == 2
    assert imgs[0].size == (16, 16)


if __name__ == "__main__":
    test_class_embedder_shape_and_uncond_index()
    print("shape + uncond index: OK")
    test_class_embedder_full_dropout_swaps_all()
    print("full-dropout swap:    OK")
    test_class_embedder_inference_mode_never_drops()
    print("inference mode no drop: OK")
    test_cfg_pipeline_end_to_end_cpu()
    print("CFG pipeline e2e:     OK")
    print("all smoke tests passed")
