"""Sanity tests for the OT-AdaMask Flow implementation.

Kept intentionally small and CPU-only so they run in < 5 seconds and catch
the most common regression classes:
  - interpolation / velocity math
  - OT permutation property
  - MAE-style mask round-trip (shape only)
  - timestep schedule monotonicity
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# Allow `pytest tests/` from repo root without installing the package.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from models.dit import AsymmetricMaskedDiT  # noqa: E402
from schedulers.scheduling_flow import RectifiedFlowScheduler  # noqa: E402


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def tiny_dit():
    """A tiny DiT suitable for shape tests on CPU."""
    model = AsymmetricMaskedDiT(
        input_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=32,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        use_ada_mask=True,
    )
    model.train(False)  # disable dropout (mask ratio is gated by training flag)
    return model


# ------------------------------------------------------------------ #
# Scheduler math
# ------------------------------------------------------------------ #
def test_interpolate_boundaries():
    """t=0 -> x_0, t=1 -> x_1 (straight-line interpolation)."""
    s = RectifiedFlowScheduler()
    x0 = torch.randn(4, 3, 8, 8)
    x1 = torch.randn(4, 3, 8, 8)
    t0 = torch.zeros(4)
    t1 = torch.ones(4)
    assert torch.allclose(s.interpolate(x0, x1, t0), x0)
    assert torch.allclose(s.interpolate(x0, x1, t1), x1)


def test_velocity_target_shape_and_value():
    s = RectifiedFlowScheduler()
    x0 = torch.randn(4, 3, 8, 8)
    x1 = torch.randn(4, 3, 8, 8)
    v = s.velocity_target(x0, x1)
    assert v.shape == x0.shape
    assert torch.allclose(v, x1 - x0)


def test_pair_ot_identity_when_disabled():
    """With use_ot=False, pair_ot must return the input tensor unchanged."""
    s = RectifiedFlowScheduler(use_ot=False)
    x0 = torch.randn(8, 3, 8, 8)
    x1 = torch.randn_like(x0)
    out = s.pair_ot(x0, x1)
    assert out is x0


def test_pair_ot_is_a_permutation():
    """With use_ot=True, output must be a row-permutation of x0."""
    torch.manual_seed(0)
    s = RectifiedFlowScheduler(use_ot=True, ot_max_batch=64)
    x0 = torch.randn(8, 3, 8, 8)
    x1 = torch.randn(8, 3, 8, 8)
    paired = s.pair_ot(x0, x1)
    assert paired.shape == x0.shape
    # Every row of x0 must appear exactly once in paired (sorted equality check).
    flat_in = x0.reshape(8, -1)
    flat_out = paired.reshape(8, -1)
    # Sort rows by their first element (stable) and compare.
    idx_in = flat_in[:, 0].argsort()
    idx_out = flat_out[:, 0].argsort()
    assert torch.allclose(flat_in[idx_in], flat_out[idx_out])


def test_timesteps_monotone_and_cover_unit_interval():
    s = RectifiedFlowScheduler(num_inference_steps=10)
    ts = s.timesteps
    assert ts.shape == (10,)
    assert ts[0].item() == 0.0
    # Strictly increasing
    assert torch.all(ts[1:] > ts[:-1])
    # Final step of length dt lands exactly on t=1.
    assert math.isclose(ts[-1].item() + s.dt, 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_add_noise_raises_not_implemented():
    """Flow has reversed t semantics; the DDPM-style shim must error loudly."""
    s = RectifiedFlowScheduler()
    with pytest.raises(NotImplementedError):
        s.add_noise(torch.zeros(1), torch.zeros(1), torch.zeros(1))


# ------------------------------------------------------------------ #
# Mask round-trip
# ------------------------------------------------------------------ #
def test_mask_roundtrip_preserves_shape(tiny_dit):
    """_apply_mask then _unmask should restore the original sequence length."""
    B, N, D = 2, tiny_dit.num_patches, tiny_dit.hidden_size
    x = torch.randn(B, N, D)
    # Use internal helpers directly (mask_ratio 0.5)
    x_vis, ids_restore = tiny_dit._apply_mask(x, mask_ratio=0.5)
    assert x_vis.shape[0] == B
    assert x_vis.shape[1] < N  # some tokens were dropped
    x_full = tiny_dit._unmask(x_vis, ids_restore)
    assert x_full.shape == x.shape


def test_mask_ratio_zero_in_eval_mode(tiny_dit):
    """In eval mode, _compute_mask_ratio must return 0 regardless of flags."""
    tiny_dit.train(False)
    t = torch.linspace(0.0, 1.0, 4)
    assert tiny_dit._compute_mask_ratio(t) == 0.0


def test_const_mask_ratio_only_active_in_train_mode():
    """const_mask_ratio must NOT leak into eval-mode forward passes."""
    model = AsymmetricMaskedDiT(
        input_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=32,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        const_mask_ratio=0.5,
    )
    model.train(False)
    t = torch.rand(4)
    assert model._compute_mask_ratio(t) == 0.0
    model.train(True)
    assert model._compute_mask_ratio(t) == 0.5
