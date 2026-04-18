"""Rectified Flow scheduler with optional Minibatch Optimal Transport pairing.

Implements the forward interpolation, velocity target, OT pairing, and ODE
integration step used by OT-AdaMask Flow (see plan).

Interface notes:
- Forward (training): ``interpolate(x_0, x_1, t)`` and ``velocity_target(x_0, x_1)``
  are called directly from ``train.py`` (not via ``add_noise``) so that the
  caller owns the noise and data tensors.
- Inference: ``set_timesteps(n)`` + iterate over ``self.timesteps`` and call
  ``step(v_pred, t, x_t)`` which advances one Euler step of size ``dt``.
- OT: ``pair_ot(x_0, x_1)`` returns a permuted ``x_0`` aligned to ``x_1`` by
  minibatch Hungarian matching. Identity when ``use_ot`` is ``False``.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class RectifiedFlowScheduler(nn.Module):
    """Continuous-time Rectified Flow with optional Minibatch OT.

    Convention:
        t = 0  -> pure noise   x_0 ~ N(0, I)
        t = 1  -> data latent  x_1 ~ P_data
        X_t   = t * x_1 + (1 - t) * x_0
        v*    = x_1 - x_0
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 10,
        solver: str = "euler",
        use_ot: bool = False,
        ot_max_batch: int = 64,
    ) -> None:
        super().__init__()
        if solver not in {"euler", "heun"}:
            raise ValueError(f"Unknown solver '{solver}'. Use 'euler' or 'heun'.")

        self.num_train_timesteps = int(num_train_timesteps)
        self.num_inference_steps = int(num_inference_steps)
        self.solver = solver
        self.use_ot = bool(use_ot)
        self.ot_max_batch = int(ot_max_batch)

        self.set_timesteps(self.num_inference_steps)

    # ------------------------------------------------------------------ #
    # Inference timestep schedule
    # ------------------------------------------------------------------ #
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """Build an evenly-spaced float schedule in [0, 1).

        We skip t=1 because the final step advances *into* t=1.  ``dt`` is
        stored so the pipeline does not need to re-derive it.
        """
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")

        self.num_inference_steps = int(num_inference_steps)
        steps = np.linspace(0.0, 1.0, num_inference_steps + 1, dtype=np.float32)[:-1]
        self.timesteps = torch.from_numpy(steps)
        if device is not None:
            self.timesteps = self.timesteps.to(device)
        self.dt = 1.0 / num_inference_steps

    def __len__(self) -> int:
        return self.num_inference_steps

    # ------------------------------------------------------------------ #
    # Training-time primitives (called from train.py)
    # ------------------------------------------------------------------ #
    @staticmethod
    def interpolate(x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Straight-line interpolation: X_t = t*x_1 + (1-t)*x_0.

        ``t`` is broadcast to match the spatial dims of ``x_0`` / ``x_1``.
        """
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0 {x_0.shape} and x_1 {x_1.shape} must match.")
        t_b = t.view(-1, *([1] * (x_1.dim() - 1))).to(x_1.dtype)
        return t_b * x_1 + (1.0 - t_b) * x_0

    @staticmethod
    def velocity_target(x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """v* = x_1 - x_0 (constant along the straight-line path)."""
        return x_1 - x_0

    def pair_ot(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Minibatch optimal transport pairing; returns permuted ``x_0``.

        Uses Hungarian (O(B^3)) for ``B <= ot_max_batch``.  Above that, a
        Sinkhorn fallback is required; we raise so the user gets a clear
        signal rather than silent slowdown.
        """
        if not self.use_ot:
            return x_0

        batch_size = x_0.shape[0]
        if batch_size > self.ot_max_batch:
            raise RuntimeError(
                f"OT batch size {batch_size} exceeds ot_max_batch={self.ot_max_batch}. "
                "Switch to Sinkhorn (e.g. via POT's `ot.sinkhorn`) for large batches."
            )

        with torch.no_grad():
            x_0_flat = x_0.reshape(batch_size, -1).float()
            x_1_flat = x_1.reshape(batch_size, -1).float()
            cost = torch.cdist(x_1_flat, x_0_flat, p=2).pow(2)
            _, col = linear_sum_assignment(cost.detach().cpu().numpy())
        perm = torch.as_tensor(col, device=x_0.device, dtype=torch.long)
        return x_0.index_select(0, perm)

    # ------------------------------------------------------------------ #
    # Inference-time ODE integrator
    # ------------------------------------------------------------------ #
    def step(
        self,
        model_output: torch.Tensor,
        t: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """One Euler step: x_{t+dt} = x_t + v_pred * dt.

        Heun is handled at the pipeline level since it needs a *second* model
        call per step, which the scheduler cannot issue on its own.
        """
        del generator, t  # not used for Euler
        return sample + model_output * self.dt

    def step_heun(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Heun 2nd-order step: x_{t+dt} = x_t + 0.5 * (v1 + v2) * dt."""
        return sample + 0.5 * (v1 + v2) * self.dt

    # ------------------------------------------------------------------ #
    # Compatibility shim
    # ------------------------------------------------------------------ #
    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Intentionally not implemented.

        DDPM's ``add_noise(x_0, noise, t)`` has *t large = noisier* semantics.
        Rectified Flow's ``interpolate(x_0=noise, x_1=data, t)`` has the
        *opposite* convention (t large = more data). A silent shim mapping
        the two would be a footgun; callers must use ``interpolate`` directly.
        """
        raise NotImplementedError(
            "RectifiedFlowScheduler has reversed t semantics vs DDPM. "
            "Use scheduler.interpolate(x_0=noise, x_1=data, t) explicitly."
        )
