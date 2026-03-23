from typing import Optional, Union

import torch
import numpy as np

from utils import randn_tensor
from .scheduling_ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # BUG FIX #1: num_inference_steps can be None from the parent default.
        # Fall back to num_train_timesteps so set_timesteps never receives None.
        steps = self.num_inference_steps if self.num_inference_steps is not None else self.num_train_timesteps
        self.set_timesteps(steps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
    ):
        """
        Override to build evenly-spaced sub-sequence of timesteps for DDIM.
        DDIM typically uses far fewer steps than the 1000 used during training.
        """
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps` ({num_inference_steps}) cannot exceed "
                f"`num_train_timesteps` ({self.num_train_timesteps})."
            )
        self.num_inference_steps = num_inference_steps
        # Evenly spaced indices in descending order
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def _get_variance(self, t):
        """
        DDIM variance (Eq. 16 from DDIM paper):
          sigma_t^2 = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_{t-1})

        BUG FIX #2: clamp to 0 — float arithmetic can produce tiny negatives.
        """
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.tensor(1.0, device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype)
        )
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)

        # BUG FIX #2 — prevent sqrt of negative from float imprecision
        variance = torch.clamp(variance, min=0.0)

        return variance

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM reverse step (Eq. 12 from DDIM paper):
          x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_hat
                  + sqrt(1 - alpha_bar_{t-1} - sigma_t^2) * eps_theta
                  + sigma_t * noise

        eta=0  → fully deterministic (standard DDIM)
        eta=1  → stochastic, reduces to DDPM
        """
        t = int(timestep)
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        # device-aware fallback (same fix as DDPM)
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.tensor(1.0, device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype)
        )
        beta_prod_t = 1.0 - alpha_prod_t

        # 1. Predict x_0 from predicted noise
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        # 2. Clip predicted x_0
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # 3. Sigma_t = eta * sqrt(variance)
        variance = self._get_variance(t)           # already clamped >= 0
        std_dev_t = eta * (variance ** 0.5)

        # 4. "Direction pointing to x_t"
        #    coeff = sqrt(1 - alpha_bar_{t-1} - sigma_t^2)
        # BUG FIX #3: clamp before sqrt — can be tiny-negative due to float precision
        direction_coeff = (1.0 - alpha_prod_t_prev - std_dev_t ** 2).clamp(min=0.0) ** 0.5
        pred_sample_direction = direction_coeff * pred_epsilon

        # 5. x_{t-1} (deterministic part)
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        # 6. Optional stochastic noise (only when eta > 0)
        if eta > 0:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator,
                device=model_output.device, dtype=model_output.dtype
            )
            prev_sample = prev_sample + std_dev_t * variance_noise

        return prev_sample