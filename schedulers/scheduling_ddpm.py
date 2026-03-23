from typing import Optional, Union

import torch
import torch.nn as nn
import numpy as np

from utils import randn_tensor


class DDPMScheduler(nn.Module):

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: Optional[int] = None,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear',
        variance_type: str = "fixed_small",
        prediction_type: str = 'epsilon',
        clip_sample: bool = True,
        clip_sample_range: float = 1.0,
    ):
        super(DDPMScheduler, self).__init__()

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.variance_type = variance_type
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        # ---- Beta schedule ----
        if self.beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Beta schedule '{beta_schedule}' not implemented.")
        self.register_buffer("betas", betas)

        # ---- Alphas ----
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)

        # ---- Cumulative product of alphas ----
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        # ---- Timesteps for reverse process ----
        timesteps = torch.arange(num_train_timesteps - 1, -1, -1)
        self.register_buffer("timesteps", timesteps)

    def set_timesteps(
        self,
        num_inference_steps: int = 250,
        device: Union[str, torch.device] = None,
    ):
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than "
                f"`self.num_train_timesteps`: {self.num_train_timesteps}."
            )
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

    def __len__(self):
        return self.num_train_timesteps

    def previous_timestep(self, timestep):
        """
        Get the previous timestep.
        BUG FIX #7: force timestep to int so comparisons always return a plain bool,
        not a 0-d tensor (which is truthy even when 0).
        """
        t = int(timestep)
        num_inference_steps = (
            self.num_inference_steps if self.num_inference_steps else self.num_train_timesteps
        )
        prev_t = t - self.num_train_timesteps // num_inference_steps
        return prev_t

    def _get_variance(self, t):
        """
        Posterior variance sigma_t^2.
        BUG FIX #4: create the fallback tensor on the same device as alphas_cumprod
        so CUDA tensors never mix with CPU tensors.
        """
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        # BUG FIX #4 — device-aware fallback
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.tensor(1.0, device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype)
        )
        current_beta_t = 1.0 - alpha_prod_t / alpha_prod_t_prev

        variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        if self.variance_type == "fixed_small":
            variance = variance
        elif self.variance_type == "fixed_large":
            variance = current_beta_t
        else:
            raise NotImplementedError(f"Variance type {self.variance_type} not implemented.")

        return variance

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps
        """
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Reverse diffusion step: x_{t-1} ~ p_theta(x_{t-1} | x_t)
        BUG FIX #5: device-aware fallback for alpha_prod_t_prev.
        """
        t = int(timestep)
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        # BUG FIX #5 — device-aware fallback
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t]
            if prev_t >= 0
            else torch.tensor(1.0, device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype)
        )
        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1.0 - current_alpha_t

        # Predict x_0
        if self.prediction_type == 'epsilon':
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        else:
            raise NotImplementedError(f"Prediction type {self.prediction_type} not implemented.")

        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )

        # Posterior mean coefficients
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t ** 0.5 * beta_prod_t_prev) / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # Add noise only when t > 0
        variance = 0
        if t > 0:
            variance_noise = randn_tensor(
                model_output.shape, generator=generator,
                device=model_output.device, dtype=model_output.dtype
            )
            variance = (self._get_variance(t) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample