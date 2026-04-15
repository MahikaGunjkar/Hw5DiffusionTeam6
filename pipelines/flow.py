"""Flow-Matching inference pipeline (Euler / Heun ODE integrator).

Mirrors the interface of ``DDPMPipeline`` so ``inference.py`` and
``generate_submission.py`` only need to select which pipeline class to
construct.  CFG and VAE decoding logic are kept in parity with
``DDPMPipeline``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from PIL import Image
from tqdm import tqdm

from utils import randn_tensor


class FlowMatchingPipeline:
    """Rectified Flow / OT-AdaMask sampler.

    Trajectory: starts from pure noise at ``t=0`` and advances toward data at
    ``t=1`` via an ODE step of size ``dt = 1 / num_inference_steps``.
    """

    def __init__(self, unet, scheduler, vae=None, class_embedder=None) -> None:
        self.unet = unet
        self.scheduler = scheduler
        self.vae = vae
        self.class_embedder = class_embedder

    # ------------------------------------------------------------------ #
    # Helpers reused from DDPMPipeline conventions
    # ------------------------------------------------------------------ #
    def numpy_to_pil(self, images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            return [Image.fromarray(img.squeeze(), mode="L") for img in images]
        return [Image.fromarray(img) for img in images]

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        if total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        raise ValueError("Either `total` or `iterable` has to be defined.")

    # ------------------------------------------------------------------ #
    # Main sampler
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 10,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        # ---- Determine input shape (same logic as DDPMPipeline) ----
        if self.vae is not None:
            latent_size = self.unet.input_size
            image_shape = (batch_size, self.unet.input_ch, latent_size, latent_size)
        else:
            image_shape = (
                batch_size, self.unet.input_ch,
                self.unet.input_size, self.unet.input_size,
            )

        if device is None:
            device = next(self.unet.parameters()).device

        use_cfg = (
            self.class_embedder is not None
            and classes is not None
            and guidance_scale is not None
            and guidance_scale != 1.0
        )

        class_embeds = None
        uncond_embeds = None
        if self.class_embedder is not None and classes is not None:
            if isinstance(classes, int):
                classes = torch.tensor([classes] * batch_size, device=device)
            elif isinstance(classes, list):
                assert len(classes) == batch_size, "len(classes) must equal batch_size"
                classes = torch.tensor(classes, device=device)
            class_embeds = self.class_embedder(classes)

            if use_cfg:
                uncond_classes = torch.full(
                    (batch_size,), self.class_embedder.num_classes,
                    dtype=torch.long, device=device,
                )
                uncond_embeds = self.class_embedder(uncond_classes)

        # ---- Start from pure Gaussian noise (t=0) ----
        image = randn_tensor(image_shape, generator=generator, device=device)

        # ---- Set timesteps ----
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        dt = self.scheduler.dt
        solver = getattr(self.scheduler, "solver", "euler")

        # ---- ODE integration loop ----
        for t_scalar in self.progress_bar(self.scheduler.timesteps):
            t_batch = t_scalar.expand(batch_size).to(device)

            v1 = self._model_velocity(
                image, t_batch, class_embeds, uncond_embeds, use_cfg, guidance_scale
            )

            if solver == "heun":
                # Predictor step
                x_pred = image + v1 * dt
                t_next = torch.clamp(t_batch + dt, max=1.0)
                v2 = self._model_velocity(
                    x_pred, t_next, class_embeds, uncond_embeds, use_cfg, guidance_scale
                )
                image = self.scheduler.step_heun(v1, v2, image)
            else:
                image = self.scheduler.step(v1, t_batch, image)

        # ---- VAE decode (if latent) ----
        if self.vae is not None:
            image = self.vae.decode(image / 0.1845)
            image = image.clamp(-1.0, 1.0)

        # ---- Rescale [-1, 1] -> [0, 1] and convert to PIL ----
        image = (image + 1.0) / 2.0
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        return self.numpy_to_pil(image)

    # ------------------------------------------------------------------ #
    # Internal: one velocity forward pass (with optional CFG)
    # ------------------------------------------------------------------ #
    def _rescale_t(self, t: torch.Tensor) -> torch.Tensor:
        """Some denoisers (e.g. UNet) use a discrete timestep embedding table
        and require integer indices.  Detect via the ``.T`` attribute that
        ``UNet`` exposes as its max-timestep count; DiT leaves ``t`` alone.
        """
        T_max = getattr(self.unet, "T", None)
        if T_max is None:
            return t
        return (t * T_max).long().clamp(0, T_max - 1)

    def _model_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_embeds: Optional[torch.Tensor],
        uncond_embeds: Optional[torch.Tensor],
        use_cfg: bool,
        guidance_scale: Optional[float],
    ) -> torch.Tensor:
        t_model = self._rescale_t(t)
        if use_cfg:
            model_input = torch.cat([x_t, x_t], dim=0)
            t_in = torch.cat([t_model, t_model], dim=0)
            c = torch.cat([uncond_embeds, class_embeds], dim=0)
            v = self.unet(model_input, t_in, c=c)
            uncond_v, cond_v = v.chunk(2)
            return uncond_v + guidance_scale * (cond_v - uncond_v)

        return self.unet(x_t, t_model, c=class_embeds)
