from typing import List, Optional, Union
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn

from utils import randn_tensor


class DDPMPipeline:
    def __init__(self, unet, scheduler, vae=None, class_embedder=None):
        self.unet = unet
        self.scheduler = scheduler

        # BUG FIX #6: always set both attributes (even as None) so hasattr checks
        # are consistent and we never get AttributeError mid-inference.
        self.vae = vae
        self.class_embedder = class_embedder

    def numpy_to_pil(self, images):
        """Convert a numpy image or batch to PIL images."""
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, "
                f"but is {type(self._progress_bar_config)}."
            )
        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1000,
        classes: Optional[Union[int, List[int]]] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        device=None,
    ):
        # Determine input shape:
        # - If using VAE (latent DDPM) the UNet operates on the latent spatial size.
        #   The VAE downsamples by 8x, so 128 -> 16, and latent channels = 4.
        # - Otherwise use raw image dimensions from the UNet config.
        if self.vae is not None:
            # latent spatial size = image_size // 8, latent ch = 4
            latent_size = self.unet.input_size  # already set to latent size in config
            image_shape = (batch_size, self.unet.input_ch, latent_size, latent_size)
        else:
            image_shape = (batch_size, self.unet.input_ch, self.unet.input_size, self.unet.input_size)

        if device is None:
            device = next(self.unet.parameters()).device

        # ---- CFG setup ----
        use_cfg = (
            self.class_embedder is not None
            and classes is not None
            and guidance_scale is not None
            and guidance_scale != 1.0
        )

        class_embeds = None
        uncond_embeds = None
        if self.class_embedder is not None and classes is not None:
            # Convert classes to tensor
            if isinstance(classes, int):
                classes = torch.tensor([classes] * batch_size, device=device)
            elif isinstance(classes, list):
                assert len(classes) == batch_size, "len(classes) must equal batch_size"
                classes = torch.tensor(classes, device=device)

            class_embeds = self.class_embedder(classes)

            if use_cfg:
                # Unconditional token = num_classes (the extra embedding index)
                uncond_classes = torch.full(
                    (batch_size,), self.class_embedder.num_classes,
                    dtype=torch.long, device=device
                )
                uncond_embeds = self.class_embedder(uncond_classes)

        # ---- Start from pure Gaussian noise ----
        image = randn_tensor(image_shape, generator=generator, device=device)

        # ---- Set timesteps ----
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        # ---- Reverse diffusion loop ----
        for t in self.progress_bar(self.scheduler.timesteps):

            if use_cfg:
                # Dual forward pass: concatenate along batch dim
                model_input = torch.cat([image, image], dim=0)
                c = torch.cat([uncond_embeds, class_embeds], dim=0)
            else:
                model_input = image
                c = class_embeds  # None if not using class conditioning

            # Predict noise
            model_output = self.unet(model_input, t, c=c)

            if use_cfg:
                # Split unconditional and conditional predictions
                uncond_out, cond_out = model_output.chunk(2)
                # CFG: guided_eps = uncond + w*(cond - uncond)
                model_output = uncond_out + guidance_scale * (cond_out - uncond_out)

            # Scheduler step: x_t -> x_{t-1}
            image = self.scheduler.step(model_output, t, image, generator=generator)

        # ---- Decode from latent space if using VAE ----
        if self.vae is not None:
            # Undo the VAE.scaling_factor scaling applied during encoding
            image = self.vae.decode(image / self.vae.scaling_factor)
            image = image.clamp(-1.0, 1.0)

        # ---- Rescale [-1, 1] -> [0, 1] ----
        image = (image + 1.0) / 2.0

        # ---- Convert to PIL ----
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = self.numpy_to_pil(image)

        return image