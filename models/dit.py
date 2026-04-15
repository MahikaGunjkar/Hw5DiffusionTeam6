"""Diffusion Transformer (DiT) with optional Asymmetric Masked encoder-decoder.

Ported from facebookresearch/DiT (Peebles & Xie, ICCV 2023) with two
extensions:
  1. ``mask_ratio`` support — MaskDiT-style random patch dropping during
     training with a shallow unmask-aware decoder.
  2. Time-adaptive mask schedule — ``gamma(t) = gamma_max * cos^2(pi*t/2)``
     wired in via :class:`AsymmetricMaskedDiT.forward` when
     ``self.use_ada_mask`` is set (called from ``train.py``).

Interface mirrors the existing ``UNet`` so ``pipelines/flow.py`` does not
need to branch on model type:

    y = model(x, t, c=class_emb)

with ``x: (B, C, H, W)``, ``t: (B,) float in [0, 1]``, and ``c`` optional
class-conditioning already emitted by :class:`ClassEmbedder` — its
``embed_dim`` MUST equal ``hidden_size`` (enforced with an ``assert``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN modulation: x * (1 + scale) + shift, broadcasting (B, D) -> (B, 1, D)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal embedding for a (B,) float time tensor."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _build_2d_sincos_posembed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sinusoidal position embedding.  Returns (grid_size**2, embed_dim)."""
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos."
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(grid_h, grid_w, indexing="ij"), dim=0)  # (2, H, W)
    grid = grid.reshape(2, -1)

    half_dim = embed_dim // 2
    omega = torch.arange(half_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (half_dim / 2)))

    out = []
    for axis in (0, 1):
        ang = grid[axis].unsqueeze(-1) * omega.unsqueeze(0)
        out.append(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))
    return torch.cat(out, dim=-1)  # (N, embed_dim)


# ---------------------------------------------------------------------- #
# Time embedder (continuous t in [0, 1] -> hidden_size)
# ---------------------------------------------------------------------- #
class TimestepEmbedder(nn.Module):
    """Rescales t by 1000 (conventional) then applies sinusoidal + MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.freq_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Accept t in [0, 1] (flow) or [0, num_train_timesteps) (ddpm).
        # In both cases * 1000 yields the range DiT was designed for.
        if t.dtype != torch.float32:
            t = t.float()
        if t.max() <= 1.0:
            t = t * 1000.0
        emb = _sinusoidal_embedding(t, self.freq_size)
        return self.mlp(emb)


# ---------------------------------------------------------------------- #
# Patchify
# ---------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """Naive patch embedding via Conv2d (equivalent to the timm version)."""

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        in_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        assert input_size % patch_size == 0, "input_size must be divisible by patch_size."
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # (B, D, H', W')
        return x.flatten(2).transpose(1, 2)    # (B, N, D)


# ---------------------------------------------------------------------- #
# DiT block with AdaLN-Zero
# ---------------------------------------------------------------------- #
class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning (Peebles & Xie 2023)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c_emb).chunk(6, dim=1)
        )
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


class FinalLayer(nn.Module):
    """AdaLN + linear projection back to patch pixels.  Zero-initialized."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c_emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------- #
# Main model: Asymmetric Masked DiT
# ---------------------------------------------------------------------- #
class AsymmetricMaskedDiT(nn.Module):
    """DiT with deep encoder + shallow decoder + optional random patch masking.

    ``mask_ratio == 0`` recovers the symmetric DiT of Peebles & Xie (the
    decoder then acts as additional depth, which is fine).

    ``use_ada_mask`` enables the cosine-squared time-adaptive schedule during
    training; at inference the mask is always off.
    """

    # NOTE: mirror the UNet attribute names so pipelines can read them
    # uniformly.
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 768,
        encoder_depth: int = 12,
        decoder_depth: int = 2,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        c_dim: Optional[int] = None,       # external class-embed size; None = hidden_size
        use_ada_mask: bool = False,
        ada_mask_max: float = 0.75,
        const_mask_ratio: Optional[float] = None,   # if set, overrides ada schedule
        num_train_timesteps_ref: int = 1000,        # used to normalize integer DDPM t -> [0,1]
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.input_ch = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.use_ada_mask = use_ada_mask
        self.ada_mask_max = ada_mask_max
        self.const_mask_ratio = const_mask_ratio
        self.num_train_timesteps_ref = int(num_train_timesteps_ref)

        c_dim = c_dim if c_dim is not None else hidden_size
        self.c_proj = (
            nn.Identity() if c_dim == hidden_size
            else nn.Linear(c_dim, hidden_size, bias=True)
        )

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        num_patches = self.x_embedder.num_patches
        pos_embed = _build_2d_sincos_posembed(hidden_size, self.x_embedder.grid_size)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.encoder_blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(encoder_depth)]
        )
        self.decoder_blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(decoder_depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self.num_patches = num_patches
        self._initialize_weights()

    # ------------------------------------------------------------------ #
    # Init (AdaLN-Zero is the critical piece)
    # ------------------------------------------------------------------ #
    def _initialize_weights(self) -> None:
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)

        # Patch embed: init as standard Xavier (we already replaced via apply)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight.view(self.x_embedder.proj.weight.size(0), -1))
        nn.init.zeros_(self.x_embedder.proj.bias)

        # Mask token: small Gaussian so it is not identical to data-carrying tokens
        nn.init.normal_(self.mask_token, std=0.02)

        # Timestep MLP output: small Gaussian
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # ====== AdaLN-Zero: the life-or-death init ======
        for block in list(self.encoder_blocks) + list(self.decoder_blocks):
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

        # Final layer: zero-init modulation and output projection so model
        # starts as an identity over latent coordinates.
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # ------------------------------------------------------------------ #
    # Patch-level mask helper
    # ------------------------------------------------------------------ #
    def _compute_mask_ratio(self, t: torch.Tensor) -> float:
        """Time-adaptive mask schedule — scalar per-batch for simplicity.

        The schedule is defined in *flow-time coordinates* where ``t=0``
        corresponds to pure noise (max mask) and ``t=1`` to data (no mask):

            gamma(t_flow) = ada_mask_max * cos^2(pi/2 * t_flow)

        Inputs can be either:
          * Flow-matching continuous t ∈ [0, 1]  — used directly.
          * DDPM integer timesteps in [0, T)      — higher t = more noise, so
            we convert to flow-time via ``t_flow = 1 - t / T`` before applying
            the schedule.
        """
        if self.const_mask_ratio is not None:
            return float(self.const_mask_ratio)
        if not self.use_ada_mask or not self.training:
            return 0.0

        t_float = t.float()
        if t_float.max().item() > 1.0:
            # DDPM-style: convert to flow-time (invert the noise axis).
            t_float = 1.0 - t_float / float(self.num_train_timesteps_ref)

        t_mean = float(t_float.mean().item())
        # Clamp defensively in case of numerical drift above [0, 1].
        t_mean = max(0.0, min(1.0, t_mean))
        return self.ada_mask_max * math.cos(math.pi * t_mean / 2.0) ** 2

    def _apply_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Random patch drop.  Returns (visible_tokens, ids_restore)."""
        if mask_ratio <= 0:
            return x, None

        B, N, D = x.shape
        len_keep = max(1, int(round(N * (1 - mask_ratio))))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        return x_vis, ids_restore

    def _unmask(
        self,
        x_vis: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Re-insert mask tokens at the correct positions."""
        B, _, D = x_vis.shape
        N = ids_restore.shape[1]
        num_masked = N - x_vis.shape[1]
        mask_tokens = self.mask_token.expand(B, num_masked, D)
        x_full = torch.cat([x_vis, mask_tokens], dim=1)
        return torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))

    def _unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) -> (B, C, H, W)."""
        p = self.patch_size
        c = self.out_channels
        h = w = self.x_embedder.grid_size
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- Embed ----
        x = self.x_embedder(x) + self.pos_embed       # (B, N, D)
        t_emb = self.t_embedder(t)                    # (B, D)
        c_emb = t_emb if c is None else t_emb + self.c_proj(c)

        # ---- Mask (train-time only) ----
        mask_ratio = self._compute_mask_ratio(t)
        x_enc, ids_restore = self._apply_mask(x, mask_ratio)

        # ---- Deep encoder (only sees visible tokens) ----
        for blk in self.encoder_blocks:
            x_enc = blk(x_enc, c_emb)

        # ---- Unmask + shallow decoder ----
        x_full = self._unmask(x_enc, ids_restore) if ids_restore is not None else x_enc
        for blk in self.decoder_blocks:
            x_full = blk(x_full, c_emb)

        # ---- Project back to patch pixels ----
        x_out = self.final_layer(x_full, c_emb)
        return self._unpatchify(x_out)


# ---------------------------------------------------------------------- #
# Registry of presets
# ---------------------------------------------------------------------- #
_DIT_PRESETS = {
    "dit_s": dict(hidden_size=384, encoder_depth=12, decoder_depth=2, num_heads=6),
    "dit_b": dict(hidden_size=768, encoder_depth=12, decoder_depth=2, num_heads=12),
    "dit_l": dict(hidden_size=1024, encoder_depth=24, decoder_depth=2, num_heads=16),
    "dit_xl": dict(hidden_size=1152, encoder_depth=28, decoder_depth=2, num_heads=16),
}


def build_dit(
    preset: str,
    input_size: int = 32,
    patch_size: int = 2,
    in_channels: int = 3,
    c_dim: Optional[int] = None,
    use_ada_mask: bool = False,
    ada_mask_max: float = 0.75,
    const_mask_ratio: Optional[float] = None,
    decoder_depth: Optional[int] = None,
    num_train_timesteps_ref: int = 1000,
) -> AsymmetricMaskedDiT:
    """Build a preset DiT (e.g. ``dit_b``) with the given latent dims.

    Passing ``decoder_depth`` overrides the preset (used by the "DecDepth=0"
    ablation to test necessity of the non-symmetric decoder).
    """
    if preset not in _DIT_PRESETS:
        raise ValueError(f"Unknown DiT preset '{preset}'. Choose from {list(_DIT_PRESETS)}.")
    cfg = dict(_DIT_PRESETS[preset])
    if decoder_depth is not None:
        cfg["decoder_depth"] = int(decoder_depth)
    cfg.update(
        input_size=input_size,
        patch_size=patch_size,
        in_channels=in_channels,
        c_dim=c_dim,
        use_ada_mask=use_ada_mask,
        ada_mask_max=ada_mask_max,
        const_mask_ratio=const_mask_ratio,
        num_train_timesteps_ref=num_train_timesteps_ref,
    )
    return AsymmetricMaskedDiT(**cfg)
