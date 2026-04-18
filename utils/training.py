from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass(frozen=True)
class AmpConfig:
    """Runtime AMP settings derived from device + user precision flag."""

    device_type: str
    autocast_dtype: Optional[torch.dtype]
    scaler_enabled: bool

    def autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.autocast_dtype)


def resolve_amp_config(
    device: torch.device | str,
    mixed_precision: str,
) -> AmpConfig:
    """Resolve autocast/scaler policy for the current runtime."""
    device = torch.device(device)
    device_type = device.type

    if mixed_precision in {"none", "fp32"}:
        return AmpConfig(device_type=device_type, autocast_dtype=None, scaler_enabled=False)

    if mixed_precision == "bf16":
        if device_type == "cuda":
            if torch.cuda.is_bf16_supported():
                return AmpConfig(
                    device_type="cuda",
                    autocast_dtype=torch.bfloat16,
                    scaler_enabled=False,
                )
            # V100-class GPUs do not support bf16 autocast. Fall back to fp16
            # so the flag still enables reduced-precision training.
            return AmpConfig(
                device_type="cuda",
                autocast_dtype=torch.float16,
                scaler_enabled=True,
            )
        if device_type == "cpu":
            return AmpConfig(
                device_type="cpu",
                autocast_dtype=torch.bfloat16,
                scaler_enabled=False,
            )
        return AmpConfig(device_type=device_type, autocast_dtype=None, scaler_enabled=False)

    if mixed_precision == "fp16":
        if device_type == "cuda":
            return AmpConfig(
                device_type="cuda",
                autocast_dtype=torch.float16,
                scaler_enabled=True,
            )
        return AmpConfig(device_type=device_type, autocast_dtype=None, scaler_enabled=False)

    raise ValueError(f"Unsupported mixed_precision mode: {mixed_precision}")


def create_grad_scaler(amp_config: AmpConfig) -> torch.amp.GradScaler:
    """Create a GradScaler that is active only when needed."""
    return torch.amp.GradScaler(
        device=amp_config.device_type,
        enabled=amp_config.scaler_enabled,
    )


@contextmanager
def evaluation_mode(*modules) -> Iterator[None]:
    """Temporarily switch modules to eval mode and restore their prior state."""
    states = []
    active_modules = [module for module in modules if module is not None]
    for module in active_modules:
        states.append((module, module.training))
        module.eval()

    try:
        yield
    finally:
        for module, was_training in reversed(states):
            module.train(was_training)
