from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from utils.training import evaluation_mode, resolve_amp_config


def test_evaluation_mode_switches_modules_and_restores():
    unet = nn.Linear(4, 4)
    class_embedder = nn.Embedding(5, 4)

    unet.train(True)
    class_embedder.train(True)

    with evaluation_mode(unet, class_embedder):
        assert not unet.training
        assert not class_embedder.training

    assert unet.training
    assert class_embedder.training


def test_evaluation_mode_ignores_none_entries():
    module = nn.Linear(2, 2)
    module.train(True)

    with evaluation_mode(module, None):
        assert not module.training

    assert module.training


def test_resolve_amp_config_disables_autocast_for_none():
    cfg = resolve_amp_config(torch.device("cpu"), "none")

    assert cfg.device_type == "cpu"
    assert cfg.autocast_dtype is None
    assert not cfg.scaler_enabled


def test_resolve_amp_config_uses_bf16_on_cpu():
    cfg = resolve_amp_config(torch.device("cpu"), "bf16")

    assert cfg.device_type == "cpu"
    assert cfg.autocast_dtype == torch.bfloat16
    assert not cfg.scaler_enabled


def test_resolve_amp_config_disables_fp16_on_cpu():
    cfg = resolve_amp_config(torch.device("cpu"), "fp16")

    assert cfg.device_type == "cpu"
    assert cfg.autocast_dtype is None
    assert not cfg.scaler_enabled
