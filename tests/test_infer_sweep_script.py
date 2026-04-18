from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_infer_sweep_reuses_training_config_for_dit_ckpt(tmp_path):
    exp_dir = tmp_path / "exp-03"
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    ckpt_path = ckpt_dir / "checkpoint_epoch_299.pth"
    ckpt_path.touch()

    config_path = exp_dir / "config.yaml"
    config_path.write_text(
        "framework: flow_matching\n"
        "model_type: dit_b\n"
        "use_cfg: true\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["DRY_RUN"] = "1"

    result = subprocess.run(
        ["bash", "scripts/infer_sweep.sh", str(ckpt_path), "dit_b"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert str(config_path) in result.stdout
    assert "--cfg_guidance_scale 1.5" in result.stdout
