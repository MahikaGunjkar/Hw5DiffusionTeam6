import math
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from schedulers import DDPMScheduler, DDIMScheduler
from train import (
    checkpoint_artifact_alias_for_resume,
    checkpoint_artifact_name,
    bind_wandb_resume_config,
    download_wandb_training_checkpoint,
    upload_wandb_checkpoints,
)
from utils.checkpoint import (
    infer_resume_global_step,
    load_checkpoint,
    restore_lr_scheduler_progress,
    save_checkpoint,
)


class CheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_load_checkpoint_skips_timestep_shape_mismatch(self):
        model = torch.nn.Linear(1, 1)
        saved_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=1000,
        )
        save_checkpoint(model, saved_scheduler, epoch=0, save_dir=self.tmpdir)

        resume_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            num_inference_steps=50,
        )
        checkpoint = load_checkpoint(
            model,
            resume_scheduler,
            checkpoint_path=os.path.join(self.tmpdir, "checkpoint_epoch_0.pth"),
        )

        self.assertEqual(checkpoint["epoch"], 0)
        self.assertEqual(resume_scheduler.timesteps.shape[0], 50)
        self.assertTrue(torch.allclose(resume_scheduler.alphas, saved_scheduler.alphas))

    def test_infer_resume_global_step_uses_scheduler_progress_when_present(self):
        checkpoint = {
            "epoch": 2,
            "lr_scheduler_state_dict": {
                "last_epoch": 30,
                "_step_count": 31,
            },
        }
        self.assertEqual(infer_resume_global_step(checkpoint, steps_per_epoch=10), 30)

    def test_infer_resume_global_step_falls_back_to_completed_epochs_for_legacy_checkpoint(self):
        checkpoint = {"epoch": 4}
        self.assertEqual(infer_resume_global_step(checkpoint, steps_per_epoch=10), 50)

    def test_restore_lr_scheduler_progress_preserves_current_t_max(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=200,
            eta_min=1e-6,
        )

        restore_lr_scheduler_progress(scheduler, global_step=30)

        expected_model = torch.nn.Linear(1, 1)
        expected_optimizer = torch.optim.AdamW(expected_model.parameters(), lr=1e-3)
        expected_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            expected_optimizer,
            T_max=200,
            eta_min=1e-6,
        )
        for _ in range(30):
            expected_optimizer.step()
            expected_scheduler.step()

        self.assertEqual(scheduler.T_max, 200)
        self.assertEqual(scheduler.last_epoch, 30)
        self.assertTrue(
            math.isclose(
                optimizer.param_groups[0]["lr"],
                expected_optimizer.param_groups[0]["lr"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def test_bind_wandb_resume_config_derives_log_target_from_ckpt_run_path(self):
        args = SimpleNamespace(
            resume_enabled=True,
            wandb_run_id=None,
            wandb_resume=None,
            wandb_ckpt_run_path="team-a/ddpm/run123",
        )

        bind_wandb_resume_config(args, environ={})

        self.assertEqual(args.wandb_entity, "team-a")
        self.assertEqual(args.wandb_project, "ddpm")
        self.assertEqual(args.wandb_run_id, "run123")
        self.assertEqual(args.wandb_resume, "must")

    def test_bind_wandb_resume_config_rejects_conflicting_run_ids(self):
        args = SimpleNamespace(
            resume_enabled=True,
            wandb_run_id="different",
            wandb_resume="must",
            wandb_ckpt_run_path="team-a/ddpm/run123",
        )

        with self.assertRaisesRegex(ValueError, "must match"):
            bind_wandb_resume_config(args, environ={})

    def test_bind_wandb_resume_config_requires_explicit_run_target_when_enabled(self):
        args = SimpleNamespace(
            resume_enabled=True,
            wandb_run_id=None,
            wandb_resume=None,
            wandb_ckpt_run_path=None,
        )

        with self.assertRaisesRegex(ValueError, "requires a specific W&B run target"):
            bind_wandb_resume_config(args, environ={})

    def test_upload_wandb_checkpoints_raises_when_upload_fails(self):
        fake_run = mock.Mock()
        fake_wandb = mock.Mock()
        fake_artifact = mock.Mock()
        fake_wandb.Artifact.return_value = fake_artifact
        fake_run.log_artifact.side_effect = RuntimeError("upload failed")

        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            upload_wandb_checkpoints(
                fake_run,
                wandb_module=fake_wandb,
                run_id="run123",
                ckpt_path="/tmp/output/checkpoints/checkpoint_epoch_0.pth",
                ema_ckpt_path="/tmp/output/checkpoints/ema_checkpoint_epoch_0.pth",
                epoch=0,
                global_step=10,
                is_last_epoch=False,
            )

    def test_checkpoint_artifact_name_is_derived_from_run_id(self):
        self.assertEqual(
            checkpoint_artifact_name("run123"),
            "run-run123-training-checkpoints",
        )

    def test_checkpoint_artifact_alias_for_resume_uses_latest_by_default(self):
        self.assertEqual(checkpoint_artifact_alias_for_resume(None), "latest")
        self.assertEqual(checkpoint_artifact_alias_for_resume(7), "epoch-7")

    def test_upload_wandb_checkpoints_logs_model_artifact(self):
        fake_run = mock.Mock()
        fake_wandb = mock.Mock()
        fake_artifact = mock.Mock()
        fake_wandb.Artifact.return_value = fake_artifact

        with mock.patch("train.shutil.copy2"):
            upload_wandb_checkpoints(
                fake_run,
                wandb_module=fake_wandb,
                run_id="run123",
                ckpt_path="/tmp/output/checkpoints/checkpoint_epoch_3.pth",
                ema_ckpt_path="/tmp/output/checkpoints/ema_checkpoint_epoch_3.pth",
                epoch=3,
                global_step=40,
                is_last_epoch=True,
            )

        fake_wandb.Artifact.assert_called_once()
        fake_artifact.add_file.assert_any_call(
            local_path="/tmp/output/checkpoints/checkpoint_epoch_3.pth",
            name="checkpoint.pth",
        )
        fake_artifact.add_file.assert_any_call(
            local_path="/tmp/output/checkpoints/ema_checkpoint_epoch_3.pth",
            name="ema_checkpoint.pth",
        )
        fake_run.log_artifact.assert_called_once()
        _, kwargs = fake_run.log_artifact.call_args
        self.assertEqual(kwargs["aliases"], ["epoch-3", "last"])

    def test_download_wandb_training_checkpoint_does_not_fallback_to_legacy_run_files(self):
        with mock.patch(
            "train._download_wandb_training_checkpoint_from_artifact",
            side_effect=FileNotFoundError("artifact missing"),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "artifact missing"):
                download_wandb_training_checkpoint(
                    "team-a/ddpm/run123",
                    epoch=None,
                    cache_root=self.tmpdir,
                )


if __name__ == "__main__":
    unittest.main()
