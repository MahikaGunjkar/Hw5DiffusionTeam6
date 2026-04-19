import os
import sys
import copy
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from datetime import timedelta
from logging import getLogger as get_logger
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from models import UNet, VAE, ClassEmbedder, build_dit
from schedulers import DDPMScheduler, DDIMScheduler, RectifiedFlowScheduler
from pipelines import DDPMPipeline, FlowMatchingPipeline
from utils import (
    seed_everything,
    init_distributed_device,
    is_primary,
    AverageMeter,
    str2bool,
    save_checkpoint,
    save_checkpoint_atomic,
    extract_tar_if_needed,
)
from utils.checkpoint import load_checkpoint
from utils.training import create_grad_scaler, evaluation_mode, resolve_amp_config
# fid_utils is imported lazily inside the FID evaluation block to avoid
# crashing --help when torchmetrics/transformers versions mismatch.

logger = get_logger(__name__)


# ===================== EMA Helper =====================

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        """Swap model params with EMA shadow params (for inference)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model params after inference."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ===================== Argument Parsing =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM model.")

    # config file(s) - later files override earlier ones, so an ablation
    # YAML can ship only the deltas from the base config.
    parser.add_argument("--config", type=str, nargs='+', default=['configs/ddpm.yaml'],
                        help="One or more YAML config files; later override earlier.")

    # ---- Dataset ----
    # BUG FIX #8 / new feature: use a tar path instead of a pre-extracted data_dir
    parser.add_argument("--tar_path", type=str, default=None,
                        help="Path to imagenet100_128x128.tar.gz. "
                             "The archive is extracted on first run; subsequent runs skip extraction.")
    parser.add_argument("--extract_dir", type=str, default="./dataset_extracted",
                        help="Directory to stream-extract the tar into.")
    # data_dir is still supported for users who already have extracted data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Pre-extracted train/ directory. Ignored when --tar_path is set.")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=100)

    # ---- Training ----
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default='none',
                        choices=['fp16', 'bf16', 'fp32', 'none'])

    # ---- DDPM ----
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=0.0002)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default='linear')
    parser.add_argument("--variance_type", type=str, default='fixed_small')
    parser.add_argument("--prediction_type", type=str, default='epsilon')
    parser.add_argument("--clip_sample", type=str2bool, default=True)
    parser.add_argument("--clip_sample_range", type=float, default=1.0)

    # ---- UNet ----
    parser.add_argument("--unet_in_size", type=int, default=128,
                        help="UNet input spatial size. Auto-set to 32 when latent_ddpm is enabled.")
    parser.add_argument("--unet_in_ch", type=int, default=3,
                        help="UNet input channels.")
    parser.add_argument("--unet_ch", type=int, default=128)
    parser.add_argument("--unet_ch_mult", type=int, default=[1, 2, 2, 2], nargs='+')
    parser.add_argument("--unet_attn", type=int, default=[1, 2, 3], nargs='+')
    parser.add_argument("--unet_num_res_blocks", type=int, default=2)
    parser.add_argument("--unet_dropout", type=float, default=0.0)

    # ---- Advanced: OT-AdaMask Flow framework flags ----
    parser.add_argument("--framework", type=str, default='ddpm',
                        choices=['ddpm', 'flow_matching'],
                        help="Training framework. 'ddpm' (baseline) or 'flow_matching' (Rectified Flow).")
    parser.add_argument("--model_type", type=str, default='unet',
                        choices=['unet', 'dit_s', 'dit_b', 'dit_l', 'dit_xl'],
                        help="Denoising network architecture.")
    parser.add_argument("--use_ot", type=str2bool, default=False,
                        help="Minibatch OT pairing (flow_matching only).")
    parser.add_argument("--ot_max_batch", type=int, default=64,
                        help="Max batch size for Hungarian OT before requiring Sinkhorn.")
    parser.add_argument("--use_ada_mask", type=str2bool, default=False,
                        help="Time-adaptive random patch masking (DiT only).")
    parser.add_argument("--ada_mask_max", type=float, default=0.75,
                        help="Maximum AdaMask drop ratio at t=0.")
    parser.add_argument("--const_mask_ratio", type=float, default=None,
                        help="If set, override AdaMask schedule with a constant ratio.")
    parser.add_argument("--dit_patch_size", type=int, default=2)
    parser.add_argument("--dit_decoder_depth", type=int, default=2)
    parser.add_argument("--flow_solver", type=str, default='euler', choices=['euler', 'heun'])
    parser.add_argument("--num_inference_steps_flow", type=int, default=10)

    # ---- VAE ----
    parser.add_argument("--latent_ddpm", type=str2bool, default=True,
                        help="Use VAE for latent DDPM. Automatically sets unet_in_size=32, unet_in_ch=3.")

    # ---- CFG ----
    parser.add_argument("--use_cfg", type=str2bool, default=False)
    parser.add_argument("--cfg_guidance_scale", type=float, default=2.0)

    # ---- DDIM ----
    parser.add_argument("--use_ddim", type=str2bool, default=False)

    # ---- Checkpoint (inference) ----
    parser.add_argument("--ckpt", type=str, default=None)

    # ---- Wandb ----
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity (team or user slug). Overrides yaml.")
    parser.add_argument("--wandb_project", type=str, default='ddpm',
                        help="Wandb project name.")

    # ---- Resume ----
    parser.add_argument("--resume_ckpt_path", type=str, default=None,
                        help="Local Ocean checkpoint (.pth) to resume from. "
                             "Takes precedence over --resume_enabled. When the path "
                             "is under an existing experiment dir, that dir is reused "
                             "instead of spawning exp-N-<run_name>.")
    parser.add_argument("--resume_enabled", type=str2bool, default=False,
                        help="Download artifact from wandb and resume training.")
    parser.add_argument("--wandb_resume", type=str,
                        choices=['allow', 'must', 'never', 'auto'], default='allow',
                        help="Wandb run resume mode passed to wandb.init.")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Wandb run ID to resume. Env WANDB_RUN_ID takes priority.")
    parser.add_argument("--wandb_ckpt_run_path", type=str, default=None,
                        help="entity/project/run_id for artifact download on resume.")
    parser.add_argument("--wandb_ckpt_alias", type=str, default='latest',
                        help="Artifact alias to pull on resume, e.g. latest/best.")
    parser.add_argument("--wandb_ckpt_cache_dir", type=str, default=None,
                        help="Local dir for artifact download cache.")

    # ---- Checkpoint saving ----
    parser.add_argument("--save_every_n_epochs", type=int, default=5,
                        help="Upload wandb artifact every N epochs.")

    # ---- FID evaluation ----
    parser.add_argument("--fid_every_n_epochs", type=int, default=0,
                        help="Compute FID every N epochs (0 = disabled).")
    parser.add_argument("--fid_num_samples", type=int, default=2048,
                        help="Number of generated samples for FID computation.")
    parser.add_argument("--fid_ref_stats_path", type=str, default=None,
                        help="Path to reference .npz stats file. Auto-computed on first run if None.")

    # First parse - get config file path
    args = parser.parse_args()

    # Load YAML config and set as new defaults
    if args.config:
        merged = {}
        file_yaml = yaml.YAML()
        for cfg_path in args.config:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = file_yaml.load(f) or {}
            # Strip the `slurm:` block - it is only for scripts/submit.sh.
            cfg.pop('slurm', None)
            merged.update(cfg)
        parser.set_defaults(**merged)

    # Re-parse - command-line overrides YAML
    args = parser.parse_args()

    # Auto-override UNet config when using latent DDPM
    if args.latent_ddpm:
        args.unet_in_size = 32
        args.unet_in_ch = 3

    return args


# ===================== Resume Helper =====================

def _resume_from_wandb_artifact(args, unet, scheduler, vae, class_embedder, optimizer):
    """
    Download checkpoint artifact from wandb and load it.
    Returns start_epoch (int): the epoch to resume from (saved_epoch + 1).
    """
    if not args.wandb_ckpt_run_path:
        logger.warning("resume_enabled=True but wandb_ckpt_run_path is not set. Starting from epoch 0.")
        return 0

    cache_dir = args.wandb_ckpt_cache_dir or './wandb_resume_cache'
    artifact_name = f"{args.wandb_ckpt_run_path}:{args.wandb_ckpt_alias}"
    logger.info(f"Downloading artifact {artifact_name} to {cache_dir}")

    try:
        api = wandb.Api()
        artifact = api.artifact(artifact_name)
        ckpt_dir = artifact.download(root=cache_dir)
        # Find the checkpoint file inside the artifact directory
        ckpt_files = list(Path(ckpt_dir).glob('*.pth'))
        if not ckpt_files:
            raise FileNotFoundError(f"No .pth file found in artifact download dir: {ckpt_dir}")
        ckpt_path = str(ckpt_files[0])
        logger.info(f"Loaded artifact checkpoint: {ckpt_path}")
        saved_epoch = load_checkpoint(unet, scheduler, vae, class_embedder, optimizer, ckpt_path)
        start_epoch = saved_epoch + 1
        logger.info(f"Resuming from epoch {start_epoch}")
        return start_epoch
    except Exception as exc:
        logger.error(f"Resume artifact download failed: {exc}. Starting from epoch 0.")
        return 0


# ===================== Main =====================

def main():
    args = parse_args()
    seed_everything(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    device = init_distributed_device(args)
    if args.distributed:
        logger.info(
            f'Training in distributed mode. '
            f'Process {args.rank}, total {args.world_size}, device {args.device}.'
        )
    else:
        logger.info(f'Training with a single process on device {args.device}.')
    assert args.rank >= 0

    # ---- BUG FIX #8: create output_dir BEFORE listdir ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Dataset: extract tar on the fly if tar_path is provided ----
    if args.tar_path is not None:
        # Primary process extracts; others wait (distributed barrier below)
        if is_primary(args):
            data_dir = extract_tar_if_needed(args.tar_path, args.extract_dir)
            # Write resolved path so other ranks can read it
            path_file = os.path.join(args.extract_dir, ".train_path")
            with open(path_file, "w") as f:
                f.write(data_dir)
        if args.distributed:
            torch.distributed.barrier()
        path_file = os.path.join(args.extract_dir, ".train_path")
        with open(path_file) as f:
            data_dir = f.read().strip()
    elif args.data_dir is not None:
        data_dir = args.data_dir
    else:
        raise ValueError(
            "You must provide either --tar_path (path to .tar.gz) or "
            "--data_dir (path to already-extracted train/ folder)."
        )

    logger.info(f"Using dataset: {data_dir}")

    # ===================== Dataset =====================
    logger.info("Creating dataset")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),                       # [0, 1]
        transforms.Normalize([0.5], [0.5]),           # -> [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    shuffle = False if sampler else True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    total_batch_size = args.batch_size * args.world_size
    args.total_batch_size = total_batch_size

    # ---- Experiment folder ----
    # Resume precedence: if --resume_ckpt_path points into <output_dir>/<existing_run>/checkpoints/<x>.pth,
    # reuse the existing exp-N-<run_name> directory. Otherwise allocate a fresh index.
    reused_run_dir = None
    if args.resume_ckpt_path:
        ckpt_p = Path(args.resume_ckpt_path).resolve()
        out_p = Path(args.output_dir).resolve()
        try:
            rel = ckpt_p.relative_to(out_p)
            # Expect <run_name>/checkpoints/<file>.pth
            if len(rel.parts) >= 2 and rel.parts[1] == 'checkpoints':
                reused_run_dir = rel.parts[0]
        except ValueError:
            reused_run_dir = None

    if reused_run_dir is not None:
        args.run_name = reused_run_dir
    elif args.run_name is None:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}'
    else:
        args.run_name = f'exp-{len(os.listdir(args.output_dir))}-{args.run_name}'
    output_dir = os.path.join(args.output_dir, args.run_name)
    save_dir = os.path.join(output_dir, 'checkpoints')
    if is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    # ===================== Model Setup =====================
    logger.info(f"Creating model: framework={args.framework}, model_type={args.model_type}")

    # Class-conditioning dimension depends on architecture
    _DIT_HIDDEN = {'dit_s': 384, 'dit_b': 768, 'dit_l': 1024, 'dit_xl': 1152}
    if args.model_type == 'unet':
        class_emb_dim = args.unet_ch
    else:
        class_emb_dim = _DIT_HIDDEN[args.model_type]

    if args.model_type == 'unet':
        unet = UNet(
            input_size=args.unet_in_size,
            input_ch=args.unet_in_ch,
            T=args.num_train_timesteps,
            ch=args.unet_ch,
            ch_mult=args.unet_ch_mult,
            attn=args.unet_attn,
            num_res_blocks=args.unet_num_res_blocks,
            dropout=args.unet_dropout,
            conditional=args.use_cfg,
            c_dim=args.unet_ch,
        )
    else:
        unet = build_dit(
            preset=args.model_type,
            input_size=args.unet_in_size,
            patch_size=args.dit_patch_size,
            in_channels=args.unet_in_ch,
            c_dim=class_emb_dim if args.use_cfg else None,
            use_ada_mask=args.use_ada_mask,
            ada_mask_max=args.ada_mask_max,
            const_mask_ratio=args.const_mask_ratio,
            decoder_depth=args.dit_decoder_depth,
            num_train_timesteps_ref=args.num_train_timesteps,
        )
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"{args.model_type} parameters: {num_params / 1e6:.2f}M")

    if args.framework == 'flow_matching':
        noise_scheduler = RectifiedFlowScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps_flow,
            solver=args.flow_solver,
            use_ot=args.use_ot,
            ot_max_batch=args.ot_max_batch,
        )
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            prediction_type=args.prediction_type,
            clip_sample=args.clip_sample,
            clip_sample_range=args.clip_sample_range,
        )

    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()

    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            embed_dim=class_emb_dim,
            n_classes=args.num_classes,
            cond_drop_rate=0.1,
        )

    unet = unet.to(device)
    noise_scheduler = noise_scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # ===================== Optimizer =====================
    params = list(unet.parameters())
    if class_embedder:
        params += list(class_embedder.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    amp_config = resolve_amp_config(device, args.mixed_precision)
    grad_scaler = create_grad_scaler(amp_config)
    if is_primary(args):
        autocast_name = str(amp_config.autocast_dtype).replace("torch.", "") if amp_config.autocast_dtype else "disabled"
        logger.info(
            "AMP config: autocast=%s, grad_scaler=%s",
            autocast_name,
            amp_config.scaler_enabled,
        )

    num_update_steps_per_epoch = len(train_loader)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_train_steps, eta_min=1e-6
    )

    # ===================== EMA =====================
    ema = EMA(unet, decay=0.9999)

    # ===================== Distributed =====================
    if args.distributed:
        unet = torch.nn.parallel.DistributedDataParallel(
            unet, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
        unet_wo_ddp = unet.module
        if class_embedder:
            class_embedder = torch.nn.parallel.DistributedDataParallel(
                class_embedder, device_ids=[args.device], output_device=args.device, find_unused_parameters=False)
            class_embedder_wo_ddp = class_embedder.module
        else:
            class_embedder_wo_ddp = None
    else:
        unet_wo_ddp = unet
        class_embedder_wo_ddp = class_embedder
    vae_wo_ddp = vae

    # Inference scheduler and pipeline selection
    if args.framework == 'flow_matching':
        inference_scheduler = noise_scheduler     # reuse the flow scheduler
        pipeline = FlowMatchingPipeline(
            unet=unet_wo_ddp,
            scheduler=inference_scheduler,
            vae=vae_wo_ddp,
            class_embedder=class_embedder_wo_ddp,
        )
    else:
        if args.use_ddim:
            inference_scheduler = DDIMScheduler(
                num_train_timesteps=args.num_train_timesteps,
                num_inference_steps=args.num_inference_steps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                beta_schedule=args.beta_schedule,
                variance_type=args.variance_type,
                prediction_type=args.prediction_type,
                clip_sample=args.clip_sample,
                clip_sample_range=args.clip_sample_range,
            ).to(device)
        else:
            inference_scheduler = noise_scheduler

        pipeline = DDPMPipeline(
            unet=unet_wo_ddp,
            scheduler=inference_scheduler,
            vae=vae_wo_ddp,
            class_embedder=class_embedder_wo_ddp,
        )

    # ===================== Config Dump =====================
    if is_primary(args):
        experiment_config = vars(args)
        with open(os.path.join(output_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            file_yaml = yaml.YAML()
            file_yaml.dump(experiment_config, f)

    # ===================== Wandb Init =====================
    if is_primary(args):
        # env var takes priority over argparse (resume / CI injection)
        run_id = os.environ.get("WANDB_RUN_ID") or args.wandb_run_id
        resume = os.environ.get("WANDB_RESUME") or args.wandb_resume
        wandb_logger = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            id=run_id,
            resume=resume,
            config=vars(args),
            dir=os.environ.get("WANDB_DIR"),
            tags=[args.model_type, args.framework,
                  f"exp{os.environ.get('EXP_ID', 'local')}"],
        )
        # Step monotonicity - rules/ml-training-wandb.md section 1
        wandb_logger.define_metric("global_step")
        wandb_logger.define_metric("train/*", step_metric="global_step")
        wandb_logger.define_metric("val/*", step_metric="global_step")
        wandb_logger.define_metric("val/fid", summary="min")
        wandb_logger.define_metric("train/loss", summary="min")
    else:
        wandb_logger = None

    # ===================== Resume (local Ocean ckpt > wandb artifact) =====================
    start_epoch = 0
    best_fid = float('inf')
    resume_state = None
    if args.resume_ckpt_path:
        if not os.path.exists(args.resume_ckpt_path):
            raise FileNotFoundError(f"--resume_ckpt_path not found: {args.resume_ckpt_path}")
        logger.info(f"Resuming from local checkpoint: {args.resume_ckpt_path}")
        resume_state = load_checkpoint(
            unet_wo_ddp, inference_scheduler, vae_wo_ddp,
            class_embedder_wo_ddp, optimizer,
            checkpoint_path=args.resume_ckpt_path,
            lr_scheduler=lr_scheduler,
            grad_scaler=grad_scaler,
        )
        start_epoch = resume_state['epoch'] + 1
        logger.info(f"Resume start_epoch={start_epoch}")
    elif args.resume_enabled:
        start_epoch = _resume_from_wandb_artifact(
            args, unet_wo_ddp, inference_scheduler, vae_wo_ddp,
            class_embedder_wo_ddp, optimizer
        )

    # Restore EMA shadow into the live EMA helper and best_fid tracker.
    if resume_state is not None:
        if resume_state.get('ema_shadow') is not None:
            for name, tensor in resume_state['ema_shadow'].items():
                if name in ema.shadow:
                    ema.shadow[name] = tensor.to(ema.shadow[name].device)
            logger.info("EMA shadow restored from checkpoint.")
        if resume_state.get('best_fid') is not None:
            best_fid = float(resume_state['best_fid'])
            logger.info(f"best_fid restored = {best_fid:.4f}")
        # Legacy ckpts without lr_scheduler: fast-forward via step()
        if not resume_state.get('has_lr_scheduler') and start_epoch > 0:
            replay_steps = start_epoch * num_update_steps_per_epoch
            logger.info(f"Legacy ckpt: advancing lr_scheduler by {replay_steps} steps")
            for _ in range(replay_steps):
                lr_scheduler.step()

    # ===================== FID Reference Stats (lazy init, rank-0 only) =====================
    ref_mu = None
    ref_sigma = None
    if is_primary(args) and args.fid_every_n_epochs > 0:
        from fid_utils import extract_features_from_tensors, compute_statistics, compute_fid, save_stats_npz, load_stats_npz  # noqa: lazy import
        ref_stats_path = args.fid_ref_stats_path
        if ref_stats_path is None:
            proj_root = os.environ.get("PROJ_ROOT", os.path.dirname(os.path.abspath(__file__)))
            ref_stats_path = os.path.join(proj_root, "fid_ref_imagenet100_128.npz")
        if os.path.exists(ref_stats_path):
            logger.info(f"Loading cached FID reference stats from {ref_stats_path}")
            ref_mu, ref_sigma = load_stats_npz(ref_stats_path)
        else:
            logger.info("Reference FID stats not found - will compute from training data on first FID epoch.")

    # ===================== Training Loop =====================
    if is_primary(args):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples           = {len(train_dataset)}")
        logger.info(f"  Num Epochs             = {args.num_epochs}")
        logger.info(f"  Batch size per device  = {args.batch_size}")
        logger.info(f"  Total batch size       = {total_batch_size}")
        logger.info(f"  Steps per epoch        = {num_update_steps_per_epoch}")
        logger.info(f"  Total steps            = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not is_primary(args))

    for epoch in range(start_epoch, args.num_epochs):

        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        args.epoch = epoch
        if is_primary(args):
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")

        loss_m = AverageMeter()
        unet.train()
        if class_embedder:
            class_embedder.train()

        for step, (images, labels) in enumerate(train_loader):
            batch_size = images.size(0)
            images = images.to(device)
            labels = labels.to(device)

            # Latent DDPM - encode with frozen VAE
            if vae:
                with torch.no_grad():
                    images = vae.encode(images)
                images = images * vae.scaling_factor   # scale to ~unit std

            # Deterministic per-step seed so the DiT mask sampling
            # (models/dit.py::_apply_mask) is reproducible across runs and
            # distinct across ranks. Offset by rank so each DDP worker masks
            # its own micro-batch differently.
            global_step = epoch * num_update_steps_per_epoch + step
            torch.manual_seed(
                args.seed + global_step * max(args.world_size, 1) + args.rank
            )

            optimizer.zero_grad()

            with amp_config.autocast_context():
                # CFG class embeddings (with 10 % unconditional dropout inside ClassEmbedder)
                class_emb = class_embedder(labels) if class_embedder is not None else None

                # ================= Forward + loss =================
                if args.framework == 'flow_matching':
                    # Rectified Flow with optional OT pairing.
                    # Convention: t=0 -> noise, t=1 -> data; v* = x_1 - x_0.
                    x_1 = images                              # real latents
                    x_0 = torch.randn_like(images)            # pure Gaussian noise
                    x_0 = noise_scheduler.pair_ot(x_0, x_1)   # identity if use_ot=False

                    t = torch.rand(batch_size, device=device)
                    x_t = noise_scheduler.interpolate(x_0, x_1, t)
                    target = noise_scheduler.velocity_target(x_0, x_1)

                    # UNet uses a discrete timestep embedding table; rescale
                    # continuous t -> integer index. DiT accepts continuous t.
                    if args.model_type == 'unet':
                        t_in = (t * args.num_train_timesteps).long().clamp(
                            0, args.num_train_timesteps - 1)
                    else:
                        t_in = t

                    model_pred = unet(x_t, t_in, c=class_emb)
                    loss = F.mse_loss(model_pred, target)
                else:
                    # Standard DDPM epsilon-prediction baseline.
                    noise = torch.randn_like(images)
                    timesteps = torch.randint(
                        0, args.num_train_timesteps, (batch_size,), device=device
                    ).long()
                    noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

                    model_pred = unet(noisy_images, timesteps, c=class_emb)

                    if args.prediction_type == 'epsilon':
                        target = noise
                    else:
                        raise NotImplementedError(
                            f"prediction_type {args.prediction_type} not supported."
                        )

                    loss = F.mse_loss(model_pred, target)

            # ---- NaN/Inf early exit (exit 42 = do not retry) ----
            if not torch.isfinite(loss):
                if is_primary(args):
                    Path(output_dir, "NAN_DETECTED").write_text(
                        f"epoch={epoch} step={step}\n"
                    )
                    try:
                        wandb_logger.alert(
                            title="NaN loss",
                            text=f"exp {args.run_name} died at epoch={epoch} step={step}",
                        )
                    except Exception:
                        pass
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
                sys.exit(42)

            loss_m.update(loss.item())

            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                if args.grad_clip:
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
            lr_scheduler.step()
            ema.update(unet_wo_ddp)
            progress_bar.update(1)

            if step % 100 == 0 and is_primary(args):
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs}, "
                    f"Step {step}/{num_update_steps_per_epoch}, "
                    f"Loss {loss.item():.4f} (avg {loss_m.avg:.4f})"
                )
                wandb_logger.log({
                    'train/loss': loss_m.avg,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'global_step': global_step,
                })

        # ===================== Epoch-end: Sample & Save =====================
        ema.apply_shadow(unet_wo_ddp)
        with evaluation_mode(unet, class_embedder):
            generator = torch.Generator(device=device)
            generator.manual_seed(epoch + args.seed)

            infer_steps = (
                args.num_inference_steps_flow
                if args.framework == 'flow_matching'
                else args.num_inference_steps
            )

            if args.use_cfg:
                classes = torch.randint(0, args.num_classes, (4,), device=device)
                gen_images = pipeline(
                    batch_size=4,
                    num_inference_steps=infer_steps,
                    classes=classes.tolist(),
                    guidance_scale=args.cfg_guidance_scale,
                    generator=generator,
                    device=device,
                )
            else:
                gen_images = pipeline(
                    batch_size=4,
                    num_inference_steps=infer_steps,
                    generator=generator,
                    device=device,
                )

            grid_image = Image.new('RGB', (4 * args.image_size, args.image_size))
            for i, image in enumerate(gen_images):
                grid_image.paste(image, (i * args.image_size, 0))

            if is_primary(args):
                wandb_logger.log({
                    'val/gen_images': wandb.Image(grid_image),
                    'global_step': global_step,
                })

        ema.restore(unet_wo_ddp)

        # ===================== FID Evaluation (rank-0, every N epochs) =====================
        val_fid = None
        if is_primary(args) and args.fid_every_n_epochs > 0 and (epoch + 1) % args.fid_every_n_epochs == 0:
            logger.info(f"Computing FID at epoch {epoch+1}...")
            from fid_utils import extract_features_from_tensors, compute_statistics, compute_fid, save_stats_npz, load_stats_npz  # noqa: lazy import

            # Lazily build reference stats from training data if not yet available.
            # exp01-04 may hit this concurrently — use an exclusive file lock so
            # only one job generates the shared npz and peers load the result.
            if ref_mu is None:
                ref_stats_path = args.fid_ref_stats_path
                if ref_stats_path is None:
                    proj_root = os.environ.get('PROJ_ROOT', os.path.dirname(os.path.abspath(__file__)))
                    ref_stats_path = os.path.join(proj_root, 'fid_ref_imagenet100_128.npz')

                import fcntl
                lock_path = ref_stats_path + '.lock'
                Path(ref_stats_path).parent.mkdir(parents=True, exist_ok=True)
                with open(lock_path, 'w') as lock_fh:
                    fcntl.flock(lock_fh, fcntl.LOCK_EX)
                    try:
                        if os.path.exists(ref_stats_path):
                            logger.info(f"FID ref stats exist (built by peer): {ref_stats_path}")
                            ref_mu, ref_sigma = load_stats_npz(ref_stats_path)
                        else:
                            logger.info("Computing reference FID stats (one-time, under lock)...")
                            ref_transform_fid = transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size)),
                                transforms.ToTensor(),   # [0, 1]
                            ])
                            ref_dataset_fid = datasets.ImageFolder(root=data_dir, transform=ref_transform_fid)
                            n_ref = min(args.fid_num_samples, len(ref_dataset_fid))
                            indices = torch.randperm(len(ref_dataset_fid))[:n_ref].tolist()
                            ref_subset = torch.utils.data.Subset(ref_dataset_fid, indices)
                            ref_loader_fid = torch.utils.data.DataLoader(
                                ref_subset, batch_size=64, shuffle=False,
                                num_workers=args.num_workers,
                            )
                            ref_tensors_list = []
                            for imgs_fid, _ in ref_loader_fid:
                                ref_tensors_list.append(imgs_fid)
                            ref_tensors_cat = torch.cat(ref_tensors_list, dim=0)
                            ref_features = extract_features_from_tensors(
                                ref_tensors_cat, device=str(device)
                            )
                            ref_mu, ref_sigma = compute_statistics(ref_features)
                            # Atomic write: tmp + rename so peers never load a partial file.
                            tmp_stats = ref_stats_path + '.tmp'
                            tmp_actual = save_stats_npz(ref_mu, ref_sigma, tmp_stats)
                            os.replace(tmp_actual, ref_stats_path)
                            logger.info(f"Reference FID stats saved to {ref_stats_path}")
                    finally:
                        fcntl.flock(lock_fh, fcntl.LOCK_UN)

            # Generate samples for FID scoring
            with evaluation_mode(unet, class_embedder):
                gen_fid_generator = torch.Generator(device=device)
                gen_fid_generator.manual_seed(epoch + args.seed + 9999)

                gen_imgs_list = []
                n_remaining = args.fid_num_samples
                fid_batch = 50
                while n_remaining > 0:
                    cur_batch = min(fid_batch, n_remaining)
                    if args.use_cfg:
                        fid_classes = torch.randint(0, args.num_classes, (cur_batch,), device=device)
                        batch_imgs = pipeline(
                            batch_size=cur_batch,
                            num_inference_steps=infer_steps,
                            classes=fid_classes.tolist(),
                            guidance_scale=args.cfg_guidance_scale,
                            generator=gen_fid_generator,
                            device=device,
                        )
                    else:
                        batch_imgs = pipeline(
                            batch_size=cur_batch,
                            num_inference_steps=infer_steps,
                            generator=gen_fid_generator,
                            device=device,
                        )
                    gen_imgs_list.extend(batch_imgs)
                    n_remaining -= cur_batch

            # Convert PIL images to float tensor [0, 1]
            gen_tensors_fid = torch.stack([
                transforms.ToTensor()(img) for img in gen_imgs_list
            ])
            gen_features = extract_features_from_tensors(gen_tensors_fid, device=str(device))
            gen_mu, gen_sigma = compute_statistics(gen_features)
            val_fid = compute_fid(ref_mu, ref_sigma, gen_mu, gen_sigma)
            logger.info(f"Epoch {epoch+1} FID = {val_fid:.4f}")
            wandb_logger.log({'val/fid': val_fid, 'global_step': global_step})

        # ===================== Checkpoint Save =====================
        if is_primary(args):
            # Full training state (optimizer, lr_scheduler, grad_scaler, EMA
            # shadow, best_fid) so resume is byte-identical-ish, not warm-start.
            latest_path = save_checkpoint_atomic(
                unet_wo_ddp, inference_scheduler,
                vae_wo_ddp, class_embedder_wo_ddp,
                optimizer, epoch, ema_shadow=ema.shadow,
                filename='latest.pth', save_dir=save_dir,
                lr_scheduler=lr_scheduler,
                grad_scaler=grad_scaler,
                best_fid=best_fid,
            )

            best_path = None
            if val_fid is not None and val_fid < best_fid:
                best_fid = val_fid
                best_path = save_checkpoint_atomic(
                    unet_wo_ddp, inference_scheduler,
                    vae_wo_ddp, class_embedder_wo_ddp,
                    optimizer, epoch, ema_shadow=ema.shadow,
                    filename='best.pth', save_dir=save_dir,
                    lr_scheduler=lr_scheduler,
                    grad_scaler=grad_scaler,
                    best_fid=best_fid,
                )
                wandb_logger.summary['best_val_fid'] = best_fid
                wandb_logger.summary['best_epoch'] = epoch
                logger.info(f"New best checkpoint saved (FID={best_fid:.4f})")

            # W&B artifact: log as file:// reference (no byte upload). Ocean
            # remains source of truth; W&B only tracks versioned metadata.
            if (epoch + 1) % args.save_every_n_epochs == 0:
                try:
                    latest_abs = os.path.abspath(latest_path)
                    art = wandb.Artifact(
                        f"model-{args.run_name}",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "val_fid": val_fid,
                            "best_fid": best_fid,
                            "ocean_path": latest_abs,
                            "run_name": args.run_name,
                        },
                    )
                    art.add_reference(f"file://{latest_abs}", checksum=True)
                    is_best = (best_path is not None)
                    aliases = ["latest"] + (["best"] if is_best else [])
                    wandb_logger.log_artifact(art, aliases=aliases)
                    art.ttl = timedelta(days=60 if is_best else 3)
                    if is_best:
                        best_abs = os.path.abspath(best_path)
                        art_best = wandb.Artifact(
                            f"model-{args.run_name}-best",
                            type="model",
                            metadata={"epoch": epoch, "val_fid": val_fid,
                                      "ocean_path": best_abs, "run_name": args.run_name},
                        )
                        art_best.add_reference(f"file://{best_abs}", checksum=True)
                        wandb_logger.log_artifact(art_best, aliases=["best"])
                        art_best.ttl = timedelta(days=60)
                except Exception as exc:
                    logger.warning(f"wandb artifact reference logging failed (non-fatal): {exc}")


if __name__ == '__main__':
    main()
