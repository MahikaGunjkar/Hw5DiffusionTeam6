import os
import sys
import copy
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb
import logging
from logging import getLogger as get_logger
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
    extract_tar_if_needed,
)
from utils.training import create_grad_scaler, evaluation_mode, resolve_amp_config

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

    # config file(s) — later files override earlier ones, so an ablation
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

    # First parse — get config file path
    args = parser.parse_args()

    # Load YAML config and set as new defaults
    if args.config:
        merged = {}
        file_yaml = yaml.YAML()
        for cfg_path in args.config:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = file_yaml.load(f) or {}
            # Strip the `slurm:` block — it is only for scripts/submit.sh.
            cfg.pop('slurm', None)
            merged.update(cfg)
        parser.set_defaults(**merged)

    # Re-parse — command-line overrides YAML
    args = parser.parse_args()

    # Auto-override UNet config when using latent DDPM
    if args.latent_ddpm:
        args.unet_in_size = 32
        args.unet_in_ch = 3

    return args


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
    if args.run_name is None:
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

    if is_primary(args):
        wandb_logger = wandb.init(project='ddpm', name=args.run_name, config=vars(args))

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

    for epoch in range(args.num_epochs):

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

            # Latent DDPM — encode with frozen VAE
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
                wandb_logger.log({'loss': loss_m.avg, 'lr': optimizer.param_groups[0]['lr']})

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
                wandb_logger.log({'gen_images': wandb.Image(grid_image)})

        ema.restore(unet_wo_ddp)

        if is_primary(args):
            save_checkpoint(
                unet_wo_ddp, inference_scheduler,
                vae_wo_ddp, class_embedder_wo_ddp,
                optimizer, epoch, save_dir=save_dir
            )
            ema_ckpt_path = os.path.join(save_dir, f'ema_checkpoint_epoch_{epoch}.pth')
            torch.save({'ema_shadow': ema.shadow}, ema_ckpt_path)
            logger.info(f"EMA checkpoint saved: {ema_ckpt_path}")


if __name__ == '__main__':
    main()
