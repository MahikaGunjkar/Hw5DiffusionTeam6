import os
import sys
import torch

# ── Patch torch.load for PyTorch 2.6 compatibility ──────────────────────────
_real_load = torch.load
def _load_patch(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    return _real_load(f, map_location=map_location, weights_only=False, **kwargs)
torch.load = _load_patch
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np
import ruamel.yaml as yaml
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
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def main():
    # parse arguments
    args = parse_args()

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # ===================== Model Setup =====================
    logger.info(f"Creating model: framework={args.framework}, model_type={args.model_type}")

    _DIT_HIDDEN = {'dit_s': 384, 'dit_b': 768, 'dit_l': 1024, 'dit_xl': 1152}
    class_emb_dim = args.unet_ch if args.model_type == 'unet' else _DIT_HIDDEN[args.model_type]

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
            use_ada_mask=False,   # inference: mask is always off
            ada_mask_max=args.ada_mask_max,
            const_mask_ratio=None,
            decoder_depth=args.dit_decoder_depth,
            num_train_timesteps_ref=args.num_train_timesteps,
        )
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"{args.model_type} parameters: {num_params / 10 ** 6:.2f}M")

    # VAE
    vae = None
    if args.latent_ddpm:
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()

    # CFG class embedder
    class_embedder = None
    if args.use_cfg:
        class_embedder = ClassEmbedder(
            embed_dim=class_emb_dim,
            n_classes=args.num_classes,
            cond_drop_rate=0.0,
        )

    # Scheduler
    if args.framework == 'flow_matching':
        scheduler = RectifiedFlowScheduler(
            num_train_timesteps=args.num_train_timesteps,
            num_inference_steps=args.num_inference_steps_flow,
            solver=args.flow_solver,
            use_ot=False,   # inference never uses OT
            ot_max_batch=args.ot_max_batch,
        )
    elif args.use_ddim:
        scheduler = DDIMScheduler(
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
    else:
        scheduler = DDPMScheduler(
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

    # Send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)

    # Load checkpoint — uses patched load_checkpoint (skips scheduler state)
    assert args.ckpt is not None, "Please provide --ckpt"

    # Inline load to avoid scheduler size mismatch
    print("loading checkpoint")
    ckpt = torch.load(args.ckpt, weights_only=False)
    print("loading unet")
    unet.load_state_dict(ckpt['unet_state_dict'])
    print("skipping scheduler (rebuilt from args)")
    if vae is not None and 'vae_state_dict' in ckpt:
        print("loading vae")
        vae.load_state_dict(ckpt['vae_state_dict'])
    if class_embedder is not None and 'class_embedder_state_dict' in ckpt:
        print("loading class_embedder")
        class_embedder.load_state_dict(ckpt['class_embedder_state_dict'])

    # Load EMA weights if available
    ema_ckpt_path = args.ckpt.replace('checkpoint_epoch_', 'ema_checkpoint_epoch_')
    if os.path.exists(ema_ckpt_path):
        logger.info(f"Loading EMA weights from {ema_ckpt_path}")
        ema_data = torch.load(ema_ckpt_path, weights_only=False)
        for name, param in unet.named_parameters():
            if name in ema_data.get('ema_shadow', {}):
                param.data.copy_(ema_data['ema_shadow'][name].to(device))
        logger.info("EMA weights loaded")

    unet.eval()

    PipelineCls = FlowMatchingPipeline if args.framework == 'flow_matching' else DDPMPipeline
    pipeline = PipelineCls(
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        class_embedder=class_embedder,
    )

    infer_steps = (
        args.num_inference_steps_flow
        if args.framework == 'flow_matching'
        else args.num_inference_steps
    )

    # ===================== Generate Images =====================
    logger.info("***** Running Inference *****")

    all_images = []
    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, 'generated_images')
    os.makedirs(save_dir, exist_ok=True)

    if args.use_cfg:
        for cls_idx in tqdm(range(args.num_classes), desc="Generating per class"):
            batch_size = 50
            classes = torch.full((batch_size,), cls_idx, dtype=torch.long, device=device)
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=infer_steps,
                classes=classes.tolist(),
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)
            for i, img in enumerate(gen_images):
                img.save(os.path.join(save_dir, f'class{cls_idx:03d}_img{i:02d}.png'))
    else:
        batch_size = 50
        num_batches = 5000 // batch_size
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=infer_steps,
                generator=generator,
                device=device,
            )
            all_images.extend(gen_images)
            for i, img in enumerate(gen_images):
                img_idx = batch_idx * batch_size + i
                img.save(os.path.join(save_dir, f'img{img_idx:05d}.png'))

    logger.info(f"Generated {len(all_images)} images -> {save_dir}")

    # ===================== FID & IS =====================
    logger.info("Computing FID and IS...")

    all_tensors = []
    for img in all_images:
        t = transforms.ToTensor()(img)
        t = (t * 255).to(torch.uint8)
        all_tensors.append(t)
    gen_tensor = torch.stack(all_tensors)

    logger.info("Loading reference images...")
    ref_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    ref_dataset = datasets.ImageFolder(root=args.data_dir, transform=ref_transform)
    ref_loader = torch.utils.data.DataLoader(
        ref_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    ref_tensors = []
    for imgs, _ in tqdm(ref_loader, desc="Loading reference"):
        imgs = (imgs * 255).to(torch.uint8)
        ref_tensors.append(imgs)
    ref_tensor = torch.cat(ref_tensors, dim=0)

    # FID
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    for i in tqdm(range(0, ref_tensor.shape[0], 64), desc="FID real"):
        fid_metric.update(ref_tensor[i:i+64].to(device), real=True)
    for i in tqdm(range(0, gen_tensor.shape[0], 64), desc="FID fake"):
        fid_metric.update(gen_tensor[i:i+64].to(device), real=False)
    fid_score = fid_metric.compute().item()

    # IS
    from torchmetrics.image.inception import InceptionScore
    is_metric = InceptionScore().to(device)
    for i in tqdm(range(0, gen_tensor.shape[0], 64), desc="IS"):
        is_metric.update(gen_tensor[i:i+64].to(device))
    is_mean, is_std = is_metric.compute()

    logger.info("=" * 50)
    logger.info(f"  FID  = {fid_score:.4f}")
    logger.info(f"  IS   = {is_mean:.4f} +/- {is_std:.4f}")
    logger.info("=" * 50)

    # Kaggle submission
    logger.info("Generating Kaggle submission CSV...")
    from generate_submission import generate_submission_from_tensors
    gen_float = gen_tensor.float() / 255.0
    submission_csv = os.path.join(args.output_dir, 'submission.csv')
    generate_submission_from_tensors(gen_float, output_csv=submission_csv, device=str(device))
    logger.info(f"Submission CSV -> {submission_csv}")


if __name__ == '__main__':
    main()