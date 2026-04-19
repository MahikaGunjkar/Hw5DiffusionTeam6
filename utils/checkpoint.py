import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def load_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    checkpoint_path: str = 'checkpoints/checkpoint.pth',
    lr_scheduler=None,
    grad_scaler=None,
) -> Dict[str, Any]:
    """Load model checkpoint and return full resume state.

    Backward-compatible with legacy checkpoints that only stored
    ``epoch``/``ema_shadow``. Missing fields come back as ``None`` so the
    caller can apply sensible defaults (e.g. reconstruct LR scheduler from
    saved epoch, initialize best_fid to +inf).
    """

    print(f"loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print("loading unet")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("loading scheduler")
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if vae is not None and 'vae_state_dict' in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint['vae_state_dict'])

    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        print("loading optimizer")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
        print("loading lr_scheduler")
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    if grad_scaler is not None and 'grad_scaler_state_dict' in checkpoint:
        try:
            print("loading grad_scaler")
            grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])
        except Exception as exc:  # noqa: BLE001
            print(f"grad_scaler state incompatible ({exc}); leaving scaler fresh.")

    return {
        'epoch': checkpoint.get('epoch', -1),
        'ema_shadow': checkpoint.get('ema_shadow'),
        'best_fid': checkpoint.get('best_fid'),
        'has_lr_scheduler': 'lr_scheduler_state_dict' in checkpoint,
    }


def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None,
                    epoch=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')

    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    manage_checkpoints(save_dir, keep_last_n=10)


def save_checkpoint_atomic(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    epoch: Optional[int] = None,
    ema_shadow: Optional[Dict[str, torch.Tensor]] = None,
    filename: str = 'latest.pth',
    save_dir: str = 'checkpoints',
    lr_scheduler=None,
    grad_scaler=None,
    best_fid: Optional[float] = None,
) -> str:
    """Atomic checkpoint save: write to .tmp, fsync, then rename.

    Stores full training state (optimizer, lr_scheduler, grad_scaler, EMA
    shadow, best_fid) so resume can continue rather than restart warm.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = save_dir / f".{filename}.tmp"
    final_path = save_dir / filename

    checkpoint: Dict[str, Any] = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    if grad_scaler is not None:
        try:
            checkpoint['grad_scaler_state_dict'] = grad_scaler.state_dict()
        except Exception:
            pass
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if ema_shadow is not None:
        checkpoint['ema_shadow'] = ema_shadow
    if best_fid is not None:
        checkpoint['best_fid'] = best_fid

    torch.save(checkpoint, tmp_path)

    with open(tmp_path, 'rb') as fh:
        os.fsync(fh.fileno())

    tmp_path.rename(final_path)
    print(f"Checkpoint atomically saved at {final_path}")
    return str(final_path)


def manage_checkpoints(save_dir, keep_last_n=10):
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    if len(checkpoints) > keep_last_n + 1:
        for checkpoint_file in checkpoints[:-keep_last_n - 1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
