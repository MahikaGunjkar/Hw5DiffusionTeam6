import os
import torch
from pathlib import Path


def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None,
                    checkpoint_path='checkpoints/checkpoint.pth') -> int:
    """Load model checkpoint. Returns the saved epoch number (or -1 if not found)."""

    print("loading checkpoint")
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

    return checkpoint.get('epoch', -1)


def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None,
                    epoch=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
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

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

    # Manage checkpoint history
    manage_checkpoints(save_dir, keep_last_n=10)


def save_checkpoint_atomic(unet, scheduler, vae=None, class_embedder=None, optimizer=None,
                           epoch=None, ema_shadow=None, filename='latest.pth',
                           save_dir='checkpoints') -> str:
    """
    Atomic checkpoint save: write to .tmp, fsync, then rename.
    Prevents partial reads on Lustre / NFS parallel filesystems.

    When ``ema_shadow`` is provided it is packaged into the same checkpoint
    file under the ``ema_shadow`` key, so inference.py can load weights and
    EMA shadow from a single file without per-epoch sidecar sprawl.

    Returns the path of the final checkpoint file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tmp_path = save_dir / f".{filename}.tmp"
    final_path = save_dir / filename

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

    if ema_shadow is not None:
        checkpoint['ema_shadow'] = ema_shadow

    torch.save(checkpoint, tmp_path)

    # fsync to flush kernel buffer to disk before rename
    with open(tmp_path, 'rb') as fh:
        os.fsync(fh.fileno())

    tmp_path.rename(final_path)
    print(f"Checkpoint atomically saved at {final_path}")
    return str(final_path)


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n - 1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
