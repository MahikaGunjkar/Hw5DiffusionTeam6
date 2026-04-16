import os
import warnings

import torch


def read_checkpoint(checkpoint_path='checkpoints/checkpoint.pth'):
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_scheduler_state(scheduler, scheduler_state_dict):
    if scheduler is None or scheduler_state_dict is None:
        return

    current_state = scheduler.state_dict()
    merged_state = dict(current_state)
    skipped_keys = []

    for key, value in scheduler_state_dict.items():
        if key == "timesteps":
            skipped_keys.append(key)
            continue

        if key not in current_state:
            skipped_keys.append(key)
            continue

        current_value = current_state[key]
        if torch.is_tensor(current_value) and torch.is_tensor(value):
            if current_value.shape != value.shape:
                skipped_keys.append(key)
                continue

        merged_state[key] = value

    scheduler.load_state_dict(merged_state)
    if skipped_keys:
        print(
            "warning: skipped scheduler state keys during checkpoint load: "
            + ", ".join(sorted(skipped_keys))
        )


def infer_resume_global_step(checkpoint, steps_per_epoch):
    if checkpoint is None:
        return None

    global_step = checkpoint.get('global_step')
    if global_step is not None:
        return int(global_step)

    lr_state = checkpoint.get('lr_scheduler_state_dict')
    if isinstance(lr_state, dict):
        if lr_state.get('last_epoch') is not None:
            return int(lr_state['last_epoch'])
        if lr_state.get('_step_count') is not None:
            return max(0, int(lr_state['_step_count']) - 1)

    epoch = checkpoint.get('epoch')
    if epoch is not None and steps_per_epoch is not None:
        return (int(epoch) + 1) * int(steps_per_epoch)

    return None


def restore_lr_scheduler_progress(lr_scheduler, global_step):
    if lr_scheduler is None or global_step is None:
        return

    global_step = int(global_step)
    if global_step < 0:
        raise ValueError(f"global_step must be >= 0, got {global_step}")
    if global_step == 0:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        lr_scheduler.step(global_step)

def load_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    checkpoint_path='checkpoints/checkpoint.pth',
):
    print("loading checkpoint")
    checkpoint = read_checkpoint(checkpoint_path)

    print("loading unet")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        print("loading scheduler")
        _load_scheduler_state(scheduler, checkpoint['scheduler_state_dict'])

    if vae is not None and 'vae_state_dict' in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint['vae_state_dict'])

    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        print("loading optimizer")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
    
    
        

def save_checkpoint(
    unet,
    scheduler,
    vae=None,
    class_embedder=None,
    optimizer=None,
    lr_scheduler=None,
    epoch=None,
    global_step=None,
    save_dir='checkpoints',
):
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

    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if global_step is not None:
        checkpoint['global_step'] = int(global_step)
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history
    manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
