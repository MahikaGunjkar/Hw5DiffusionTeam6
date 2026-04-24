#!/usr/bin/env python3
"""Watch a checkpoint and snapshot it once a target epoch is reached."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch


def should_snapshot(saved_epoch: int, target_epoch: int) -> bool:
    """Return True once the saved checkpoint reaches the human target epoch.

    Training stores zero-based epoch indices in checkpoints, while the user
    talks about epochs starting from 1. So target epoch 400 corresponds to
    saved_epoch >= 399.
    """

    return int(saved_epoch) + 1 >= int(target_epoch)


def atomic_snapshot_copy(source: Path, dest: Path) -> bool:
    """Create dest exactly once from source using an atomic rename."""

    source = Path(source)
    dest = Path(dest)
    if dest.exists():
        return False

    tmp_dest = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
    shutil.copy2(source, tmp_dest)
    os.replace(tmp_dest, dest)
    return True


def load_saved_epoch(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "epoch" not in checkpoint:
        raise KeyError(f"'epoch' missing in checkpoint: {checkpoint_path}")
    return int(checkpoint["epoch"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to latest.pth")
    parser.add_argument("--snapshot", required=True, help="Destination snapshot path")
    parser.add_argument("--target-epoch", type=int, required=True, help="Human epoch number")
    parser.add_argument("--poll-seconds", type=int, default=120, help="Polling interval")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="0 means no timeout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    snapshot_path = Path(args.snapshot)
    target_epoch = int(args.target_epoch)
    poll_seconds = max(5, int(args.poll_seconds))
    timeout_seconds = max(0, int(args.timeout_seconds))
    start_time = time.time()
    last_reported_epoch: int | None = None

    print(
        f"[watcher] watching {checkpoint_path} for epoch {target_epoch}; "
        f"will snapshot to {snapshot_path}",
        flush=True,
    )

    while True:
        if snapshot_path.exists():
            print(f"[watcher] snapshot already exists: {snapshot_path}", flush=True)
            return 0

        if checkpoint_path.exists():
            try:
                saved_epoch = load_saved_epoch(checkpoint_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[watcher] failed to read checkpoint: {exc}", flush=True)
            else:
                human_epoch = saved_epoch + 1
                if human_epoch != last_reported_epoch:
                    print(
                        f"[watcher] observed checkpoint epoch={saved_epoch} "
                        f"(human epoch {human_epoch})",
                        flush=True,
                    )
                    last_reported_epoch = human_epoch
                if should_snapshot(saved_epoch, target_epoch):
                    created = atomic_snapshot_copy(checkpoint_path, snapshot_path)
                    if created:
                        print(
                            f"[watcher] snapshot created at human epoch {human_epoch}: "
                            f"{snapshot_path}",
                            flush=True,
                        )
                    else:
                        print(f"[watcher] snapshot already created: {snapshot_path}", flush=True)
                    return 0
        else:
            print(f"[watcher] checkpoint not found yet: {checkpoint_path}", flush=True)

        if timeout_seconds and (time.time() - start_time) >= timeout_seconds:
            print("[watcher] timeout reached before target epoch", flush=True)
            return 2

        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
