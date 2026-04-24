#!/usr/bin/env python3
"""Snapshot latest.pth after a target epoch appears in the Slurm log."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path


EPOCH_RE = re.compile(r"\bEpoch\s+(\d+)/\d+\b")
def latest_epoch_seen(log_text: str) -> int | None:
    latest: int | None = None
    for match in EPOCH_RE.finditer(log_text):
        latest = int(match.group(1))
    return latest


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Slurm log path that contains Epoch lines")
    parser.add_argument("--checkpoint", required=True, help="Path to latest.pth")
    parser.add_argument("--snapshot", required=True, help="Destination snapshot path")
    parser.add_argument("--target-epoch", type=int, required=True, help="Human epoch number")
    parser.add_argument("--poll-seconds", type=int, default=15, help="Polling interval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    checkpoint_path = Path(args.checkpoint)
    snapshot_path = Path(args.snapshot)
    target_epoch = int(args.target_epoch)
    poll_seconds = max(5, int(args.poll_seconds))
    last_reported_epoch: int | None = None
    target_seen = False
    checkpoint_mtime_at_target: int | None = None

    print(
        f"[watcher] watching {log_path} for epoch {target_epoch}; "
        f"will snapshot {checkpoint_path} after the next checkpoint save to {snapshot_path}",
        flush=True,
    )

    while True:
        if snapshot_path.exists():
            print(f"[watcher] snapshot already exists: {snapshot_path}", flush=True)
            return 0

        if not log_path.exists():
            print(f"[watcher] log not found yet: {log_path}", flush=True)
            time.sleep(poll_seconds)
            continue

        try:
            log_text = log_path.read_text(errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"[watcher] failed to read log: {exc}", flush=True)
            time.sleep(poll_seconds)
            continue

        epoch = latest_epoch_seen(log_text)
        if epoch is not None and epoch != last_reported_epoch:
            print(f"[watcher] latest epoch seen in log: {epoch}", flush=True)
            last_reported_epoch = epoch

        if epoch is not None and epoch >= target_epoch and not target_seen:
            target_seen = True
            if not checkpoint_path.exists():
                print(f"[watcher] target epoch seen; checkpoint missing: {checkpoint_path}", flush=True)
                time.sleep(poll_seconds)
                continue
            checkpoint_mtime_at_target = checkpoint_path.stat().st_mtime_ns
            print(
                f"[watcher] target epoch {epoch} seen; waiting for next latest.pth update",
                flush=True,
            )

        if target_seen:
            if not checkpoint_path.exists():
                print(f"[watcher] checkpoint missing after target epoch: {checkpoint_path}", flush=True)
                time.sleep(poll_seconds)
                continue
            current_mtime = checkpoint_path.stat().st_mtime_ns
            if checkpoint_mtime_at_target is not None and current_mtime == checkpoint_mtime_at_target:
                time.sleep(poll_seconds)
                continue
            created = atomic_snapshot_copy(checkpoint_path, snapshot_path)
            if created:
                print(f"[watcher] snapshot created: {snapshot_path}", flush=True)
            else:
                print(f"[watcher] snapshot already created: {snapshot_path}", flush=True)
            return 0

        time.sleep(poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
