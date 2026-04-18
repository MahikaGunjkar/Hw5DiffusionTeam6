"""Dataset utilities shared by training and inference scripts."""

from __future__ import annotations

import logging
import os
import tarfile

logger = logging.getLogger(__name__)


def extract_tar_if_needed(tar_path: str, extract_to: str) -> str:
    """Stream-extract a .tar.gz archive into ``extract_to`` if not already done.

    A hidden ``.extracted`` flag file is written so repeated runs skip extraction.

    Expected archive layout::

        imagenet100_128x128/
            train/
                class_folder_0/  *.JPEG
                ...
            val/
                ...

    Returns the path to the ``train/`` split inside the extracted directory.
    """
    flag = os.path.join(extract_to, ".extracted")
    if os.path.exists(flag):
        logger.info(f"Dataset already extracted at '{extract_to}' — skipping extraction.")
    else:
        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                f"Tar file not found: '{tar_path}'\n"
                "Please set --tar_path to point at imagenet100_128x128.tar.gz"
            )
        os.makedirs(extract_to, exist_ok=True)
        logger.info(f"Extracting '{tar_path}' -> '{extract_to}' (streaming, no temp copy) ...")

        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            total = len(members)
            for i, member in enumerate(members):
                tar.extract(member, extract_to, set_attrs=False)
                if i % 10_000 == 0 and i > 0:
                    logger.info(f"  Extraction progress: {i}/{total} ({100 * i / total:.1f}%)")

        with open(flag, "w") as f:
            f.write("done")
        logger.info("Extraction complete.")

    for candidate in [
        os.path.join(extract_to, "imagenet100_128x128"),
        os.path.join(extract_to, "imagenet100"),
        extract_to,
    ]:
        train_candidate = os.path.join(candidate, "train")
        if os.path.isdir(train_candidate):
            return train_candidate

    raise RuntimeError(
        f"Could not find a 'train/' subfolder inside '{extract_to}'. "
        "Please check the archive structure."
    )
