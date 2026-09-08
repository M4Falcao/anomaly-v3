"""Print the AUROC history stored inside a training checkpoint.

Training checkpoints keep the per-epoch ``image_aurocs`` and ``pixel_aurocs``
lists. This tool loads a checkpoint on CPU and tabulates them.

Usage:
    python scripts/tools/read_checkpoint_auroc.py [checkpoint.pt]

When no path is given, a default checkpoint under ``checkpoints/`` is used.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from core.paths import project_path


def main(checkpoint_path):
    """Load a checkpoint and print its per-epoch AUROC table.

    Args:
        checkpoint_path: Path to a ``.pt`` file saved by the training loop.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: file not found: {checkpoint_path}")
        return

    print(f"Reading checkpoint: {checkpoint_path}")
    # Load on CPU to avoid allocating GPU memory just to inspect metadata.
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(ckpt, dict):
        epoch_saved = ckpt.get('epoch', 'Unknown')
        print(f"Final epoch of the checkpoint: {epoch_saved}")

        image_aurocs = ckpt.get('image_aurocs', [])
        pixel_aurocs = ckpt.get('pixel_aurocs', [])

        print("\n--- AUROC history per epoch ---")
        max_len = max(len(image_aurocs), len(pixel_aurocs))
        if max_len == 0:
            print("No AUROC metrics stored in the checkpoint lists.")
            return

        print(f"{'Epoch':<10} | {'Image AUROC':<15} | {'Pixel AUROC':<15}")
        print("-" * 45)
        for i in range(max_len):
            img_auc = image_aurocs[i] if i < len(image_aurocs) else 'N/A'
            pix_auc = pixel_aurocs[i] if i < len(pixel_aurocs) else 'N/A'

            img_str = f"{img_auc:.4f}" if isinstance(img_auc, (float, int)) else str(img_auc)
            pix_str = f"{pix_auc:.4f}" if isinstance(pix_auc, (float, int)) else str(pix_auc)

            print(f"{i+1:<10} | {img_str:<15} | {pix_str:<15}")
    else:
        print("The file is not in the expected dictionary format.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = project_path(
            "checkpoints",
            "lightning-rod-suspension_se_differnet_lightning_rod_suspension_100_1_epoch_150.pt",
        )
    main(path)
