"""Generate placeholder ground-truth masks for the dummy dataset.

Creates one mask per test image under ``dummy_dataset/<class>/ground_truth``:
fully black for the ``good`` split and fully white for the ``anomaly`` split.
Useful for exercising the pixel-level evaluation path without real annotations.

Usage:
    python scripts/tools/generate_dummy_gt.py
"""

import os
from PIL import Image
import numpy as np

dataset_path = r"c:\Users\teo-s\OneDrive\Documentos\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\dummy_dataset"
class_name = "dummy_class"

test_dir = os.path.join(dataset_path, class_name, "test")
gt_dir = os.path.join(dataset_path, class_name, "ground_truth")

if not os.path.exists(gt_dir):
    os.makedirs(gt_dir)

for subdir in ["good", "anomaly"]:
    src_subdir = os.path.join(test_dir, subdir)
    dst_subdir = os.path.join(gt_dir, subdir)
    
    if not os.path.exists(dst_subdir):
        os.makedirs(dst_subdir)
        
    for filename in os.listdir(src_subdir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Create a black image for good, white for anomaly (just as a dummy)
            # Actually, usually GT is black for good (no anomaly) and white/mask for anomaly.
            # For dummy purpose, let's make anomaly fully white.
            
            img = Image.new('L', (448, 448), color=0) # Black
            if subdir == "anomaly":
                img = Image.new('L', (448, 448), color=255) # White
                
            img.save(os.path.join(dst_subdir, filename))
            print(f"Created mask for {subdir}/{filename}")
