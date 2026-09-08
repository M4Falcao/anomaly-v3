"""
Smoke Test — Trivial Baseline Accuracy
=======================================

If a model predicts ALL pixels as "not anomalous" (i.e. all zeros),
what would its pixel-level accuracy, pixel AUROC, IoU, etc. be?

This sets the absolute minimum bar any real model must beat.

For each class this script reports:
  - Total test images (normal + anomaly)
  - % of normal images (image-level baseline)
  - % of normal pixels across all anomalous images (pixel-level baseline)
  - Pixel "accuracy" of the trivial predictor (= % normal pixels overall)
  - What Pixel AUROC would be (always 0.5 for a constant predictor)

Usage:
    python scripts/analysis/smoke_test_baseline.py
    python scripts/analysis/smoke_test_baseline.py --dataset_path <path>
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import config as c


def analyze_class(dataset_path, class_name):
    """Compute trivial baseline stats for one class."""
    test_dir = os.path.join(dataset_path, class_name, 'test')
    gt_dir = os.path.join(dataset_path, class_name, 'ground_truth')

    if not os.path.isdir(test_dir):
        print(f"  [SKIP] Test dir not found: {test_dir}")
        return None

    has_gt = os.path.isdir(gt_dir)

    # Count images per category
    n_normal = 0
    n_anomaly = 0
    anomaly_class_names = []

    categories = sorted(os.listdir(test_dir))
    for cat in categories:
        cat_dir = os.path.join(test_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        imgs = [f for f in os.listdir(cat_dir)
                if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')]
        if cat == 'good':
            n_normal += len(imgs)
        else:
            n_anomaly += len(imgs)
            anomaly_class_names.append(cat)

    total_images = n_normal + n_anomaly

    # Pixel-level stats: iterate GT masks of anomalous images
    total_pixels = 0
    anomaly_pixels = 0
    normal_pixels_in_anomaly_imgs = 0
    total_pixels_in_anomaly_imgs = 0
    per_defect_stats = {}

    if has_gt:
        for cat in anomaly_class_names:
            gt_cat_dir = os.path.join(gt_dir, cat)
            if not os.path.isdir(gt_cat_dir):
                continue

            cat_anom_px = 0
            cat_total_px = 0

            mask_files = sorted([f for f in os.listdir(gt_cat_dir)
                                 if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')])

            for mf in mask_files:
                mask = np.array(Image.open(os.path.join(gt_cat_dir, mf)).convert('L'))
                binary = (mask > 127).astype(int)

                n_pix = binary.size
                n_anom = binary.sum()

                total_pixels += n_pix
                anomaly_pixels += n_anom
                total_pixels_in_anomaly_imgs += n_pix
                normal_pixels_in_anomaly_imgs += (n_pix - n_anom)
                cat_anom_px += n_anom
                cat_total_px += n_pix

            if cat_total_px > 0:
                per_defect_stats[cat] = {
                    'n_masks': len(mask_files),
                    'total_pixels': int(cat_total_px),
                    'anomaly_pixels': int(cat_anom_px),
                    'anomaly_ratio': cat_anom_px / cat_total_px,
                }

    # Also account for normal images (all pixels normal)
    # Estimate using the average image size from GT masks (or use config)
    img_h, img_w = c.img_size
    normal_img_pixels = n_normal * img_h * img_w
    total_pixels_all = total_pixels + normal_img_pixels

    # The trivial predictor says "ALL pixels are normal"
    # Its pixel accuracy = (correctly predicted normal pixels) / (total pixels)
    # = (all_normal_pixels) / (total_pixels)
    # = 1 - (anomaly_pixels / total_pixels_all)

    if total_pixels_all > 0:
        trivial_pixel_accuracy = 1.0 - (anomaly_pixels / total_pixels_all)
    else:
        trivial_pixel_accuracy = 1.0

    # Among ONLY anomalous images (what matters for pixel AUROC):
    if total_pixels_in_anomaly_imgs > 0:
        anomaly_pixel_ratio = anomaly_pixels / total_pixels_in_anomaly_imgs
        trivial_accuracy_anomaly_only = 1.0 - anomaly_pixel_ratio
    else:
        anomaly_pixel_ratio = 0.0
        trivial_accuracy_anomaly_only = 1.0

    return {
        'class_name': class_name,
        'total_images': total_images,
        'normal_images': n_normal,
        'anomaly_images': n_anomaly,
        'normal_image_pct': n_normal / total_images * 100 if total_images > 0 else 0,
        'anomaly_pixel_ratio_in_anomaly_imgs': anomaly_pixel_ratio * 100,
        'normal_pixel_ratio_in_anomaly_imgs': (1 - anomaly_pixel_ratio) * 100,
        'trivial_pixel_accuracy_all': trivial_pixel_accuracy * 100,
        'trivial_pixel_accuracy_anomaly_only': trivial_accuracy_anomaly_only * 100,
        'trivial_pixel_auroc': 0.5,  # constant predictor always = 0.5
        'trivial_image_auroc': 0.5,  # constant predictor always = 0.5
        'trivial_iou': 0.0,         # predicts no anomaly -> 0 intersection
        'trivial_dice': 0.0,        # same
        'total_anomaly_pixels': int(anomaly_pixels),
        'total_pixels_anomaly_imgs': int(total_pixels_in_anomaly_imgs),
        'defect_types': anomaly_class_names,
        'per_defect': per_defect_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Smoke Test — Trivial Baseline")
    parser.add_argument('--dataset_path', type=str, default=c.dataset_path)
    args = parser.parse_args()

    # Auto-discover classes
    classes = sorted([d for d in os.listdir(args.dataset_path)
                      if os.path.isdir(os.path.join(args.dataset_path, d))])

    print("=" * 80)
    print("  SMOKE TEST — TRIVIAL BASELINE (All pixels predicted as NORMAL)")
    print("=" * 80)
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Classes: {classes}")
    print("=" * 80)

    results = []
    for cls in classes:
        print(f"\n{'-' * 60}")
        print(f"  Class: {cls}")
        print(f"{'-' * 60}")

        stats = analyze_class(args.dataset_path, cls)
        if stats is None:
            continue

        results.append(stats)

        print(f"  Images:          {stats['total_images']} "
              f"({stats['normal_images']} normal, {stats['anomaly_images']} anomaly)")
        print(f"  Normal img %:    {stats['normal_image_pct']:.1f}%")
        print(f"  Defect types:    {stats['defect_types']}")
        print()
        print(f"  Among ANOMALOUS images:")
        print(f"    Anomaly pixels:  {stats['anomaly_pixel_ratio_in_anomaly_imgs']:.2f}%")
        print(f"    Normal pixels:   {stats['normal_pixel_ratio_in_anomaly_imgs']:.2f}%")
        print()
        print(f"  TRIVIAL BASELINE (predict all zeros):")
        print(f"    Pixel Accuracy (all imgs):      {stats['trivial_pixel_accuracy_all']:.2f}%")
        print(f"    Pixel Accuracy (anomaly only):   {stats['trivial_pixel_accuracy_anomaly_only']:.2f}%")
        print(f"    Pixel AUROC:                     {stats['trivial_pixel_auroc']:.4f}")
        print(f"    Image AUROC:                     {stats['trivial_image_auroc']:.4f}")
        print(f"    IoU:                             {stats['trivial_iou']:.4f}")
        print(f"    Dice:                            {stats['trivial_dice']:.4f}")

        if stats['per_defect']:
            print(f"\n  Per-defect breakdown:")
            for defect, dstats in stats['per_defect'].items():
                print(f"    {defect}:")
                print(f"      Masks: {dstats['n_masks']}, "
                      f"Anomaly ratio: {dstats['anomaly_ratio']*100:.2f}%")

    # Summary table
    print("\n\n" + "=" * 80)
    print("  SUMMARY TABLE")
    print("=" * 80)

    summary_rows = []
    for r in results:
        summary_rows.append({
            'Class': r['class_name'],
            'Images': r['total_images'],
            'Normal%': f"{r['normal_image_pct']:.1f}%",
            'Anom Pixels%': f"{r['anomaly_pixel_ratio_in_anomaly_imgs']:.2f}%",
            'Baseline PixAcc': f"{r['trivial_pixel_accuracy_anomaly_only']:.2f}%",
            'Baseline PixAUROC': f"{r['trivial_pixel_auroc']:.4f}",
            'Baseline ImgAUROC': f"{r['trivial_image_auroc']:.4f}",
            'Baseline IoU': f"{r['trivial_iou']:.4f}",
        })

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "analysis", "runs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "smoke_test_baseline.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    print("\n" + "=" * 80)
    print("  INTERPRETATION")
    print("=" * 80)
    print("  A trivial model that always predicts 'normal' for every pixel will achieve")
    print("  the Pixel Accuracy shown above — simply because anomalous pixels are RARE.")
    print()
    print("  This is why Pixel Accuracy alone is MISLEADING for anomaly detection.")
    print("  Use Pixel AUROC, IoU, Dice, and AUPRO instead — they account for imbalance.")
    print()
    print("  Any real model MUST exceed:")
    print("    - Pixel AUROC > 0.5 (random chance)")
    print("    - IoU > 0.0 (trivial baseline)")
    print("    - Dice > 0.0 (trivial baseline)")
    print("=" * 80)


if __name__ == "__main__":
    main()
