"""Evaluate the CFLOW pixel head WITHOUT Gaussian smoothing (and optionally compare).

Gaussian smoothing is a post-processing step applied to the CFLOW anomaly map
before pixel metrics are computed. This script isolates its contribution: the
raw score map is computed once per image, then scored under each requested
sigma, so every configuration sees exactly the same network output.

``--sigmas 0`` disables smoothing entirely (scipy's ``gaussian_filter`` with
sigma 0 is a no-op, so the raw map is used as-is).

Only pixel-level metrics are affected. The image-level score comes from the
SEDifferNet flow head and never passes through the smoothing filter, so it is
reported once per class.

Usage:
    # No smoothing at all (the default)
    python scripts/eval/evaluate_cflow_no_smoothing.py

    # Ablation: no smoothing vs the production sigma
    python scripts/eval/evaluate_cflow_no_smoothing.py --sigmas 0,4

    # Quick check on one class
    python scripts/eval/evaluate_cflow_no_smoothing.py --classes vari-grip --limit 20

Model and metric code is imported from ``evaluate_full_pipeline.py`` so both
scripts stay in sync. On the full test set the numbers reproduce
``evaluate_full_pipeline.py --sigma 0`` exactly; ``--limit`` draws a balanced random
subset, so limited runs are only comparable to other runs with the same
``--limit`` and ``--seed``.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import time

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as c
from core.model import SEDifferNet, load_weights
from core.utils import load_datasets
from core.paths import project_path

from evaluate_full_pipeline import (
    ALL_CLASSES,
    DEVICE,
    CFlowPixelHead,
    SEBackboneFeatureExtractor,
    compute_best_f1,
    compute_pro_rd_style,
    find_nf_head_checkpoint,
    find_se_checkpoint,
)
from core.cflow import (
    DEFAULT_ATTENTION,
    TRAIN_CLAMP_SCALE,
    TRAIN_OUT_SIZE,
    TRAIN_SCORE_MODE,
    SCORE_MODES,
    infer_n_blocks,
    read_cflow_hparams,
    resolve_hparam,
)


def build_test_loader(class_name, dataset_path, limit=None, seed=42):
    """Build a deterministic, single-transform test loader for one class.

    Pixel-level evaluation must not rotate the input, otherwise the anomaly map
    no longer aligns with the ground-truth mask.

    Args:
        class_name: Dataset class to load.
        dataset_path: Root directory holding the dataset classes.
        limit: Optional cap on the number of test images, balanced between
            normal and anomalous samples.
        seed: Seed for the subsampling RNG.

    Returns:
        Tuple ``(loader, n_images)``.
    """
    c.class_name = class_name
    c.n_transforms_test = 1
    c.transf_rotations = False

    _, testset = load_datasets(dataset_path, class_name, aligned=True)
    testset.n_transforms = 1
    testset.get_fixed = False
    testset.transform = transforms.Compose([
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ])

    if limit and limit > 0:
        normal_idx = [i for i, (_, label, _) in enumerate(testset.samples) if label == 0]
        anomaly_idx = [i for i, (_, label, _) in enumerate(testset.samples) if label > 0]

        if limit < len(anomaly_idx):
            # Dedicated RNG: building the models reseeds numpy's global RNG
            # (FrEIA's permute_layer calls np.random.seed), which would make the
            # sampled subset depend on model-construction side effects.
            rng = np.random.default_rng(seed)
            n_anomaly = min(limit // 2, len(anomaly_idx))
            n_normal = min(limit - n_anomaly, len(normal_idx))
            indices = (rng.choice(anomaly_idx, n_anomaly, replace=False).tolist()
                       + rng.choice(normal_idx, n_normal, replace=False).tolist())
            rng.shuffle(indices)
        else:
            indices = list(range(min(limit, len(testset))))
        testset = Subset(testset, indices)

    loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return loader, len(testset)


def load_models(class_name, args):
    """Load the SEDifferNet backbone and its trained CFLOW pixel head.

    Returns:
        Tuple ``(model, backbone, cflow)``, or ``None`` if a checkpoint is
        missing or has an unrecognized layout.
    """
    se_ckpt_path = find_se_checkpoint(class_name)
    nf_ckpt_path = find_nf_head_checkpoint(class_name)

    if se_ckpt_path is None:
        print(f"  ERROR: SEDifferNet checkpoint not found for '{class_name}'")
        return None
    if nf_ckpt_path is None:
        print(f"  ERROR: NF Head checkpoint not found for '{class_name}'")
        return None

    print(f"  SE model:  {os.path.basename(se_ckpt_path)}")
    print(f"  NF Head:   {os.path.basename(nf_ckpt_path)}")

    model = SEDifferNet()
    model, _ = load_weights(model, se_ckpt_path)
    model.to(DEVICE).eval()

    nf_data = torch.load(nf_ckpt_path, map_location=DEVICE, weights_only=False)
    if 'cflow_state_dict' in nf_data:
        cflow_sd = nf_data['cflow_state_dict']
    elif 'model_state_dict' in nf_data:
        cflow_sd = nf_data['model_state_dict']
    else:
        print(f"  ERROR: unknown NF Head checkpoint format. Keys: {list(nf_data.keys())}")
        return None

    hp = read_cflow_hparams(nf_data)
    out_size, src_out = resolve_hparam('out_size', args.out_size, hp, TRAIN_OUT_SIZE)
    clamp, src_clamp = resolve_hparam('clamp_scale', args.clamp_scale, hp, TRAIN_CLAMP_SCALE)
    n_blocks, _ = resolve_hparam('n_blocks', args.cflow_n_blocks, hp,
                                 infer_n_blocks(cflow_sd), warn=False)
    attention = args.attention if args.attention != 'auto' else hp.get('attention', 'auto')
    print(f"  out_size={out_size} ({src_out}) | clamp_scale={clamp} ({src_clamp}) | "
          f"n_blocks={n_blocks} | attention={attention} | score={args.score_norm}")

    backbone = SEBackboneFeatureExtractor(
        model, out_size=out_size, reflect_pad=args.reflect_pad,
        attention=attention).to(DEVICE).eval()

    cflow = CFlowPixelHead(
        layer_channels=backbone.layer_channels,
        cond_dim=args.cflow_cond_dim,
        n_blocks=n_blocks,
        hidden=args.cflow_hidden,
        clamp_scale=clamp,
    ).to(DEVICE)
    cflow.load_state_dict(cflow_sd)
    cflow.eval()

    return model, backbone, cflow


def collect_raw_outputs(model, backbone, cflow, loader, class_name, score_mode='raw'):
    """Run inference once and keep the UNSMOOTHED CFLOW maps.

    Returns:
        Tuple ``(image_scores, image_labels, raw_maps, masks, pixel_latency_ms)``
        where ``raw_maps[i]`` is the 2-D anomaly map for image ``i``.
    """
    image_scores, image_labels = [], []
    raw_maps, masks_bin = [], []
    times_pixel = []

    with torch.no_grad():
        for data in tqdm(loader, desc=f"  [{class_name}]"):
            images, labels, masks = data
            images_flat = images.to(DEVICE).view(-1, *images.shape[-3:])

            z = model(images_flat)
            image_scores.append(float(torch.mean(z ** 2).item()))
            image_labels.append(1 if int(labels[0].item()) > 0 else 0)

            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, feats = backbone(images_flat)
            smap = cflow.score_map(feats, c.img_size[0], mode=score_mode)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            times_pixel.append(time.perf_counter() - t0)

            smap_np = smap.cpu().numpy()
            raw_maps.append(smap_np[0] if smap_np.ndim == 3 else smap_np)

            mask_np = masks.squeeze(0).squeeze(0).numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            masks_bin.append((mask_np > 0.5).astype(np.uint8))

    latency_ms = float(np.mean(times_pixel) * 1000) if times_pixel else 0.0
    return image_scores, image_labels, raw_maps, masks_bin, latency_ms


def score_at_sigma(raw_maps, masks_bin, image_labels, sigma):
    """Compute pixel-level metrics for one smoothing sigma.

    Args:
        raw_maps: Unsmoothed anomaly maps, one 2-D array per image.
        masks_bin: Binary ground-truth masks aligned with ``raw_maps``.
        image_labels: Per-image labels, used to restrict AUPRO to anomalies.
        sigma: Gaussian sigma; ``0`` leaves the map untouched.

    Returns:
        Dict with ``pixel_auroc``, ``pixel_ap``, ``pixel_f1`` and ``aupro``.
    """
    maps = raw_maps if sigma <= 0 else [gaussian_filter(m, sigma=sigma) for m in raw_maps]

    scores_flat = np.concatenate([m.flatten() for m in maps])
    labels_flat = np.concatenate([m.flatten() for m in masks_bin]).astype(np.uint8)

    if labels_flat.sum() == 0 or (labels_flat == 0).sum() == 0:
        return {'pixel_auroc': float('nan'), 'pixel_ap': float('nan'),
                'pixel_f1': float('nan'), 'aupro': float('nan')}

    pix_f1, _ = compute_best_f1(labels_flat, scores_flat)
    aupro_list = [compute_pro_rd_style(masks_bin[i], maps[i])
                  for i in range(len(maps)) if image_labels[i] > 0]

    return {
        'pixel_auroc': float(roc_auc_score(labels_flat, scores_flat)),
        'pixel_ap': float(average_precision_score(labels_flat, scores_flat)),
        'pixel_f1': float(pix_f1),
        'aupro': float(np.mean(aupro_list)) if aupro_list else 0.0,
    }


def plot_sigma_comparison(df, save_path):
    """Plot pixel metrics against smoothing sigma, averaged over classes."""
    metrics = ['pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro']
    agg = df.groupby('sigma')[metrics].mean().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in metrics:
        ax.plot(agg.index, agg[m], marker='o', label=m)
    ax.set_xlabel('Gaussian smoothing sigma (0 = disabled)')
    ax.set_ylabel('Score (mean over classes)')
    ax.set_title('Effect of Gaussian smoothing on CFLOW pixel metrics')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the CFLOW pixel head without Gaussian smoothing")

    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class names. Default: all 5 classes.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test images per class (None = all)')
    parser.add_argument('--sigmas', type=str, default='0',
                        help="Comma-separated sigmas to score. 0 disables smoothing. "
                             "Use '0,4' to compare against the production setting.")
    parser.add_argument('--cflow_cond_dim', type=int, default=64)
    parser.add_argument('--cflow_n_blocks', type=int, default=None)
    parser.add_argument('--cflow_hidden', type=int, default=256)
    parser.add_argument('--out_size', type=int, default=None,
                        help='Feature grid the flow is conditioned on; must match training.')
    parser.add_argument('--clamp_scale', type=float, default=None,
                        help='Affine coupling clamp; must match training.')
    parser.add_argument('--attention', type=str, default=DEFAULT_ATTENTION,
                        choices=['auto', 'se', 'cbam', 'legacy_train'])
    parser.add_argument('--score_norm', type=str, default=TRAIN_SCORE_MODE,
                        choices=list(SCORE_MODES))
    parser.add_argument('--reflect_pad', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sigmas = [float(s.strip()) for s in args.sigmas.split(',') if s.strip()]
    class_list = ([cl.strip() for cl in args.classes.split(',') if cl.strip()]
                  if args.classes else ALL_CLASSES)

    run_dir = project_path('resultado_analise_final', 'ablacao_smoothing',
                           f'run_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 70)
    print("  CFLOW PIXEL EVALUATION - GAUSSIAN SMOOTHING ABLATION")
    print("=" * 70)
    print(f"  Device:   {DEVICE}")
    print(f"  Classes:  {', '.join(class_list)}")
    print(f"  Sigmas:   {sigmas}   (0 = no smoothing)")
    print(f"  Limit:    {args.limit if args.limit else 'all images'}")
    print(f"  Output:   {run_dir}")

    original_class = c.class_name
    original_n_transforms = c.n_transforms_test
    original_rotations = c.transf_rotations

    rows = []
    try:
        for class_name in class_list:
            print(f"\n{'=' * 70}\n  Class: {class_name}\n{'=' * 70}")

            loaded = load_models(class_name, args)
            if loaded is None:
                continue
            model, backbone, cflow = loaded

            loader, n_test = build_test_loader(class_name, args.dataset,
                                               args.limit, args.seed)
            print(f"  Dataset: {n_test} test images")

            (image_scores, image_labels, raw_maps,
             masks_bin, latency_ms) = collect_raw_outputs(
                model, backbone, cflow, loader, class_name, args.score_norm)

            # Image-level metrics never pass through the smoothing filter.
            if len(np.unique(image_labels)) > 1:
                img_auroc = float(roc_auc_score(image_labels, image_scores))
                img_ap = float(average_precision_score(image_labels, image_scores))
                img_f1, _ = compute_best_f1(np.array(image_labels), np.array(image_scores))
            else:
                img_auroc = img_ap = img_f1 = float('nan')

            print(f"\n  Image-level (independent of sigma): "
                  f"AUROC={img_auroc:.4f}  AP={img_ap:.4f}  F1={img_f1:.4f}")
            print(f"\n  {'sigma':<8} {'PixAUROC':<11} {'PixAP':<11} {'PixF1':<11} {'AUPRO':<11}")
            print("  " + "-" * 54)

            for sigma in sigmas:
                m = score_at_sigma(raw_maps, masks_bin, image_labels, sigma)
                label = 'none' if sigma <= 0 else f'{sigma:g}'
                print(f"  {label:<8} {m['pixel_auroc']:<11.4f} {m['pixel_ap']:<11.4f} "
                      f"{m['pixel_f1']:<11.4f} {m['aupro']:<11.4f}")

                rows.append({
                    'class': class_name, 'sigma': sigma, 'smoothing': label,
                    'n_images': n_test,
                    'image_auroc': img_auroc, 'image_ap': img_ap, 'image_f1': img_f1,
                    'latency_pixel_ms': latency_ms, **m,
                })

            del model, backbone, cflow, raw_maps, masks_bin
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
    finally:
        c.class_name = original_class
        c.n_transforms_test = original_n_transforms
        c.transf_rotations = original_rotations

    if not rows:
        print("\nNo class could be evaluated. Check the checkpoint paths.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, 'cflow_smoothing_ablation.csv')
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 70}\n  SUMMARY (mean over classes)\n{'=' * 70}")
    summary = df.groupby('smoothing')[
        ['pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro']].mean()
    print(summary.to_string())

    if len(sigmas) > 1:
        baseline = df[df['sigma'] <= 0]
        if not baseline.empty:
            base = baseline.groupby('class')['pixel_auroc'].mean()
            print("\n  Pixel AUROC delta vs no smoothing:")
            for sigma in sorted(s for s in sigmas if s > 0):
                sub = df[df['sigma'] == sigma].groupby('class')['pixel_auroc'].mean()
                delta = (sub - base).mean()
                print(f"    sigma={sigma:g}: {delta:+.4f}")

        plot_path = os.path.join(run_dir, 'smoothing_ablation.png')
        plot_sigma_comparison(df, plot_path)
        print(f"\n  Plot saved to: {plot_path}")

    print(f"  CSV saved to:  {csv_path}")


if __name__ == '__main__':
    main()
