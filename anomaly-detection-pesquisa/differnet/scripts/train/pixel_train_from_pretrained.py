"""Train pixel-level head on top of a pretrained SEDifferNet.

Loads a frozen SEDifferNet checkpoint and trains:
- CFLOW-AD (conditional NF) for pixel anomaly localization
- PatchCore memory bank as complementary method
- Applies Gaussian smoothing (A) as post-processing

Final validation stage produces:
- Image-level and pixel-level AUROC, AUPRO, F1, AP
- Score histograms (normal vs anomaly)
- Per-image score distribution plots
- ROC curves (image and pixel)
- Anomaly map visualizations
- All logged to MLflow

Usage:
    python scripts/train/pixel_train_from_pretrained.py
    python scripts/train/pixel_train_from_pretrained.py --checkpoint path/to/model.pt --epochs 80
"""
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import math

# Local imports
import config as c
from core.model import SEDifferNet, CBAMDifferNet, load_weights
from core.utils import load_datasets, make_dataloaders, t2np
from core.cflow import (
    DEFAULT_ATTENTION,
    PAD_MODES,
    SCORE_MODES,
    TRAIN_CLAMP_SCALE,
    TRAIN_OUT_SIZE,
    TRAIN_SCORE_MODE,
    CFlowPixelHead,
    SEBackboneFeatureExtractor,
    apply_feature_stats,
    border_mask,
    compute_feature_stats,
    compute_level_stats,
    gaussian_smooth,
    image_score_from_map,
    minmax_norm,
    save_cflow_checkpoint,
    select_levels,
)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════════════════
# Backbone / CFLOW definitions now live in core.cflow (see imports above)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_arch(checkpoint_path):
    """Detect model architecture from the checkpoint file name.

    ``SEDifferNet`` and ``CBAMDifferNet`` declare the same submodules, so their
    state dicts share every key and cannot be told apart. Checkpoint names
    follow ``<class>_<se|cbam>_differnet_...``; anything not marked CBAM is SE.
    """
    name = os.path.basename(checkpoint_path).lower()
    return 'cbam' if 'cbam' in name else 'se'


def clip_flow_grads(cflow, max_norm):
    """Clip each level's flow on its own; a global norm lets the widest level starve the others."""
    for flow in cflow.flows:
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm)


class PatchCoreBank:
    """Memory bank for PatchCore pixel scoring."""
    
    def __init__(self, coreset_ratio=0.1, k=3, seed=42):
        self.coreset_ratio = coreset_ratio
        self.k = k
        self.seed = seed
        self.bank = None
    
    @torch.no_grad()
    def fit(self, backbone, train_loader):
        feats_all = []
        for data in tqdm(train_loader, desc='[PatchCore] Building memory bank'):
            if isinstance(data, (list, tuple)):
                images = data[0]
            else:
                images = data
            images = images.to(DEVICE).view(-1, *images.shape[-3:])
            concat, _ = backbone(images)
            B, C, H, W = concat.shape
            f = concat.permute(0, 2, 3, 1).reshape(-1, C)
            feats_all.append(f.cpu())
        
        feats = torch.cat(feats_all, dim=0)
        n_select = max(int(len(feats) * self.coreset_ratio), 1024)
        n_select = min(n_select, len(feats))
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(feats), n_select, replace=False)
        self.bank = feats[idx].to(DEVICE)
        print(f"[PatchCore] Memory bank: {self.bank.shape}")
    
    @torch.no_grad()
    def score_map(self, concat, img_size):
        B, C, H, W = concat.shape
        f = concat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        chunk = 2048
        dists_min = torch.empty(B * H * W, device=DEVICE)
        for i in range(0, f.shape[0], chunk):
            d = torch.cdist(f[i:i+chunk], self.bank)
            topk = torch.topk(d, k=self.k, dim=1, largest=False).values
            dists_min[i:i+chunk] = topk.mean(dim=1)
        score = dists_min.view(B, H, W)
        up = F.interpolate(score.unsqueeze(1), size=(img_size, img_size),
                           mode='bilinear', align_corners=False).squeeze(1)
        return up


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aupro(masks, scores, max_fpr=0.3, n_thresholds=200, rng_seed=0):
    """Compute Area Under Per-Region Overlap (AUPRO).

    Thresholds come from quantiles of the *normal* pixel scores instead of a
    linear grid over ``[min, max]``.  Raw NLL maps are heavy-tailed: a handful of
    extreme pixels stretch the range so much that a linear grid puts ~199 of 200
    thresholds in the region where FPR is already above ``max_fpr``, leaving
    fewer than two usable points and making the function return 0.0 (this is what
    happened to vari-grip in run_20260907_145952).  Because ``FPR(th) = P(normal
    score >= th)``, taking ``th = quantile(normal, 1 - f)`` places every
    threshold exactly inside the useful ``FPR in [0, max_fpr]`` window.
    """
    from scipy.ndimage import label as connected_components

    normal_scores = scores[masks == 0]
    total_normal_pixels = normal_scores.size
    if total_normal_pixels == 0:
        return 0.0

    # Quantiles over a subsample: full-resolution maps hold ~1e8 normal pixels.
    max_sample = 2_000_000
    if total_normal_pixels > max_sample:
        idx = np.random.default_rng(rng_seed).choice(
            total_normal_pixels, max_sample, replace=False)
        normal_scores = normal_scores[idx]
    thresholds = np.unique(
        np.quantile(normal_scores, np.linspace(1.0 - max_fpr, 1.0, n_thresholds)))
    del normal_scores

    pro_values = []
    fpr_values = []

    # Pre-extract connected components once (masks do not change with threshold)
    regions = []
    for i in range(masks.shape[0]):
        if masks[i].max() == 0:
            continue
        labeled, n_regions = connected_components(masks[i])
        for region_id in range(1, n_regions + 1):
            region_mask = (labeled == region_id)
            region_size = int(region_mask.sum())
            if region_size > 0:
                regions.append((i, region_mask, region_size))

    if not regions:
        return 0.0
    
    for th in thresholds:
        binary_pred = (scores >= th).astype(np.uint8)
        
        # FPR
        fp = ((binary_pred == 1) & (masks == 0)).sum()
        fpr = fp / total_normal_pixels
        if fpr > max_fpr:
            continue
        
        # Per-region overlap
        overlaps = [
            float((binary_pred[i] & region_mask).sum()) / region_size
            for i, region_mask, region_size in regions
        ]
        
        if overlaps:
            pro_values.append(np.mean(overlaps))
            fpr_values.append(fpr)
    
    if len(fpr_values) < 2:
        return 0.0
    
    # Sort by FPR and compute AUC
    sorted_idx = np.argsort(fpr_values)
    fpr_sorted = np.array(fpr_values)[sorted_idx]
    pro_sorted = np.array(pro_values)[sorted_idx]
    trapz_fn = getattr(np, 'trapezoid', np.trapz)
    aupro = trapz_fn(pro_sorted, fpr_sorted) / max_fpr
    return float(aupro)


def compute_best_f1(labels, scores):
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return f1_scores[best_idx], thresholds[min(best_idx, len(thresholds)-1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def plot_score_histogram(normal_scores, anomaly_scores, title, save_path):
    """Plot histogram of anomaly scores for normal vs anomaly."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(labels, scores, title, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pixel_histogram(normal_pixels, anomaly_pixels, title, save_path):
    """Plot histogram of pixel-level scores."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Subsample for plotting speed
    n_max = 100000
    if len(normal_pixels) > n_max:
        normal_pixels = np.random.choice(normal_pixels, n_max, replace=False)
    if len(anomaly_pixels) > n_max:
        anomaly_pixels = np.random.choice(anomaly_pixels, n_max, replace=False)
    ax.hist(normal_pixels, bins=100, alpha=0.6, label='Normal pixels', color='green', density=True)
    ax.hist(anomaly_pixels, bins=100, alpha=0.6, label='Anomaly pixels', color='red', density=True)
    ax.set_xlabel('Pixel Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(history, save_path):
    """Plot loss, pixel/image AUROC and AUPRO curves (RD++ style monitor)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('NLL Loss')
    axes[0].set_title('Training Loss (CFLOW)')

    eval_epochs = history['eval_epochs']
    panels = [
        (1, 'pixel_auroc', 'r-o', 'Pixel AUROC'),
        (2, 'image_auroc', 'g-o', 'Image AUROC (map-derived)'),
        (3, 'aupro', 'm-o', 'AUPRO (max_fpr=0.3)'),
    ]
    for ax_idx, key, style, title in panels:
        ax = axes[ax_idx]
        raw = history.get(key) or []
        # AUPRO is None whenever the metric could not be computed for that epoch.
        pairs = [(e, v) for e, v in zip(eval_epochs, raw) if v is not None]
        if pairs:
            xs = [e for e, _ in pairs]
            ys = [v for _, v in pairs]
            ax.plot(xs, ys, style, markersize=4)
            best_i = int(np.argmax(ys))
            ax.axvline(xs[best_i], color='k', ls='--', lw=0.8)
            ax.set_title(f'{title} - best {ys[best_i]:.4f} @ ep {xs[best_i]}')
            ax.set_ylim([0.0 if key == 'aupro' else 0.5, 1.0])
        else:
            ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_anomaly_maps(images, masks, score_maps, save_path, n_samples=8):
    """Visualize anomaly maps alongside original images and GT masks."""
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(n):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy().clip(0, 1)
        
        mask = masks[i].squeeze().numpy()
        smap = score_maps[i]
        smap_norm = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(smap_norm, cmap='hot')
        axes[i, 2].set_title('Anomaly Map')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(img)
        axes[i, 3].imshow(smap_norm, cmap='hot', alpha=0.5)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Training & Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train pixel-level head on pretrained SEDifferNet")
    
    # Model
    parser.add_argument('--checkpoint', type=str,
                        default=r"C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\lightning-rod-suspension\lightning-rod-suspension_se_differnet_lightning_rod_suspension_100_1_epoch_41.pt")
    parser.add_argument('--arch', type=str, default='auto', choices=['auto', 'se', 'cbam'],
                        help="Backbone class of --checkpoint. 'auto' reads it from the file name "
                             "(SE and CBAM checkpoints share identical state-dict keys).")
    
    # Data
    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--class_name', type=str, default=c.class_name)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--out_size', type=int, default=96)
    
    # CFLOW training
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--cond_dim', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--clamp_scale', type=float, default=TRAIN_CLAMP_SCALE,
                        help='Affine coupling clamp; stored in the checkpoint so evaluation '
                             'can reproduce the exact transform.')
    parser.add_argument('--attention', type=str, default=DEFAULT_ATTENTION,
                        choices=['auto', 'se', 'cbam', 'legacy_train'],
                        help="Which attention blocks feed the CFLOW head. Defaults to 'se', "
                             "the path SEDifferNet actually trains. 'legacy_train' reproduces "
                             "the old behaviour of reading CBAM blocks even for SE models.")
    parser.add_argument('--score_norm', type=str, default=TRAIN_SCORE_MODE,
                        choices=list(SCORE_MODES),
                        help='How per-level NLL maps are aggregated during in-training validation '
                             'and checkpoint selection. Defaults to "raw" (TRAIN_SCORE_MODE) so '
                             'best_pixel_auroc.pt is selected by raw NLL fusion aligned with deployment.')
    parser.add_argument('--reflect_pad', type=int, default=0,
                        help='Reflect-pad the input by N pixels to reduce border artifacts. '
                             'Snapped up to the nearest value that keeps the feature grid '
                             'registered with the mask.')
    parser.add_argument('--pad_mode', type=str, default='reflect', choices=list(PAD_MODES),
                        help='Fill used by --reflect_pad: mirror (reflect) or edge value (replicate).')
    parser.add_argument('--feat_pool', type=int, default=0,
                        help='Stride-1 average-pool kernel applied to each feature level before '
                             'the flow (PatchCore-style neighbourhood aggregation). 0 = off.')
    parser.add_argument('--feat_norm', action='store_true',
                        help='Standardize features per channel with train-set statistics before '
                             'the flow; the statistics are stored in the checkpoint.')
    parser.add_argument('--feat_norm_levels', type=str, default=None,
                        help="Comma-separated feature levels to standardize per channel (0=L1, 1=L2, 2=L3). "
                             "E.g. '0' standardizes L1 only (selective standardization). "
                             "Enables feature standardization automatically.")
    parser.add_argument('--levels', type=str, default=None,
                        help="Comma-separated feature levels to train flows on (0=L1, 1=L2, 2=L3). "
                             "Default: all three. E.g. '1,2' drops the border-biased L1.")
    parser.add_argument('--no_rotation', action='store_true',
                        help='Disable the random-rotation augmentation for pixel-head training. '
                             'Rotation fills the corners with black, which the flow never sees at '
                             'test time and which biases the positional prior at the border.')
    parser.add_argument('--border_margin', type=int, default=0,
                        help='Exclude an N-pixel frame from pixel metrics.')
    parser.add_argument('--image_score', type=str, default='max', choices=['max', 'topk'],
                        help="Pixel-map reduction for the image score. 'max' is decided by a "
                             "single pixel, so one border false positive can dominate.")
    parser.add_argument('--image_score_topk', type=float, default=1.0,
                        help='Percentage of pixels averaged when --image_score topk')
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', help='Enable AMP (off by default for flow stability)')
    parser.add_argument('--eval_aupro', action='store_true', default=True,
                        help='Compute AUPRO at every evaluation and enable the composite '
                             '(pixel_auroc + image_auroc + aupro)/3 checkpoint, as RD++ does. '
                             'Cheap now that connected components are pre-extracted.')
    parser.add_argument('--no_eval_aupro', dest='eval_aupro', action='store_false',
                        help='Skip the per-epoch AUPRO (falls back to pixel-AUROC selection only).')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping: stop after N evaluations with no improvement of the '
                             'selection metric. 0 disables it. Measured on run_20260907_145952, '
                             'yoke peaked at epoch 4/80 and lost 0.030 AUROC by the end.')
    
    # PatchCore
    parser.add_argument('--patchcore_coreset', type=float, default=0.1)
    parser.add_argument('--patchcore_k', type=int, default=3)
    parser.add_argument('--disable_patchcore', action='store_true', help="Disable PatchCore to save memory")
    
    # Post-processing
    parser.add_argument('--sigma', type=float, default=6.0)
    
    # Output
    parser.add_argument('--out_dir', type=str, default='./pixel_head_runs')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    levels = None
    if args.levels:
        levels = sorted({int(x) for x in args.levels.split(',') if x.strip()})
        if any(l < 0 or l > 2 for l in levels):
            parser.error('--levels must be a subset of 0,1,2')

    norm_levels = None
    if args.feat_norm_levels:
        norm_levels = sorted({int(x) for x in args.feat_norm_levels.split(',') if x.strip()})
        if any(l < 0 or l > 2 for l in norm_levels):
            parser.error('--feat_norm_levels must be a subset of 0,1,2')
        if levels is not None:
            missing = [l for l in norm_levels if l not in levels]
            if missing:
                parser.error(f'--feat_norm_levels pede {missing}, mas a head foi configurada com --levels {levels}')
        args.feat_norm = True
    elif args.feat_norm:
        norm_levels = levels

    if args.no_rotation:
        c.transf_rotations = False
    
    # ─── Create output directory ─────────────────────────────────────────────
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.out_dir, f"run_{run_timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    best_dir = os.path.join(run_dir, "best_models")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Device: {DEVICE}")
    
    # ─── MLflow ──────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(c.mlflow_tracking_uri)
    mlflow.set_experiment("Pixel_Head_SEDifferNet")
    mlflow.start_run(run_name=f"pixel_{args.class_name}_{run_timestamp}")
    
    mlflow.log_params({
        'checkpoint': os.path.basename(args.checkpoint),
        'img_size': args.img_size,
        'out_size': args.out_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'cond_dim': args.cond_dim,
        'n_blocks': args.n_blocks,
        'hidden': args.hidden,
        'sigma': args.sigma,
        'patchcore_coreset': args.patchcore_coreset,
        'patchcore_k': args.patchcore_k,
        'disable_patchcore': args.disable_patchcore,
        'class_name': args.class_name,
        'warmup_epochs': args.warmup_epochs,
        'grad_clip': args.grad_clip,
        'use_amp': args.use_amp,
        'reflect_pad': args.reflect_pad,
        'pad_mode': args.pad_mode,
        'feat_pool': args.feat_pool,
        'feat_norm': args.feat_norm,
        'feat_norm_levels': str(norm_levels) if norm_levels is not None else 'all',
        'levels': args.levels or '0,1,2',
        'no_rotation': args.no_rotation,
        'clamp_scale': args.clamp_scale,
        'score_norm': args.score_norm,
        'eval_interval': args.eval_interval,
        'patience': args.patience,
    })
    
    # ─── Load pretrained model (SE vs CBAM) ───────────────────────────────────────
    arch = args.arch if args.arch != 'auto' else _detect_arch(args.checkpoint)
    model_cls = CBAMDifferNet if arch == 'cbam' else SEDifferNet
    print(f"\nLoading pretrained {model_cls.__name__} (arch={arch}) from:\n  {args.checkpoint}")
    mlflow.log_param('backbone_arch', arch)
    model = model_cls()
    model, checkpoint_data = load_weights(model, args.checkpoint)
    model.to(DEVICE).eval()
    
    if checkpoint_data and 'image_aurocs' in checkpoint_data:
        best_img = max(checkpoint_data['image_aurocs'])
        print(f"  Pretrained Image AUROC: {best_img:.4f} (epoch {checkpoint_data.get('epoch', '?')})")
        mlflow.log_metric("pretrained_image_auroc", best_img)
    
    # ─── Build feature extractor ─────────────────────────────────────────────
    backbone = SEBackboneFeatureExtractor(
        model, out_size=args.out_size, reflect_pad=args.reflect_pad,
        attention=args.attention, pad_mode=args.pad_mode,
        feat_pool=args.feat_pool).to(DEVICE).eval()
    attn_type = 'CBAM' if backbone._use_cbam else 'SE'
    print(f"  Backbone: {attn_type} attention | channels: {backbone.layer_channels} -> concat {backbone.concat_channels}")
    head_channels = select_levels(backbone.layer_channels, levels)
    print(f"  Flow levels: {levels if levels is not None else [0, 1, 2]} -> channels {head_channels}")
    
    # ─── Load data ───────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.class_name}")
    trainset, testset = load_datasets(args.dataset, args.class_name, aligned=True)
    # Only the first training view is used, so do not generate the other n_transforms-1.
    trainset.n_transforms = 1
    # load_datasets() gives the test set the *training* transform: with rotation on it
    # rotates the image but not the mask, and it builds n_transforms_test (64) views per
    # image. Pixel evaluation needs one deterministic, unrotated view aligned to the GT.
    testset.n_transforms = 1
    testset.get_fixed = False
    testset.transform = transforms.Compose([
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ])
    train_loader, _ = make_dataloaders(trainset, testset)
    # Use a small batch size for test_loader to avoid OOM during dense CFLOW pixel evaluation
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)
    print(f"  Train: {len(trainset)} samples")
    print(f"  Test:  {len(testset)} samples (batch_size=1 for eval)")
    print(f"  Rotation augmentation: {'OFF' if args.no_rotation else 'ON'}")

    # ─── Feature standardization (optional) ──────────────────────────────────
    feat_stats = None
    if args.feat_norm:
        norm_desc = ','.join(f"L{l+1}" for l in norm_levels) if norm_levels is not None else 'all'
        print(f"\n--- Computing per-channel feature statistics ({norm_desc}) ---")
        feat_stats = compute_feature_stats(backbone, train_loader, DEVICE, norm_levels=norm_levels)
        feat_stats = select_levels(feat_stats, levels)

    def extract(images):
        """Frozen features for the levels the head is trained on."""
        with torch.no_grad():
            concat, feats = backbone(images)
        feats = apply_feature_stats(select_levels(feats, levels), feat_stats)
        return concat, feats
    
    # ─── Build PatchCore memory bank ─────────────────────────────────────────
    if not args.disable_patchcore:
        print("\n--- Building PatchCore Memory Bank ---")
        patchcore = PatchCoreBank(coreset_ratio=args.patchcore_coreset, 
                                  k=args.patchcore_k, seed=args.seed)
        patchcore.fit(backbone, train_loader)
        mlflow.log_metric("patchcore_bank_size", patchcore.bank.shape[0])
    else:
        print("\n--- PatchCore Disabled ---")
        patchcore = None
    
    # ─── Build CFLOW pixel head ──────────────────────────────────────────────
    print("\n--- Initializing CFLOW Pixel Head ---")
    cflow = CFlowPixelHead(
        layer_channels=head_channels,
        cond_dim=args.cond_dim,
        n_blocks=args.n_blocks,
        hidden=args.hidden,
        clamp_scale=args.clamp_scale,
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in cflow.parameters())
    print(f"  CFLOW parameters: {n_params:,}")
    mlflow.log_param("cflow_params", n_params)

    # Travels with every checkpoint so evaluation reproduces this exact transform.
    cflow_hparams = {
        'out_size': args.out_size,
        'cond_dim': args.cond_dim,
        'n_blocks': args.n_blocks,
        'hidden': args.hidden,
        'img_size': args.img_size,
        'score_mode': args.score_norm,
        'reflect_pad': args.reflect_pad,
        'clamp_scale': args.clamp_scale,
        'attention': backbone.attention,
        'pad_mode': args.pad_mode,
        'feat_pool': args.feat_pool,
        'levels': levels,
        'feat_norm': args.feat_norm,
        'feat_norm_levels': norm_levels,
        'no_rotation': args.no_rotation,
    }
    # Feature statistics travel with every checkpoint (None when --feat_norm is off).
    ckpt_extra = {'feat_stats': feat_stats}
    
    optimizer = torch.optim.Adam(cflow.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.use_amp)
    
    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ─── Training CFLOW ──────────────────────────────────────────────────────
    if args.feat_norm:
        norm_str = f"selective(L{','.join(str(l+1) for l in norm_levels)})" if norm_levels is not None else 'ON'
    else:
        norm_str = 'OFF'
    print(f"\n{'='*60}")
    print(f"Training CFLOW Pixel Head ({args.epochs} epochs)")
    print(f"  LR: {args.lr} | Warmup: {args.warmup_epochs} ep | Grad clip: {args.grad_clip}")
    print(f"  AMP: {'ON' if args.use_amp else 'OFF'} | out_size: {args.out_size} | "
          f"sigma: {args.sigma} | score_norm: {args.score_norm} | feat_norm: {norm_str}")
    print(f"{'='*60}")
    
    best_pixel_auroc = 0.0
    best_image_auroc = 0.0
    best_composite = 0.0
    evals_since_improvement = 0
    history = {'train_loss': [], 'pixel_auroc': [], 'image_auroc': [], 'aupro': [],
               'composite': [], 'eval_epochs': [], 'learning_rate': []}
    history_path = os.path.join(run_dir, 'history.json')
    monitor_path = os.path.join(plots_dir, 'training_curves.png')
    
    for epoch in range(args.epochs):
        # Train
        cflow.train()
        epoch_losses = []
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            if len(data) == 3:
                images, labels, masks = data
            else:
                images, labels = data
            
            # For pixel training, use only first transform (rotations don't help)
            images = images.to(DEVICE)
            if images.dim() == 5:
                images = images[:, 0]  # (B, n_transforms, C, H, W) -> (B, C, H, W)
            
            _, feats = extract(images)
            
            optimizer.zero_grad()
            if args.use_amp:
                with autocast('cuda'):
                    nll_maps = cflow.nll_per_level(feats)
                    loss = sum(m.mean() for m in nll_maps)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_flow_grads(cflow, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nll_maps = cflow.nll_per_level(feats)
                loss = sum(m.mean() for m in nll_maps)
                loss.backward()
                clip_flow_grads(cflow, args.grad_clip)
                optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(float(avg_loss))
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(float(current_lr))
        mlflow.log_metric("cflow_train_loss", avg_loss, step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            cflow.eval()
            torch.cuda.empty_cache()
            all_pixel_scores = []
            all_pixel_labels = []
            all_image_scores = []
            all_image_labels = []
            
            with torch.no_grad():
                for data in tqdm(test_loader, desc=f"Eval {epoch+1}", leave=False):
                    if len(data) == 3:
                        images, labels, masks = data
                    else:
                        images, labels = data
                        masks = torch.zeros(images.shape[0], 1, args.img_size, args.img_size)
                    
                    # Only first transform for pixel eval
                    images = images.to(DEVICE)
                    if images.dim() == 5:
                        images = images[:, 0]
                    _, feats = extract(images)
                    
                    # CFLOW score
                    smap = cflow.score_map(feats, args.img_size, mode=args.score_norm)
                    smap_np = gaussian_smooth(t2np(smap), sigma=args.sigma)

                    img_scores = image_score_from_map(
                        smap_np, mode=args.image_score, top_percent=args.image_score_topk)
                    
                    all_pixel_scores.append(smap_np)
                    all_pixel_labels.append(masks.squeeze(1).numpy())
                    all_image_scores.extend(img_scores.tolist())
                    all_image_labels.extend([1 if l > 0 else 0 for l in labels.numpy()])
            
            pixel_scores = np.concatenate(all_pixel_scores, axis=0)
            pixel_labels = np.concatenate(all_pixel_labels, axis=0)
            
            # Pixel AUROC
            keep = border_mask(pixel_scores.shape[-1], args.border_margin)
            pix_flat = pixel_scores[:, keep].reshape(-1)
            lab_flat = pixel_labels[:, keep].reshape(-1).astype(np.uint8)
            pixel_auroc = roc_auc_score(lab_flat, pix_flat) if lab_flat.sum() > 0 else 0.5
            
            # Image AUROC
            image_auroc = roc_auc_score(all_image_labels, all_image_scores) \
                if len(np.unique(all_image_labels)) > 1 else 0.5

            # AUPRO. RD++ tracks it every epoch and selects on it; AUROC, AP and
            # AUPRO disagree systematically, so pixel AUROC alone picks the wrong
            # epoch for localization quality.
            aupro = float('nan')
            if args.eval_aupro and lab_flat.sum() > 0:
                try:
                    aupro = compute_aupro(pixel_labels.astype(np.uint8), pixel_scores)
                except Exception as exc:  # noqa: BLE001 - never abort training on a metric
                    print(f"    AUPRO falhou: {exc}")

            composite = (pixel_auroc + image_auroc + aupro) / 3 \
                if aupro == aupro else (pixel_auroc + image_auroc) / 2

            history['pixel_auroc'].append(pixel_auroc)
            history['image_auroc'].append(image_auroc)
            history['aupro'].append(None if aupro != aupro else aupro)
            history['composite'].append(composite)
            history['eval_epochs'].append(epoch + 1)

            mlflow.log_metric("pixel_auroc", pixel_auroc, step=epoch)
            mlflow.log_metric("image_auroc", image_auroc, step=epoch)
            mlflow.log_metric("composite_score", composite, step=epoch)
            if aupro == aupro:
                mlflow.log_metric("aupro", aupro, step=epoch)

            aupro_str = f" | aupro={aupro:.4f}" if aupro == aupro else ""
            print(f"  Ep {epoch+1}: loss={avg_loss:.4f} | pix={pixel_auroc:.4f} "
                  f"({args.score_norm}) | img={image_auroc:.4f}{aupro_str} | comp={composite:.4f}")

            improved = False

            # Save best
            if pixel_auroc > best_pixel_auroc:
                best_pixel_auroc = pixel_auroc
                improved = True
                save_cflow_checkpoint(
                    os.path.join(best_dir, "best_pixel_auroc.pt"), cflow, cflow_hparams,
                    extra={'epoch': epoch + 1, 'pixel_auroc': pixel_auroc,
                           'image_auroc': image_auroc, 'aupro': aupro,
                           'score_mode': args.score_norm, **ckpt_extra})
                print(f"    -> New best PIXEL AUROC ({args.score_norm}): {pixel_auroc:.4f}")
            
            if image_auroc > best_image_auroc:
                best_image_auroc = image_auroc
                save_cflow_checkpoint(
                    os.path.join(best_dir, "best_image_auroc.pt"), cflow, cflow_hparams,
                    extra={'epoch': epoch + 1, 'pixel_auroc': pixel_auroc,
                           'image_auroc': image_auroc, 'aupro': aupro,
                           'score_mode': args.score_norm, **ckpt_extra})
                print(f"    -> New best IMAGE AUROC: {image_auroc:.4f}")

            if composite > best_composite:
                best_composite = composite
                improved = True
                save_cflow_checkpoint(
                    os.path.join(best_dir, "best_composite.pt"), cflow, cflow_hparams,
                    extra={'epoch': epoch + 1, 'pixel_auroc': pixel_auroc,
                           'image_auroc': image_auroc, 'aupro': aupro,
                           'composite': composite,
                           'score_mode': args.score_norm, **ckpt_extra})
                print(f"    -> New best COMPOSITE: {composite:.4f}")

            # Persist history and refresh the monitor plot at every evaluation so
            # a run can be diagnosed while it is still going (and after a crash).
            best_i = int(np.argmax(history['pixel_auroc']))
            with open(history_path, 'w', encoding='utf-8') as fh:
                json.dump({**history,
                           'best_pixel_auroc': best_pixel_auroc,
                           'best_image_auroc': best_image_auroc,
                           'best_composite': best_composite,
                           'best_pixel_epoch': history['eval_epochs'][best_i]},
                          fh, indent=2)
            plot_training_curves(history, monitor_path)

            evals_since_improvement = 0 if improved else evals_since_improvement + 1
            if args.patience and evals_since_improvement >= args.patience:
                print(f"  Early stopping: {args.patience} avaliacoes sem melhora "
                      f"(melhor pixel AUROC {best_pixel_auroc:.4f}).")
                break
        
        # Regular checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_cflow_checkpoint(
                os.path.join(ckpt_dir, f"cflow_epoch_{epoch+1}.pt"), cflow, cflow_hparams,
                extra={'epoch': epoch + 1,
                       'optimizer_state_dict': optimizer.state_dict(), **ckpt_extra})
    
    # ─── Plot training curves ────────────────────────────────────────────────
    plot_training_curves(history, os.path.join(plots_dir, "training_curves.png"))
    mlflow.log_artifact(os.path.join(plots_dir, "training_curves.png"))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("FINAL VALIDATION")
    print(f"{'='*60}")
    
    # Load best pixel model
    best_ckpt = torch.load(os.path.join(best_dir, "best_pixel_auroc.pt"), 
                           map_location=DEVICE, weights_only=False)
    cflow.load_state_dict(best_ckpt['cflow_state_dict'])
    cflow.eval()
    torch.cuda.empty_cache()
    # Full evaluation with all metrics
    all_cflow_maps = []
    all_patchcore_maps = []
    all_masks = []
    all_images_for_viz = []
    all_image_scores_cflow = []
    all_image_scores_patchcore = []
    all_image_labels = []
    
    print("\nRunning inference on test set...")
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Final eval"):
            if len(data) == 3:
                images, labels, masks = data
            else:
                images, labels = data
                masks = torch.zeros(images.shape[0], 1, args.img_size, args.img_size)
            
            # Only first transform for pixel eval
            images_dev = images.to(DEVICE)
            if images_dev.dim() == 5:
                images_dev = images_dev[:, 0]
            concat, feats = extract(images_dev)
            
            # CFLOW score map
            cflow_map = t2np(cflow.score_map(feats, args.img_size, mode=args.score_norm))
            cflow_map = gaussian_smooth(cflow_map, sigma=args.sigma)
            all_cflow_maps.append(cflow_map)
            
            # PatchCore score map
            if not args.disable_patchcore:
                pc_map = t2np(patchcore.score_map(concat, args.img_size))
                pc_map = gaussian_smooth(pc_map, sigma=args.sigma)
                all_patchcore_maps.append(pc_map)
            
            all_masks.append(masks.squeeze(1).numpy())
            
            # Save images for visualization (first batch only)
            if len(all_images_for_viz) < 16:
                all_images_for_viz.extend([images_dev[i] for i in range(images_dev.shape[0])])
            
            # Image-level scores
            cflow_img = cflow_map.reshape(cflow_map.shape[0], -1).max(axis=1)
            all_image_scores_cflow.extend(cflow_img.tolist())
            
            if not args.disable_patchcore:
                pc_img = pc_map.reshape(pc_map.shape[0], -1).max(axis=1)
                all_image_scores_patchcore.extend(pc_img.tolist())
                
            all_image_labels.extend([1 if l > 0 else 0 for l in labels.numpy()])
    
    # Concatenate
    cflow_maps = np.concatenate(all_cflow_maps, axis=0)
    gt_masks = np.concatenate(all_masks, axis=0)
    image_labels = np.array(all_image_labels)
    
    # ─── Compute all metrics ─────────────────────────────────────────────────
    print("\n--- Computing Metrics ---")
    
    methods_eval = {
        'CFLOW': (cflow_maps, all_image_scores_cflow),
    }
    
    if not args.disable_patchcore:
        pc_maps = np.concatenate(all_patchcore_maps, axis=0)
        # Ensemble: normalized CFLOW + PatchCore
        cflow_norm = minmax_norm(cflow_maps)
        pc_norm = minmax_norm(pc_maps)
        ensemble_maps = 0.6 * cflow_norm + 0.4 * pc_norm
        
        methods_eval['PatchCore'] = (pc_maps, all_image_scores_patchcore)
        methods_eval['Ensemble'] = (ensemble_maps, (0.6 * minmax_norm(np.array(all_image_scores_cflow).reshape(-1, 1)) + 
                                      0.4 * minmax_norm(np.array(all_image_scores_patchcore).reshape(-1, 1))).flatten().tolist())
    
    results_table = []
    
    for method_name, (pixel_maps, img_scores) in methods_eval.items():
        pix_flat = pixel_maps.reshape(-1)
        lab_flat = gt_masks.reshape(-1).astype(np.uint8)
        img_scores_arr = np.array(img_scores)
        
        # Pixel metrics
        pixel_auroc = roc_auc_score(lab_flat, pix_flat) if lab_flat.sum() > 0 else 0.5
        pixel_ap = average_precision_score(lab_flat, pix_flat) if lab_flat.sum() > 0 else 0.0
        pixel_f1, pixel_thresh = compute_best_f1(lab_flat, pix_flat)
        
        # AUPRO (slower, compute only for best method)
        aupro = 0.0
        if method_name in ('CFLOW', 'Ensemble'):
            try:
                aupro = compute_aupro(gt_masks, pixel_maps)
            except Exception as e:
                print(f"  AUPRO failed for {method_name}: {e}")
        
        # Image metrics
        image_auroc = roc_auc_score(image_labels, img_scores_arr) \
            if len(np.unique(image_labels)) > 1 else 0.5
        image_ap = average_precision_score(image_labels, img_scores_arr) \
            if len(np.unique(image_labels)) > 1 else 0.0
        image_f1, _ = compute_best_f1(image_labels, img_scores_arr)
        
        results_table.append({
            'method': method_name,
            'pixel_auroc': pixel_auroc,
            'pixel_ap': pixel_ap,
            'pixel_f1': pixel_f1,
            'aupro': aupro,
            'image_auroc': image_auroc,
            'image_ap': image_ap,
            'image_f1': image_f1,
        })
        
        # Log to MLflow
        mlflow.log_metrics({
            f'{method_name}_pixel_auroc': pixel_auroc,
            f'{method_name}_pixel_ap': pixel_ap,
            f'{method_name}_pixel_f1': pixel_f1,
            f'{method_name}_aupro': aupro,
            f'{method_name}_image_auroc': image_auroc,
            f'{method_name}_image_ap': image_ap,
            f'{method_name}_image_f1': image_f1,
        })
    
    # ─── Print results ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"{'Method':<12} {'Pix AUROC':>10} {'Pix AP':>8} {'Pix F1':>8} {'AUPRO':>8} "
          f"{'Img AUROC':>10} {'Img AP':>8} {'Img F1':>8}")
    print("-" * 80)
    for r in results_table:
        print(f"{r['method']:<12} {r['pixel_auroc']:>10.4f} {r['pixel_ap']:>8.4f} "
              f"{r['pixel_f1']:>8.4f} {r['aupro']:>8.4f} {r['image_auroc']:>10.4f} "
              f"{r['image_ap']:>8.4f} {r['image_f1']:>8.4f}")
    
    # ─── Generate plots ──────────────────────────────────────────────────────
    print("\n--- Generating Plots ---")
    
    # 1. Image-level score histogram
    normal_img_scores = np.array(all_image_scores_cflow)[image_labels == 0]
    anomaly_img_scores = np.array(all_image_scores_cflow)[image_labels == 1]
    plot_score_histogram(normal_img_scores, anomaly_img_scores,
                         "CFLOW Image-Level Score Distribution",
                         os.path.join(plots_dir, "hist_image_scores_cflow.png"))
    
    if not args.disable_patchcore:
        normal_pc_scores = np.array(all_image_scores_patchcore)[image_labels == 0]
        anomaly_pc_scores = np.array(all_image_scores_patchcore)[image_labels == 1]
        plot_score_histogram(normal_pc_scores, anomaly_pc_scores,
                             "PatchCore Image-Level Score Distribution",
                             os.path.join(plots_dir, "hist_image_scores_patchcore.png"))
    
    # 2. Pixel-level score histogram
    normal_pix = cflow_maps[gt_masks == 0]
    anomaly_pix = cflow_maps[gt_masks > 0]
    plot_pixel_histogram(normal_pix, anomaly_pix,
                         "CFLOW Pixel-Level Score Distribution",
                         os.path.join(plots_dir, "hist_pixel_scores_cflow.png"))
    
    if not args.disable_patchcore:
        normal_pix_pc = pc_maps[gt_masks == 0]
        anomaly_pix_pc = pc_maps[gt_masks > 0]
        plot_pixel_histogram(normal_pix_pc, anomaly_pix_pc,
                             "PatchCore Pixel-Level Score Distribution",
                             os.path.join(plots_dir, "hist_pixel_scores_patchcore.png"))
    
    # 3. ROC curves
    plot_roc_curve(image_labels, np.array(all_image_scores_cflow),
                   "CFLOW Image-Level ROC Curve",
                   os.path.join(plots_dir, "roc_image_cflow.png"))
    
    lab_flat = gt_masks.reshape(-1).astype(np.uint8)
    # Subsample for ROC plotting (full pixel set is too large)
    n_sample = min(500000, len(lab_flat))
    idx = np.random.choice(len(lab_flat), n_sample, replace=False)
    plot_roc_curve(lab_flat[idx], cflow_maps.reshape(-1)[idx],
                   "CFLOW Pixel-Level ROC Curve",
                   os.path.join(plots_dir, "roc_pixel_cflow.png"))
    
    if not args.disable_patchcore:
        plot_roc_curve(lab_flat[idx], ensemble_maps.reshape(-1)[idx],
                       "Ensemble Pixel-Level ROC Curve",
                       os.path.join(plots_dir, "roc_pixel_ensemble.png"))
    
    # 4. Anomaly map visualization (anomaly samples only)
    anomaly_indices = [i for i, l in enumerate(all_image_labels) if l == 1]
    if anomaly_indices and len(all_images_for_viz) > 0:
        viz_images = []
        viz_masks = []
        viz_maps = []
        for idx in anomaly_indices[:8]:
            if idx < len(all_images_for_viz):
                viz_images.append(all_images_for_viz[idx])
                viz_masks.append(torch.tensor(gt_masks[idx]))
                viz_maps.append(cflow_maps[idx])
        if viz_images:
            plot_anomaly_maps(viz_images, viz_masks, viz_maps,
                              os.path.join(plots_dir, "anomaly_maps_cflow.png"))
            
            # Also for ensemble
            if not args.disable_patchcore:
                viz_maps_ens = [ensemble_maps[anomaly_indices[i]] 
                               for i in range(min(8, len(anomaly_indices)))
                               if anomaly_indices[i] < len(ensemble_maps)]
                if viz_maps_ens:
                    plot_anomaly_maps(viz_images[:len(viz_maps_ens)], 
                                      viz_masks[:len(viz_maps_ens)], viz_maps_ens,
                                      os.path.join(plots_dir, "anomaly_maps_ensemble.png"))
    
    # 5. Method comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods_names = [r['method'] for r in results_table]
    
    # Pixel metrics
    x = np.arange(len(methods_names))
    width = 0.25
    axes[0].bar(x - width, [r['pixel_auroc'] for r in results_table], width, label='AUROC')
    axes[0].bar(x, [r['pixel_ap'] for r in results_table], width, label='AP')
    axes[0].bar(x + width, [r['pixel_f1'] for r in results_table], width, label='F1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods_names)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Pixel-Level Metrics')
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    
    # Image metrics
    axes[1].bar(x - width, [r['image_auroc'] for r in results_table], width, label='AUROC')
    axes[1].bar(x, [r['image_ap'] for r in results_table], width, label='AP')
    axes[1].bar(x + width, [r['image_f1'] for r in results_table], width, label='F1')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods_names)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Image-Level Metrics')
    axes[1].legend()
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_comparison.png"), dpi=150)
    plt.close()
    
    # ─── Log all plots to MLflow ─────────────────────────────────────────────
    for f in os.listdir(plots_dir):
        mlflow.log_artifact(os.path.join(plots_dir, f), artifact_path="plots")
    
    # ─── Save results CSV ────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, "final_results.csv")
    with open(csv_path, 'w') as f:
        f.write("method,pixel_auroc,pixel_ap,pixel_f1,aupro,image_auroc,image_ap,image_f1\n")
        for r in results_table:
            f.write(f"{r['method']},{r['pixel_auroc']:.6f},{r['pixel_ap']:.6f},"
                    f"{r['pixel_f1']:.6f},{r['aupro']:.6f},{r['image_auroc']:.6f},"
                    f"{r['image_ap']:.6f},{r['image_f1']:.6f}\n")
    mlflow.log_artifact(csv_path)
    
    # ─── Save best models as MLflow artifacts ────────────────────────────────
    mlflow.log_artifacts(best_dir, artifact_path="best_models")
    
    # ─── Final summary ───────────────────────────────────────────────────────
    best_pix_method = max(results_table, key=lambda r: r['pixel_auroc'])
    best_img_method = max(results_table, key=lambda r: r['image_auroc'])
    
    print(f"\n{'='*60}")
    print(f"BEST PIXEL:  {best_pix_method['method']} -> AUROC={best_pix_method['pixel_auroc']:.4f}")
    print(f"BEST IMAGE:  {best_img_method['method']} -> AUROC={best_img_method['image_auroc']:.4f}")
    print(f"{'='*60}")
    print(f"\nRun directory: {run_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    mlflow.log_metrics({
        'final_best_pixel_auroc': best_pix_method['pixel_auroc'],
        'final_best_image_auroc': best_img_method['image_auroc'],
    })
    
    mlflow.end_run()
    print("MLflow run ended.")


if __name__ == '__main__':
    main()
