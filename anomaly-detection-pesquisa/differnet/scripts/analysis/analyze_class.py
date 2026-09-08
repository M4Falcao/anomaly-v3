"""
Standalone Class-Level Anomaly Analysis
========================================

Given:
  1. A pretrained SEDifferNet (image-level anomaly detection)
  2. A trained CFLOW pixel head checkpoint (pixel-level localization)

This script produces a complete analysis for a single class:
  - Per-image anomaly scores (image-level)
  - Per-image anomaly heatmaps (pixel-level, only for anomalous images)
  - Aggregated metrics: Image AUROC, Pixel AUROC, AUPRO, IoU, Dice, F1, AP
  - Score distribution histograms
  - ROC curves (image and pixel)
  - Per-image 4-panel visualizations (Original | GT Mask | Heatmap | Overlay)
  - Summary CSV and JSON

Usage:
    python analyze_class.py ^
        --model_path  <path_to_sediffernet.pt> ^
        --cflow_checkpoint <path_to_cflow_head.pt> ^
        --class_name polymer-insulator-upper-shackle ^
        --dataset_path <path_to_insplad-seg>

All outputs are saved under:
    ./analysis/runs/<timestamp>_<class_name>/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import datetime
import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter, label as connected_components
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score
)
from skimage.filters import threshold_otsu
from tqdm import tqdm
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- Make the project root importable regardless of the working directory ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import config as c
from core.model import SEDifferNet, CBAMDifferNet, load_weights
from core.utils import load_datasets, t2np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture Components (mirrors pixel_train_from_pretrained.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SEBackboneFeatureExtractor(nn.Module):
    """Extracts multi-layer spatial features from a pretrained SEDifferNet."""

    def __init__(self, model, out_size=56):
        super().__init__()
        self.alexnet = model.alexnet
        self._use_cbam = hasattr(model, 'cbam1')
        if self._use_cbam:
            self.attn1, self.attn2 = model.cbam1, model.cbam2
            self.attn3, self.attn4 = model.cbam3, model.cbam4
        else:
            self.attn1, self.attn2 = model.simsa1, model.simsa2
            self.attn3, self.attn4 = model.simsa3, model.simsa4
        self.out_size = out_size
        self.layer_channels = [64, 192, 256]
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        feats = []
        x = self.alexnet.features[0](x)
        x = self.alexnet.features[1](x)
        x = self.attn1(x)
        feats.append(x)
        x = self.alexnet.features[2](x)
        x = self.alexnet.features[3](x)
        x = self.alexnet.features[4](x)
        x = self.attn2(x)
        feats.append(x)
        x = self.alexnet.features[5](x)
        x = self.alexnet.features[6](x)
        x = self.alexnet.features[7](x)
        x = self.attn3(x)
        x = self.alexnet.features[8](x)
        x = self.alexnet.features[9](x)
        x = self.alexnet.features[10](x)
        x = self.alexnet.features[11](x)
        x = self.attn4(x)
        feats.append(x)
        resized = [F.interpolate(f, size=(self.out_size, self.out_size),
                                 mode='bilinear', align_corners=False) for f in feats]
        concat = torch.cat(resized, dim=1)
        return concat, resized


def _pos_encoding(h, w, dim=64, device=DEVICE):
    assert dim % 4 == 0
    d = dim // 4
    y = torch.arange(h, device=device).float()
    x = torch.arange(w, device=device).float()
    div = torch.exp(torch.arange(0, d, device=device).float() * (-math.log(10000.0) / d))
    yy = y.unsqueeze(1) * div.unsqueeze(0)
    xx = x.unsqueeze(1) * div.unsqueeze(0)
    pe_y = torch.cat([yy.sin(), yy.cos()], dim=1)
    pe_x = torch.cat([xx.sin(), xx.cos()], dim=1)
    pe_y = pe_y.unsqueeze(1).expand(h, w, dim // 2)
    pe_x = pe_x.unsqueeze(0).expand(h, w, dim // 2)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)


class CondCouplingBlock(nn.Module):
    def __init__(self, channels, cond_dim, hidden=256):
        super().__init__()
        self.c_split = channels // 2
        c_pass = channels - self.c_split
        self.net = nn.Sequential(
            nn.Linear(c_pass + cond_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * self.c_split),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, cond):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        st = self.net(torch.cat([x2, cond], dim=1))
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 2.0
        y1 = x1 * torch.exp(s) + t
        return torch.cat([y1, x2], dim=1), s.sum(dim=1)


class CondFlow(nn.Module):
    def __init__(self, channels, cond_dim, n_blocks=6, hidden=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            CondCouplingBlock(channels, cond_dim, hidden) for _ in range(n_blocks)
        ])
        self.register_buffer('perm_idx',
                             torch.tensor([(i + channels // 2) % channels for i in range(channels)]))

    def forward(self, x, cond):
        ld_sum = torch.zeros(x.size(0), device=x.device)
        for blk in self.blocks:
            x, ld = blk(x, cond)
            ld_sum += ld
            x = x.index_select(1, self.perm_idx)
        return x, ld_sum


class CFlowPixelHead(nn.Module):
    def __init__(self, layer_channels, cond_dim=64, n_blocks=6, hidden=256):
        super().__init__()
        self.cond_dim = cond_dim
        self.flows = nn.ModuleList([
            CondFlow(ch, cond_dim, n_blocks=n_blocks, hidden=hidden)
            for ch in layer_channels
        ])

    def nll_per_level(self, feats_per_level):
        maps = []
        for feat, flow in zip(feats_per_level, self.flows):
            B, C, H, W = feat.shape
            pe = _pos_encoding(H, W, dim=self.cond_dim, device=feat.device)
            pe = pe.unsqueeze(0).expand(B, -1, -1)
            x = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            cond = pe.reshape(B * H * W, self.cond_dim)
            z, log_det = flow(x, cond)
            log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * C * math.log(2 * math.pi)
            maps.append((-log_prob - log_det).view(B, H, W))
        return maps

    def score_map(self, feats_per_level, img_size):
        nll = self.nll_per_level(feats_per_level)
        up = [F.interpolate(m.unsqueeze(1), size=(img_size, img_size),
                            mode='bilinear', align_corners=False).squeeze(1) for m in nll]
        return sum(up)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def minmax_norm(arr, eps=1e-8):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + eps) if (mx - mn) > eps else np.zeros_like(arr)


def compute_aupro(masks, scores, max_fpr=0.3, n_thresholds=200):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    pro_vals, fpr_vals = [], []
    total_normal = (masks == 0).sum()
    for th in thresholds:
        bp = (scores >= th).astype(np.uint8)
        fp = ((bp == 1) & (masks == 0)).sum()
        fpr = fp / max(total_normal, 1)
        if fpr > max_fpr:
            continue
        overlaps = []
        for i in range(masks.shape[0]):
            if masks[i].max() == 0:
                continue
            labeled, n = connected_components(masks[i])
            for rid in range(1, n + 1):
                rm = (labeled == rid)
                rs = rm.sum()
                if rs == 0:
                    continue
                overlaps.append((bp[i] & rm).sum() / rs)
        if overlaps:
            pro_vals.append(np.mean(overlaps))
            fpr_vals.append(fpr)
    if len(fpr_vals) < 2:
        return 0.0
    idx = np.argsort(fpr_vals)
    return np.trapz(np.array(pro_vals)[idx], np.array(fpr_vals)[idx]) / max_fpr


def compute_pixel_auroc(pred, gt):
    gt_bin = (gt > 0.5).astype(int).flatten()
    if gt_bin.sum() == 0 or gt_bin.sum() == len(gt_bin):
        return np.nan
    try:
        return roc_auc_score(gt_bin, pred.flatten())
    except ValueError:
        return np.nan


def compute_iou_dice(pred, gt):
    gt_bin = (gt > 0.5).astype(int)
    if gt_bin.sum() == 0:
        return np.nan, np.nan
    try:
        th = threshold_otsu(pred)
        pred_bin = (pred >= th).astype(int)
    except ValueError:
        return np.nan, np.nan
    inter = (pred_bin * gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum() - inter
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (pred_bin.sum() + gt_bin.sum()) if (pred_bin.sum() + gt_bin.sum()) > 0 else 0.0
    return iou, dice


def compute_best_f1(labels, scores):
    prec, rec, ths = precision_recall_curve(labels, scores)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    best = np.argmax(f1s)
    return f1s[best], ths[min(best, len(ths) - 1)]


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def denorm(tensor):
    """(C,H,W) tensor -> (H,W,3) np [0,1]."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * STD + MEAN, 0, 1)


# Custom colormap: transparent blue -> cyan -> yellow -> red
from matplotlib.colors import LinearSegmentedColormap
ANOMALY_CMAP = LinearSegmentedColormap.from_list(
    'anomaly_blue_red',
    [(0.0, (0.0, 0.0, 0.5, 0.0)),    # transparent blue
     (0.2, (0.0, 0.0, 1.0, 0.3)),    # blue
     (0.4, (0.0, 1.0, 1.0, 0.5)),    # cyan
     (0.6, (1.0, 1.0, 0.0, 0.6)),    # yellow
     (0.8, (1.0, 0.5, 0.0, 0.8)),    # orange
     (1.0, (0.9, 0.0, 0.0, 1.0))],   # red
    N=256
)


def save_per_image_viz(img_tensor, mask_np, heatmap, img_score, pixel_auc,
                       save_path, fname, label_str, pred_label, threshold):
    """Save a 4-panel visualization: Original | GT Mask | Heatmap | Overlay.
    
    Shows both the ground truth label and the model prediction.
    """
    img = denorm(img_tensor)
    hmap = minmax_norm(heatmap)

    # Determine if prediction is correct
    gt_is_anomaly = (label_str == 'anomaly')
    pred_is_anomaly = (pred_label == 'ANOMALY')
    is_correct = (gt_is_anomaly == pred_is_anomaly)
    verdict_str = 'CORRECT' if is_correct else 'WRONG'
    verdict_color = '#2ecc71' if is_correct else '#e74c3c'

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    # Panel 1: Input with GT label + model prediction
    axes[0].imshow(img)
    title_lines = f'Input (GT: {label_str})\n'
    title_lines += f'Pred: {pred_label} [{verdict_str}]'
    axes[0].set_title(title_lines, fontsize=11,
                      color=verdict_color, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground Truth mask
    axes[1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth Mask', fontsize=11)
    axes[1].axis('off')

    # Panel 3: CFLOW heatmap (blue -> red)
    im2 = axes[2].imshow(hmap, cmap='jet', vmin=0, vmax=1)
    hmap_title = 'CFLOW Anomaly Map'
    if not np.isnan(pixel_auc):
        hmap_title += f'\nPixel AUROC: {pixel_auc:.4f}'
    axes[2].set_title(hmap_title, fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Overlay (image + red heatmap)
    axes[3].imshow(img)
    axes[3].imshow(hmap, cmap=ANOMALY_CMAP, vmin=0, vmax=1)
    # Mark max activation
    max_loc = np.unravel_index(np.argmax(hmap), hmap.shape)
    axes[3].plot(max_loc[1], max_loc[0], 'wx', markersize=14, markeredgewidth=3)
    axes[3].plot(max_loc[1], max_loc[0], 'rx', markersize=12, markeredgewidth=2)
    axes[3].set_title(f'Overlay\nScore: {img_score:.1f} (th: {threshold:.1f})', fontsize=11)
    axes[3].axis('off')

    plt.suptitle(fname, fontsize=9, color='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc(labels, scores, title, save_path):
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUROC = {auroc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_histogram(normal, anomaly, title, xlabel, save_path, n_max=100000):
    if len(normal) > n_max:
        normal = np.random.choice(normal, n_max, replace=False)
    if len(anomaly) > n_max:
        anomaly = np.random.choice(anomaly, n_max, replace=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normal, bins=60, alpha=0.6, label='Normal', color='#2ecc71', density=True)
    ax.hist(anomaly, bins=60, alpha=0.6, label='Anomaly', color='#e74c3c', density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_metrics_summary(metrics, save_path):
    """Bar chart of all scalar metrics."""
    keys = ['image_auroc', 'pixel_auroc', 'aupro', 'iou', 'dice', 'f1', 'ap']
    vals = [metrics.get(k, 0) for k in keys]
    labels = ['Image\nAUROC', 'Pixel\nAUROC', 'AUPRO', 'IoU', 'Dice', 'F1', 'AP']

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#1abc9c', '#f39c12', '#e67e22', '#2ecc71']
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=1.2)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title(f"Anomaly Detection Metrics — {metrics.get('class_name', '')}")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Class-Level Anomaly Analysis")

    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to pretrained SEDifferNet .pt checkpoint")
    parser.add_argument('--cflow_checkpoint', type=str, required=True,
                        help="Path to CFLOW pixel head .pt checkpoint")
    parser.add_argument('--class_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, default=c.dataset_path)

    # Architecture params (must match training)
    parser.add_argument('--img_size', type=int, default=448)
    parser.add_argument('--out_size', type=int, default=56)
    parser.add_argument('--cond_dim', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=6)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--sigma', type=float, default=4.0,
                        help="Gaussian smoothing sigma for heatmaps")

    # Optional
    parser.add_argument('--limit', type=int, default=None,
                        help="Limit number of test samples")

    args = parser.parse_args()

    # ── Setup output dir ──────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.class_name}"
    run_dir = os.path.join(PROJECT_ROOT, "analysis", "runs", run_name)
    images_dir = os.path.join(run_dir, "images")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print(f"  CLASS ANOMALY ANALYSIS")
    print(f"  Class:          {args.class_name}")
    print(f"  Model:          {os.path.basename(args.model_path)}")
    print(f"  CFLOW Head:     {os.path.basename(args.cflow_checkpoint)}")
    print(f"  Output:         {run_dir}")
    print("=" * 70)

    # ── 1. Load Model ─────────────────────────────────────────────────────
    print("\n[1/5] Loading SEDifferNet model...")
    base_model = SEDifferNet()
    model, checkpoint = load_weights(base_model, args.model_path)
    model.to(DEVICE).eval()

    # ── 2. Load CFLOW Head ────────────────────────────────────────────────
    print("[2/5] Loading CFLOW pixel head...")
    backbone = SEBackboneFeatureExtractor(model, out_size=args.out_size).to(DEVICE).eval()

    cflow = CFlowPixelHead(
        layer_channels=backbone.layer_channels,
        cond_dim=args.cond_dim,
        n_blocks=args.n_blocks,
        hidden=args.hidden,
    ).to(DEVICE)

    cflow_data = torch.load(args.cflow_checkpoint, map_location=DEVICE, weights_only=False)
    if isinstance(cflow_data, dict) and 'cflow_state_dict' in cflow_data:
        cflow.load_state_dict(cflow_data['cflow_state_dict'])
    else:
        raise RuntimeError("CFLOW checkpoint does not contain 'cflow_state_dict' key.")
    cflow.eval()
    print(f"  CFLOW loaded successfully ({sum(p.numel() for p in cflow.parameters()):,} params)")

    # ── 3. Load Dataset ───────────────────────────────────────────────────
    print("[3/5] Loading dataset...")
    c.n_transforms_test = 1
    c.transf_rotations = False
    c.img_size = (args.img_size, args.img_size)

    trainset, testset = load_datasets(args.dataset_path, args.class_name, aligned=True)
    total = len(testset)
    if args.limit:
        total = min(args.limit, total)
    print(f"  Test samples: {total}")

    # ── 4. Run Inference (two passes) ───────────────────────────────────
    # Pass 1: Collect all scores + maps (no visualization yet)
    print(f"[4/6] Pass 1 — Collecting scores on {total} samples...")

    all_image_scores = []
    all_image_labels = []
    all_cflow_maps = []
    all_masks = []
    all_image_tensors = []  # store for pass 2
    all_filenames = []
    all_label_strs = []

    for idx in tqdm(range(total), desc="Pass 1 (scores)"):
        images, label, mask = testset[idx]
        images = images.to(DEVICE).view(-1, *images.shape[-3:])
        mask_np = mask.squeeze().numpy()
        label_val = int(label) if isinstance(label, (int, float)) else int(label.item())
        is_anomaly = label_val > 0

        # Image-level score
        with torch.no_grad():
            z = model(images[:1])
            if isinstance(z, tuple):
                z = z[0]
            if z.dim() == 1:
                z = z.unsqueeze(0)
            img_score = torch.mean(torch.sum(z ** 2, dim=1)).item()

        all_image_scores.append(img_score)
        all_image_labels.append(1 if is_anomaly else 0)

        # Pixel-level score
        with torch.no_grad():
            concat, feats = backbone(images[:1])
            cflow_map = cflow.score_map(feats, args.img_size).squeeze(0).cpu().numpy()
            cflow_map = gaussian_filter(cflow_map, sigma=args.sigma)

        cflow_norm = minmax_norm(cflow_map)
        all_cflow_maps.append(cflow_norm)
        all_masks.append(mask_np)
        all_image_tensors.append(images[0].cpu())

        img_path = testset.samples[idx][0]
        all_filenames.append(os.path.basename(img_path))
        all_label_strs.append("anomaly" if is_anomaly else "normal")

        torch.cuda.empty_cache()

    # Compute optimal image-level threshold (best F1)
    image_labels = np.array(all_image_labels)
    image_scores = np.array(all_image_scores)

    if len(np.unique(image_labels)) > 1:
        _, best_threshold = compute_best_f1(image_labels, image_scores)
        print(f"  Optimal image threshold (best F1): {best_threshold:.4f}")
    else:
        best_threshold = float(np.median(image_scores))
        print(f"  Only one class present. Using median as threshold: {best_threshold:.4f}")

    # Pass 2: Generate visualizations with prediction labels
    print(f"[5/6] Pass 2 — Generating visualizations with predictions...")

    per_image_records = []
    n_correct = 0

    for idx in tqdm(range(total), desc="Pass 2 (viz)"):
        img_tensor = all_image_tensors[idx]
        mask_np = all_masks[idx]
        cflow_norm = all_cflow_maps[idx]
        img_score = all_image_scores[idx]
        fname = all_filenames[idx]
        label_str = all_label_strs[idx]

        # Model prediction using optimal threshold
        pred_is_anomaly = img_score >= best_threshold
        pred_label = "ANOMALY" if pred_is_anomaly else "NORMAL"
        gt_is_anomaly = (label_str == "anomaly")
        is_correct = (pred_is_anomaly == gt_is_anomaly)
        if is_correct:
            n_correct += 1

        # Per-image metrics
        pixel_auc = compute_pixel_auroc(cflow_norm, mask_np)
        iou_val, dice_val = compute_iou_dice(cflow_norm, mask_np)

        # Save visualization
        verdict = "correct" if is_correct else "wrong"
        save_name = f"{idx:04d}_{label_str}_{verdict}_{fname}"
        save_path = os.path.join(images_dir, save_name.replace('.', '_') + '.png')

        save_per_image_viz(
            img_tensor, mask_np, cflow_norm, img_score, pixel_auc,
            save_path, fname, label_str, pred_label, best_threshold
        )

        per_image_records.append({
            'index': idx,
            'filename': fname,
            'gt_label': label_str,
            'pred_label': pred_label,
            'correct': is_correct,
            'image_score': img_score,
            'threshold': best_threshold,
            'pixel_auroc': pixel_auc if not np.isnan(pixel_auc) else None,
            'iou': iou_val if not np.isnan(iou_val) else None,
            'dice': dice_val if not np.isnan(dice_val) else None,
        })

    img_accuracy = n_correct / total * 100
    print(f"  Image classification accuracy: {n_correct}/{total} ({img_accuracy:.1f}%)")

    # ── 6. Aggregate Metrics ──────────────────────────────────────────────
    print("[6/6] Computing aggregated metrics...")

    cflow_maps = np.stack(all_cflow_maps, axis=0)
    gt_masks = np.stack(all_masks, axis=0)
    gt_binary = (gt_masks > 0.5).astype(int)

    metrics = {'class_name': args.class_name}
    metrics['image_accuracy'] = img_accuracy
    metrics['image_threshold'] = float(best_threshold)

    # Image AUROC
    if len(np.unique(image_labels)) > 1:
        metrics['image_auroc'] = roc_auc_score(image_labels, image_scores)
        metrics['image_ap'] = average_precision_score(image_labels, image_scores)
        f1_img, _ = compute_best_f1(image_labels, image_scores)
        metrics['image_f1'] = f1_img
    else:
        metrics['image_auroc'] = np.nan
        metrics['image_ap'] = np.nan

    # Pixel AUROC (only anomalous)
    anom_mask = image_labels == 1
    if anom_mask.sum() > 0:
        anom_maps = cflow_maps[anom_mask]
        anom_gts = gt_binary[anom_mask]
        flat_pred = anom_maps.flatten()
        flat_gt = anom_gts.flatten()
        if flat_gt.sum() > 0 and flat_gt.sum() < len(flat_gt):
            metrics['pixel_auroc'] = roc_auc_score(flat_gt, flat_pred)
        else:
            metrics['pixel_auroc'] = np.nan

        # AUPRO
        metrics['aupro'] = compute_aupro(anom_gts, anom_maps)

        # Avg IoU / Dice
        ious, dices = [], []
        for rec in per_image_records:
            if rec['gt_label'] == 'anomaly':
                if rec['iou'] is not None:
                    ious.append(rec['iou'])
                if rec['dice'] is not None:
                    dices.append(rec['dice'])
        metrics['iou'] = np.mean(ious) if ious else np.nan
        metrics['dice'] = np.mean(dices) if dices else np.nan

        # Pixel F1 / AP
        metrics['pixel_ap'] = average_precision_score(flat_gt, flat_pred) if flat_gt.sum() > 0 else np.nan
        f1_pix, th_pix = compute_best_f1(flat_gt, flat_pred)
        metrics['f1'] = f1_pix
        metrics['ap'] = metrics['pixel_ap']
    else:
        metrics['pixel_auroc'] = np.nan
        metrics['aupro'] = np.nan
        metrics['iou'] = np.nan
        metrics['dice'] = np.nan
        metrics['f1'] = np.nan
        metrics['ap'] = np.nan

    # ── Save results ──────────────────────────────────────────────────────

    # Per-image CSV
    df_images = pd.DataFrame(per_image_records)
    df_images.to_csv(os.path.join(run_dir, 'per_image_results.csv'), index=False)

    # Aggregated metrics JSON
    metrics_clean = {k: (float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else v)
                     for k, v in metrics.items()}
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_clean, f, indent=2, default=str)

    # Aggregated metrics CSV
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(os.path.join(run_dir, 'metrics_summary.csv'), index=False)

    # ── Plots ─────────────────────────────────────────────────────────────

    # 1. Metrics summary bar chart
    plot_metrics_summary(metrics, os.path.join(plots_dir, 'metrics_summary.png'))

    # 2. Image-level score histogram
    if len(np.unique(image_labels)) > 1:
        plot_histogram(
            image_scores[image_labels == 0],
            image_scores[image_labels == 1],
            'Image-Level Score Distribution (SEDifferNet)',
            'Anomaly Score ||z||²',
            os.path.join(plots_dir, 'hist_image_scores.png')
        )
        # 3. Image-level ROC
        plot_roc(image_labels, image_scores,
                 'Image-Level ROC (SEDifferNet)',
                 os.path.join(plots_dir, 'roc_image.png'))

    # 4. Pixel-level score histogram
    if anom_mask.sum() > 0:
        normal_pix = cflow_maps[gt_binary == 0]
        anomaly_pix = cflow_maps[gt_binary > 0]
        if len(anomaly_pix) > 0:
            plot_histogram(
                normal_pix, anomaly_pix,
                'Pixel-Level Score Distribution (CFLOW)',
                'Pixel Anomaly Score',
                os.path.join(plots_dir, 'hist_pixel_scores.png')
            )

        # 5. Pixel-level ROC
        if flat_gt.sum() > 0 and flat_gt.sum() < len(flat_gt):
            # Subsample for speed
            n_sub = min(500000, len(flat_gt))
            sub_idx = np.random.choice(len(flat_gt), n_sub, replace=False)
            plot_roc(flat_gt[sub_idx], flat_pred[sub_idx],
                     'Pixel-Level ROC (CFLOW)',
                     os.path.join(plots_dir, 'roc_pixel.png'))

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    for k, v in metrics.items():
        if k == 'class_name':
            continue
        if isinstance(v, float) and not np.isnan(v):
            print(f"  {k:20s}: {v:.4f}")
        else:
            print(f"  {k:20s}: {v}")
    print("=" * 70)
    print(f"\n  Output saved to: {run_dir}")
    print(f"  Per-image visualizations: {images_dir}")
    print(f"  Plots: {plots_dir}")
    n_anomaly_imgs = sum(1 for r in per_image_records if r['label'] == 'anomaly')
    print(f"  Total images analyzed: {total} ({n_anomaly_imgs} anomalous)")
    print("=" * 70)


if __name__ == "__main__":
    main()
