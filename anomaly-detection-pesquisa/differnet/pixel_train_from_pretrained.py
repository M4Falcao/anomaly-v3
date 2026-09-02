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
    python pixel_train_from_pretrained.py
    python pixel_train_from_pretrained.py --checkpoint path/to/model.pt --epochs 80
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
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
from model import SEDifferNet, CBAMDifferNet, load_weights
from utils import load_datasets, make_dataloaders, t2np


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ═══════════════════════════════════════════════════════════════════════════════
# Backbone Feature Extractor (from pretrained SEDifferNet)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_arch(checkpoint_path):
    """Detect model architecture from checkpoint state_dict keys."""
    data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    keys = data.get('model_state_dict', data).keys()
    if any('cbam' in k for k in keys):
        return 'cbam'
    return 'se'


class SEBackboneFeatureExtractor(nn.Module):
    """Extracts multi-layer spatial features from a pretrained SEDifferNet or CBAMDifferNet.
    
    Extracts features BEFORE GAP at 3 semantic levels:
      L1: after attn1 -> 64 channels
      L2: after attn2 -> 192 channels  
      L3: after attn4 -> 256 channels
    
    All maps are bilinearly upsampled to `out_size`.
    Works transparently for both SE and CBAM model variants.
    """
    
    def __init__(self, model, out_size=56):
        super().__init__()
        self.alexnet = model.alexnet
        # CBAM models use cbam* for spatial path; SE models use simsa*
        # Both variants expose the same forward logic; we store whichever exists
        self._use_cbam = hasattr(model, 'cbam1')
        if self._use_cbam:
            self.attn1 = model.cbam1
            self.attn2 = model.cbam2
            self.attn3 = model.cbam3
            self.attn4 = model.cbam4
        else:
            self.attn1 = model.simsa1
            self.attn2 = model.simsa2
            self.attn3 = model.simsa3
            self.attn4 = model.simsa4
        self.out_size = out_size
        self.layer_channels = [64, 192, 256]
        
        # Freeze everything
        for p in self.parameters():
            p.requires_grad_(False)
    
    @torch.no_grad()
    def forward(self, x_input):
        """Extract multi-layer features (no multi-scale to preserve spatial info).
        
        Returns: (concat, [L1, L2, L3]) where concat is all layers concatenated.
        """
        feats = []
        
        x = self.alexnet.features[0](x_input)   # Conv1
        x = self.alexnet.features[1](x)         # ReLU
        x = self.attn1(x)                       # 64ch
        feats.append(x)                         # L1: 64ch
        
        x = self.alexnet.features[2](x)         # MaxPool
        x = self.alexnet.features[3](x)         # Conv2
        x = self.alexnet.features[4](x)         # ReLU
        x = self.attn2(x)                       # 192ch
        feats.append(x)                         # L2: 192ch
        
        x = self.alexnet.features[5](x)         # MaxPool
        x = self.alexnet.features[6](x)         # Conv3
        x = self.alexnet.features[7](x)         # ReLU
        x = self.attn3(x)                       # 384ch (skip for concat)
        x = self.alexnet.features[8](x)         # Conv4
        x = self.alexnet.features[9](x)         # ReLU
        x = self.alexnet.features[10](x)        # Conv5
        x = self.alexnet.features[11](x)        # ReLU
        x = self.attn4(x)                       # 256ch
        feats.append(x)                         # L3: 256ch
        
        # Upsample all to common spatial size
        resized = [F.interpolate(f, size=(self.out_size, self.out_size),
                                 mode='bilinear', align_corners=False) for f in feats]
        concat = torch.cat(resized, dim=1)  # (B, 512, H, W)
        return concat, resized
    
    @property
    def concat_channels(self):
        return sum(self.layer_channels)


# ═══════════════════════════════════════════════════════════════════════════════
# CFLOW-AD 
# ═══════════════════════════════════════════════════════════════════════════════

def _pos_encoding(h, w, dim=64, device=DEVICE):
    """2D sinusoidal positional encoding -> (h*w, dim)."""
    assert dim % 4 == 0
    d_quarter = dim // 4
    y_pos = torch.arange(h, device=device).float()
    x_pos = torch.arange(w, device=device).float()
    div = torch.exp(torch.arange(0, d_quarter, device=device).float() *
                    (-math.log(10000.0) / d_quarter))
    yy = y_pos.unsqueeze(1) * div.unsqueeze(0)
    xx = x_pos.unsqueeze(1) * div.unsqueeze(0)
    pe_y = torch.cat([yy.sin(), yy.cos()], dim=1)
    pe_x = torch.cat([xx.sin(), xx.cos()], dim=1)
    pe_y = pe_y.unsqueeze(1).expand(h, w, dim // 2)
    pe_x = pe_x.unsqueeze(0).expand(h, w, dim // 2)
    pe = torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)
    return pe


class CondCouplingBlock(nn.Module):
    def __init__(self, channels, cond_dim, hidden=256):
        super().__init__()
        self.c_split = channels // 2
        c_pass = channels - self.c_split
        self.net = nn.Sequential(
            nn.Linear(c_pass + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * self.c_split),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, cond):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        st = self.net(torch.cat([x2, cond], dim=1))
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 0.5  # Conservative scale for training stability
        y1 = x1 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([y1, x2], dim=1), log_det


class CondFlow(nn.Module):
    def __init__(self, channels, cond_dim, n_blocks=4, hidden=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            CondCouplingBlock(channels, cond_dim, hidden) for _ in range(n_blocks)
        ])
        self.register_buffer(
            'perm_idx',
            torch.tensor([(i + channels // 2) % channels for i in range(channels)])
        )

    def forward(self, x, cond):
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        for blk in self.blocks:
            x, ld = blk(x, cond)
            log_det_sum = log_det_sum + ld
            x = x.index_select(1, self.perm_idx)
        return x, log_det_sum


class CFlowPixelHead(nn.Module):
    """CFLOW-AD pixel head: one conditional flow per feature level."""
    
    def __init__(self, layer_channels, cond_dim=64, n_blocks=4, hidden=256):
        super().__init__()
        self.cond_dim = cond_dim
        self.flows = nn.ModuleList([
            CondFlow(c, cond_dim, n_blocks=n_blocks, hidden=hidden)
            for c in layer_channels
        ])
    
    def nll_per_level(self, feats_per_level):
        """Compute NLL map per level. Returns list of (B, H, W) tensors.
        Each level's NLL is normalized by its channel count for balanced loss.
        Uses chunked processing to avoid OOM on large spatial maps."""
        maps = []
        for feat, flow in zip(feats_per_level, self.flows):
            B, C, H, W = feat.shape
            pe = _pos_encoding(H, W, dim=self.cond_dim, device=feat.device)
            pe = pe.unsqueeze(0).expand(B, -1, -1)  # (B, HW, cd)
            x = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            cond = pe.reshape(B * H * W, self.cond_dim)
            
            # Process in chunks to save VRAM
            chunk_size = 4096
            n_total = x.shape[0]
            nll_chunks = []
            for i in range(0, n_total, chunk_size):
                x_chunk = x[i:i+chunk_size]
                c_chunk = cond[i:i+chunk_size]
                z, log_det = flow(x_chunk, c_chunk)
                log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * C * math.log(2 * math.pi)
                log_p = log_prob + log_det
                nll_chunks.append(-log_p / C)
            
            nll = torch.cat(nll_chunks, dim=0).view(B, H, W)
            maps.append(nll)
        return maps
    
    def score_map(self, feats_per_level, img_size):
        """Compute aggregated NLL score map with per-level normalization.
        Normalization at native res, upsample on CPU to save VRAM."""
        nll_maps = self.nll_per_level(feats_per_level)
        normalized = []
        for m in nll_maps:
            # Normalize at native (out_size) resolution - much cheaper
            B = m.shape[0]
            flat = m.view(B, -1)
            mn = flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
            mx = flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
            m_norm = (m - mn) / (mx - mn + 1e-8)
            normalized.append(m_norm)
        # Sum normalized maps at native res, then upsample on CPU
        combined = sum(normalized).cpu()
        return F.interpolate(combined.unsqueeze(1), size=(img_size, img_size),
                             mode='bilinear', align_corners=False).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# PatchCore Memory Bank
# ═══════════════════════════════════════════════════════════════════════════════

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
# Post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_smooth(score_map, sigma=4.0):
    out = np.empty_like(score_map)
    for b in range(score_map.shape[0]):
        out[b] = gaussian_filter(score_map[b], sigma=sigma)
    return out


def minmax_norm(arr, eps=1e-8):
    flat = arr.reshape(arr.shape[0], -1)
    mn = flat.min(axis=1, keepdims=True)
    mx = flat.max(axis=1, keepdims=True)
    return ((flat - mn) / (mx - mn + eps)).reshape(arr.shape)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aupro(masks, scores, max_fpr=0.3, n_thresholds=200):
    """Compute Area Under Per-Region Overlap (AUPRO)."""
    from scipy.ndimage import label as connected_components
    
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    pro_values = []
    fpr_values = []
    
    total_normal_pixels = (masks == 0).sum()
    
    for th in thresholds:
        binary_pred = (scores >= th).astype(np.uint8)
        
        # FPR
        fp = ((binary_pred == 1) & (masks == 0)).sum()
        fpr = fp / max(total_normal_pixels, 1)
        if fpr > max_fpr:
            continue
        
        # Per-region overlap
        overlaps = []
        for i in range(masks.shape[0]):
            if masks[i].max() == 0:
                continue
            labeled, n_regions = connected_components(masks[i])
            for region_id in range(1, n_regions + 1):
                region_mask = (labeled == region_id)
                region_size = region_mask.sum()
                if region_size == 0:
                    continue
                overlap = (binary_pred[i] & region_mask).sum() / region_size
                overlaps.append(overlap)
        
        if overlaps:
            pro_values.append(np.mean(overlaps))
            fpr_values.append(fpr)
    
    if len(fpr_values) < 2:
        return 0.0
    
    # Sort by FPR and compute AUC
    sorted_idx = np.argsort(fpr_values)
    fpr_sorted = np.array(fpr_values)[sorted_idx]
    pro_sorted = np.array(pro_values)[sorted_idx]
    aupro = np.trapz(pro_sorted, fpr_sorted) / max_fpr
    return aupro


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
    """Plot training loss and AUROC curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('NLL Loss')
    axes[0].set_title('Training Loss (CFLOW)')
    
    if history['pixel_auroc']:
        eval_epochs = history['eval_epochs']
        axes[1].plot(eval_epochs, history['pixel_auroc'], 'r-o', markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Pixel AUROC')
        axes[1].set_title('Pixel AUROC over Training')
        axes[1].set_ylim([0.5, 1.0])
    
    if history['image_auroc']:
        eval_epochs = history['eval_epochs']
        axes[2].plot(eval_epochs, history['image_auroc'], 'g-o', markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Image AUROC')
        axes[2].set_title('Image AUROC over Training')
        axes[2].set_ylim([0.5, 1.0])
    
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
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', help='Enable AMP (off by default for flow stability)')
    
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
    })
    
    # ─── Load pretrained model (auto-detect SE vs CBAM) ─────────────────────
    arch = _detect_arch(args.checkpoint)
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
    backbone = SEBackboneFeatureExtractor(model, out_size=args.out_size).to(DEVICE).eval()
    attn_type = 'CBAM' if backbone._use_cbam else 'SE'
    print(f"  Backbone: {attn_type} attention | channels: {backbone.layer_channels} -> concat {backbone.concat_channels}")
    
    # ─── Load data ───────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.class_name}")
    trainset, testset = load_datasets(args.dataset, args.class_name, aligned=True)
    train_loader, _ = make_dataloaders(trainset, testset)
    # Use a small batch size for test_loader to avoid OOM during dense CFLOW pixel evaluation
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True)
    print(f"  Train: {len(trainset)} samples")
    print(f"  Test:  {len(testset)} samples (batch_size=1 for eval)")
    
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
        layer_channels=backbone.layer_channels,
        cond_dim=args.cond_dim,
        n_blocks=args.n_blocks,
        hidden=args.hidden
    ).to(DEVICE)
    
    n_params = sum(p.numel() for p in cflow.parameters())
    print(f"  CFLOW parameters: {n_params:,}")
    mlflow.log_param("cflow_params", n_params)
    
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
    print(f"\n{'='*60}")
    print(f"Training CFLOW Pixel Head ({args.epochs} epochs)")
    print(f"  LR: {args.lr} | Warmup: {args.warmup_epochs} ep | Grad clip: {args.grad_clip}")
    print(f"  AMP: {'ON' if args.use_amp else 'OFF'} | out_size: {args.out_size} | sigma: {args.sigma}")
    print(f"{'='*60}")
    
    best_pixel_auroc = 0.0
    best_image_auroc = 0.0
    history = {'train_loss': [], 'pixel_auroc': [], 'image_auroc': [], 'eval_epochs': []}
    
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
            
            with torch.no_grad():
                _, feats = backbone(images)
            
            optimizer.zero_grad()
            if args.use_amp:
                with autocast('cuda'):
                    nll_maps = cflow.nll_per_level(feats)
                    loss = sum(m.mean() for m in nll_maps)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(cflow.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nll_maps = cflow.nll_per_level(feats)
                loss = sum(m.mean() for m in nll_maps)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cflow.parameters(), args.grad_clip)
                optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        history['train_loss'].append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
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
                    _, feats = backbone(images)
                    
                    # CFLOW score
                    smap = cflow.score_map(feats, args.img_size)
                    smap_np = gaussian_smooth(t2np(smap), sigma=args.sigma)
                    
                    # Image score = max of pixel scores
                    img_scores = smap_np.reshape(smap_np.shape[0], -1).max(axis=1)
                    
                    all_pixel_scores.append(smap_np)
                    all_pixel_labels.append(masks.squeeze(1).numpy())
                    all_image_scores.extend(img_scores.tolist())
                    all_image_labels.extend([1 if l > 0 else 0 for l in labels.numpy()])
            
            pixel_scores = np.concatenate(all_pixel_scores, axis=0)
            pixel_labels = np.concatenate(all_pixel_labels, axis=0)
            
            # Pixel AUROC
            pix_flat = pixel_scores.reshape(-1)
            lab_flat = pixel_labels.reshape(-1).astype(np.uint8)
            pixel_auroc = roc_auc_score(lab_flat, pix_flat) if lab_flat.sum() > 0 else 0.5
            
            # Image AUROC
            image_auroc = roc_auc_score(all_image_labels, all_image_scores) \
                if len(np.unique(all_image_labels)) > 1 else 0.5
            
            history['pixel_auroc'].append(pixel_auroc)
            history['image_auroc'].append(image_auroc)
            history['eval_epochs'].append(epoch + 1)
            
            mlflow.log_metric("pixel_auroc", pixel_auroc, step=epoch)
            mlflow.log_metric("image_auroc", image_auroc, step=epoch)
            
            print(f"  Ep {epoch+1}: loss={avg_loss:.4f} | pix={pixel_auroc:.4f} | img={image_auroc:.4f}")
            
            # Save best
            if pixel_auroc > best_pixel_auroc:
                best_pixel_auroc = pixel_auroc
                torch.save({
                    'epoch': epoch + 1,
                    'cflow_state_dict': cflow.state_dict(),
                    'pixel_auroc': pixel_auroc,
                    'image_auroc': image_auroc,
                }, os.path.join(best_dir, "best_pixel_auroc.pt"))
                print(f"    -> New best PIXEL AUROC: {pixel_auroc:.4f}")
            
            if image_auroc > best_image_auroc:
                best_image_auroc = image_auroc
                torch.save({
                    'epoch': epoch + 1,
                    'cflow_state_dict': cflow.state_dict(),
                    'pixel_auroc': pixel_auroc,
                    'image_auroc': image_auroc,
                }, os.path.join(best_dir, "best_image_auroc.pt"))
                print(f"    -> New best IMAGE AUROC: {image_auroc:.4f}")
        
        # Regular checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'cflow_state_dict': cflow.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(ckpt_dir, f"cflow_epoch_{epoch+1}.pt"))
    
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
            concat, feats = backbone(images_dev)
            
            # CFLOW score map
            cflow_map = t2np(cflow.score_map(feats, args.img_size))
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
