"""Avaliação final completa: SEDifferNet (image-level) + CFLOW NF Head (pixel-level).

Combina os modelos finais de ambas as pipelines e avalia o dataset InSPLAD-Seg
com métricas abrangentes, gráficos e salvamento de imagens com erro.

Usage:
    python evaluate_final.py
    python evaluate_final.py --limit 10
    python evaluate_final.py --classes "glass-insulator,vari-grip"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
import math
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, f1_score, confusion_matrix
)
from scipy.ndimage import gaussian_filter, label as connected_components
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Local imports
import config as c
from model import SEDifferNet, load_weights
from utils import load_datasets, make_dataloaders, t2np, AlignedTestDataset

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ALL_CLASSES = [
    'glass-insulator',
    'lightning-rod-suspension',
    'polymer-insulator-upper-shackle',
    'vari-grip',
    'yoke-suspension',
]

SE_MODEL_DIR = os.path.join(os.path.dirname(__file__), 'final_models', 'SEDiffernet')
NF_HEAD_DIR  = os.path.join(os.path.dirname(__file__), 'final_models', 'NF Head')


# ═══════════════════════════════════════════════════════════════════════════════
# Backbone Feature Extractor (reused from pixel_train_from_pretrained.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SEBackboneFeatureExtractor(nn.Module):
    """Extracts multi-layer spatial features from a pretrained SEDifferNet."""

    def __init__(self, model, out_size=56):
        super().__init__()
        self.alexnet = model.alexnet
        self.attn1 = model.simsa1
        self.attn2 = model.simsa2
        self.attn3 = model.simsa3
        self.attn4 = model.simsa4
        self.out_size = out_size
        self.layer_channels = [64, 192, 256]

        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_input):
        feats = []
        x = self.alexnet.features[0](x_input)
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


# ═══════════════════════════════════════════════════════════════════════════════
# CFLOW Pixel Head (reused architecture)
# ═══════════════════════════════════════════════════════════════════════════════

def _pos_encoding(h, w, dim=64, device=DEVICE):
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
    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)


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
        s = torch.tanh(s) * 2.0
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
    def __init__(self, layer_channels, cond_dim=64, n_blocks=4, hidden=256):
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
            log_p = log_prob + log_det
            maps.append((-log_p).view(B, H, W))
        return maps

    def score_map(self, feats_per_level, img_size):
        nll_maps = self.nll_per_level(feats_per_level)
        upsampled = [F.interpolate(m.unsqueeze(1), size=(img_size, img_size),
                                   mode='bilinear', align_corners=False).squeeze(1)
                     for m in nll_maps]
        return sum(upsampled)


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pro_rd_style(mask, amap, num_th=200):
    """Compute AUPRO exactly like RD++ (per-image, ignoring normal images)."""
    from skimage.measure import label, regionprops
    from sklearn.metrics import auc
    
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
        amap = amap[np.newaxis, ...]
        
    min_th = amap.min()
    max_th = amap.max()
    if max_th == min_th:
        return 0.0
        
    delta = (max_th - min_th) / num_th
    
    pro_list = []
    fpr_list = []
    
    regions = regionprops(label(mask[0]))
    inverse_mask = 1 - mask[0]
    total_inverse = inverse_mask.sum()
    if total_inverse == 0:
        return 0.0

    for th in np.arange(min_th, max_th, delta):
        binary_amap = (amap[0] > th).astype(np.uint8)
        
        pros = []
        for region in regions:
            axes0_ids = region.coords[:, 0]
            axes1_ids = region.coords[:, 1]
            tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
            pros.append(tp_pixels / region.area)
            
        fp_pixels = (inverse_mask & binary_amap).sum()
        fpr = fp_pixels / total_inverse
        
        pro_list.append(np.mean(pros) if pros else 0.0)
        fpr_list.append(fpr)
        
    pro_arr = np.array(pro_list)
    fpr_arr = np.array(fpr_list)
    
    valid_idx = fpr_arr < 0.3
    if not np.any(valid_idx):
        return 0.0
        
    fpr_valid = fpr_arr[valid_idx]
    pro_valid = pro_arr[valid_idx]
    
    if fpr_valid.max() > 0:
        fpr_valid = fpr_valid / fpr_valid.max()
    else:
        return 0.0
        
    sort_idx = np.argsort(fpr_valid)
    return float(auc(fpr_valid[sort_idx], pro_valid[sort_idx]))


def compute_best_f1(labels, scores):
    """Find threshold that maximizes F1 score. Returns (best_f1, threshold)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return float(f1_scores[best_idx]), float(thresholds[min(best_idx, len(thresholds) - 1)])


def gaussian_smooth(score_map, sigma=4.0):
    out = np.empty_like(score_map)
    for b in range(score_map.shape[0]):
        out[b] = gaussian_filter(score_map[b], sigma=sigma)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Model Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def find_se_checkpoint(class_name):
    """Find the SEDifferNet checkpoint .pt file for a given class."""
    class_dir = os.path.join(SE_MODEL_DIR, class_name)
    if not os.path.isdir(class_dir):
        return None
    for f in os.listdir(class_dir):
        if f.endswith('.pt'):
            return os.path.join(class_dir, f)
    return None


def find_nf_head_checkpoint(class_name):
    """Find the best CFLOW NF Head checkpoint for a given class."""
    best_path = os.path.join(NF_HEAD_DIR, class_name, 'best_models', 'best_pixel_auroc.pt')
    if os.path.exists(best_path):
        return best_path
    # Fallback to best_image_auroc
    alt_path = os.path.join(NF_HEAD_DIR, class_name, 'best_models', 'best_image_auroc.pt')
    if os.path.exists(alt_path):
        return alt_path
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def denormalize(tensor):
    """Denormalize image tensor to [0,1] for display."""
    mean = np.array(c.norm_mean)
    std = np.array(c.norm_std)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)


def save_error_image(img_tensor, mask_np, anomaly_map, img_idx, label, predicted_label,
                     img_score, pix_score_max, output_dir):
    """Save misclassified image with anomaly map and GT mask side by side."""
    error_type = "FN" if label == 1 and predicted_label == 0 else "FP"

    img_np = denormalize(img_tensor)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Imagem Original', fontsize=12)
    axes[0].axis('off')

    # Anomaly map overlay
    amap_norm = anomaly_map.copy()
    amin, amax = amap_norm.min(), amap_norm.max()
    if amax - amin > 1e-8:
        amap_norm = (amap_norm - amin) / (amax - amin)
    axes[1].imshow(img_np)
    axes[1].imshow(amap_norm, cmap='hot', alpha=0.55)
    axes[1].set_title('Mapa de Anomalia (CFLOW)', fontsize=12)
    axes[1].axis('off')

    # GT mask
    axes[2].imshow(mask_np, cmap='gray')
    axes[2].set_title('Ground Truth', fontsize=12)
    axes[2].axis('off')

    fig.suptitle(
        f'{error_type} | Idx: {img_idx} | Image Score: {img_score:.4f} | '
        f'Pixel Max: {pix_score_max:.4f} | Label: {"Anomalia" if label else "Normal"}',
        fontsize=13, fontweight='bold',
        color='red' if error_type == 'FN' else 'orange'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"{error_type}_{img_idx:04d}_imgscore_{img_score:.3f}_pixscore_{pix_score_max:.3f}.png"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Class Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_score_histogram(normal_scores, anomaly_scores, threshold, title, save_path):
    """Histogram of image-level anomaly scores, normal vs anomaly."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normal_scores, bins=40, alpha=0.6, label='Normal', color='#2ecc71', density=True)
    ax.hist(anomaly_scores, bins=40, alpha=0.6, label='Anomalia', color='#e74c3c', density=True)
    if threshold is not None:
        ax.axvline(threshold, color='#2c3e50', linestyle='--', linewidth=2,
                   label=f'Threshold (F1 ótimo) = {threshold:.3f}')
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pixel_histogram(normal_pixels, anomaly_pixels, title, save_path):
    """Histogram of pixel-level scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    n_max = 200000
    if len(normal_pixels) > n_max:
        normal_pixels = np.random.choice(normal_pixels, n_max, replace=False)
    if len(anomaly_pixels) > n_max:
        anomaly_pixels = np.random.choice(anomaly_pixels, n_max, replace=False)
    ax.hist(normal_pixels, bins=80, alpha=0.6, label='Pixels Normais', color='#2ecc71', density=True)
    ax.hist(anomaly_pixels, bins=80, alpha=0.6, label='Pixels Anômalos', color='#e74c3c', density=True)
    ax.set_xlabel('Pixel Anomaly Score', fontsize=12)
    ax.set_ylabel('Densidade', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(labels, scores, auroc_val, title, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUROC = {auroc_val:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.15, color='blue')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_precision_recall(labels_img, scores_img, labels_pix, scores_pix, title, save_path):
    """Precision-Recall curves for image and pixel levels."""
    fig, ax = plt.subplots(figsize=(9, 7))

    if len(np.unique(labels_img)) > 1:
        prec_i, rec_i, _ = precision_recall_curve(labels_img, scores_img)
        ap_i = average_precision_score(labels_img, scores_img)
        ax.plot(rec_i, prec_i, 'b-', linewidth=2, label=f'Image-level (AP={ap_i:.4f})')

    if len(np.unique(labels_pix)) > 1:
        prec_p, rec_p, _ = precision_recall_curve(labels_pix, scores_pix)
        ap_p = average_precision_score(labels_pix, scores_pix)
        ax.plot(rec_p, prec_p, 'r-', linewidth=2, label=f'Pixel-level (AP={ap_p:.4f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_anomaly_map_examples(images, masks, score_maps, labels, save_path, n_samples=8):
    """Grid showing anomaly map examples (prioritize anomalous images)."""
    # Separate anomaly and normal indices
    anom_idx = [i for i in range(len(labels)) if labels[i] > 0]
    norm_idx = [i for i in range(len(labels)) if labels[i] == 0]

    # Show mostly anomalous, some normal
    n_anom = min(len(anom_idx), max(n_samples - 2, n_samples))
    n_norm = min(len(norm_idx), n_samples - n_anom)
    show_idx = anom_idx[:n_anom] + norm_idx[:n_norm]
    show_idx = show_idx[:n_samples]

    if not show_idx:
        return

    n = len(show_idx)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    mean_t = torch.tensor(c.norm_mean).view(3, 1, 1)
    std_t = torch.tensor(c.norm_std).view(3, 1, 1)

    for row, idx in enumerate(show_idx):
        img = images[idx].cpu() * std_t + mean_t
        img_np = img.permute(1, 2, 0).numpy().clip(0, 1)

        mask = masks[idx].squeeze() if masks[idx] is not None else np.zeros(img_np.shape[:2])
        smap = score_maps[idx]
        smap_norm = (smap - smap.min()) / (smap.max() - smap.min() + 1e-8)

        lbl = "Anomalia" if labels[idx] > 0 else "Normal"

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'Input ({lbl})', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title('Ground Truth', fontsize=10)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(smap_norm, cmap='hot')
        axes[row, 2].set_title('Anomaly Map', fontsize=10)
        axes[row, 2].axis('off')

        axes[row, 3].imshow(img_np)
        axes[row, 3].imshow(smap_norm, cmap='hot', alpha=0.5)
        axes[row, 3].set_title('Overlay', fontsize=10)
        axes[row, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregated Cross-Class Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_metrics_bar_chart(df, save_path):
    """Grouped bar chart of metrics per class."""
    metric_cols = [c for c in ['image_auroc', 'image_ap', 'image_f1',
                                'pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro']
                   if c in df.columns]
    if not metric_cols:
        return

    df_plot = df.melt(id_vars=['class'], value_vars=metric_cols,
                      var_name='Métrica', value_name='Valor')

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=df_plot, x='Métrica', y='Valor', hue='class', ax=ax)
    ax.set_title('Métricas por Classe', fontsize=16, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13)
    ax.set_xlabel('')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Classe', fontsize=9, title_fontsize=10,
              bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_radar_chart(df, save_path):
    """Radar/spider chart per class."""
    metric_cols = [c for c in ['image_auroc', 'image_ap', 'image_f1',
                                'pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro']
                   if c in df.columns]
    if len(metric_cols) < 3:
        return

    labels = metric_cols
    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row[m] if pd.notna(row.get(m)) else 0 for m in labels]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['class'], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title('Perfil de Performance por Classe', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap(df, save_path):
    """Heatmap of metrics per class."""
    metric_cols = [c for c in ['image_auroc', 'image_ap', 'image_f1',
                                'pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro',
                                'fps_image', 'fps_pixel']
                   if c in df.columns]
    if not metric_cols:
        return

    df_hm = df.set_index('class')[metric_cols].astype(float)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df_hm, annot=True, fmt='.3f', cmap='RdYlGn', linewidths=0.5,
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Score'})
    ax.set_title('Heatmap de Métricas por Classe', fontsize=14, fontweight='bold')
    ax.set_ylabel('')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_boxplot_scores(all_class_scores, save_path):
    """Box plot of image-level scores per class."""
    data = []
    for cls_name, scores_dict in all_class_scores.items():
        for s in scores_dict.get('normal', []):
            data.append({'Classe': cls_name, 'Score': s, 'Tipo': 'Normal'})
        for s in scores_dict.get('anomaly', []):
            data.append({'Classe': cls_name, 'Score': s, 'Tipo': 'Anomalia'})

    if not data:
        return

    df_box = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=df_box, x='Classe', y='Score', hue='Tipo',
                palette={'Normal': '#2ecc71', 'Anomalia': '#e74c3c'}, ax=ax)
    ax.set_title('Distribuição de Scores Image-Level por Classe', fontsize=14, fontweight='bold')
    ax.set_ylabel('Anomaly Score', fontsize=12)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fps_comparison(df, save_path):
    """Bar chart comparing FPS across classes."""
    fps_cols = [c for c in ['fps_image', 'fps_pixel'] if c in df.columns]
    if not fps_cols:
        return

    df_fps = df.melt(id_vars=['class'], value_vars=fps_cols,
                     var_name='Pipeline', value_name='FPS')
    df_fps['Pipeline'] = df_fps['Pipeline'].map({
        'fps_image': 'Image-Level',
        'fps_pixel': 'Pixel-Level'
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_fps, x='class', y='FPS', hue='Pipeline', ax=ax,
                palette=['#3498db', '#e67e22'])
    ax.set_title('Velocidade de Inferência (FPS) por Classe', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frames por Segundo (FPS)', fontsize=12)
    ax.set_xlabel('')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation for One Class
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_class(class_name, args, class_output_dir):
    """Run full evaluation for a single class. Returns metrics dict."""
    print(f"\n{'='*70}")
    print(f"  Avaliando: {class_name}")
    print(f"{'='*70}")

    # ── Find model files ──────────────────────────────────────────────────
    se_ckpt_path = find_se_checkpoint(class_name)
    nf_ckpt_path = find_nf_head_checkpoint(class_name)

    if se_ckpt_path is None:
        print(f"  ERRO: SEDifferNet checkpoint não encontrado para '{class_name}'")
        return None
    if nf_ckpt_path is None:
        print(f"  ERRO: NF Head checkpoint não encontrado para '{class_name}'")
        return None

    print(f"  SE model:  {os.path.basename(se_ckpt_path)}")
    print(f"  NF Head:   {os.path.basename(nf_ckpt_path)}")

    # ── Load SEDifferNet ──────────────────────────────────────────────────
    model = SEDifferNet()
    model, _ = load_weights(model, se_ckpt_path)
    model.to(DEVICE).eval()

    # ── Build backbone + load CFLOW ───────────────────────────────────────
    out_size = max(1, c.img_size[0] // 8)
    backbone = SEBackboneFeatureExtractor(model, out_size=out_size).to(DEVICE).eval()

    cflow = CFlowPixelHead(
        layer_channels=backbone.layer_channels,
        cond_dim=args.cflow_cond_dim,
        n_blocks=args.cflow_n_blocks,
        hidden=args.cflow_hidden,
    ).to(DEVICE)

    nf_data = torch.load(nf_ckpt_path, map_location=DEVICE, weights_only=False)
    if 'cflow_state_dict' in nf_data:
        cflow.load_state_dict(nf_data['cflow_state_dict'])
    elif 'model_state_dict' in nf_data:
        cflow.load_state_dict(nf_data['model_state_dict'])
    else:
        print(f"  AVISO: formato de checkpoint NF Head desconhecido. Keys: {list(nf_data.keys())}")
        return None
    cflow.eval()
    print(f"  CFLOW NF Head carregado com sucesso.")

    # ── Load dataset ──────────────────────────────────────────────────────
    # Override config for this class
    original_class = c.class_name
    c.class_name = class_name

    # For pixel-level: 1 transform, no rotation
    original_n_transforms = c.n_transforms_test
    original_rotations = c.transf_rotations
    c.n_transforms_test = 1
    c.transf_rotations = False

    trainset, testset = load_datasets(c.dataset_path, class_name, aligned=True)
    testset.n_transforms = 1
    testset.get_fixed = False

    # Apply deterministic transform (no rotations)
    from torchvision import transforms
    eval_transform = transforms.Compose([
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ])
    testset.transform = eval_transform

    if args.limit and args.limit > 0:
        from torch.utils.data import Subset
        
        normal_idx = [i for i, (_, label, _) in enumerate(testset.samples) if label == 0]
        anomaly_idx = [i for i, (_, label, _) in enumerate(testset.samples) if label > 0]
        
        if args.limit < len(anomaly_idx):
            n_anomaly = args.limit // 2
            n_normal = args.limit - n_anomaly
            
            n_anomaly = min(n_anomaly, len(anomaly_idx))
            n_normal = min(n_normal, len(normal_idx))
            
            chosen_anomaly = np.random.choice(anomaly_idx, n_anomaly, replace=False).tolist()
            chosen_normal = np.random.choice(normal_idx, n_normal, replace=False).tolist()
            
            indices = chosen_anomaly + chosen_normal
            np.random.shuffle(indices)
        else:
            indices = list(range(min(args.limit, len(testset))))
            
        testset_eval = Subset(testset, indices)
    else:
        testset_eval = testset

    test_loader = DataLoader(testset_eval, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)

    n_test = len(testset_eval)
    print(f"  Dataset: {n_test} imagens de teste")

    # ── Evaluate ──────────────────────────────────────────────────────────
    all_image_scores = []
    all_image_labels = []
    all_pixel_scores_list = []
    all_pixel_labels_list = []
    all_images_for_viz = []
    all_masks_for_viz = []
    all_score_maps = []
    all_labels_for_viz = []

    # Speed tracking
    times_image = []
    times_pixel = []

    # Warm-up
    print("  Warm-up (5 iterações)...")
    warmup_done = 0
    with torch.no_grad():
        for data in test_loader:
            if warmup_done >= 5:
                break
            images, labels, masks = data
            images_flat = images.to(DEVICE).view(-1, *images.shape[-3:])
            _ = model(images_flat)
            _, feats = backbone(images_flat)
            _ = cflow.score_map(feats, c.img_size[0])
            warmup_done += 1
    torch.cuda.synchronize() if DEVICE == 'cuda' else None

    print("  Executando avaliação...")
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"  [{class_name}]"):
            images, labels, masks = data
            images_flat = images.to(DEVICE).view(-1, *images.shape[-3:])
            label_val = int(labels[0].item())
            is_anomaly = 1 if label_val > 0 else 0

            # ── Image-level score (via SEDifferNet NF head) ───────────
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            z = model(images_flat)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_image.append(t1 - t0)

            img_score = float(torch.mean(z ** 2).item())
            all_image_scores.append(img_score)
            all_image_labels.append(is_anomaly)

            # ── Pixel-level score (via CFLOW) ─────────────────────────
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, feats = backbone(images_flat)
            smap = cflow.score_map(feats, c.img_size[0])
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_pixel.append(t1 - t0)

            smap_np = gaussian_smooth(t2np(smap), sigma=args.sigma)

            mask_np = masks.squeeze(0).squeeze(0).numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_binary = (mask_np > 0.5).astype(np.uint8)

            all_pixel_scores_list.append(smap_np.flatten())
            all_pixel_labels_list.append(mask_binary.flatten())

            # Store for viz
            all_images_for_viz.append(images_flat[0].cpu())
            all_masks_for_viz.append(mask_np)
            all_score_maps.append(smap_np[0] if smap_np.ndim == 3 else smap_np)
            all_labels_for_viz.append(label_val)

    # ── Compute Metrics ───────────────────────────────────────────────────
    image_labels_np = np.array(all_image_labels)
    image_scores_np = np.array(all_image_scores)

    pixel_scores_flat = np.concatenate(all_pixel_scores_list)
    pixel_labels_flat = np.concatenate(all_pixel_labels_list).astype(np.uint8)

    metrics = {'class': class_name}

    # Image-level
    if len(np.unique(image_labels_np)) > 1:
        metrics['image_auroc'] = float(roc_auc_score(image_labels_np, image_scores_np))
        metrics['image_ap'] = float(average_precision_score(image_labels_np, image_scores_np))
        img_f1, img_thresh = compute_best_f1(image_labels_np, image_scores_np)
        metrics['image_f1'] = img_f1
        metrics['image_threshold'] = img_thresh
    else:
        metrics['image_auroc'] = float('nan')
        metrics['image_ap'] = float('nan')
        metrics['image_f1'] = float('nan')
        img_thresh = None

    # Pixel-level
    if pixel_labels_flat.sum() > 0 and (pixel_labels_flat == 0).sum() > 0:
        metrics['pixel_auroc'] = float(roc_auc_score(pixel_labels_flat, pixel_scores_flat))
        metrics['pixel_ap'] = float(average_precision_score(pixel_labels_flat, pixel_scores_flat))
        pix_f1, pix_thresh = compute_best_f1(pixel_labels_flat, pixel_scores_flat)
        metrics['pixel_f1'] = pix_f1

        # AUPRO (RD++ style: per-image average on anomalous images only)
        aupro_list = []
        for i in range(len(all_masks_for_viz)):
            # Only calculate AUPRO if the image is actually anomalous
            if all_labels_for_viz[i] > 0:
                mask_np = all_masks_for_viz[i]
                mask_bin = (mask_np > 0.5).astype(np.uint8)
                amap_np = all_score_maps[i]
                img_aupro = compute_pro_rd_style(mask_bin, amap_np)
                aupro_list.append(img_aupro)
        
        metrics['aupro'] = float(np.mean(aupro_list)) if aupro_list else 0.0
    else:
        metrics['pixel_auroc'] = float('nan')
        metrics['pixel_ap'] = float('nan')
        metrics['pixel_f1'] = float('nan')
        metrics['aupro'] = float('nan')
        pix_thresh = None

    # Speed
    avg_time_img = np.mean(times_image) if times_image else 0.0
    avg_time_pix = np.mean(times_pixel) if times_pixel else 0.0
    metrics['latency_image_ms'] = avg_time_img * 1000
    metrics['latency_pixel_ms'] = avg_time_pix * 1000
    metrics['fps_image'] = 1.0 / avg_time_img if avg_time_img > 0 else 0.0
    metrics['fps_pixel'] = 1.0 / avg_time_pix if avg_time_pix > 0 else 0.0

    # ── Print metrics ─────────────────────────────────────────────────────
    print(f"\n  Resultados para {class_name}:")
    print(f"    Image AUROC:    {metrics['image_auroc']:.4f}")
    print(f"    Image AP:       {metrics['image_ap']:.4f}")
    print(f"    Image F1:       {metrics['image_f1']:.4f}")
    print(f"    Pixel AUROC:    {metrics['pixel_auroc']:.4f}")
    print(f"    Pixel AP:       {metrics['pixel_ap']:.4f}")
    print(f"    Pixel F1:       {metrics['pixel_f1']:.4f}")
    print(f"    AUPRO:          {metrics['aupro']:.4f}")
    print(f"    FPS (image):    {metrics['fps_image']:.1f}")
    print(f"    FPS (pixel):    {metrics['fps_pixel']:.1f}")
    print(f"    Latência img:   {metrics['latency_image_ms']:.1f} ms")
    print(f"    Latência pix:   {metrics['latency_pixel_ms']:.1f} ms")

    # ── Save error images ─────────────────────────────────────────────────
    errors_dir = os.path.join(class_output_dir, 'erros')
    n_errors = 0

    if img_thresh is not None:
        for i in range(len(all_image_labels)):
            true_label = all_image_labels[i]
            pred_label = 1 if all_image_scores[i] >= img_thresh else 0
            if pred_label != true_label:
                save_error_image(
                    img_tensor=all_images_for_viz[i],
                    mask_np=all_masks_for_viz[i],
                    anomaly_map=all_score_maps[i],
                    img_idx=i,
                    label=true_label,
                    predicted_label=pred_label,
                    img_score=all_image_scores[i],
                    pix_score_max=float(all_score_maps[i].max()),
                    output_dir=errors_dir,
                )
                n_errors += 1
    print(f"    Imagens com erro salvas: {n_errors}")

    # ── Per-class plots ───────────────────────────────────────────────────
    print("  Gerando gráficos...")

    # Score separation for histograms and boxplot
    normal_img_scores = [all_image_scores[i] for i in range(len(all_image_labels)) if all_image_labels[i] == 0]
    anomaly_img_scores = [all_image_scores[i] for i in range(len(all_image_labels)) if all_image_labels[i] == 1]

    # Histogram image-level
    if normal_img_scores and anomaly_img_scores:
        plot_score_histogram(
            normal_img_scores, anomaly_img_scores, img_thresh,
            f'Distribuição de Scores Image-Level — {class_name}',
            os.path.join(class_output_dir, 'histograma_scores_image.png')
        )

    # Histogram pixel-level
    if pixel_labels_flat.sum() > 0:
        normal_pix = pixel_scores_flat[pixel_labels_flat == 0]
        anomaly_pix = pixel_scores_flat[pixel_labels_flat == 1]
        if len(normal_pix) > 0 and len(anomaly_pix) > 0:
            plot_pixel_histogram(
                normal_pix, anomaly_pix,
                f'Distribuição de Scores Pixel-Level — {class_name}',
                os.path.join(class_output_dir, 'histograma_scores_pixel.png')
            )

    # ROC image
    if len(np.unique(image_labels_np)) > 1:
        plot_roc_curve(
            image_labels_np, image_scores_np, metrics['image_auroc'],
            f'ROC Curve Image-Level — {class_name}',
            os.path.join(class_output_dir, 'roc_image.png')
        )

    # ROC pixel
    if pixel_labels_flat.sum() > 0 and (pixel_labels_flat == 0).sum() > 0:
        # Subsample for speed
        n_sample = min(500000, len(pixel_scores_flat))
        if n_sample < len(pixel_scores_flat):
            idx_sub = np.random.choice(len(pixel_scores_flat), n_sample, replace=False)
            pix_s_sub = pixel_scores_flat[idx_sub]
            pix_l_sub = pixel_labels_flat[idx_sub]
        else:
            pix_s_sub = pixel_scores_flat
            pix_l_sub = pixel_labels_flat

        plot_roc_curve(
            pix_l_sub, pix_s_sub, metrics['pixel_auroc'],
            f'ROC Curve Pixel-Level — {class_name}',
            os.path.join(class_output_dir, 'roc_pixel.png')
        )

    # Precision-Recall
    if len(np.unique(image_labels_np)) > 1:
        pix_s_sub_pr = pix_s_sub if pixel_labels_flat.sum() > 0 else np.array([])
        pix_l_sub_pr = pix_l_sub if pixel_labels_flat.sum() > 0 else np.array([])
        plot_precision_recall(
            image_labels_np, image_scores_np,
            pix_l_sub_pr, pix_s_sub_pr,
            f'Precision-Recall — {class_name}',
            os.path.join(class_output_dir, 'precision_recall.png')
        )

    # Anomaly map examples
    if all_images_for_viz:
        plot_anomaly_map_examples(
            all_images_for_viz, all_masks_for_viz, all_score_maps,
            all_labels_for_viz,
            os.path.join(class_output_dir, 'exemplos_anomaly_maps.png'),
            n_samples=min(8, len(all_images_for_viz))
        )

    # ── Cleanup ───────────────────────────────────────────────────────────
    c.class_name = original_class
    c.n_transforms_test = original_n_transforms
    c.transf_rotations = original_rotations

    del model, backbone, cflow
    del all_images_for_viz, all_masks_for_viz, all_score_maps
    gc.collect()
    torch.cuda.empty_cache()

    # Return scores for cross-class boxplot
    class_scores = {
        'normal': normal_img_scores,
        'anomaly': anomaly_img_scores,
    }

    return metrics, class_scores


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Avaliação final SEDifferNet + CFLOW NF Head")

    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class names. Default: all 5 classes.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test images per class (None = all)')
    parser.add_argument('--sigma', type=float, default=4.0,
                        help='Gaussian smoothing sigma for pixel maps')
    parser.add_argument('--cflow_cond_dim', type=int, default=64)
    parser.add_argument('--cflow_n_blocks', type=int, default=6)
    parser.add_argument('--cflow_hidden', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Parse classes
    if args.classes:
        class_list = [cl.strip() for cl in args.classes.split(',') if cl.strip()]
    else:
        class_list = ALL_CLASSES

    # Create run directory
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(os.path.dirname(__file__), 'resultado_analise_final')
    run_dir = os.path.join(base_dir, f'run_{run_timestamp}')
    agg_dir = os.path.join(run_dir, 'graficos_agregados')
    os.makedirs(agg_dir, exist_ok=True)

    print("=" * 70)
    print("  AVALIAÇÃO FINAL — SEDifferNet + CFLOW NF Head")
    print("=" * 70)
    print(f"  Device:     {DEVICE}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Classes:    {class_list}")
    print(f"  Limit:      {args.limit if args.limit else 'Todas'}")
    print(f"  Sigma:      {args.sigma}")
    print(f"  Output:     {run_dir}")
    print("=" * 70)

    # Save config
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")
        f.write(f"device={DEVICE}\n")
        f.write(f"img_size={c.img_size}\n")
        f.write(f"n_scales={c.n_scales}\n")
        f.write(f"timestamp={run_timestamp}\n")

    # ── Evaluate each class ───────────────────────────────────────────────
    all_results = []
    all_class_scores = {}

    for class_name in class_list:
        class_output_dir = os.path.join(run_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        try:
            result = evaluate_class(class_name, args, class_output_dir)
            if result is not None:
                metrics, class_scores = result
                all_results.append(metrics)
                all_class_scores[class_name] = class_scores
            else:
                print(f"  Classe '{class_name}' retornou None — pulando.")
        except Exception as e:
            print(f"  ERRO ao avaliar '{class_name}': {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("\nNenhum resultado obtido. Verifique os modelos e o dataset.")
        return

    # ── Build results DataFrame ───────────────────────────────────────────
    df = pd.DataFrame(all_results)

    # Add mean row
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    mean_row = {'class': 'MÉDIA'}
    for col in numeric_cols:
        vals = df[col].dropna()
        mean_row[col] = float(vals.mean()) if len(vals) > 0 else float('nan')
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # Save CSVs
    csv_per_class = os.path.join(run_dir, 'results_per_class.csv')
    csv_summary = os.path.join(run_dir, 'results_summary.csv')

    # Column order
    col_order = ['class', 'image_auroc', 'image_ap', 'image_f1',
                 'pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro',
                 'fps_image', 'fps_pixel', 'latency_image_ms', 'latency_pixel_ms',
                 'image_threshold']
    col_order = [c for c in col_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in col_order]
    df_ordered = df[col_order + extra_cols]

    df_ordered.to_csv(csv_per_class, index=False, float_format='%.6f')
    df_ordered[df_ordered['class'] == 'MÉDIA'].to_csv(csv_summary, index=False, float_format='%.6f')

    # ── Print final table ─────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("  RESULTADOS FINAIS")
    print("=" * 100)

    display_cols = ['class', 'image_auroc', 'image_ap', 'image_f1',
                    'pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro',
                    'fps_image', 'fps_pixel']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df_ordered[display_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    print("\n--- Legenda ---")
    print("  image_auroc (+)  |  image_ap (+)  |  image_f1 (+)")
    print("  pixel_auroc (+)  |  pixel_ap (+)  |  pixel_f1 (+)  |  aupro (+)")
    print("  fps_image (+)    |  fps_pixel (+)")
    print("=" * 100)

    # ── Aggregated plots ──────────────────────────────────────────────────
    print("\nGerando gráficos agregados...")
    df_classes = df[df['class'] != 'MÉDIA'].copy()

    if len(df_classes) > 1:
        plot_metrics_bar_chart(df_classes, os.path.join(agg_dir, 'metricas_por_classe.png'))
        plot_radar_chart(df_classes, os.path.join(agg_dir, 'radar_performance.png'))
        plot_heatmap(df_classes, os.path.join(agg_dir, 'heatmap_metricas.png'))
        plot_fps_comparison(df_classes, os.path.join(agg_dir, 'comparativo_fps.png'))

    if all_class_scores:
        plot_boxplot_scores(all_class_scores, os.path.join(agg_dir, 'boxplot_scores.png'))

    print(f"\nResultados salvos em: {run_dir}")
    print(f"  CSV por classe:   {csv_per_class}")
    print(f"  CSV resumo:       {csv_summary}")
    print(f"  Gráficos:         {agg_dir}")
    print("Avaliação completa!")


if __name__ == '__main__':
    main()
