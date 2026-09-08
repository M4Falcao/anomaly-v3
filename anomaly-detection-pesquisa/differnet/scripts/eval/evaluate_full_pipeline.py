"""Full final evaluation: SEDifferNet (image-level) + CFLOW NF head (pixel-level).

Combines the final models of both pipelines and evaluates the InSPLAD-Seg
dataset with comprehensive metrics, plots and dumps of the misclassified
images.

Usage:
    python scripts/eval/evaluate_full_pipeline.py
    python scripts/eval/evaluate_full_pipeline.py --limit 10
    python scripts/eval/evaluate_full_pipeline.py --classes "glass-insulator,vari-grip"
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

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
from core.model import SEDifferNet, load_weights
from core.utils import load_datasets, make_dataloaders, t2np, AlignedTestDataset
from core.paths import project_path
from core.cflow import (
    DEFAULT_ATTENTION,
    LEGACY_EVAL_CLAMP_SCALE,
    SCORE_MODES,
    TRAIN_CLAMP_SCALE,
    TRAIN_OUT_SIZE,
    TTA_MODES,
    CFlowPixelHead,
    SEBackboneFeatureExtractor,
    border_mask,
    compute_level_stats,
    gaussian_smooth,
    infer_n_blocks,
    nll_with_tta,
    read_cflow_hparams,
    resolve_hparam,
    select_levels,
)

# Re-exported: ablation_nf_head.py and evaluate_cflow_no_smoothing.py import
# these from this module.
from core.eval_pipeline import (
    ALL_CLASSES,
    DEVICE,
    NF_HEAD_DIR,
    SE_MODEL_DIR,
    binary_metrics,
    compute_best_f1,
    compute_pro_rd_style,
    denormalize,
    find_nf_head_checkpoint,
    find_se_checkpoint,
    resolve_score_levels,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
# DEVICE, ALL_CLASSES, SE_MODEL_DIR and NF_HEAD_DIR are imported from
# core.eval_pipeline above and re-exported here for the dependent scripts.


# ═══════════════════════════════════════════════════════════════════════════════
# Backbone, CFLOW head, metrics and checkpoint discovery live in core.cflow and
# core.eval_pipeline, so the image-only, pixel-only and full-pipeline scripts
# all measure the same thing.
# ═══════════════════════════════════════════════════════════════════════════════


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
    nf_ckpt_path = find_nf_head_checkpoint(class_name, base_dir=args.nf_dir)

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
    nf_data = torch.load(nf_ckpt_path, map_location=DEVICE, weights_only=False)
    if 'cflow_state_dict' in nf_data:
        cflow_sd = nf_data['cflow_state_dict']
    elif 'model_state_dict' in nf_data:
        cflow_sd = nf_data['model_state_dict']
    else:
        print(f"  AVISO: formato de checkpoint NF Head desconhecido. Keys: {list(nf_data.keys())}")
        return None

    hp = read_cflow_hparams(nf_data)
    out_size, src_out = resolve_hparam('out_size', args.out_size, hp, TRAIN_OUT_SIZE)
    clamp, src_clamp = resolve_hparam('clamp_scale', args.clamp_scale, hp, TRAIN_CLAMP_SCALE)
    n_blocks, _ = resolve_hparam('n_blocks', args.cflow_n_blocks, hp,
                                 infer_n_blocks(cflow_sd), warn=False)
    attention = args.attention

    # Recipe hyperparameters stored by the training script; legacy heads have none.
    trained_levels = list(hp.get('levels') or [0, 1, 2])
    feat_pool = int(hp.get('feat_pool') or 0)
    pad_mode = hp.get('pad_mode') or 'reflect'
    feat_stats = nf_data.get('feat_stats')
    channels = select_levels([64, 192, 256], trained_levels)
    try:
        score_idx, score_levels = resolve_score_levels(
            args.levels, trained_levels, class_name)
    except ValueError as err:
        print(f"  ERRO: {err}")
        return None

    if feat_stats is None:
        norm_desc = 'OFF'
    elif any(st is None for st in feat_stats):
        norm_lvls = [LEVEL_NAMES[l] for l, st in zip(trained_levels, feat_stats) if st is not None]
        norm_desc = f"selective({','.join(norm_lvls)})"
    else:
        norm_desc = 'ON'

    print(f"  out_size={out_size} ({src_out}) | clamp_scale={clamp} ({src_clamp}) | "
          f"n_blocks={n_blocks} | attention={attention} | score={args.score_norm}")
    print(f"  levels treinados={trained_levels} -> usados={score_levels} | "
          f"feat_norm={norm_desc} | feat_pool={feat_pool} | "
          f"pad={args.reflect_pad} ({pad_mode}) | tta={args.tta} | sigma={args.sigma}")

    backbone = SEBackboneFeatureExtractor(
        model, out_size=out_size, reflect_pad=args.reflect_pad,
        attention=attention, pad_mode=pad_mode, feat_pool=feat_pool).to(DEVICE).eval()

    cflow = CFlowPixelHead(
        layer_channels=channels,
        cond_dim=args.cflow_cond_dim,
        n_blocks=n_blocks,
        hidden=args.cflow_hidden,
        clamp_scale=clamp,
    ).to(DEVICE)
    cflow.load_state_dict(cflow_sd)
    cflow.eval()
    print(f"  CFLOW NF Head carregado com sucesso.")

    per_channel = args.score_norm == 'per_level_minmax'
    # One-element list so the closure below sees the stats filled in later, once
    # the training loader exists (only 'per_level_std' needs them).
    level_stats_ref = [None]

    def pixel_score_map(images_flat):
        """Anomaly map for one batch under the configured levels, TTA and aggregation."""
        nll = nll_with_tta(backbone, cflow, images_flat, levels=trained_levels,
                           feat_stats=feat_stats, tta=args.tta, per_channel=per_channel)
        return CFlowPixelHead.aggregate_nll(
            nll, channels, c.img_size[0], mode=args.score_norm,
            level_stats=level_stats_ref[0], levels=score_idx)

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
    keep_mask = None
    if args.score_norm == 'per_level_std':
        stats_loader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)
        level_stats_ref[0] = compute_level_stats(
            backbone, cflow, stats_loader, DEVICE, levels=trained_levels,
            feat_stats=feat_stats, desc=f'  [{class_name}] level stats')
        print(f"  Level stats (mean, std): "
              + ", ".join(f"({m:.1f}, {s:.1f})" for m, s in level_stats_ref[0]))

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
            _ = pixel_score_map(images_flat)
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
            smap = pixel_score_map(images_flat)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_pixel.append(t1 - t0)

            smap_np = gaussian_smooth(t2np(smap), sigma=args.sigma)

            mask_np = masks.squeeze(0).squeeze(0).numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np[0]
            mask_binary = (mask_np > 0.5).astype(np.uint8)

            if keep_mask is None:
                keep_mask = border_mask(mask_binary.shape[-1], args.border_margin)

            all_pixel_scores_list.append(smap_np[0][keep_mask].flatten()
                                         if smap_np.ndim == 3 else smap_np[keep_mask].flatten())
            all_pixel_labels_list.append(mask_binary[keep_mask].flatten())

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
    parser.add_argument('--nf_dir', type=str, default=None,
                        help='Root directory with per-class folders containing best_models/best_pixel_auroc.pt.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test images per class (None = all)')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='Gaussian smoothing sigma for pixel maps (0 disables it). '
                             'Default 8 (defeitos >= 1%% da area); use 2 para defeitos pequenos.')
    parser.add_argument('--cflow_cond_dim', type=int, default=64)
    parser.add_argument('--cflow_n_blocks', type=int, default=None,
                        help='Default: read from checkpoint, else inferred from weights')
    parser.add_argument('--cflow_hidden', type=int, default=256)
    parser.add_argument('--out_size', type=int, default=None,
                        help='Feature grid the flow is conditioned on. MUST match training. '
                             f'Default: read from checkpoint, else {TRAIN_OUT_SIZE} (the '
                             'training default) with a warning.')
    parser.add_argument('--clamp_scale', type=float, default=None,
                        help=f'Affine coupling clamp. Training used {TRAIN_CLAMP_SCALE}; '
                             f'evaluation historically used {LEGACY_EVAL_CLAMP_SCALE}, which '
                             f'makes the flow compute a different transform than it was fitted '
                             f'for. Default: checkpoint value, else {TRAIN_CLAMP_SCALE}.')
    parser.add_argument('--attention', type=str, default=DEFAULT_ATTENTION,
                        choices=['auto', 'se', 'cbam', 'legacy_train'],
                        help="Which attention blocks feed the CFLOW head. Defaults to 'se', "
                             "the path SEDifferNet actually trains. 'legacy_train' reproduces "
                             "the old training script, which read CBAM blocks even for SE models.")
    parser.add_argument('--score_norm', type=str, default='raw',
                        choices=list(SCORE_MODES),
                        help='Como os mapas de NLL por nivel sao agregados. Default "raw": venceu '
                             'em 12/12 heads das 3 classes estudadas (+0.044 a +0.057 sobre '
                             'per_level_minmax). Use per_level_minmax para reproduzir resultados '
                             'antigos.')
    parser.add_argument('--tta', type=str, default='flips', choices=list(TTA_MODES),
                        help='Test-time augmentation por espelhamento. Default "flips" (4 vistas): '
                             'ganho de +0.007 a +0.036 de AUROC em todas as classes, ao custo de 4x '
                             'o tempo de inferencia pixel. Use "none" para medir latencia real.')
    parser.add_argument('--levels', type=str, default='auto',
                        help='Niveis de features usados no score, por id original (0=L1, 1=L2, '
                             '2=L3). Default "auto" = L1+L3 quando a head os tem, que e o melhor '
                             'compromisso medido nas 3 classes (L2 nunca e o melhor nivel). '
                             '"per_class" usa o melhor subconjunto medido por classe '
                             '(vari-grip=0, lightning-rod=2, demais=02). '
                             'Use "all" para todos, ou ex. "2" para so L3.')
    parser.add_argument('--reflect_pad', type=int, default=112,
                        help='Reflect-pad the input by N pixels to reduce conv zero-padding '
                             'artifacts at the image border (0 = off). Default 112: melhora o '
                             'AUPRO em praticamente toda head medida. E tecnica de INFERENCIA - '
                             'treinar com pad piora.')
    parser.add_argument('--border_margin', type=int, default=0,
                        help='Exclude an N-pixel frame from pixel metrics (0 = off). Mantenha 0: '
                             'margem > 0 e confundidor de selecao (esconde falso positivo de borda '
                             'do AUROC sem melhorar o AUPRO).')
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
    base_dir = project_path('resultado_analise_final')
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
    print(f"  Receita:    score={args.score_norm} | pad={args.reflect_pad} | tta={args.tta} | "
          f"levels={args.levels} | sigma={args.sigma} | margin={args.border_margin}")
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
