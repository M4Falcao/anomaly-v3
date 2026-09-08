"""Full ablation of the CFLOW NF head (pixel-level) on one InsPLAD class.

Two stages, both driven from this one entry point:

``--stage posthoc`` (default)
    Take a trained NF head (by default ``final_models/NF Head/<class>``) and
    sweep every *post-processing* factor without retraining:

    * input padding before the frozen backbone (``--pads``, ``--pad_modes``)
    * flip test-time augmentation (``--ttas``)
    * per-level aggregation rule (``--score_modes``)
    * which feature levels are summed (``--level_sets``)
    * positional (per-cell) calibration of the deep levels (``--pos_calibs``)
    * Gaussian smoothing sigma (``--sigmas``)
    * border frame excluded from the pixel metrics (``--margins``)
    * map-to-image reduction (``--image_scores``)

    The flow is run once per (padding, TTA) pair and the per-level NLL maps
    are cached at native resolution, so the whole grid is re-aggregated from
    identical network outputs. Metrics for the grid are computed on a
    stratified *validation* subset of the test split (``--val_fraction``);
    only the top configurations (``--top_n``) are then scored once, at full
    resolution and with AUPRO, on the held-out remainder. This is the
    selection protocol requested in ``docs/cflow_correcoes_e_fase3.md``.

``--stage retrain``
    Retrain the head under named training variants (see ``TRAIN_VARIANTS``)
    through ``scripts/train/pixel_train_from_pretrained.py`` - always on top of
    the frozen SEDifferNet *final model* of the class - and run the post-hoc
    stage on every resulting ``best_pixel_auroc.pt``.

``--stage all`` runs both.

Usage:
    # Post-hoc ablation of the released vari-grip head
    python scripts/eval/ablation_nf_head.py --class_name vari-grip

    # Smoke test (small grid, few images)
    python scripts/eval/ablation_nf_head.py --class_name vari-grip --quick --limit 16

    # Retrain three variants for 40 epochs and ablate each
    python scripts/eval/ablation_nf_head.py --class_name vari-grip --stage retrain `
        --variants baseline,norot,norot_pad112 --epochs 40

Outputs (under ``resultado_analise_final/ablacao_nf_head/run_<timestamp>/``):
    grid_val.csv            every grid configuration, validation metrics
    holdout_test_top.csv    top-N (+ reference configs) on the held-out split
    per_level_auroc.csv     single-level AUROC per (padding, TTA) - diagnostic
    main_effects.png        marginal effect of each factor on val pixel AUROC
    border_profile.png      mean score vs distance to the border (normals)
    maps_comparison.png     input | GT | baseline map | best map
    summary.txt             human-readable recap
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, SCRIPT_DIR)

import argparse
import itertools
import json
import subprocess
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
from core.paths import project_path, PROJECT_ROOT
from core.cflow import (
    DEFAULT_ATTENTION,
    PAD_MODES,
    SCORE_MODES,
    TRAIN_CLAMP_SCALE,
    TRAIN_OUT_SIZE,
    TTA_MODES,
    CFlowPixelHead,
    SEBackboneFeatureExtractor,
    apply_feature_stats,
    border_mask,
    compute_positional_stats,
    image_score_from_map,
    infer_n_blocks,
    nll_with_tta,
    read_cflow_hparams,
    resolve_hparam,
    select_levels,
    snap_reflect_pad,
)
from evaluate_full_pipeline import (
    DEVICE,
    compute_best_f1,
    compute_pro_rd_style,
    find_nf_head_checkpoint,
    find_se_checkpoint,
)

LEVEL_NAMES = {0: 'L1', 1: 'L2', 2: 'L3'}
POS_CALIB_MODES = {'none': set(), 'l23': {1, 2}, 'all': {0, 1, 2}}
FACTORS = ['pad', 'pad_mode', 'tta', 'score_mode', 'levels', 'pos_calib',
           'sigma', 'margin', 'image_score']

# Reference configurations that are always carried into the held-out evaluation.
REFERENCE_CONFIGS = {
    'baseline_producao': dict(pad=0, pad_mode='reflect', tta='none',
                              score_mode='per_level_minmax', levels='012',
                              pos_calib='none', sigma=4.0, margin=0, image_score='max'),
    'producao_pad112_m8': dict(pad=112, pad_mode='reflect', tta='none',
                               score_mode='per_level_minmax', levels='012',
                               pos_calib='none', sigma=4.0, margin=8, image_score='max'),
}

# Training variants for --stage retrain. Every one trains on the frozen
# SEDifferNet final model of the class; only the pixel-head recipe changes.
TRAIN_VARIANTS = {
    'baseline':        [],
    'norot':           ['--no_rotation'],
    'pad112':          ['--reflect_pad', '112'],
    'norot_pad112':    ['--no_rotation', '--reflect_pad', '112'],
    'l23':             ['--levels', '1,2'],
    'norot_l23':       ['--no_rotation', '--levels', '1,2'],
    'pool3':           ['--feat_pool', '3'],
    'featnorm':        ['--feat_norm'],
    'clamp19':         ['--clamp_scale', '1.9'],
    'blocks4':         ['--n_blocks', '4'],
    'hidden512':       ['--hidden', '512'],
    'combo':           ['--no_rotation', '--reflect_pad', '112', '--feat_pool', '3', '--feat_norm'],
    'combo_l23':       ['--no_rotation', '--reflect_pad', '112', '--feat_pool', '3',
                        '--feat_norm', '--levels', '1,2'],
    # Round 2 (after run_20260904_092255): feat_norm and clamp 1.9 were the two recipes whose
    # raw NLL + positional calibration beat every min-max fusion; rotation stayed ON because it
    # acted as a useful augmentation (baseline kept improving until epoch 40, norot peaked at 5).
    'featnorm_pad112':         ['--feat_norm', '--reflect_pad', '112'],
    'featnorm_clamp19':        ['--feat_norm', '--clamp_scale', '1.9'],
    'featnorm_pad112_clamp19': ['--feat_norm', '--reflect_pad', '112', '--clamp_scale', '1.9'],
    'featnorm_pool3':          ['--feat_norm', '--feat_pool', '3'],
    'featnorm_l1':             ['--feat_norm', '--levels', '0'],
    'featnorm_l12':            ['--feat_norm', '--levels', '0,1'],
    # Selective standardization (feat_norm only on L1):
    # Standardizes color/texture L1 channels without equalizing noise in multi-channel L2/L3.
    'featnorm_l1_only':         ['--feat_norm_levels', '0'],
    'featnorm_l1_only_02':      ['--levels', '0,2', '--feat_norm_levels', '0'],
    'featnorm_l1_only_clamp19': ['--feat_norm_levels', '0', '--clamp_scale', '1.9'],
}


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

def build_datasets(class_name, dataset_path, limit=None, seed=42):
    """Deterministic (unrotated) train/test datasets plus test labels.

    Returns:
        ``(trainset, testset, labels)`` where ``labels`` is a per-image
        0/1 array aligned with ``testset``.
    """
    c.class_name = class_name
    c.n_transforms_test = 1
    c.transf_rotations = False  # also affects the train loader used for statistics

    trainset, testset = load_datasets(dataset_path, class_name, aligned=True)
    testset.n_transforms = 1
    testset.get_fixed = False
    testset.transform = transforms.Compose([
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ])

    labels = np.array([1 if lbl > 0 else 0 for _, lbl, _ in testset.samples])
    if limit and 0 < limit < len(testset):
        rng = np.random.default_rng(seed)
        anomaly_idx = np.flatnonzero(labels == 1)
        normal_idx = np.flatnonzero(labels == 0)
        n_anomaly = min(limit // 2, len(anomaly_idx))
        n_normal = min(limit - n_anomaly, len(normal_idx))
        indices = np.concatenate([rng.choice(anomaly_idx, n_anomaly, replace=False),
                                  rng.choice(normal_idx, n_normal, replace=False)])
        indices = np.sort(indices)
        testset = Subset(testset, indices.tolist())
        labels = labels[indices]
    return trainset, testset, labels


def stratified_split(labels, val_fraction, seed):
    """Split image indices into (val, holdout), stratified by label."""
    if val_fraction <= 0:
        all_idx = np.arange(len(labels))
        return all_idx, all_idx
    rng = np.random.default_rng(seed)
    val, hold = [], []
    for lbl in (0, 1):
        idx = np.flatnonzero(labels == lbl)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) > 1 else 0
        val.extend(idx[:n_val].tolist())
        hold.extend(idx[n_val:].tolist())
    return np.array(sorted(val)), np.array(sorted(hold))


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

def load_head(nf_ckpt_path, args):
    """Load a CFLOW head with the hyperparameters it was trained with."""
    nf_data = torch.load(nf_ckpt_path, map_location=DEVICE, weights_only=False)
    if 'cflow_state_dict' in nf_data:
        sd = nf_data['cflow_state_dict']
    elif 'model_state_dict' in nf_data:
        sd = nf_data['model_state_dict']
    else:
        raise ValueError(f"unknown NF head checkpoint layout: {list(nf_data.keys())}")

    hp = read_cflow_hparams(nf_data)
    out_size, _ = resolve_hparam('out_size', args.out_size, hp, TRAIN_OUT_SIZE)
    clamp, _ = resolve_hparam('clamp_scale', args.clamp_scale, hp, TRAIN_CLAMP_SCALE)
    n_blocks, _ = resolve_hparam('n_blocks', None, hp, infer_n_blocks(sd), warn=False)
    cond_dim, _ = resolve_hparam('cond_dim', None, hp, 64, warn=False)
    hidden, _ = resolve_hparam('hidden', None, hp, 256, warn=False)
    levels = hp.get('levels') or [0, 1, 2]
    feat_stats = nf_data.get('feat_stats')

    info = {
        'out_size': out_size, 'clamp_scale': clamp, 'n_blocks': n_blocks,
        'cond_dim': cond_dim, 'hidden': hidden, 'levels': list(levels),
        'feat_pool': int(hp.get('feat_pool') or 0),
        'pad_mode': hp.get('pad_mode') or 'reflect',
        'train_reflect_pad': int(hp.get('reflect_pad') or 0),
        'feat_norm': bool(hp.get('feat_norm')),
        'feat_norm_levels': hp.get('feat_norm_levels'),
        'no_rotation': bool(hp.get('no_rotation')),
        'attention': hp.get('attention') or DEFAULT_ATTENTION,
        'epoch': nf_data.get('epoch'),
        'train_pixel_auroc': nf_data.get('pixel_auroc'),
    }

    channels = select_levels([64, 192, 256], levels)
    cflow = CFlowPixelHead(channels, cond_dim=cond_dim, n_blocks=n_blocks,
                           hidden=hidden, clamp_scale=clamp).to(DEVICE)
    cflow.load_state_dict(sd)
    cflow.eval()
    return cflow, feat_stats, info


def build_backbone(model, info, pad, pad_mode):
    return SEBackboneFeatureExtractor(
        model, out_size=info['out_size'], reflect_pad=pad, attention=info['attention'],
        pad_mode=pad_mode, feat_pool=info['feat_pool']).to(DEVICE).eval()


# ═══════════════════════════════════════════════════════════════════════════════
# Inference with cache
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_nll(model, backbone, cflow, loader, levels, feat_stats, tta, desc,
                want_images=False, want_se_scores=False):
    """Run the frozen pipeline once and cache native-resolution NLL maps.

    Returns a dict with ``nll`` (list of ``(N, H, W)`` float32 tensors, one per
    trained level), ``masks`` ``(N, S, S)`` uint8, optional ``images``
    ``(N, S, S, 3)`` uint8, optional ``se_scores`` and ``latency_ms``.
    """
    nll_acc, masks, images_u8, se_scores, times = None, [], [], [], []
    mean_t = torch.tensor(c.norm_mean).view(3, 1, 1)
    std_t = torch.tensor(c.norm_std).view(3, 1, 1)

    for images, labels, mask in tqdm(loader, desc=desc, leave=False):
        x = images.to(DEVICE).view(-1, *images.shape[-3:])
        if want_se_scores:
            z = model(x)
            se_scores.append(float(torch.mean(z ** 2).item()))

        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        nll = nll_with_tta(backbone, cflow, x, levels, feat_stats, tta)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

        nll = [m.float().cpu() for m in nll]
        nll_acc = [[m] for m in nll] if nll_acc is None else [a + [m] for a, m in zip(nll_acc, nll)]

        m = mask.squeeze(0).squeeze(0).numpy()
        if m.ndim == 3:
            m = m[0]
        masks.append((m > 0.5).astype(np.uint8))
        if want_images:
            img = (images.view(-1, *images.shape[-3:])[0] * std_t + mean_t).clamp(0, 1)
            images_u8.append((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    out = {
        'nll': [torch.cat(level_maps, dim=0) for level_maps in nll_acc],
        'masks': np.stack(masks),
        'latency_ms': float(np.mean(times) * 1000),
    }
    if want_images:
        out['images'] = np.stack(images_u8)
    if want_se_scores:
        out['se_scores'] = np.array(se_scores)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def fast_binary_metrics(scores, labels, n_bins=65536):
    """Histogram-based AUROC / AP / best-F1 in O(N + n_bins).

    Scores are quantized to ``n_bins`` equal-width bins on their own range,
    so the quantization error is bounded by 1/n_bins of the score range.
    Ties inside a bin are resolved the same way scikit-learn resolves ties
    (trapezoid on the ROC, step on the PR curve).
    """
    s = scores.astype(np.float64, copy=False)
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-12 or labels.sum() == 0 or labels.sum() == labels.size:
        return float('nan'), float('nan'), float('nan')
    b = ((s - mn) / (mx - mn) * (n_bins - 1)).astype(np.int64)
    pos = np.bincount(b[labels == 1], minlength=n_bins).astype(np.float64)
    neg = np.bincount(b[labels == 0], minlength=n_bins).astype(np.float64)
    P, N = pos.sum(), neg.sum()
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    tpr = np.concatenate([[0.0], tp / P])
    fpr = np.concatenate([[0.0], fp / N])
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    auroc = float(trapezoid(tpr, fpr))
    prec = tp / np.maximum(tp + fp, 1.0)
    rec = tp / P
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    ap = float(np.sum((rec - rec_prev) * prec))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return auroc, ap, float(f1.max())


def pixel_metrics(maps, masks, margin, stride=1, exact=False):
    """Pixel AUROC/AP/F1 on a strided, border-masked view of the maps."""
    size = masks.shape[-1]
    keep = border_mask(size, margin)[::stride, ::stride]
    s = maps[:, ::stride, ::stride][:, keep].reshape(-1)
    y = masks[:, ::stride, ::stride][:, keep].reshape(-1).astype(np.uint8)
    if y.sum() == 0 or y.sum() == y.size:
        return {'pixel_auroc': float('nan'), 'pixel_ap': float('nan'), 'pixel_f1': float('nan')}
    if exact:
        auroc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
        _, _, f1 = fast_binary_metrics(s, y)
    else:
        auroc, ap, f1 = fast_binary_metrics(s, y)
    return {'pixel_auroc': auroc, 'pixel_ap': ap, 'pixel_f1': f1}


def aupro_rd(maps, masks, labels):
    vals = [compute_pro_rd_style(masks[i], maps[i]) for i in range(len(labels)) if labels[i] > 0]
    return float(np.mean(vals)) if vals else float('nan')


def image_auroc_from_map(maps, labels, mode, topk=1.0):
    if len(np.unique(labels)) < 2:
        return float('nan')
    scores = image_score_from_map(maps, mode=mode, top_percent=topk)
    return float(roc_auc_score(labels, scores))


def border_profile(maps, bin_px=8, max_px=224):
    """Mean per-image-normalized score as a function of distance to the border."""
    n, h, w = maps.shape
    ii, jj = np.mgrid[0:h, 0:w]
    dist = np.minimum.reduce([ii, jj, h - 1 - ii, w - 1 - jj])
    flat = maps.reshape(n, -1)
    mn = flat.min(axis=1, keepdims=True)
    mx = flat.max(axis=1, keepdims=True)
    norm = ((flat - mn) / (mx - mn + 1e-8)).reshape(n, h, w).mean(axis=0)
    centers, means = [], []
    for lo in range(0, max_px, bin_px):
        sel = (dist >= lo) & (dist < lo + bin_px)
        if sel.any():
            centers.append(lo + bin_px / 2)
            means.append(float(norm[sel].mean()))
    return np.array(centers), np.array(means)


# ═══════════════════════════════════════════════════════════════════════════════
# Aggregation from cached NLL
# ═══════════════════════════════════════════════════════════════════════════════

def parse_level_set(name, trained_levels):
    """'12' -> indices into the trained-level list for original levels L2, L3."""
    wanted = sorted({int(ch) for ch in name})
    if any(l not in trained_levels for l in wanted):
        return None
    return [trained_levels.index(l) for l in wanted]


def build_map(cache, pos_cache, cfg, trained_levels, channels, img_size):
    """Aggregate cached NLL maps for one configuration -> (N, S, S) float32."""
    nll = cache['nll']
    mode = cfg['score_mode']
    if mode == 'per_level_minmax':
        nll = [m / C for m, C in zip(nll, channels)]

    pos_stats = None
    calib = POS_CALIB_MODES[cfg['pos_calib']]
    if calib:
        pos_stats = []
        for i, lvl in enumerate(trained_levels):
            if lvl in calib:
                mu, sd = pos_cache['pos'][i]
                if mode == 'per_level_minmax':
                    mu, sd = mu / channels[i], sd / channels[i]
                pos_stats.append((mu, sd))
            else:
                pos_stats.append(None)

    level_stats = None
    if mode == 'per_level_std':
        level_stats = pos_cache['scalar']
        if calib:
            # Calibrated levels are already ~N(0,1); leave them untouched.
            level_stats = [(0.0, 1.0) if st is not None else ls
                           for st, ls in zip(pos_stats, level_stats)]

    sel = parse_level_set(cfg['levels'], trained_levels)
    smap = CFlowPixelHead.aggregate_nll(
        nll, channels, img_size, mode=mode, level_stats=level_stats,
        levels=sel, pos_stats=pos_stats)
    return smap.numpy().astype(np.float32)


def smooth(maps, sigma):
    if sigma <= 0:
        return maps
    out = np.empty_like(maps)
    for i in range(maps.shape[0]):
        out[i] = gaussian_filter(maps[i], sigma=sigma)
    return out


def scalar_from_positional(pos):
    """Scalar (mean, std) per level derived exactly from positional maps."""
    stats = []
    for mu, sd in pos:
        mean = float(mu.mean())
        second = float((sd ** 2 + mu ** 2).mean())
        stats.append((mean, float(np.sqrt(max(second - mean ** 2, 0.0)))))
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_main_effects(df, metric, save_path):
    factors = [f for f in FACTORS if df[f].nunique() > 1]
    if not factors:
        return
    fig, axes = plt.subplots(1, len(factors), figsize=(3.2 * len(factors), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, factors):
        agg = df.groupby(f)[metric].agg(['mean', 'max']).sort_index()
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg['mean'], width=0.4, label='média')
        ax.bar(x + 0.2, agg['max'], width=0.4, label='máximo')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in agg.index], rotation=30, ha='right')
        ax.set_title(f)
        ax.grid(alpha=0.3, axis='y')
    axes[0].set_ylabel(f'{metric} (val)')
    axes[0].legend(fontsize=8)
    lo = df[metric].min()
    axes[0].set_ylim(max(0.0, lo - 0.05), 1.0)
    fig.suptitle('Efeitos marginais por fator (grade completa, validação)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_border_profiles(profiles, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, (x, y) in profiles.items():
        ax.plot(x, y, marker='o', markersize=3, label=name)
    ax.set_xlabel('distância até a borda (px, imagem 448)')
    ax.set_ylabel('score normalizado médio (imagens normais)')
    ax.set_title('Perfil de borda do mapa de anomalia')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_map_comparison(images, masks, named_maps, indices, save_path):
    if len(indices) == 0:
        return
    cols = 2 + len(named_maps)
    fig, axes = plt.subplots(len(indices), cols, figsize=(3.2 * cols, 3.2 * len(indices)))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(indices):
        axes[r, 0].imshow(images[idx])
        axes[r, 0].set_title('input', fontsize=9)
        axes[r, 1].imshow(masks[idx], cmap='gray')
        axes[r, 1].set_title('ground truth', fontsize=9)
        for k, (name, maps) in enumerate(named_maps.items()):
            m = maps[idx]
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            axes[r, 2 + k].imshow(images[idx])
            axes[r, 2 + k].imshow(m, cmap='hot', alpha=0.55)
            axes[r, 2 + k].set_title(name, fontsize=9)
        for ax in axes[r]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Post-hoc stage
# ═══════════════════════════════════════════════════════════════════════════════

def cfg_name(cfg):
    return (f"pad{cfg['pad']}-{cfg['pad_mode']}|tta={cfg['tta']}|{cfg['score_mode']}|"
            f"L={cfg['levels']}|calib={cfg['pos_calib']}|s={cfg['sigma']:g}|"
            f"m={cfg['margin']}|img={cfg['image_score']}")


def run_posthoc(args, nf_ckpt_path, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    img_size = c.img_size[0]
    print(f"\n{'=' * 70}\n  POST-HOC ABLATION [{tag}]\n{'=' * 70}")
    print(f"  NF head: {nf_ckpt_path}")

    se_ckpt = args.se_checkpoint or find_se_checkpoint(args.class_name)
    if se_ckpt is None:
        raise FileNotFoundError(f"SEDifferNet final model not found for {args.class_name}")
    print(f"  SE model: {os.path.basename(se_ckpt)}")

    model = SEDifferNet()
    model, _ = load_weights(model, se_ckpt)
    model.to(DEVICE).eval()

    cflow, feat_stats, info = load_head(nf_ckpt_path, args)
    trained_levels = info['levels']
    channels = select_levels([64, 192, 256], trained_levels)
    print("  Head: " + ", ".join(f"{k}={v}" for k, v in info.items()))

    trainset, testset, labels = build_datasets(args.class_name, args.dataset, args.limit, args.seed)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    stats_loader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0)
    val_idx, hold_idx = stratified_split(labels, args.val_fraction, args.seed)
    print(f"  Test images: {len(labels)} ({labels.sum()} anomalous) | "
          f"val={len(val_idx)} holdout={len(hold_idx)}"
          + ("  [val == holdout: no split requested]" if args.val_fraction <= 0 else ""))

    # ── Grid definition ──────────────────────────────────────────────────────
    pads = sorted({int(p) for p in args.pads.split(',')} | {info['train_reflect_pad']})
    pads = [snap_reflect_pad(p, info['out_size'], img_size) if p > 0 else 0 for p in pads]
    pads = sorted(set(pads))
    pad_modes = [m for m in args.pad_modes.split(',') if m]
    ttas = [t for t in args.ttas.split(',') if t]
    score_modes = [m for m in args.score_modes.split(',') if m]
    level_sets = [l for l in args.level_sets.split(',') if l and parse_level_set(l, trained_levels) is not None]
    pos_calibs = [p for p in args.pos_calibs.split(',') if p]
    sigmas = [float(s) for s in args.sigmas.split(',') if s]
    margins = [int(m) for m in args.margins.split(',') if m]
    image_scores = [s for s in args.image_scores.split(',') if s]
    grid = {'pad': pads, 'pad_mode': pad_modes, 'tta': ttas, 'score_mode': score_modes,
            'levels': level_sets, 'pos_calib': pos_calibs, 'sigma': sigmas,
            'margin': margins, 'image_score': image_scores}
    n_total = int(np.prod([len(v) for v in grid.values()]))
    print("  Grid: " + " x ".join(f"{k}={v}" for k, v in grid.items()) + f"  -> {n_total} configs")

    # ── Inference passes (one per padding x tta) ─────────────────────────────
    caches, pos_caches = {}, {}
    se_scores, images_u8 = None, None
    per_level_rows = []

    def positional(backbone, pad, pad_mode):
        pos = compute_positional_stats(backbone, cflow, stats_loader, DEVICE, per_channel=False,
                                       feat_stats=feat_stats, levels=trained_levels, calibrate=None,
                                       desc=f'  stats pad={pad}')
        pos_caches[(pad, pad_mode)] = {'pos': pos, 'scalar': scalar_from_positional(pos)}

    for pad, pad_mode in itertools.product(pads, pad_modes):
        if pad == 0 and pad_mode != pad_modes[0]:
            continue  # padding mode is irrelevant without padding
        backbone = build_backbone(model, info, pad, pad_mode)
        positional(backbone, pad, pad_mode)
        for tta in ttas:
            first = se_scores is None
            cache = collect_nll(model, backbone, cflow, test_loader, trained_levels, feat_stats, tta,
                                desc=f'  infer pad={pad} {pad_mode} tta={tta}',
                                want_images=first, want_se_scores=first)
            if first:
                se_scores = cache.pop('se_scores')
                images_u8 = cache.pop('images')
            caches[(pad, pad_mode, tta)] = cache
            for i, lvl in enumerate(trained_levels):
                up = CFlowPixelHead.aggregate_nll([cache['nll'][i]], [channels[i]], img_size, mode='raw')
                m = pixel_metrics(up.numpy(), cache['masks'], 0, stride=args.pixel_stride)
                per_level_rows.append({'pad': pad, 'pad_mode': pad_mode, 'tta': tta,
                                       'level': LEVEL_NAMES[lvl], **m})
        del backbone
        torch.cuda.empty_cache() if DEVICE == 'cuda' else None

    masks = next(iter(caches.values()))['masks']
    pd.DataFrame(per_level_rows).to_csv(os.path.join(out_dir, 'per_level_auroc.csv'), index=False)
    se_auroc = float(roc_auc_score(labels, se_scores)) if len(np.unique(labels)) > 1 else float('nan')
    print(f"  SEDifferNet image AUROC (all {len(labels)} images): {se_auroc:.4f}")
    print("  Single-level pixel AUROC:")
    print(pd.DataFrame(per_level_rows).pivot_table(index=['pad', 'pad_mode', 'tta'], columns='level',
                                                    values='pixel_auroc').round(4).to_string())

    # ── Grid evaluation on the validation subset ─────────────────────────────
    rows = []
    pbar = tqdm(total=n_total, desc='  grid (val)')
    for pad, pad_mode, tta, score_mode, lv, calib in itertools.product(
            pads, pad_modes, ttas, score_modes, level_sets, pos_calibs):
        n_inner = len(sigmas) * len(margins) * len(image_scores)
        if pad == 0 and pad_mode != pad_modes[0]:
            pbar.update(n_inner)
            continue
        key = (pad, pad_mode, tta)
        cfg = dict(pad=pad, pad_mode=pad_mode if pad > 0 else '-', tta=tta,
                   score_mode=score_mode, levels=lv, pos_calib=calib)
        full = build_map(caches[key], pos_caches[(pad, pad_mode)], cfg, trained_levels, channels, img_size)
        maps_val, masks_val, labels_val = full[val_idx], masks[val_idx], labels[val_idx]
        for sigma in sigmas:
            sm = smooth(maps_val, sigma)
            for margin in margins:
                pm = pixel_metrics(sm, masks_val, margin, stride=args.pixel_stride)
                for img_mode in image_scores:
                    row = dict(cfg, sigma=sigma, margin=margin, image_score=img_mode, **pm)
                    row['image_auroc_map'] = image_auroc_from_map(sm, labels_val, img_mode, args.topk_percent)
                    row['latency_pixel_ms'] = caches[key]['latency_ms']
                    row['name'] = cfg_name(row)
                    rows.append(row)
                    pbar.update(1)
    pbar.close()

    df = pd.DataFrame(rows)
    df = df.sort_values(args.select_metric, ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, 'grid_val.csv'), index=False)
    plot_main_effects(df, args.select_metric, os.path.join(out_dir, 'main_effects.png'))

    print(f"\n  Top {min(10, len(df))} on validation by {args.select_metric}:")
    print(df.head(10)[FACTORS + ['pixel_auroc', 'pixel_ap', 'pixel_f1', 'image_auroc_map']]
          .round(4).to_string(index=False))

    # ── Held-out evaluation: top-N + references, full resolution + AUPRO ────
    chosen = []
    for _, r in df.head(args.top_n).iterrows():
        chosen.append(('top', {f: r[f] for f in FACTORS}))
    for name, ref in REFERENCE_CONFIGS.items():
        ref = dict(ref)
        ref['pad'] = snap_reflect_pad(ref['pad'], info['out_size'], img_size) if ref['pad'] > 0 else 0
        if parse_level_set(ref['levels'], trained_levels) is None:
            ref['levels'] = ''.join(str(l) for l in trained_levels)
        chosen.append((name, ref))

    hold_rows, hold_maps = [], {}
    for role, cfg in tqdm(chosen, desc='  holdout (full-res + AUPRO)'):
        pad_mode = cfg['pad_mode'] if cfg['pad'] > 0 and cfg['pad_mode'] != '-' else pad_modes[0]
        cfg = dict(cfg, pad=int(cfg['pad']), margin=int(cfg['margin']), sigma=float(cfg['sigma']),
                   pad_mode=pad_mode if cfg['pad'] > 0 else '-')
        key = (cfg['pad'], pad_mode, cfg['tta'])
        if key not in caches:
            backbone = build_backbone(model, info, cfg['pad'], pad_mode)
            caches[key] = collect_nll(model, backbone, cflow, test_loader, trained_levels, feat_stats,
                                      cfg['tta'], desc=f'  infer {key}')
            if (cfg['pad'], pad_mode) not in pos_caches:
                positional(backbone, cfg['pad'], pad_mode)
            del backbone
        full = build_map(caches[key], pos_caches[(cfg['pad'], pad_mode)], cfg, trained_levels, channels, img_size)
        sm = smooth(full, cfg['sigma'])
        pm = pixel_metrics(sm[hold_idx], masks[hold_idx], cfg['margin'], stride=1, exact=True)
        row = dict(cfg, role=role, **pm)
        row['aupro'] = aupro_rd(sm[hold_idx], masks[hold_idx], labels[hold_idx])
        row['image_auroc_map'] = image_auroc_from_map(sm[hold_idx], labels[hold_idx],
                                                      cfg['image_score'], args.topk_percent)
        row['image_auroc_sediffernet'] = (float(roc_auc_score(labels[hold_idx], se_scores[hold_idx]))
                                          if len(np.unique(labels[hold_idx])) > 1 else float('nan'))
        row['n_holdout'] = len(hold_idx)
        row['name'] = cfg_name(cfg)
        hold_rows.append(row)
        if role != 'top':
            hold_maps[role] = sm
        elif 'best' not in hold_maps:
            hold_maps['best'] = sm

    hold_df = pd.DataFrame(hold_rows)
    hold_df.to_csv(os.path.join(out_dir, 'holdout_test_top.csv'), index=False)
    print("\n  Held-out results (full resolution):")
    print(hold_df[['role'] + FACTORS + ['pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro', 'image_auroc_map']]
          .round(4).to_string(index=False))

    # ── Diagnostics: border profile + visual comparison ──────────────────────
    normal_idx = np.flatnonzero(labels == 0)
    profiles = {name: border_profile(m[normal_idx]) for name, m in hold_maps.items()}
    plot_border_profiles(profiles, os.path.join(out_dir, 'border_profile.png'))
    anom_hold = [i for i in hold_idx if labels[i] == 1][:6]
    plot_map_comparison(images_u8, masks, hold_maps, anom_hold, os.path.join(out_dir, 'maps_comparison.png'))

    # ── Summary ──────────────────────────────────────────────────────────────
    best_val = df.iloc[0]
    best_hold = hold_df[hold_df['role'] == 'top'].iloc[0]
    base_hold = hold_df[hold_df['role'] == 'baseline_producao'].iloc[0]
    lines = [
        f"tag: {tag}", f"class: {args.class_name}", f"nf_head: {nf_ckpt_path}", f"se_model: {se_ckpt}",
        "head hparams: " + json.dumps(info, default=str),
        f"images: {len(labels)} (val {len(val_idx)}, holdout {len(hold_idx)}), val_fraction={args.val_fraction}",
        f"grid size: {len(df)} | selection metric: {args.select_metric} | pixel_stride (grid)={args.pixel_stride}",
        f"SEDifferNet image AUROC (all): {se_auroc:.4f}",
        "",
        f"BEST on val:      {best_val['name']}  -> {args.select_metric}={best_val[args.select_metric]:.4f}",
        f"  -> holdout:     pixAUROC={best_hold['pixel_auroc']:.4f} AP={best_hold['pixel_ap']:.4f} "
        f"F1={best_hold['pixel_f1']:.4f} AUPRO={best_hold['aupro']:.4f}",
        f"BASELINE holdout: pixAUROC={base_hold['pixel_auroc']:.4f} AP={base_hold['pixel_ap']:.4f} "
        f"F1={base_hold['pixel_f1']:.4f} AUPRO={base_hold['aupro']:.4f}",
        f"DELTA (best - baseline, holdout): pixAUROC={best_hold['pixel_auroc'] - base_hold['pixel_auroc']:+.4f} "
        f"AP={best_hold['pixel_ap'] - base_hold['pixel_ap']:+.4f} AUPRO={best_hold['aupro'] - base_hold['aupro']:+.4f}",
    ]
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    del model, cflow, caches
    torch.cuda.empty_cache() if DEVICE == 'cuda' else None
    return {'tag': tag, 'nf_head': nf_ckpt_path, **{k: info[k] for k in ('levels', 'train_reflect_pad',
            'feat_pool', 'feat_norm', 'feat_norm_levels', 'no_rotation', 'epoch', 'train_pixel_auroc')},
            'best_val_name': best_val['name'], 'best_val_metric': float(best_val[args.select_metric]),
            'holdout_best_pixel_auroc': float(best_hold['pixel_auroc']),
            'holdout_best_pixel_ap': float(best_hold['pixel_ap']),
            'holdout_best_aupro': float(best_hold['aupro']),
            'holdout_baseline_pixel_auroc': float(base_hold['pixel_auroc']),
            'holdout_baseline_aupro': float(base_hold['aupro'])}


# ═══════════════════════════════════════════════════════════════════════════════
# Retrain stage
# ═══════════════════════════════════════════════════════════════════════════════

def run_retrain(args, run_dir):
    se_ckpt = args.se_checkpoint or find_se_checkpoint(args.class_name)
    if se_ckpt is None:
        raise FileNotFoundError(f"SEDifferNet final model not found for {args.class_name}")
    variants = [v for v in args.variants.split(',') if v] if args.variants else list(TRAIN_VARIANTS)
    unknown = [v for v in variants if v not in TRAIN_VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; available: {list(TRAIN_VARIANTS)}")

    train_script = project_path('scripts', 'train', 'pixel_train_from_pretrained.py')
    heads, rows = [], []
    for variant in variants:
        out_dir = os.path.join(run_dir, 'train', variant)
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, train_script,
               '--checkpoint', se_ckpt, '--class_name', args.class_name,
               '--dataset', args.dataset, '--epochs', str(args.epochs),
               '--eval_interval', str(args.eval_interval), '--disable_patchcore',
               '--out_dir', out_dir, '--seed', str(args.seed),
               '--sigma', '4', '--score_norm', args.train_score_mode,
               *TRAIN_VARIANTS[variant]]
        print(f"\n{'=' * 70}\n  RETRAIN [{variant}]\n{'=' * 70}\n  " + " ".join(cmd))
        t0 = time.time()
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        elapsed = time.time() - t0
        run_dirs = sorted(d for d in os.listdir(out_dir) if d.startswith('run_'))
        head = None
        if run_dirs:
            cand = os.path.join(out_dir, run_dirs[-1], 'best_models', 'best_pixel_auroc.pt')
            head = cand if os.path.exists(cand) else None
        rows.append({'variant': variant, 'flags': " ".join(TRAIN_VARIANTS[variant]),
                     'returncode': ret.returncode, 'train_minutes': elapsed / 60, 'head': head})
        if head is None:
            print(f"  WARNING: no best_pixel_auroc.pt produced for {variant} (returncode {ret.returncode})")
            continue
        heads.append((variant, head))
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, 'retrain_variants.csv'), index=False)
    return heads


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Full ablation of the CFLOW NF head (pixel level)")
    parser.add_argument('--class_name', type=str, default='vari-grip')
    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--stage', type=str, default='posthoc', choices=['posthoc', 'retrain', 'all'])
    parser.add_argument('--se_checkpoint', type=str, default=None,
                        help='SEDifferNet final model. Default: final_models/SEDiffernet/<class>/*.pt')
    parser.add_argument('--nf_checkpoint', type=str, default=None,
                        help='NF head to ablate (posthoc). Default: final_models/NF Head/<class>/best_models/best_pixel_auroc.pt')
    parser.add_argument('--limit', type=int, default=None, help='Balanced cap on test images')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_fraction', type=float, default=0.3,
                        help='Fraction of test images (stratified) used to SELECT the configuration; '
                             'the rest is reported once. 0 = select and report on the same images.')
    parser.add_argument('--select_metric', type=str, default='pixel_auroc',
                        choices=['pixel_auroc', 'pixel_ap', 'pixel_f1', 'image_auroc_map'])
    parser.add_argument('--top_n', type=int, default=12,
                        help='Number of best validation configs re-scored on the held-out split')
    parser.add_argument('--pixel_stride', type=int, default=2,
                        help='Pixel subsampling for the grid metrics (held-out uses stride 1)')
    parser.add_argument('--topk_percent', type=float, default=1.0)
    # Head hyperparameters (only needed for legacy checkpoints without metadata)
    parser.add_argument('--out_size', type=int, default=None)
    parser.add_argument('--clamp_scale', type=float, default=None)
    # Grid
    parser.add_argument('--pads', type=str, default='0,112')
    parser.add_argument('--pad_modes', type=str, default='reflect', help=f"Subset of {PAD_MODES}")
    parser.add_argument('--ttas', type=str, default='none,flips', help=f"Subset of {list(TTA_MODES)}")
    parser.add_argument('--score_modes', type=str, default=','.join(SCORE_MODES))
    parser.add_argument('--level_sets', type=str, default='012,12,01,02,0,2',
                        help='Original level ids to sum (0=L1,1=L2,2=L3); sets not covered by the head are skipped')
    parser.add_argument('--pos_calibs', type=str, default='none,l23', help=f"Subset of {list(POS_CALIB_MODES)}")
    parser.add_argument('--sigmas', type=str, default='0,4,8')
    parser.add_argument('--margins', type=str, default='0,8')
    parser.add_argument('--image_scores', type=str, default='max,topk')
    parser.add_argument('--quick', action='store_true', help='Small grid for smoke tests')
    # Retrain
    parser.add_argument('--variants', type=str, default=None,
                        help=f"Comma-separated subset of {list(TRAIN_VARIANTS)} (default: all)")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--train_score_mode', type=str, default='raw',
                        choices=list(SCORE_MODES),
                        help='Aggregation mode used during in-training validation to select '
                             'best_pixel_auroc.pt. Defaults to "raw" so checkpoint selection '
                             'mirrors deployment inference.')
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    if args.quick:
        args.ttas = 'none'
        args.score_modes = 'per_level_minmax,per_level_prob'
        args.level_sets = '012,12'
        args.pos_calibs = 'none,l23'
        args.sigmas = '0,4'
        args.margins = '0,8'
        args.image_scores = 'max'
        args.top_n = min(args.top_n, 4)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_dir = args.out_dir or project_path('resultado_analise_final', 'ablacao_nf_head',
                                           f'run_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")
        f.write(f"device={DEVICE}\nimg_size={c.img_size}\n")
    print(f"Output: {run_dir}")

    original = (c.class_name, c.n_transforms_test, c.transf_rotations)
    summaries = []
    try:
        if args.stage in ('posthoc', 'all'):
            nf = args.nf_checkpoint or find_nf_head_checkpoint(args.class_name)
            if nf is None:
                raise FileNotFoundError(f"NF head not found for {args.class_name}")
            summaries.append(run_posthoc(args, nf, os.path.join(run_dir, 'posthoc_released'), 'released'))
        if args.stage in ('retrain', 'all'):
            for variant, head in run_retrain(args, run_dir):
                summaries.append(run_posthoc(args, head, os.path.join(run_dir, f'posthoc_{variant}'), variant))
    finally:
        c.class_name, c.n_transforms_test, c.transf_rotations = original

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(os.path.join(run_dir, 'ablation_summary.csv'), index=False)
        print(f"\n{'=' * 70}\n  ABLATION SUMMARY\n{'=' * 70}")
        print(summary_df[['tag', 'best_val_metric', 'holdout_baseline_pixel_auroc',
                          'holdout_best_pixel_auroc', 'holdout_best_pixel_ap',
                          'holdout_baseline_aupro', 'holdout_best_aupro']].round(4).to_string(index=False))
    print(f"\nResults: {run_dir}")


if __name__ == '__main__':
    main()
