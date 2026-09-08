"""Metrics and architecture-resolution helpers shared by the evaluation scripts.

Every function here was previously duplicated verbatim between
``scripts/eval/sweep_checkpoints.py`` and ``scripts/eval/test_single_checkpoint.py``.
Only definitions that were byte-identical in both files were moved, so the
numeric results of either script are unchanged.

Contents:

- Threshold-free scoring: :func:`calculate_pg2`, :func:`get_optimal_metrics`.
- Pixel-level scoring: :func:`calculate_pixel_level_metrics`.
- Gradient-based localization: :func:`get_grad_maps`.
- Runtime cost sampling: :func:`get_system_metrics`.
- Architecture lookup: :data:`ARCH_MAP`, :func:`load_architecture_from_group`.
"""

import os

import numpy as np
import psutil
import skimage.transform
import torch
from scipy.ndimage import gaussian_filter, rotate
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.amp import autocast
from torch.autograd import Variable

import config as c
from core.model import CBAMDifferNet, DifferNet, SEDifferNet
from core.utils import get_loss, t2np


# --- Metric helpers ----------------------------------------------------------

def calculate_pg2(y_true, y_scores):
    """Compute PG2 (Presorted Good at 2%).

    PG2 is the True Negative Rate - the fraction of good parts correctly kept -
    measured at the operating point where the False Negative Rate is 2%
    (equivalently, TPR >= 98%).

    Args:
        y_true: Binary ground-truth labels, 1 marking anomalies.
        y_scores: Anomaly scores, higher meaning more anomalous.

    Returns:
        The TNR at TPR >= 98%, or ``0.0`` if that operating point is unreachable.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= 0.98)[0]
    if len(idx) == 0:
        return 0.0
    idx = idx[0]
    tnr = 1.0 - fpr[idx]
    return tnr


def get_optimal_metrics(y_true, y_scores):
    """Score predictions at the threshold that maximizes F1.

    Threshold-free metrics (AUROC, PR-AUC, PG2) are computed directly, while
    Accuracy/F1/Precision/Recall are reported at the best-F1 operating point of
    the precision-recall curve.

    Args:
        y_true: Binary ground-truth labels, 1 marking anomalies.
        y_scores: Anomaly scores, higher meaning more anomalous.

    Returns:
        Dict with keys ``Accuracy``, ``F1``, ``Precision``, ``Recall``,
        ``PR-AUC``, ``PG2`` and ``AUROC``. All values are NaN when ``y_true``
        contains a single class.
    """
    if len(np.unique(y_true)) < 2:
        return {
            'Accuracy': np.nan, 'F1': np.nan, 'Precision': np.nan,
            'Recall': np.nan, 'PR-AUC': np.nan, 'PG2': np.nan, 'AUROC': np.nan
        }

    auroc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    pg2 = calculate_pg2(y_true, y_scores)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    denominator = (precisions + recalls)
    f1_scores = np.divide(2 * (precisions * recalls), denominator, out=np.zeros_like(denominator), where=denominator != 0)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred = (y_scores >= best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': prec,
        'Recall': rec,
        'PR-AUC': pr_auc,
        'PG2': pg2,
        'AUROC': auroc
    }


def calculate_pixel_level_metrics(predictions, ground_truth_masks, fg_mask=None):
    """Score an anomaly map against a ground-truth mask.

    Args:
        predictions: Anomaly map, resized to the mask resolution if needed.
        ground_truth_masks: Ground-truth mask; any non-zero value counts as anomalous.
        fg_mask: Optional foreground mask. When given, background pixels are
            excluded so the score is not inflated by trivially normal background.

    Returns:
        The dict produced by :func:`get_optimal_metrics` over the flattened pixels.
    """
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    ground_truth_masks_binary = (ground_truth_masks > 0).astype(int)

    if fg_mask is not None:
        # Resize foreground mask to match ground_truth shape and apply
        fg_resized = skimage.transform.resize(fg_mask.astype(float), ground_truth_masks.shape, mode='constant') > 0.5
        predictions_flat = predictions_resized[fg_resized].reshape(-1)
        ground_truth_flat = ground_truth_masks_binary[fg_resized].reshape(-1)
    else:
        predictions_flat = predictions_resized.reshape(-1)
        ground_truth_flat = ground_truth_masks_binary.reshape(-1)

    return get_optimal_metrics(ground_truth_flat, predictions_flat)


# --- System metrics ----------------------------------------------------------

def get_system_metrics():
    """Sample the current process RSS and the peak CUDA allocation.

    Returns:
        Tuple ``(ram_mb, vram_mb)``. ``vram_mb`` is ``0.0`` without CUDA.
    """
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        vram_mb = 0.0
    return ram_mb, vram_mb


# --- Gradient-based localization ---------------------------------------------

def get_grad_maps(model, inputs, labels, optimizer):
    """Build anomaly heatmaps from the gradient of the loss w.r.t. the input.

    Each of the ``config.n_transforms_test`` rotated views is un-rotated back to
    the original frame, smoothed, and the squared mean absolute gradient across
    channels is returned as the anomaly map.

    Args:
        model: Trained model exposing a ``nf`` normalizing-flow head.
        inputs: Batch shaped ``(batch * n_transforms_test, C, H, W)``.
        labels: Per-image labels; only anomalous images (``label > 0``) are kept.
        optimizer: Optimizer whose gradients are zeroed before the backward pass.

    Returns:
        Array of shape ``(n_anomalies, H, W)``, or ``None`` when no gradient is
        available or the batch contains no anomalies.
    """
    model.eval()
    inputs = Variable(inputs, requires_grad=True)

    with autocast('cuda'):
        z = model(inputs)
        loss = get_loss(z, model.nf.jacobian(run_forward=False))

    optimizer.zero_grad()
    loss.backward()

    if inputs.grad is None:
        return None

    grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad = grad[labels > 0]

    if grad.shape[0] == 0:
        return None

    grad = t2np(grad)
    degrees = -1 * np.arange(c.n_transforms_test) * 360.0 / c.n_transforms_test

    for i_item in range(c.n_transforms_test):
        old_shape = grad[:, i_item].shape
        img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
        img = np.transpose(img, [1, 2, 0])
        img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
        img = gaussian_filter(img, (0, 3, 3))
        grad[:, i_item] = np.reshape(img, old_shape)

    grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
    grad_img = np.mean(np.abs(grad), axis=1)
    grad_img_sq = grad_img ** 2
    return grad_img_sq


# --- Architecture resolution -------------------------------------------------

ARCH_MAP = {
    "cbam": CBAMDifferNet,
    "cbam differnet": CBAMDifferNet,
    "cbamdiffernet": CBAMDifferNet,
    "se": SEDifferNet,
    "sediffernet": SEDifferNet,
    "differnet": DifferNet,
}


def load_architecture_from_group(group_name):
    """Select the architecture class implied by a folder or group name.

    Args:
        group_name: Folder or experiment group name, matched case-insensitively
            against :data:`ARCH_MAP`.

    Returns:
        The matching model class, or ``None`` if nothing matches.
    """
    key = group_name.strip().lower()
    for pattern, arch_cls in ARCH_MAP.items():
        if pattern in key:
            return arch_cls
    return None
