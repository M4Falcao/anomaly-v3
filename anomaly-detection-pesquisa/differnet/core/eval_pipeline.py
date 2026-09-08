"""Shared building blocks for the three evaluation entry points.

Three scripts under ``scripts/eval/`` evaluate the same system at different
depths and must agree on how models are loaded, how the test set is built and
how metrics are computed - otherwise their numbers are not comparable:

- ``evaluate_image_level.py``  - SEDifferNet only (is this image anomalous?)
- ``evaluate_pixel_level.py``  - CFLOW NF head only (which pixels are anomalous?)
- ``evaluate_full_pipeline.py`` - both branches plus latency and plots

Everything they share lives here so that a fix lands in one place.
"""

import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import config as c
from core.cflow import (
    CFlowPixelHead,
    SEBackboneFeatureExtractor,
    TRAIN_CLAMP_SCALE,
    TRAIN_OUT_SIZE,
    infer_n_blocks,
    nll_with_tta,
    read_cflow_hparams,
    resolve_hparam,
    select_levels,
)
from core.model import SEDifferNet, load_weights
from core.paths import project_path
from core.utils import load_datasets

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ALL_CLASSES = [
    'glass-insulator',
    'lightning-rod-suspension',
    'polymer-insulator-upper-shackle',
    'vari-grip',
    'yoke-suspension',
]

SE_MODEL_DIR = project_path('final_models', 'SEDiffernet')
NF_HEAD_DIR = project_path('final_models', 'NF Head')

# AlexNet stage widths the pixel head is built on; index = level id (0=L1).
LAYER_CHANNELS = [64, 192, 256]
LEVEL_NAMES = {0: 'L1', 1: 'L2', 2: 'L3'}


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════════

def find_se_checkpoint(class_name):
    """Find the SEDifferNet checkpoint .pt file for a given class."""
    class_dir = os.path.join(SE_MODEL_DIR, class_name)
    if not os.path.isdir(class_dir):
        return None
    for f in sorted(os.listdir(class_dir)):
        if f.endswith('.pt'):
            return os.path.join(class_dir, f)
    return None


def find_nf_head_checkpoint(class_name, base_dir=None):
    """Find the best CFLOW NF Head checkpoint for a given class."""
    root = base_dir or NF_HEAD_DIR
    candidates = [
        os.path.join(root, class_name, 'best_models', 'best_pixel_auroc.pt'),
        os.path.join(root, class_name, 'best_models', 'best_image_auroc.pt'),
        os.path.join(root, class_name, 'best_pixel_auroc.pt'),
        os.path.join(root, 'best_models', class_name, 'best_pixel_auroc.pt'),
        os.path.join(root, f"{class_name}_best_pixel_auroc.pt"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None


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


def binary_metrics(labels, scores, prefix=''):
    """AUROC / AP / best-F1 (and its threshold) for one binary problem.

    Returns an empty dict when the labels are single-class, so callers can
    merge it unconditionally.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return {}
    f1, thr = compute_best_f1(labels, scores)
    return {
        f'{prefix}auroc': float(roc_auc_score(labels, scores)),
        f'{prefix}ap': float(average_precision_score(labels, scores)),
        f'{prefix}f1': f1,
        f'{prefix}threshold': thr,
    }


def confusion_at(labels, scores, threshold):
    """Counts and rates at a fixed operating point."""
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = int((pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'precision': tp / max(tp + fp, 1),
        'recall': tp / max(tp + fn, 1),
        'specificity': tn / max(tn + fp, 1),
        'balanced_acc': 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def eval_transform():
    """Deterministic resize+normalize, with no augmentation.

    Mandatory for pixel metrics: ``load_datasets`` hands the test split the
    *training* transform, whose random rotation desynchronizes the image from
    its ground-truth mask.
    """
    return transforms.Compose([
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ])


def build_eval_datasets(class_name, dataset_path, n_transforms=1, limit=None, seed=42):
    """Train/test splits ready for evaluation, plus the per-image 0/1 labels.

    Args:
        class_name: Dataset class to load.
        dataset_path: Root of the INSPLAD-seg layout.
        n_transforms: Number of fixed rotations averaged per test image. ``1``
            (deterministic) is required for pixel metrics; DifferNet's native
            image-level TTA uses more.
        limit: Cap on test images, balanced so every anomaly is kept.
        seed: Controls the balanced subsample.

    Returns:
        ``(trainset, testset, labels)``.
    """
    c.class_name = class_name
    c.n_transforms_test = n_transforms
    c.transf_rotations = False

    trainset, testset = load_datasets(dataset_path, class_name, aligned=True)
    testset.n_transforms = n_transforms
    testset.get_fixed = n_transforms > 1
    testset.fixed_degrees = [i * 360.0 / n_transforms for i in range(n_transforms)]
    if n_transforms == 1:
        testset.transform = eval_transform()

    labels = np.array([1 if lbl > 0 else 0 for _, lbl, _ in testset.samples])

    if limit and 0 < limit < len(testset):
        rng = np.random.default_rng(seed)
        anomaly_idx = np.flatnonzero(labels == 1)
        normal_idx = np.flatnonzero(labels == 0)
        n_anom = min(len(anomaly_idx), limit // 2 if limit // 2 else 1)
        n_anom = min(len(anomaly_idx), max(n_anom, limit - len(normal_idx)))
        n_norm = min(len(normal_idx), limit - n_anom)
        keep = np.concatenate([rng.choice(anomaly_idx, n_anom, replace=False),
                               rng.choice(normal_idx, n_norm, replace=False)])
        keep.sort()
        testset = Subset(testset, keep.tolist())
        labels = labels[keep]

    return trainset, testset, labels


def eval_loader(testset, batch_size=1):
    return DataLoader(testset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=True)


def unpack_batch(data):
    """``(images, label, mask)`` from a loader batch, images as ``(B, 3, H, W)``."""
    images, labels, masks = data
    images = images.to(DEVICE)
    if images.dim() == 5:
        images = images.view(-1, *images.shape[-3:])
    return images, labels, masks


# ═══════════════════════════════════════════════════════════════════════════════
# Image branch (SEDifferNet)
# ═══════════════════════════════════════════════════════════════════════════════

def load_se_model(se_ckpt_path):
    """Load a frozen SEDifferNet from a checkpoint."""
    model = SEDifferNet()
    model, _ = load_weights(model, se_ckpt_path)
    return model.to(DEVICE).eval()


@torch.no_grad()
def image_score(model, images):
    """DifferNet likelihood score: mean squared latent norm over all views.

    Higher means less likely under the normal-data flow, hence more anomalous.
    """
    z = model(images)
    return float(torch.mean(z ** 2).item())


# ═══════════════════════════════════════════════════════════════════════════════
# Pixel branch (CFLOW NF head)
# ═══════════════════════════════════════════════════════════════════════════════

class PixelHead:
    """A trained CFLOW head plus everything needed to reproduce its scoring.

    The head's recipe (which feature levels it saw, whether features were
    standardized, how the input was padded) is stored in the checkpoint. Using
    a different recipe at evaluation time silently computes a different
    function, which is how several early results in this project went wrong.
    """

    def __init__(self, model, nf_ckpt_path, reflect_pad=112, attention='se',
                 out_size=None, clamp_scale=None, n_blocks=None,
                 cond_dim=64, hidden=256):
        data = torch.load(nf_ckpt_path, map_location=DEVICE, weights_only=False)
        if 'cflow_state_dict' in data:
            state = data['cflow_state_dict']
        elif 'model_state_dict' in data:
            state = data['model_state_dict']
        else:
            raise KeyError(f"{nf_ckpt_path} has no CFLOW state dict")

        hp = read_cflow_hparams(data)
        self.hparams = hp
        self.out_size, self.src_out = resolve_hparam('out_size', out_size, hp, TRAIN_OUT_SIZE)
        self.clamp, self.src_clamp = resolve_hparam('clamp_scale', clamp_scale, hp,
                                                    TRAIN_CLAMP_SCALE)
        self.n_blocks, _ = resolve_hparam('n_blocks', n_blocks, hp, infer_n_blocks(state),
                                          warn=False)

        self.trained_levels = list(hp.get('levels') or [0, 1, 2])
        self.feat_pool = int(hp.get('feat_pool') or 0)
        self.pad_mode = hp.get('pad_mode') or 'reflect'
        self.feat_stats = data.get('feat_stats')
        self.channels = select_levels(LAYER_CHANNELS, self.trained_levels)
        self.reflect_pad = reflect_pad
        self.epoch = hp.get('epoch')

        self.backbone = SEBackboneFeatureExtractor(
            model, out_size=self.out_size, reflect_pad=reflect_pad, attention=attention,
            pad_mode=self.pad_mode, feat_pool=self.feat_pool).to(DEVICE).eval()

        self.cflow = CFlowPixelHead(
            layer_channels=self.channels, cond_dim=cond_dim, n_blocks=self.n_blocks,
            hidden=hidden, clamp_scale=self.clamp).to(DEVICE)
        self.cflow.load_state_dict(state)
        self.cflow.eval()

    @torch.no_grad()
    def nll(self, images, tta='flips', per_channel=False):
        """Per-level NLL maps, averaged over the TTA views."""
        return nll_with_tta(self.backbone, self.cflow, images, levels=self.trained_levels,
                            feat_stats=self.feat_stats, tta=tta, per_channel=per_channel)

    @torch.no_grad()
    def score_map(self, images, tta='flips', score_norm='raw', levels=None, level_stats=None):
        """Anomaly map at full image resolution.

        Args:
            images: ``(B, 3, H, W)`` batch.
            tta: Key of :data:`core.cflow.TTA_MODES`.
            score_norm: Aggregation mode across levels.
            levels: Indices *into the trained levels* to keep (``None`` = all);
                use :func:`resolve_score_levels` to build it from level ids.
            level_stats: Required by ``per_level_std``.
        """
        per_channel = score_norm == 'per_level_minmax'
        nll = self.nll(images, tta=tta, per_channel=per_channel)
        return CFlowPixelHead.aggregate_nll(
            nll, self.channels, c.img_size[0], mode=score_norm,
            level_stats=level_stats, levels=levels)

    def describe(self):
        if self.feat_stats is None:
            norm = 'OFF'
        elif any(st is None for st in self.feat_stats):
            norm_lvls = [LEVEL_NAMES[lvl] for lvl, st in zip(self.trained_levels, self.feat_stats)
                         if st is not None]
            norm = f"selective({','.join(norm_lvls)})"
        else:
            norm = 'ON'
        return (f"out_size={self.out_size} ({self.src_out}) | clamp={self.clamp} "
                f"({self.src_clamp}) | n_blocks={self.n_blocks} | "
                f"levels treinados={self.trained_levels} | feat_norm={norm} | "
                f"feat_pool={self.feat_pool} | pad={self.reflect_pad} ({self.pad_mode})")


#: Score-level subset measured as best for each class on the full INSPLAD test
#: set with the deployment recipe (raw + pad112 + tta flips + sigma 8), on the
#: heads of ``resultado_analise_final/treino_nf_heads/run_20260907_145952``.
#:
#:   class                       L1      L2      L3     L1+L3   L1+L2+L3
#:   glass-insulator           0.8584     -    0.9394  0.9534      -
#:   lightning-rod-suspension  0.8015     -    0.8761  0.8622      -
#:   polymer-...-shackle       0.8279  0.7771  0.8525  0.8505   0.7987
#:   vari-grip                 0.9191     -    0.8052  0.8561      -
#:   yoke-suspension           0.8929     -    0.8960  0.9348      -
#:
#: vari-grip is the extreme case: L3 is its weakest level, and fusing it with L1
#: costs 0.063 AUROC and 0.150 AUPRO versus scoring on L1 alone.
#:
#: polymer keeps L1+L3 even though L3 alone scores 0.0020 higher: that gap is
#: below the measured split-noise floor (0.005) while L3 alone loses 0.060 AUPRO,
#: so the AUROC "win" is noise and the AUPRO loss is real.  Selecting on a single
#: metric is exactly the failure RD++'s composite score guards against.
#:
#: These are head-specific.  A head trained on a different level subset ranks its
#: levels differently, so re-measure with ``evaluate_pixel_level.py`` (which
#: writes ``per_level_auroc.csv``) after retraining rather than reusing this map.
PER_CLASS_SCORE_LEVELS = {
    'glass-insulator': '02',
    'lightning-rod-suspension': '2',
    'polymer-insulator-upper-shackle': '02',
    'vari-grip': '0',
    'yoke-suspension': '02',
}


def resolve_score_levels(requested, trained_levels, class_name=None):
    """Map a requested level subset to indices into the levels the head was trained on.

    ``requested`` uses the original level ids (0=L1, 1=L2, 2=L3), e.g. ``'02'``.
    ``'auto'`` keeps L1+L3 when the head has them - the lowest-regret choice
    measured across the five INSPLAD classes - and falls back to every trained
    level otherwise. ``'per_class'`` looks ``class_name`` up in
    :data:`PER_CLASS_SCORE_LEVELS` and falls back to ``'auto'`` for unknown
    classes.  ``'all'`` always keeps every trained level.

    Levels the head was not trained on are dropped from the per-class choice
    instead of raising, so a head trained on a subset still evaluates.

    Returns:
        Tuple ``(indices, level_ids)``; ``indices`` is ``None`` when every trained
        level is kept, which is what :meth:`CFlowPixelHead.aggregate_nll` expects.
    """
    if requested == 'per_class':
        preferred = PER_CLASS_SCORE_LEVELS.get(class_name)
        if preferred is None:
            requested = 'auto'
        else:
            wanted = [int(ch) for ch in preferred if ch.isdigit()]
            wanted = [l for l in wanted if l in trained_levels] or list(trained_levels)
            idx = [trained_levels.index(l) for l in wanted]
            return (None if idx == list(range(len(trained_levels))) else idx), wanted

    if requested in (None, '', 'all'):
        wanted = list(trained_levels)
    elif requested == 'auto':
        wanted = [l for l in (0, 2) if l in trained_levels] or list(trained_levels)
    else:
        wanted = sorted({int(ch) for ch in requested if ch.isdigit()})
        missing = [l for l in wanted if l not in trained_levels]
        if missing:
            raise ValueError(
                f"--levels pede {missing}, mas a head foi treinada apenas em {trained_levels}")
    idx = [trained_levels.index(l) for l in wanted]
    return (None if idx == list(range(len(trained_levels))) else idx), wanted


def mask_to_binary(masks):
    """First mask of a batch as a 2-D uint8 array."""
    m = masks.squeeze(0).squeeze(0).numpy()
    if m.ndim == 3:
        m = m[0]
    return (m > 0.5).astype(np.uint8)


def denormalize(tensor):
    """Denormalize an image tensor back to [0, 1] for display."""
    mean = np.array(c.norm_mean)
    std = np.array(c.norm_std)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return np.clip(img * std + mean, 0, 1)
