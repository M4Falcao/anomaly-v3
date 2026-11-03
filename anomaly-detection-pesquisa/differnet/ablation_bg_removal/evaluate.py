"""Mapas de anomalia por head e métricas por histograma (etapa 8).

Reproduz a receita de inferência gravada no checkpoint (como o notebook D-18): pad,
pad_mode, TTA, score_mode, níveis, align_corners, sigma e image_score. ``cflow_original``
não é suportado (é transdutivo e o preset ``general`` não o usa).

As métricas de pixel usam histogramas de alta resolução (bordas = quantis da amostra do
próprio mapa). O teste completo de yoke tem ~250 M pixels e não cabe no sklearn exato.
O teste ``tests/test_ablation.py`` compara AUROC/AP binados com o sklearn.
"""
import math
import os
import sys

import numpy as np

import common as cm

if cm.DIFFERNET_ROOT not in sys.path:
    sys.path.insert(0, cm.DIFFERNET_ROOT)

FINE_BINS = 65536
COARSE_FACTOR = 64  # bootstrap: 1024 bins


# ─── Métricas a partir de histogramas ────────────────────────────────────────

def auroc_from_hist(pos, neg):
    """AUROC com empates dentro do bin contados como 1/2 (bins em ordem crescente de score)."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float('nan')
    neg_below = np.concatenate(([0.0], np.cumsum(neg)[:-1]))
    return float((pos * (neg_below + 0.5 * neg)).sum() / (P * N))


def ap_from_hist(pos, neg):
    """Average precision com limiares nas bordas dos bins (score decrescente)."""
    pos = np.asarray(pos, dtype=np.float64)[::-1]
    neg = np.asarray(neg, dtype=np.float64)[::-1]
    P = pos.sum()
    if P == 0:
        return float('nan')
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    keep = pos > 0
    precision = tp[keep] / np.maximum(tp[keep] + fp[keep], 1)
    return float((pos[keep] / P * precision).sum())


def fpr_at(neg, edges, threshold):
    """Fração de negativos com score > ``threshold`` (aproximada na borda do bin)."""
    neg = np.asarray(neg, dtype=np.float64)
    total = neg.sum()
    if total == 0:
        return float('nan')
    idx = np.searchsorted(edges, threshold, side='right')
    return float(neg[idx:].sum() / total)


def threshold_at_fpr(neg, edges, fpr):
    """Menor borda de bin cujo FPR fica <= ``fpr``."""
    neg = np.asarray(neg, dtype=np.float64)
    tail = np.cumsum(neg[::-1])[::-1] / max(neg.sum(), 1)
    ok = np.flatnonzero(tail <= fpr)
    if not ok.size:
        return float(edges[-1])
    i = ok[0]
    return float(edges[i - 1]) if i > 0 else float(edges[0])


def quantile_edges(sample, n_bins=FINE_BINS):
    edges = np.unique(np.quantile(sample, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return edges


def bincount(values, edges):
    return np.bincount(np.searchsorted(edges, values, side='right'), minlength=len(edges) + 1)


def paired_bootstrap(ref_pos, ref_neg, cand_pos, cand_neg, labels, n_boot=1000, seed=0,
                     ref_img=None, cand_img=None):
    """Bootstrap pareado estratificado por imagem (normal/anômala).

    ``*_pos`` / ``*_neg``: ``(n_images, n_bins)``, contagens por imagem (bins grossos).
    Retorna deltas (cand - ref) de Pixel AUROC e, se dados, de Image AUROC.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == g) for g in (0, 1)]
    d_pix, d_img = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if len(g)])
        w = np.bincount(idx, minlength=len(labels)).astype(np.float64)
        d_pix.append(auroc_from_hist(w @ cand_pos, w @ cand_neg) - auroc_from_hist(w @ ref_pos, w @ ref_neg))
        if ref_img is not None and len(np.unique(labels[idx])) == 2:
            d_img.append(roc_auc_score(labels[idx], np.asarray(cand_img)[idx])
                         - roc_auc_score(labels[idx], np.asarray(ref_img)[idx]))

    def summary(d):
        d = np.asarray(d, dtype=np.float64)
        d = d[np.isfinite(d)]
        if not d.size:
            return dict(ci_low=float('nan'), ci_high=float('nan'), p_le0=float('nan'))
        return dict(ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975)),
                    p_le0=float((d <= 0).mean()))
    return dict(pixel=summary(d_pix), image=summary(d_img) if d_img else None, n_boot=n_boot)


def image_auroc_bootstrap(labels, ref_scores, cand_scores, n_boot=1000, seed=0):
    """IC95 do ΔImage AUROC (cand − ref), bootstrap pareado estratificado por imagem."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    ref_scores, cand_scores = np.asarray(ref_scores), np.asarray(cand_scores)
    groups = [np.flatnonzero(labels == g) for g in (0, 1)]
    if any(len(g) == 0 for g in groups):
        return dict(ci_low=float('nan'), ci_high=float('nan'), p_le0=float('nan'))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        d.append(roc_auc_score(labels[idx], cand_scores[idx]) - roc_auc_score(labels[idx], ref_scores[idx]))
    d = np.asarray(d)
    return dict(ci_low=float(np.quantile(d, 0.025)), ci_high=float(np.quantile(d, 0.975)), p_le0=float((d <= 0).mean()))


# ─── Inferência ──────────────────────────────────────────────────────────────

def load_head_payload(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


def head_recipe(payload):
    recipe = payload.get('inference_recipe') or (payload.get('training_recipe') or {}).get('inference')
    if recipe is None:
        raise ValueError('Checkpoint sem inference_recipe/training_recipe; passe uma head treinada com --recipe.')
    recipe = dict(recipe)
    if recipe['score_mode'] == 'cflow_original':
        raise NotImplementedError('cflow_original é transdutivo e não é suportado por este avaliador.')
    if recipe.get('margin', 0) != 0 or recipe.get('pos_calib', 'none') != 'none':
        raise ValueError('Receita com margem/pos_calib não suportada (D-08: margem 0).')
    return recipe


def compute_maps(head_ckpt, se_ckpt, dataset_root, class_name, out_dir, indices=None, desc=''):
    """Mapas do teste (ou do subconjunto ``indices``) em memmap ``out_dir/{maps,masks}.npy``.

    Retorna dict(maps, masks, image_scores, paths, labels, recipe, stats_source).
    """
    import torch
    from torch.utils.data import Subset
    import config as c
    from core.cflow import (CFlowPixelHead, SEBackboneFeatureExtractor, compute_positional_stats,
                            gaussian_smooth, image_score_from_map, infer_n_blocks, nll_with_tta,
                            read_cflow_hparams)
    from core.eval_pipeline import DEVICE, build_eval_datasets, eval_loader, eval_transform, load_se_model
    from tqdm import tqdm

    payload = load_head_payload(head_ckpt)
    recipe = head_recipe(payload)
    hp = read_cflow_hparams(payload)
    size = int(hp.get('img_size') or 448)
    c.img_size = (size, size)
    trained = list(hp.get('levels') or [0, 1, 2])
    if any(int(l) not in trained for l in recipe['levels']):
        raise ValueError(f'Níveis da receita {recipe["levels"]} não treinados ({trained})')
    selected = [trained.index(int(l)) for l in recipe['levels']]
    channels = [[64, 192, 256][l] for l in trained]

    trainset, testset, labels = build_eval_datasets(class_name, dataset_root, n_transforms=1, limit=None)
    class_dir = os.path.join(dataset_root, class_name)
    paths = [os.path.relpath(s[0], class_dir).replace(os.sep, '/') for s in testset.samples]
    indices = list(range(len(paths))) if indices is None else list(indices)
    paths = [paths[i] for i in indices]
    labels = np.asarray(labels)[indices]
    n = len(indices)

    model = load_se_model(se_ckpt)
    backbone = SEBackboneFeatureExtractor(
        model, out_size=hp.get('out_size') or 96, reflect_pad=recipe['pad'], attention=hp.get('attention') or 'se',
        pad_mode=recipe['pad_mode'], feat_pool=hp.get('feat_pool') or 0).to(DEVICE).eval()
    state = payload.get('cflow_state_dict', payload.get('model_state_dict'))
    head = CFlowPixelHead(channels, cond_dim=hp.get('cond_dim') or 64,
                          n_blocks=hp.get('n_blocks') or infer_n_blocks(state),
                          hidden=hp.get('hidden') or 256, clamp_scale=hp.get('clamp_scale') or 0.5).to(DEVICE)
    head.load_state_dict(state)
    head.eval()

    stats, stats_source = None, 'n/a'
    if recipe['score_mode'] == 'per_level_std':
        if payload.get('inference_recipe') == recipe and payload.get('inference_level_stats') is not None:
            stats, stats_source = payload['inference_level_stats'], 'checkpoint'
        else:
            trainset.transform = eval_transform()
            trainset.n_transforms = 1
            trainset.get_fixed = False
            pos = compute_positional_stats(backbone, head, eval_loader(trainset), DEVICE, levels=trained,
                                           feat_stats=payload.get('feat_stats'))
            stats = []
            for mean_map, std_map in pos:
                mean = float(mean_map.mean())
                stats.append((mean, math.sqrt(max(float((std_map ** 2 + mean_map ** 2).mean()) - mean ** 2, 0.0))))
            stats_source = 'recomputed_train'

    os.makedirs(out_dir, exist_ok=True)
    maps = np.lib.format.open_memmap(os.path.join(out_dir, 'maps.npy'), mode='w+', dtype=np.float32,
                                     shape=(n, size, size))
    masks = np.lib.format.open_memmap(os.path.join(out_dir, 'masks.npy'), mode='w+', dtype=np.uint8,
                                      shape=(n, size, size))
    scores = np.zeros(n, dtype=np.float64)
    loader = eval_loader(Subset(testset, indices))
    with torch.inference_mode():
        for i, (images, _, mask) in enumerate(tqdm(loader, total=n, desc=desc or class_name, leave=False)):
            images = images.to(DEVICE)
            if images.ndim == 5:
                images = images[:, 0]
            nll = nll_with_tta(backbone, head, images, levels=trained, feat_stats=payload.get('feat_stats'),
                               tta=recipe['tta'], per_channel=recipe['score_mode'] == 'per_level_minmax')
            nll = [m.float().cpu() for m in nll]
            agg = CFlowPixelHead.aggregate_nll(nll, channels, size, mode=recipe['score_mode'], level_stats=stats,
                                               levels=selected, align_corners=recipe['align_corners'])
            smap = gaussian_smooth(agg.numpy(), sigma=recipe['sigma'])
            if not np.isfinite(smap).all():
                raise ValueError(f'Mapa não finito em {paths[i]}')
            maps[i] = smap[0]
            masks[i] = (mask[0].squeeze().numpy() > 0.5).astype(np.uint8)
            scores[i] = float(image_score_from_map(smap, mode=recipe['image_score'],
                                                   top_percent=recipe['topk_percent'])[0])
    maps.flush()
    masks.flush()
    del model, backbone, head
    torch.cuda.empty_cache()
    return dict(maps=maps, masks=masks, image_scores=scores, paths=paths, labels=labels,
                recipe=recipe, stats_source=stats_source)


def encoder_image_scores(se_ckpt, dataset_root, class_name, views=64, indices=None):
    """Score DifferNet por imagem (média ||z||² sobre ``views`` rotações fixas)."""
    import torch
    from torch.utils.data import Subset
    from core.eval_pipeline import build_eval_datasets, eval_loader, image_score, load_se_model, unpack_batch
    from tqdm import tqdm
    _, testset, labels = build_eval_datasets(class_name, dataset_root, n_transforms=views, limit=None)
    indices = list(range(len(labels))) if indices is None else list(indices)
    model = load_se_model(se_ckpt)
    scores = []
    for data in tqdm(eval_loader(Subset(testset, indices)), total=len(indices), desc=f'encoder {class_name}',
                     leave=False):
        images, _, _ = unpack_batch(data)
        scores.append(image_score(model, images))
    del model
    torch.cuda.empty_cache()
    return np.asarray(scores), np.asarray(labels)[indices]


def stratified_subset(labels, partition, limit):
    """Smoke test: até ``limit`` imagens, balanceando partição × rótulo, na ordem original."""
    labels = np.asarray(labels)
    partition = np.asarray(partition)
    groups = [np.flatnonzero((partition == p) & (labels == lab)) for p in ('val', 'holdout') for lab in (0, 1)]
    per = max(2, limit // 4)
    keep = np.concatenate([g[:per] for g in groups])
    return sorted(keep.tolist())
