"""Pixel-level evaluation of the CFLOW NF head.

Answers one question only: *which pixels are anomalous?* The SEDifferNet flow
is loaded solely as a frozen feature extractor - its image-level score is never
used - so localization quality is measured without the detection branch masking
or inflating it.

The defaults are the recipe validated across the five INSPLAD classes
(see ``docs/plano_ablation_nf_head.md``): ``raw`` aggregation, reflect-pad 112,
4-view flip TTA, levels L1+L3, sigma 8.

Beyond the headline metrics the script reports the two diagnostics that made
those choices legible in the ablation:

- **per-level AUROC** - each feature level scored alone, which is what reveals
  that L2 is usually the weakest and that a level can be *anti*-correlated when
  the head and the evaluation disagree on hyperparameters.
- **``--sweep``** - re-aggregates the cached NLL maps under every combination of
  levels / sigma / tta / score_norm, so the sensitivity of the result to those
  choices is visible in one table instead of requiring five runs.

Outputs, per run, under ``resultado_analise_final/pixel/run_<ts>/``:
    results_per_class.csv     one row per class plus the mean
    per_level_auroc.csv       AUROC of each level in isolation
    sweep.csv                 only with --sweep
    config.txt                every resolved argument, plus each head's recipe
    <class>/mapas.png         input | GT | mapa | overlay

Examples:
    python scripts/eval/evaluate_pixel_level.py
    python scripts/eval/evaluate_pixel_level.py --classes glass-insulator --sigma 2
    python scripts/eval/evaluate_pixel_level.py --classes vari-grip --sweep --limit 60
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import itertools
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                '..', '..')))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

import config as c
from core.cflow import (
    DEFAULT_ATTENTION,
    SCORE_MODES,
    TTA_MODES,
    CFlowPixelHead,
    border_mask,
    compute_level_stats,
    gaussian_smooth,
)
from core.eval_pipeline import (
    ALL_CLASSES,
    DEVICE,
    LEVEL_NAMES,
    PixelHead,
    build_eval_datasets,
    compute_best_f1,
    compute_pro_rd_style,
    denormalize,
    eval_loader,
    find_nf_head_checkpoint,
    find_se_checkpoint,
    load_se_model,
    mask_to_binary,
    resolve_score_levels,
    unpack_batch,
)
from core.paths import project_path
from core.utils import t2np


def pixel_metrics(maps, masks, keep, with_aupro=True):
    """AUROC / AP / F1 over all kept pixels, plus AUPRO over anomalous images."""
    flat_scores = np.concatenate([m[keep].ravel() for m in maps])
    flat_labels = np.concatenate([m[keep].ravel() for m in masks]).astype(np.uint8)

    out = {}
    if flat_labels.max() > 0 and flat_labels.min() == 0:
        out['pixel_auroc'] = float(roc_auc_score(flat_labels, flat_scores))
        out['pixel_ap'] = float(average_precision_score(flat_labels, flat_scores))
        out['pixel_f1'] = compute_best_f1(flat_labels, flat_scores)[0]
        out['pixel_ap_trivial'] = float(flat_labels.mean())

    if with_aupro:
        pros = [compute_pro_rd_style(mk, mp) for mk, mp in zip(masks, maps) if mk.max() > 0]
        if pros:
            out['aupro'] = float(np.mean(pros))
    return out


def plot_maps(images, masks, maps, save_path, n=6):
    """input | ground truth | anomaly map | overlay, for the first anomalies."""
    idx = [i for i, m in enumerate(masks) if m.max() > 0][:n]
    if not idx:
        return
    fig, axes = plt.subplots(len(idx), 4, figsize=(13, 3.1 * len(idx)))
    axes = np.atleast_2d(axes)
    for row, i in enumerate(idx):
        img = denormalize(images[i])
        amap = maps[i]
        norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
        for col, (data, title, kw) in enumerate([
            (img, 'Input', {}),
            (masks[i], 'Ground truth', {'cmap': 'gray'}),
            (norm, 'Mapa de anomalia', {'cmap': 'jet'}),
            (img, 'Overlay', {}),
        ]):
            ax = axes[row, col]
            ax.imshow(data, **kw)
            if col == 3:
                ax.imshow(norm, cmap='jet', alpha=0.45)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def evaluate_class(class_name, args, out_dir):
    """Pixel metrics for one class. Returns ``(metrics, per_level_rows, sweep_rows)``."""
    print(f"\n{'=' * 70}\n  {class_name}\n{'=' * 70}")

    se_ckpt = args.se_checkpoint or find_se_checkpoint(class_name)
    nf_ckpt = args.nf_checkpoint or find_nf_head_checkpoint(class_name, base_dir=args.nf_dir)
    if se_ckpt is None or nf_ckpt is None:
        print('  ERRO: checkpoint SE ou NF head ausente.')
        return None

    model = load_se_model(se_ckpt)
    head = PixelHead(model, nf_ckpt, reflect_pad=args.reflect_pad, attention=args.attention,
                     out_size=args.out_size, clamp_scale=args.clamp_scale,
                     n_blocks=args.cflow_n_blocks, cond_dim=args.cflow_cond_dim,
                     hidden=args.cflow_hidden)
    print(f'  {head.describe()}')

    try:
        score_idx, score_levels = resolve_score_levels(
            args.levels, head.trained_levels, class_name)
    except ValueError as err:
        print(f'  ERRO: {err}')
        return None
    print(f'  Score: levels={score_levels} | {args.score_norm} | tta={args.tta} | '
          f'sigma={args.sigma} | margin={args.border_margin}')

    trainset, testset, labels = build_eval_datasets(
        class_name, args.dataset, n_transforms=1, limit=args.limit, seed=args.seed)
    loader = eval_loader(testset)
    print(f'  Teste: {len(testset)} imagens ({int(labels.sum())} anomalas)')

    level_stats = None
    if args.score_norm == 'per_level_std':
        level_stats = compute_level_stats(
            head.backbone, head.cflow, eval_loader(trainset, batch_size=4), DEVICE,
            levels=head.trained_levels, feat_stats=head.feat_stats,
            desc=f'  [{class_name}] level stats')

    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 3:
                break
            head.score_map(unpack_batch(data)[0], tta=args.tta, score_norm=args.score_norm,
                           levels=score_idx, level_stats=level_stats)
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    # The per-level NLL is cached so the sweep and the per-level diagnostic cost
    # no extra forward passes.
    per_channel = args.score_norm == 'per_level_minmax'
    cache, masks, images_viz, times = [], [], [], []
    with torch.no_grad():
        for data in tqdm(loader, desc=f'  [{class_name}]', leave=False):
            images, _, mask = unpack_batch(data)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            nll = head.nll(images, tta=args.tta, per_channel=per_channel)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            cache.append([m.cpu() for m in nll])
            masks.append(mask_to_binary(mask))
            if len(images_viz) < 8:
                images_viz.append(images[0].cpu())

    def aggregate(levels_idx, norm, sigma):
        maps = []
        for nll in cache:
            smap = CFlowPixelHead.aggregate_nll(
                [m.to(DEVICE) for m in nll], head.channels, c.img_size[0], mode=norm,
                level_stats=level_stats, levels=levels_idx)
            maps.append(gaussian_smooth(t2np(smap), sigma=sigma)[0])
        return maps

    keep = border_mask(masks[0].shape[-1], args.border_margin)

    norm_val = 'OFF'
    if head.feat_stats is not None:
        if any(st is None for st in head.feat_stats):
            norm_lvls = [LEVEL_NAMES[lvl] for lvl, st in zip(head.trained_levels, head.feat_stats)
                         if st is not None]
            norm_val = f"selective({','.join(norm_lvls)})"
        else:
            norm_val = 'ON'

    maps = aggregate(score_idx, args.score_norm, args.sigma)
    metrics = {'class': class_name, 'n_images': len(masks),
               'n_anomaly': int(labels.sum()), 'levels': ''.join(str(l) for l in score_levels),
               'score_norm': args.score_norm, 'tta': args.tta, 'sigma': args.sigma,
               'reflect_pad': args.reflect_pad, 'border_margin': args.border_margin,
               'feat_norm': norm_val, 'head_epoch': head.epoch}
    metrics.update(pixel_metrics(maps, masks, keep))
    lat = float(np.mean(times)) * 1000
    metrics['latency_ms'] = lat
    metrics['fps'] = 1000.0 / lat if lat > 0 else 0.0

    # Each level alone, which is the diagnostic that exposes a mis-loaded head.
    per_level_rows = []
    for pos, level_id in enumerate(head.trained_levels):
        lm = aggregate([pos], 'raw', args.sigma)
        m = pixel_metrics(lm, masks, keep, with_aupro=False)
        per_level_rows.append({'class': class_name, 'level': LEVEL_NAMES[level_id],
                               **{k: v for k, v in m.items() if k != 'pixel_ap_trivial'}})

    sweep_rows = []
    if args.sweep:
        combos = list(itertools.product(args.sweep_levels.split(','),
                                        [float(s) for s in args.sweep_sigmas.split(',')],
                                        args.sweep_norms.split(',')))
        for lv, sg, nm in tqdm(combos, desc=f'  [{class_name}] sweep', leave=False):
            try:
                idx, ids = resolve_score_levels(lv, head.trained_levels)
            except ValueError:
                continue
            m = pixel_metrics(aggregate(idx, nm, sg), masks, keep, with_aupro=False)
            sweep_rows.append({'class': class_name, 'levels': ''.join(str(i) for i in ids),
                               'sigma': sg, 'score_norm': nm, 'tta': args.tta, **m})

    os.makedirs(out_dir, exist_ok=True)
    plot_maps(images_viz, masks[:len(images_viz)], maps[:len(images_viz)],
              os.path.join(out_dir, 'mapas.png'))

    print(f"  AUROC={metrics.get('pixel_auroc', float('nan')):.4f}  "
          f"AP={metrics.get('pixel_ap', float('nan')):.4f}  "
          f"F1={metrics.get('pixel_f1', float('nan')):.4f}  "
          f"AUPRO={metrics.get('aupro', float('nan')):.4f}  {metrics['fps']:.1f} FPS")
    print('  Por nivel: ' + '  '.join(
        f"{r['level']}={r.get('pixel_auroc', float('nan')):.4f}" for r in per_level_rows))

    del model, head, cache
    torch.cuda.empty_cache()
    return metrics, per_level_rows, sweep_rows


def main():
    parser = argparse.ArgumentParser(
        description='Avaliacao nivel de PIXEL (CFLOW NF head isolada).')
    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--classes', type=str, default=None,
                        help='Lista separada por virgula; default = as 5 classes.')
    parser.add_argument('--se_checkpoint', type=str, default=None)
    parser.add_argument('--nf_checkpoint', type=str, default=None)
    parser.add_argument('--nf_dir', type=str, default=None,
                        help='Root folder with per-class trained heads (default: final_models/NF Head).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Teto balanceado de imagens de teste; preserva todas as anomalias.')

    parser.add_argument('--score_norm', type=str, default='raw', choices=list(SCORE_MODES),
                        help='Agregacao entre niveis. Default "raw": venceu em 17/18 heads.')
    parser.add_argument('--tta', type=str, default='flips', choices=list(TTA_MODES),
                        help='TTA por espelhamento. Default "flips" (4 vistas, 4x o custo).')
    parser.add_argument('--levels', type=str, default='auto',
                        help='Niveis no score por id original (0=L1,1=L2,2=L3). '
                             'Default "auto" = L1+L3. "per_class" usa o melhor '
                             'subconjunto medido por classe (vari-grip=0, '
                             'lightning-rod=2, demais=02). '
                             'Use "all" ou ex. "2".')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='Sigma da suavizacao gaussiana. Use 2 para defeitos pequenos.')
    parser.add_argument('--reflect_pad', type=int, default=112,
                        help='Padding refletido na entrada (tecnica de inferencia).')
    parser.add_argument('--border_margin', type=int, default=0,
                        help='Exclui uma moldura de N pixels das metricas. Mantenha 0.')

    parser.add_argument('--attention', type=str, default=DEFAULT_ATTENTION,
                        choices=['auto', 'se', 'cbam', 'legacy_train'])
    parser.add_argument('--out_size', type=int, default=None)
    parser.add_argument('--clamp_scale', type=float, default=None)
    parser.add_argument('--cflow_cond_dim', type=int, default=64)
    parser.add_argument('--cflow_n_blocks', type=int, default=None)
    parser.add_argument('--cflow_hidden', type=int, default=256)

    parser.add_argument('--sweep', action='store_true',
                        help='Reagrega os mapas ja calculados sob varias configs (sem custo de '
                             'inferencia extra) e grava sweep.csv.')
    parser.add_argument('--sweep_levels', type=str, default='0,2,02,012')
    parser.add_argument('--sweep_sigmas', type=str, default='0,2,4,8')
    parser.add_argument('--sweep_norms', type=str, default='raw,per_level_minmax')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class_list = ([x.strip() for x in args.classes.split(',') if x.strip()]
                  if args.classes else ALL_CLASSES)

    run_dir = args.out_dir or project_path(
        'resultado_analise_final', 'pixel', f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}={v}\n')
        f.write(f'device={DEVICE}\nimg_size={c.img_size}\n')

    print('=' * 70)
    print('  AVALIACAO NIVEL DE PIXEL - CFLOW NF Head')
    print('=' * 70)
    print(f'  Classes: {class_list}')
    print(f'  Receita: {args.score_norm} | pad={args.reflect_pad} | tta={args.tta} | '
          f'levels={args.levels} | sigma={args.sigma} | margin={args.border_margin}')
    print(f'  Saida:   {run_dir}')

    rows, level_rows, sweep_rows = [], [], []
    for class_name in class_list:
        try:
            result = evaluate_class(class_name, args, os.path.join(run_dir, class_name))
        except Exception as err:
            print(f'  ERRO em {class_name}: {err}')
            continue
        if result is None:
            continue
        metrics, per_level, sweep = result
        rows.append(metrics)
        level_rows.extend(per_level)
        sweep_rows.extend(sweep)

    if not rows:
        print('\nNenhuma classe avaliada.')
        return

    df = pd.DataFrame(rows)
    num = df.select_dtypes(include=[np.number]).columns
    mean_row = {col: df[col].mean() for col in num}
    mean_row['class'] = 'MEDIA'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    front = ['class', 'n_images', 'n_anomaly', 'pixel_auroc', 'pixel_ap', 'pixel_f1',
             'aupro', 'pixel_ap_trivial', 'fps', 'latency_ms', 'levels', 'score_norm',
             'tta', 'sigma']
    df = df[[col for col in front if col in df.columns]
            + [col for col in df.columns if col not in front]]

    out_csv = os.path.join(run_dir, 'results_per_class.csv')
    df.to_csv(out_csv, index=False)
    pd.DataFrame(level_rows).to_csv(os.path.join(run_dir, 'per_level_auroc.csv'), index=False)
    if sweep_rows:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(os.path.join(run_dir, 'sweep.csv'), index=False)

    print('\n' + '=' * 100)
    print('  RESULTADOS - NIVEL DE PIXEL')
    print('=' * 100)
    show = [col for col in ['class', 'pixel_auroc', 'pixel_ap', 'pixel_f1', 'aupro',
                            'pixel_ap_trivial', 'fps'] if col in df.columns]
    print(df[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    if level_rows:
        print('\n  AUROC por nivel isolado:')
        piv = pd.DataFrame(level_rows).pivot_table(index='class', columns='level',
                                                   values='pixel_auroc')
        print(piv.to_string(float_format=lambda v: f'{v:.4f}'))

    if sweep_rows:
        print('\n  Sweep - melhor por classe:')
        best = sweep_df.loc[sweep_df.groupby('class')['pixel_auroc'].idxmax()]
        print(best[['class', 'levels', 'sigma', 'score_norm', 'pixel_auroc', 'pixel_ap']]
              .to_string(index=False, float_format=lambda v: f'{v:.4f}'))

    print(f'\nCSV: {out_csv}')


if __name__ == '__main__':
    main()
