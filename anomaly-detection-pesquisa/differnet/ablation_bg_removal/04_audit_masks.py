"""Etapa 4 — audit das estratégias de máscara (sem treino).

Para cada classe e estratégia (``raw``, ``dil<r>``, ``hull``, ``hull+dil<r>``, ``bbox<m>``) mede,
no grid 448×448 em que o SEDifferNet vê a imagem:
  * ``area_kept``: fração média da imagem mantida (quanto fundo sobra);
  * ``gt_lost_frac``: fração dos pixels de defeito (GT) **fora** da máscara;
  * ``n_lost50`` / ``n_lost95``: imagens anômalas com mais de 50% / 95% do GT apagado;
  * ``n_full_mask``: imagens sem objeto detectado (máscara cheia = fundo mantido).

As partições usam o mesmo split do treinador da NF head (``val_fraction`` 0.4, ``split_seed``
42). A regra de poda (etapa 5) olha **só a validação**; o holdout fica registrado para
auditoria, mas não decide nada.

Saída: ``<run>/<classe>/audit/{audit_summary.csv, audit_per_image.csv, split_manifest.json,
audit_config.json, audit_tradeoff.png, audit_examples.png}``.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
import masking as mk  # noqa: E402


def load_gt_grid(gt_path):
    if gt_path is None:
        return np.zeros((mk.GRID, mk.GRID), bool)
    data = np.fromfile(gt_path, dtype=np.uint8)
    gt = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(gt, (mk.GRID, mk.GRID), interpolation=cv2.INTER_NEAREST) > 127


def init_run(args):
    if args.run_dir:
        run_dir = cm.resolve_run_dir(args.run_dir)
    else:
        run_dir = cm.new_ablation_run()
    cfg_path = os.path.join(run_dir, 'run_config.json')
    cfg = dict(orig_root=os.path.abspath(args.orig_root), val_fraction=args.val_fraction,
               split_seed=args.split_seed, mask_source=args.mask_source)
    if os.path.isfile(cfg_path):
        found = cm.read_json(cfg_path)
        diff = {k: (found.get(k), v) for k, v in cfg.items() if found.get(k) != v}
        if diff:
            raise ValueError(f'run_config.json de {run_dir} difere: {diff}')
    else:
        cm.write_json(cfg_path, {**cfg, 'created_at': cm.timestamp(), 'decision': 'D-19'})
    return run_dir


def audit_class(class_name, run_dir, args):
    import protocol as pt

    masks_dir = mk.resolve_masks_dir(class_name, args.masks_dir, args.mask_source)
    out = os.path.join(run_dir, class_name, 'audit')
    os.makedirs(out, exist_ok=True)
    inventory = pt.test_inventory(args.orig_root, class_name)
    labels = [lab for _, lab, _ in inventory]
    val, holdout = pt.split_indices(labels, args.val_fraction, args.split_seed)
    part = pt.partition_of(len(inventory), val, holdout)
    cm.write_json(os.path.join(out, 'split_manifest.json'), dict(
        class_name=class_name, val_fraction=args.val_fraction, split_seed=args.split_seed,
        samples=[dict(path=rel.replace(os.sep, '/'), label=lab, partition=str(p))
                 for (rel, lab, _), p in zip(inventory, part)]))

    rng = np.random.default_rng(0)
    train_dir = os.path.join(args.orig_root, class_name, 'train', 'good')
    train_files = cm.list_images(train_dir)
    if args.train_sample and len(train_files) > args.train_sample:
        train_files = sorted(rng.choice(train_files, args.train_sample, replace=False).tolist())

    items = [(os.path.join('train', 'good', f), 0, None, 'train_good') for f in train_files]
    items += [(rel, lab, gt, str(p)) for (rel, lab, gt), p in zip(inventory, part)]
    if args.limit:
        items = items[:args.limit]

    agg = {}
    per_image = []
    example_pool = []
    for rel, lab, gt_path, partition in items:
        base = mk.to_grid(mk.load_base_mask(masks_dir, rel))
        gt = load_gt_grid(gt_path) if lab else None
        gt_px = int(gt.sum()) if gt is not None else 0
        for strategy in args.strategies:
            m = mk.strategy_mask(base, strategy)
            a = agg.setdefault((strategy, partition), dict(n=0, area=0.0, n_full=0, n_anom=0, gt_px=0,
                                                          lost_px=0, n_lost50=0, n_lost95=0))
            a['n'] += 1
            a['area'] += float(m.mean())
            a['n_full'] += bool(m.all())
            if gt_px:
                lost = int((gt & ~m).sum())
                a['n_anom'] += 1
                a['gt_px'] += gt_px
                a['lost_px'] += lost
                a['n_lost50'] += lost > 0.5 * gt_px
                a['n_lost95'] += lost > 0.95 * gt_px
                per_image.append(dict(partition=partition, path=rel.replace(os.sep, '/'), strategy=strategy,
                                      area_kept=round(float(m.mean()), 5), gt_px=gt_px, lost_px=lost,
                                      lost_frac=round(lost / gt_px, 5)))
        if gt_px and partition == 'val':
            example_pool.append((rel, base, gt, float((gt & ~base).sum() / gt_px)))

    rows = []
    for (strategy, partition), a in sorted(agg.items(), key=lambda kv: (args.strategies.index(kv[0][0]), kv[0][1])):
        rows.append(dict(class_name=class_name, strategy=strategy, partition=partition, n_images=a['n'],
                         area_kept=round(a['area'] / max(a['n'], 1), 5), n_full_mask=a['n_full'],
                         n_anomalous=a['n_anom'], gt_px=a['gt_px'], gt_lost_px=a['lost_px'],
                         gt_lost_frac=round(a['lost_px'] / a['gt_px'], 5) if a['gt_px'] else '',
                         n_lost50=a['n_lost50'], n_lost95=a['n_lost95']))
    fields = list(rows[0].keys())
    with open(os.path.join(out, 'audit_summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if per_image:
        with open(os.path.join(out, 'audit_per_image.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            w.writerows(per_image)
    cm.write_json(os.path.join(out, 'audit_config.json'), dict(
        class_name=class_name, masks_dir=masks_dir, orig_root=os.path.abspath(args.orig_root),
        strategies=args.strategies, train_sample=args.train_sample, limit=args.limit,
        mask_manifest=cm.read_json(os.path.join(masks_dir, 'manifest.json')), created_at=cm.timestamp()))
    plot_tradeoff(rows, os.path.join(out, 'audit_tradeoff.png'), class_name)
    plot_examples(example_pool, args, class_name, os.path.join(out, 'audit_examples.png'))

    print(f'\n{class_name} (máscaras: {os.path.basename(masks_dir)})')
    print(f'   {"estratégia":12s} {"área train":>10s} {"área val":>9s} {"GT perdido val":>15s} {">50%":>5s} {">95%":>5s}')
    by = {(r['strategy'], r['partition']): r for r in rows}
    for s in args.strategies:
        tr, va = by.get((s, 'train_good'), {}), by.get((s, 'val'), {})
        print(f'   {s:12s} {tr.get("area_kept", float("nan")):>10.3f} {va.get("area_kept", float("nan")):>9.3f} '
              f'{str(va.get("gt_lost_frac", "")):>15s} {va.get("n_lost50", ""):>5} {va.get("n_lost95", ""):>5}')
    return rows


def plot_tradeoff(rows, path, class_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    val = [r for r in rows if r['partition'] == 'val' and r['gt_lost_frac'] != '']
    area = {r['strategy']: r['area_kept'] for r in rows if r['partition'] == 'train_good'}
    if not val:
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in val:
        x = area.get(r['strategy'], r['area_kept'])
        ax.scatter(x, r['gt_lost_frac'])
        ax.annotate(r['strategy'], (x, r['gt_lost_frac']), fontsize=8, xytext=(3, 3), textcoords='offset points')
    ax.axhline(0.05, ls='--', c='gray', lw=1, label='limite 5% (regra da etapa 5)')
    ax.set_xlabel('Área mantida (train/good)')
    ax.set_ylabel('GT perdido (validação)')
    ax.set_title(f'{class_name}: fundo removido × defeito preservado')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_examples(pool, args, class_name, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not pool:
        return
    pool = sorted(pool, key=lambda t: -t[3])[:args.examples]
    palette = [plt.cm.tab10(i) for i in range(10) if i != 3] + [plt.cm.Dark2(i) for i in range(8)]
    colors = palette[:len(args.strategies)]
    fig, axes = plt.subplots(1, len(pool), figsize=(3.2 * len(pool), 3.8), squeeze=False)
    for ax, (rel, base, gt, lost) in zip(axes[0], pool):
        img = cm.read_image(os.path.join(args.orig_root, class_name, rel))
        ax.imshow(cv2.resize(img, (mk.GRID, mk.GRID))[..., ::-1])
        ax.contour(gt, levels=[0.5], colors='red', linewidths=1.5)
        for s, col in zip(args.strategies, colors):
            ax.contour(mk.strategy_mask(base, s), levels=[0.5], colors=[col], linewidths=0.8)
        ax.set_title(f'GT perdido raw={lost:.0%}', fontsize=8)
        ax.axis('off')
    handles = [plt.Line2D([], [], color='red', label='GT')] + [
        plt.Line2D([], [], color=col, label=s) for s, col in zip(args.strategies, colors)]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6), fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--run_dir', default=None, help='Rodada existente (ou "latest"). Default: cria run_<ts> novo.')
    p.add_argument('--masks_dir', default=None, help='Pasta de máscaras explícita (só com uma classe).')
    p.add_argument('--mask_source', choices=['yolo', 'nobg'], default='yolo')
    p.add_argument('--orig_root', default=cm.ORIG_ROOT)
    p.add_argument('--strategies', default=','.join(mk.DEFAULT_AUDIT_STRATEGIES))
    p.add_argument('--val_fraction', type=float, default=0.4)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--train_sample', type=int, default=300, help='Imagens train/good usadas para medir área (0 = todas).')
    p.add_argument('--examples', type=int, default=6)
    p.add_argument('--limit', type=int, default=None, help='Smoke test: limita imagens por classe.')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.strategies = [mk.validate_strategy(s.strip()) for s in args.strategies.split(',') if s.strip()]
    if 'raw' not in args.strategies:
        args.strategies.insert(0, 'raw')
    classes = cm.parse_classes(args.classes)
    if args.masks_dir and len(classes) != 1:
        p.error('--masks_dir exige exatamente uma classe.')

    if args.dry_run:
        for class_name in classes:
            try:
                md = mk.resolve_masks_dir(class_name, args.masks_dir, args.mask_source)
            except FileNotFoundError as exc:
                md = f'AUSENTE ({exc})'
            print(f'[dry-run] audit {class_name}: estratégias={args.strategies} máscaras={md} '
                  f'split val={args.val_fraction} seed={args.split_seed} -> {args.run_dir or "novo run_<ts>"}')
        return

    run_dir = init_run(args)
    print(f'Rodada: {run_dir}')
    for class_name in classes:
        audit_class(class_name, run_dir, args)


if __name__ == '__main__':
    main()
