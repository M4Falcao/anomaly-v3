"""Etapa 8 — relatório: métricas val/holdout, comparação pareada com a referência COM FUNDO.

Heads avaliadas por classe:
  * ``none`` (referência com fundo): head já treinada em ``--reference_run`` (default
    ``treino_nf_heads/run_20261025_181523``), sobre o dataset original;
  * braço A / braço B: heads de ``<run>/<classe>/heads_index.json`` com ``split_check=ok``.

Por head e partição (val, holdout):
  * Pixel AUROC, Pixel AP, AUPRO global (``compute_aupro_cflow``), Image AUROC do mapa;
  * ``pixel_auroc_in_object``: AUROC só dentro de uma região comum a todas as heads
    (máscara base dilatada ``--object_region``, default ``dil32``);
  * ``gt_lost_frac`` e ``lost_gt_auroc``: GT fora da máscara da head e se ele ainda é
    detectável contra todos os pixels normais;
  * ``contour_fpr`` / ``offcontour_fpr``: FPR em pixels normais na faixa de ±8 px da borda
    da máscara × fora dela, no limiar com FPR global de 1% na validação;
  * deltas contra ``none`` com bootstrap pareado estratificado por imagem (IC95).

Nível de imagem do SEDifferNet (``--encoder_views`` rotações fixas, default 64; cache em
``<classe>/eval/encoder_scores/``):
  * ``ref``: encoder com fundo, imagem original (a referência);
  * braço A: o **mesmo** encoder com fundo recebendo a imagem mascarada só na inferência.
    É um diagnóstico sem treino: mostra se a máscara sozinha derruba o score de imagem
    (``--no_encoder_probe`` desliga);
  * braço B: encoder retreinado com a imagem mascarada.
  Colunas ``<part>_encoder_image_auroc`` e ``<part>_delta_encoder_image_auroc`` com IC95 pareado.
  Não entram no critério de seleção.

Critério pré-registrado (D-19), avaliado **só na validação**:
``gt_lost_frac <= 0.05``, ``Δ Pixel AUROC >= 0`` com IC95 superior >= 0, e
``Δ AUPRO >= -0.02``. Entre as aprovadas, a recomendada é a de maior Pixel AUROC de
validação. O holdout só confirma ou desmente.

Saída: ``<run>/summary.csv``, ``<run>/summary.md``, ``<run>/<classe>/eval/<head>/metrics.json``,
``<run>/<classe>/report_<classe>.png``.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402
import evaluate as ev  # noqa: E402
import heads as hd  # noqa: E402
import masking as mk  # noqa: E402

PARTS = ('val', 'holdout')
CRITERIA = dict(max_gt_lost=0.05, min_delta_auroc=0.0, min_delta_aupro=-0.02)


def reference_head(class_name, reference_run):
    from core.eval_pipeline import find_nf_head_checkpoint
    ckpt = find_nf_head_checkpoint(class_name, base_dir=reference_run)
    if ckpt is None:
        raise FileNotFoundError(f'Head de referência ausente para {class_name} em {reference_run}')
    se = None
    summary = os.path.join(reference_run, 'training_summary.csv')
    if os.path.isfile(summary):
        with open(summary, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('class') == class_name and row.get('se_checkpoint'):
                    se = row['se_checkpoint']
    return dict(arm='ref', variant='none', role='reference_bg', checkpoint=ckpt,
                se_checkpoint=se or hd.default_se_checkpoint(class_name), status='SUCCESS', split_check='ref')


def reference_split_ok(ckpt, class_name, val_paths):
    payload = ev.load_head_payload(ckpt)
    sel = {hd._class_relative(p, class_name) for p in payload.get('selection_validation_paths', [])}
    return bool(sel) and sel == set(val_paths)


def head_id(h):
    return f'{h["arm"]}_{h["variant"]}'


def cached_encoder_scores(key, cache_dir, class_name, indices, args):
    """Score DifferNet por imagem para (encoder, dataset), com cache em disco."""
    import hashlib
    se_ckpt, dataset_root = key
    tag = hashlib.sha1(f'{os.path.abspath(se_ckpt)}|{os.path.abspath(dataset_root)}|{args.encoder_views}|'
                       f'{list(indices)}'.encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f'{tag}.npz')
    if os.path.isfile(path):
        data = np.load(path)
        return data['scores'], data['labels']
    print(f'encoder {os.path.basename(se_ckpt)} em {os.path.basename(os.path.normpath(dataset_root))} '
          f'({args.encoder_views} rotações, {len(indices)} imagens)')
    scores, labels = ev.encoder_image_scores(se_ckpt, dataset_root, class_name, args.encoder_views, indices)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, scores=scores, labels=labels)
    cm.write_json(os.path.join(cache_dir, f'{tag}.json'), dict(se_checkpoint=se_ckpt, dataset_root=dataset_root,
                                                              views=args.encoder_views, n_images=len(indices)))
    return scores, labels


def evaluate_head(h, class_name, class_dir, split, base_grids, region_strategy, args):
    out_dir = os.path.join(class_dir, 'eval' + ('' if not args.limit else f'_limit{args.limit}'), head_id(h))
    metrics_path = os.path.join(out_dir, 'metrics.json')
    arrays_path = os.path.join(out_dir, 'per_image.npz')
    if args.skip_existing and os.path.isfile(metrics_path) and os.path.isfile(arrays_path):
        arrays = dict(np.load(arrays_path))
        return cm.read_json(metrics_path), arrays

    dataset_root = h.get('dataset_root') or args.orig_root
    res = ev.compute_maps(h['checkpoint'], h['se_checkpoint'], dataset_root, class_name, out_dir,
                          indices=split['indices'], desc=f'{class_name} {head_id(h)}')
    n = len(res['paths'])
    if res['paths'] != split['paths']:
        raise ValueError(f'{head_id(h)}: ordem do teste difere do audit; comparação pareada inválida.')
    maps, masks = res['maps'], res['masks']
    part = split['partition']
    strategy = mk.parse_variant(h['variant'])['strategy']

    rng = np.random.default_rng(0)
    sample = np.concatenate([np.asarray(maps[i]).ravel()[rng.integers(0, maps[i].size, 4096)] for i in range(n)])
    edges = ev.quantile_edges(sample, ev.FINE_BINS)
    coarse = edges[ev.COARSE_FACTOR - 1::ev.COARSE_FACTOR]
    nb = len(edges) + 1
    acc = {p: {k: np.zeros(nb, np.int64) for k in ('pos', 'neg', 'obj_pos', 'obj_neg', 'lost_pos', 'band_neg', 'off_neg')}
           for p in PARTS}
    gt_px = {p: 0 for p in PARTS}
    lost_px = {p: 0 for p in PARTS}
    pos_c = np.zeros((n, len(coarse) + 1), np.int64)
    neg_c = np.zeros((n, len(coarse) + 1), np.int64)
    for i in range(n):
        p = part[i]
        s = np.asarray(maps[i])
        gt = np.asarray(masks[i]).astype(bool)
        region = mk.strategy_mask(base_grids[i], region_strategy)
        hmask = mk.strategy_mask(base_grids[i], strategy) if strategy else None
        band = mk.contour_band(hmask if hmask is not None else mk.strategy_mask(base_grids[i], 'raw'), args.band_px)
        a = acc[p]
        a['pos'] += ev.bincount(s[gt], edges)
        a['neg'] += ev.bincount(s[~gt], edges)
        a['obj_pos'] += ev.bincount(s[gt & region], edges)
        a['obj_neg'] += ev.bincount(s[~gt & region], edges)
        a['band_neg'] += ev.bincount(s[~gt & band], edges)
        a['off_neg'] += ev.bincount(s[~gt & ~band], edges)
        if gt.any():
            gt_px[p] += int(gt.sum())
            if hmask is not None:
                lost = gt & ~hmask
                lost_px[p] += int(lost.sum())
                a['lost_pos'] += ev.bincount(s[lost], edges)
        pos_c[i] = ev.bincount(s[gt], coarse)
        neg_c[i] = ev.bincount(s[~gt], coarse)

    from core.eval_pipeline import compute_aupro_cflow
    from sklearn.metrics import roc_auc_score
    thr = ev.threshold_at_fpr(acc['val']['neg'], edges, args.contour_fpr)
    metrics = dict(arm=h['arm'], variant=h['variant'], role=h['role'], checkpoint=h['checkpoint'],
                   se_checkpoint=h['se_checkpoint'], dataset_root=dataset_root, recipe=res['recipe'],
                   stats_source=res['stats_source'], n_images=n, contour_threshold=thr,
                   encoder_selection=h.get('encoder_selection'))
    labels = res['labels']
    for p in PARTS:
        idx = np.flatnonzero(part == p)
        a = acc[p]
        m = {}
        m['pixel_auroc'] = ev.auroc_from_hist(a['pos'], a['neg'])
        m['pixel_ap'] = ev.ap_from_hist(a['pos'], a['neg'])
        m['pixel_auroc_in_object'] = ev.auroc_from_hist(a['obj_pos'], a['obj_neg'])
        m['lost_gt_auroc'] = ev.auroc_from_hist(a['lost_pos'], a['neg']) if strategy else float('nan')
        m['gt_lost_frac'] = lost_px[p] / gt_px[p] if (strategy and gt_px[p]) else (0.0 if gt_px[p] else float('nan'))
        m['contour_fpr'] = ev.fpr_at(a['band_neg'], edges, thr)
        m['offcontour_fpr'] = ev.fpr_at(a['off_neg'], edges, thr)
        m['image_auroc'] = (float(roc_auc_score(labels[idx], res['image_scores'][idx]))
                            if len(np.unique(labels[idx])) == 2 else float('nan'))
        m['aupro'] = float('nan')
        if not args.no_aupro and labels[idx].any():
            m['aupro'] = compute_aupro_cflow(np.asarray(maps[idx]), np.asarray(masks[idx]))
        metrics[p] = m
    cm.write_json(metrics_path, metrics)
    arrays = dict(pos_c=pos_c, neg_c=neg_c, image_scores=res['image_scores'], labels=labels)
    np.savez_compressed(arrays_path, **arrays)
    del maps, masks, res
    if not args.keep_maps:
        for f in ('maps.npy', 'masks.npy'):
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass
    return metrics, arrays


def recommend(rows, arm):
    ok = []
    for r in rows:
        if r['arm'] != arm or r['role'] != 'candidate':
            continue
        checks = dict(
            gt_lost=r['val_gt_lost_frac'] <= CRITERIA['max_gt_lost'],
            delta=r['val_delta_pixel_auroc'] >= CRITERIA['min_delta_auroc'] and r['val_delta_ci_high'] >= 0,
            aupro=(not math.isfinite(r['val_delta_aupro'])) or r['val_delta_aupro'] >= CRITERIA['min_delta_aupro'])
        r[f'passes_criteria'] = all(checks.values())
        r['failed_checks'] = ','.join(k for k, v in checks.items() if not v)
        if r['passes_criteria']:
            ok.append(r)
    if ok:
        best = max(ok, key=lambda r: r['val_pixel_auroc'])
        best['recommended'] = True
        return best
    return None


def plot_class(rows, path, class_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f'{r["arm"]}:{r["variant"]}' for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, ('pixel_auroc', 'aupro', 'pixel_auroc_in_object')):
        for k, p in enumerate(PARTS):
            ax.bar(x + (k - 0.5) * 0.38, [r[f'{p}_{metric}'] for r in rows], 0.38, label=p)
        ref = next((r for r in rows if r['variant'] == 'none'), None)
        if ref:
            ax.axhline(ref[f'val_{metric}'], ls='--', c='gray', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(metric)
        finite = [v for r in rows for p in PARTS for v in [r[f'{p}_{metric}']] if math.isfinite(v)]
        if finite:
            ax.set_ylim(max(0, min(finite) - 0.05), min(1, max(finite) + 0.03))
    axes[0].legend(fontsize=8)
    fig.suptitle(f'{class_name}: braço/variante (linha = referência com fundo, val)')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report_class(class_name, run_dir, run_cfg, args):
    class_dir = os.path.join(run_dir, class_name)
    audit = cm.read_json(os.path.join(class_dir, 'audit', 'split_manifest.json'))
    masks_dir = cm.read_json(os.path.join(class_dir, 'audit', 'audit_config.json'))['masks_dir']
    split = dict(paths=[s['path'] for s in audit['samples']],
                 partition=np.array([s['partition'] for s in audit['samples']], dtype=object))
    val_paths = [p for p, q in zip(split['paths'], split['partition']) if q == 'val']
    labels_all = [s['label'] for s in audit['samples']]
    indices = (ev.stratified_subset(labels_all, split['partition'], args.limit) if args.limit
               else list(range(len(split['paths']))))
    split = dict(indices=indices, paths=[split['paths'][i] for i in indices], partition=split['partition'][indices])

    ref = reference_head(class_name, args.reference_run)
    ref['reference_split_ok'] = reference_split_ok(ref['checkpoint'], class_name, val_paths)
    if not ref['reference_split_ok']:
        print(f'AVISO {class_name}: a head de referência não registrou a mesma validação; '
              f'o holdout dela pode não ser limpo.')
    index_heads = [h for h in hd.load_index(class_dir)['heads'] if h.get('status') == 'SUCCESS']
    bad = [h for h in index_heads if h.get('split_check') != 'ok']
    for h in bad:
        print(f'{class_name}: {head_id(h)} recusada (split_check={h.get("split_check")}).')
    candidates = [h for h in index_heads if h.get('split_check') == 'ok']
    if args.arms:
        candidates = [h for h in candidates if h['arm'] in args.arms]
    heads = [ref] + candidates
    if args.dry_run:
        for h in heads:
            print(f'[dry-run] {class_name}: avaliar {head_id(h)} | head={h["checkpoint"]} | encoder={h["se_checkpoint"]} '
                  f'| dataset={h.get("dataset_root") or args.orig_root}')
        return []

    base_grids = [mk.to_grid(mk.load_base_mask(masks_dir, p)) for p in split['paths']]
    results = {}
    for h in heads:
        print(f'\n--- {class_name}: {head_id(h)} ---')
        results[head_id(h)] = evaluate_head(h, class_name, class_dir, split, base_grids, args.object_region, args)

    encoder_scores = {}
    probe_arms = {'ref', 'B'} | ({'A'} if args.encoder_probe else set())
    if args.encoder_views > 0 and any(h['arm'] in probe_arms - {'ref'} for h in heads):
        cache_dir = os.path.join(class_dir, 'eval' + ('' if not args.limit else f'_limit{args.limit}'), 'encoder_scores')
        for h in heads:
            if h['arm'] in probe_arms:
                key = (h['se_checkpoint'], h.get('dataset_root') or args.orig_root)
                if key not in encoder_scores:
                    encoder_scores[key] = cached_encoder_scores(key, cache_dir, class_name, split['indices'], args)

    ref_m, ref_a = results[head_id(ref)]
    part = split['partition']
    rows = []
    for h in heads:
        m, a = results[head_id(h)]
        row = dict(class_name=class_name, arm=h['arm'], variant=h['variant'], role=h['role'],
                   recommended=False, passes_criteria='', failed_checks='',
                   reference_split_ok=ref['reference_split_ok'], encoder_selection=h.get('encoder_selection') or '')
        enc = None
        if encoder_scores and h['arm'] in probe_arms:
            enc = encoder_scores.get((h['se_checkpoint'], h.get('dataset_root') or args.orig_root))
        ref_enc = encoder_scores.get((ref['se_checkpoint'], args.orig_root)) if encoder_scores else None
        for p in PARTS:
            for k, v in m[p].items():
                row[f'{p}_{k}'] = v
            idx = np.flatnonzero(part == p)
            row[f'{p}_delta_pixel_auroc'] = m[p]['pixel_auroc'] - ref_m[p]['pixel_auroc']
            row[f'{p}_delta_aupro'] = m[p]['aupro'] - ref_m[p]['aupro']
            row[f'{p}_delta_image_auroc'] = m[p]['image_auroc'] - ref_m[p]['image_auroc']
            if h is ref:
                boot = dict(pixel=dict(ci_low=0.0, ci_high=0.0, p_le0=float('nan')), image=None)
            else:
                boot = ev.paired_bootstrap(ref_a['pos_c'][idx], ref_a['neg_c'][idx], a['pos_c'][idx], a['neg_c'][idx],
                                           a['labels'][idx], n_boot=args.n_boot, seed=args.seed,
                                           ref_img=ref_a['image_scores'][idx], cand_img=a['image_scores'][idx])
            row[f'{p}_delta_ci_low'] = boot['pixel']['ci_low']
            row[f'{p}_delta_ci_high'] = boot['pixel']['ci_high']
            row[f'{p}_delta_p_le0'] = boot['pixel']['p_le0']
            img = boot.get('image') or {}
            row[f'{p}_delta_image_ci_low'] = img.get('ci_low', float('nan'))
            row[f'{p}_delta_image_ci_high'] = img.get('ci_high', float('nan'))
            row[f'{p}_encoder_image_auroc'] = float('nan')
            for k in ('delta_encoder_image_auroc', 'delta_encoder_ci_low', 'delta_encoder_ci_high'):
                row[f'{p}_{k}'] = float('nan')
            if enc is not None:
                from sklearn.metrics import roc_auc_score
                sc, lb = enc
                row[f'{p}_encoder_image_auroc'] = (float(roc_auc_score(lb[idx], sc[idx]))
                                                   if len(np.unique(lb[idx])) == 2 else float('nan'))
                if ref_enc is not None and h is not ref:
                    ref_auc = (float(roc_auc_score(ref_enc[1][idx], ref_enc[0][idx]))
                               if len(np.unique(lb[idx])) == 2 else float('nan'))
                    row[f'{p}_delta_encoder_image_auroc'] = row[f'{p}_encoder_image_auroc'] - ref_auc
                    eb = ev.image_auroc_bootstrap(lb[idx], ref_enc[0][idx], sc[idx], n_boot=args.n_boot, seed=args.seed)
                    row[f'{p}_delta_encoder_ci_low'] = eb['ci_low']
                    row[f'{p}_delta_encoder_ci_high'] = eb['ci_high']
        rows.append(row)
    rec_a = recommend(rows, 'A')
    rec_b = recommend(rows, 'B')
    plot_class(rows, os.path.join(class_dir, f'report_{class_name}.png'), class_name)
    print(f'{class_name}: recomendada braço A = {rec_a["variant"] if rec_a else "nenhuma"}; '
          f'braço B = {rec_b["variant"] if rec_b else "nenhuma"}')
    return rows


def fmt(v, nd=4):
    if isinstance(v, float):
        return 'NaN' if not math.isfinite(v) else f'{v:.{nd}f}'
    return str(v)


def write_markdown(rows, path, run_dir, args):
    lines = ['# Ablação de máscara de objeto (D-19) — relatório', '',
             f'Rodada: `{run_dir}`  ', f'Referência com fundo: `{args.reference_run}`  ',
             f'Gerado em {cm.timestamp()}. Uma seed (D-13). O teste já foi inspecionado; o holdout **não** é '
             f'independente (D-18). Métricas de pixel por histograma ({ev.FINE_BINS} bins); IC95 = bootstrap '
             f'pareado estratificado por imagem ({args.n_boot} reamostragens, {ev.FINE_BINS // ev.COARSE_FACTOR} bins).', '',
             '**Critério (só validação):** GT perdido ≤ 5%, ΔPixel AUROC ≥ 0 com IC95 superior ≥ 0, ΔAUPRO ≥ −0.02. '
             'A recomendada é a aprovada com maior Pixel AUROC de validação; o holdout só confirma.', '']
    for class_name in dict.fromkeys(r['class_name'] for r in rows):
        lines += [f'## {class_name}', '',
                  '| braço | variante | papel | val AUROC | Δ val [IC95] | val AUPRO | ΔAUPRO val | GT perdido val | '
                  'AUROC no objeto val | contorno/fora FPR val | holdout AUROC | Δ holdout [IC95] | holdout AUPRO | '
                  'Img AUROC holdout | encoder img AUROC holdout | critério |',
                  '|' + '---|' * 16]
        for r in (r for r in rows if r['class_name'] == class_name):
            crit = '★ recomendada' if r['recommended'] else ('ok' if r['passes_criteria'] is True else
                                                            (f'falhou: {r["failed_checks"]}' if r['failed_checks'] else '—'))
            lines.append(
                f'| {r["arm"]} | `{r["variant"]}` | {r["role"]} | {fmt(r["val_pixel_auroc"])} | '
                f'{fmt(r["val_delta_pixel_auroc"])} [{fmt(r["val_delta_ci_low"])}, {fmt(r["val_delta_ci_high"])}] | '
                f'{fmt(r["val_aupro"])} | {fmt(r["val_delta_aupro"])} | {fmt(r["val_gt_lost_frac"], 3)} | '
                f'{fmt(r["val_pixel_auroc_in_object"])} | {fmt(r["val_contour_fpr"], 3)}/{fmt(r["val_offcontour_fpr"], 3)} | '
                f'{fmt(r["holdout_pixel_auroc"])} | {fmt(r["holdout_delta_pixel_auroc"])} '
                f'[{fmt(r["holdout_delta_ci_low"])}, {fmt(r["holdout_delta_ci_high"])}] | {fmt(r["holdout_aupro"])} | '
                f'{fmt(r["holdout_image_auroc"])} | {fmt(r["holdout_encoder_image_auroc"])} | {crit} |')
        if any(not r['reference_split_ok'] for r in rows if r['class_name'] == class_name):
            lines += ['', '> A head de referência não registrou a mesma validação: holdout dela não comprovadamente limpo.']
        if any(r['encoder_selection'] == 'test_full' for r in rows if r['class_name'] == class_name):
            lines += ['', '> Braço B: encoder escolhido por Image AUROC no teste inteiro (mesmo protocolo dos encoders '
                          'de referência); métricas de imagem no holdout não são independentes.']
        enc_rows = [r for r in rows if r['class_name'] == class_name
                    and math.isfinite(r.get('val_encoder_image_auroc', float('nan')))]
        if enc_rows:
            lines += ['', '### Nível de imagem — SEDifferNet (64 rotações)', '',
                      'Braço A = encoder **com fundo** congelado recebendo a imagem mascarada só na inferência '
                      '(diagnóstico: a máscara sozinha derruba o score de imagem?). Braço B = encoder retreinado com máscara.',
                      '', '| braço | variante | encoder | val Img AUROC | Δ val [IC95] | holdout Img AUROC | Δ holdout [IC95] |',
                      '|---|---|---|---|---|---|---|']
            for r in enc_rows:
                enc_kind = {'ref': 'com fundo / imagem original', 'A': 'com fundo / imagem mascarada',
                            'B': 'retreinado / imagem mascarada'}[r['arm']]
                lines.append(
                    f'| {r["arm"]} | `{r["variant"]}` | {enc_kind} | {fmt(r["val_encoder_image_auroc"])} | '
                    f'{fmt(r["val_delta_encoder_image_auroc"])} [{fmt(r["val_delta_encoder_ci_low"])}, '
                    f'{fmt(r["val_delta_encoder_ci_high"])}] | {fmt(r["holdout_encoder_image_auroc"])} | '
                    f'{fmt(r["holdout_delta_encoder_image_auroc"])} [{fmt(r["holdout_delta_encoder_ci_low"])}, '
                    f'{fmt(r["holdout_delta_encoder_ci_high"])}] |')
        lines += ['', f'![{class_name}]({class_name}/report_{class_name}.png)', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def typed_csv_row(row):
    """Linha lida do ``summary.csv`` com números/booleanos de volta aos tipos de ``report_class``."""
    out = {}
    for k, v in row.items():
        if v in ('True', 'False'):
            out[k] = v == 'True'
        elif k in ('class_name', 'arm', 'variant', 'role', 'failed_checks', 'encoder_selection'):
            out[k] = v
        else:
            try:
                out[k] = float(v) if v != '' else float('nan')
            except ValueError:
                out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--run_dir', default='latest')
    p.add_argument('--reference_run', default=cm.REFERENCE_HEAD_RUN)
    p.add_argument('--arms', default=None, help='Filtra braços (ex.: A ou A,B). Default: todos no índice.')
    p.add_argument('--object_region', default='dil32')
    p.add_argument('--band_px', type=int, default=8)
    p.add_argument('--contour_fpr', type=float, default=0.01)
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--encoder_views', type=int, default=64,
                   help='Rotações fixas para o Image AUROC do SEDifferNet (0 desliga todo diagnóstico de encoder).')
    p.add_argument('--no_encoder_probe', dest='encoder_probe', action='store_false',
                   help='Não medir o SEDifferNet de referência com a entrada mascarada do braço A.')
    p.add_argument('--no_aupro', action='store_true')
    p.add_argument('--keep_maps', action='store_true')
    p.add_argument('--skip_existing', action='store_true', help='Reaproveita metrics.json/per_image.npz já calculados.')
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke test: subconjunto estratificado (partição × rótulo) de até N imagens.')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    args.arms = [a.strip() for a in args.arms.split(',')] if args.arms else None
    mk.validate_strategy(args.object_region)
    run_dir = cm.resolve_run_dir(args.run_dir)
    run_cfg = cm.read_json(os.path.join(run_dir, 'run_config.json'))
    args.orig_root = run_cfg['orig_root']

    rows = []
    for class_name in cm.parse_classes(args.classes):
        if not os.path.isfile(os.path.join(run_dir, class_name, 'audit', 'split_manifest.json')):
            print(f'{class_name}: sem audit; pulei.')
            continue
        rows += report_class(class_name, run_dir, run_cfg, args)
    if args.dry_run or not rows:
        return
    suffix = '' if not args.limit else f'_limit{args.limit}'
    csv_path = os.path.join(run_dir, f'summary{suffix}.csv')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    existing = []
    if os.path.isfile(csv_path) and not args.limit:
        with open(csv_path, encoding='utf-8') as f:
            done = {r['class_name'] for r in rows}
            existing = [r for r in csv.DictReader(f) if r['class_name'] not in done]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing + rows)
    write_markdown([typed_csv_row(r) for r in existing] + rows, os.path.join(run_dir, f'summary{suffix}.md'), run_dir, args)
    print(f'\nRelatório: {csv_path}')


if __name__ == '__main__':
    main()
