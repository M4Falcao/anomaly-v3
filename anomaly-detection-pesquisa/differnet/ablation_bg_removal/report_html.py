"""Relatório HTML autocontido de uma rodada de ``run_all.py`` (imagens embutidas em base64).

Seções: status por classe e etapa; resumo macro com gráficos; e, por classe, YOLO26, máscaras,
audit, finalistas, nível de pixel (braço A/B × referência com fundo) e nível de imagem
(SEDifferNet com fundo × sem fundo).

Uso isolado: ``python ablation_bg_removal/report_html.py <rodada>``.
"""
import base64
import csv
import glob
import html
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402

CSS = """
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}
header{background:#1f3b57;color:#fff;padding:18px 28px}header h1{margin:0;font-size:22px}
header p{margin:4px 0 0;opacity:.85;font-size:13px}
nav{background:#fff;border-bottom:1px solid #ddd;padding:8px 28px;position:sticky;top:0;z-index:5}
nav a{margin-right:14px;color:#1f3b57;text-decoration:none;font-size:13px}
main{padding:18px 28px;max-width:1500px}
section{background:#fff;border:1px solid #e1e4e8;border-radius:6px;padding:14px 18px;margin-bottom:18px}
h2{margin-top:0;color:#1f3b57}h3{color:#33506e;margin-bottom:6px}
table{border-collapse:collapse;font-size:12.5px;margin:6px 0 12px}
th,td{border:1px solid #dde;padding:4px 8px;text-align:right}th{background:#eef2f6}
td:first-child,th:first-child{text-align:left}
.ok{background:#d9f2df}.skipped{background:#eef2f6}.failed{background:#f8d7da}.none{color:#aaa}
.pos{color:#0a7d2c;font-weight:600}.neg{color:#b3261e;font-weight:600}
.note{background:#fff8e1;border-left:4px solid #f2b600;padding:8px 12px;font-size:13px;margin:8px 0}
.imgs{display:flex;flex-wrap:wrap;gap:10px}.imgs figure{margin:0}.imgs img{max-width:100%;border:1px solid #ddd}
figcaption{font-size:11.5px;color:#555}
details summary{cursor:pointer;color:#1f3b57}
code{background:#f0f2f4;padding:1px 4px;border-radius:3px}
"""


# ─── utilidades ──────────────────────────────────────────────────────────────

def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else float('nan')
    except (TypeError, ValueError):
        return float('nan')


def fmt(v, nd=4, signed=False):
    x = num(v)
    if not math.isfinite(x):
        return '<span class="none">—</span>' if v in (None, '', 'nan', 'NaN') or isinstance(v, float) else html.escape(str(v))
    s = f'{x:+.{nd}f}' if signed else f'{x:.{nd}f}'
    if signed:
        cls = 'pos' if x > 0 else ('neg' if x < 0 else '')
        return f'<span class="{cls}">{s}</span>'
    return s


def ci(lo, hi, nd=3):
    lo, hi = num(lo), num(hi)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return ''
    return f' <small>[{lo:+.{nd}f}, {hi:+.{nd}f}]</small>'


def img_tag(path, caption='', width=None):
    if not path or not os.path.isfile(path):
        return ''
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    mime = 'jpeg' if ext in ('jpg', 'jpeg') else ext
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    style = f' style="max-width:{width}px"' if width else ''
    cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ''
    return f'<figure><img src="data:image/{mime};base64,{data}"{style}>{cap}</figure>'


def fig_tag(fig, caption=''):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode()
    cap = f'<figcaption>{html.escape(caption)}</figcaption>' if caption else ''
    return f'<figure><img src="data:image/png;base64,{data}">{cap}</figure>'


def table(headers, rows):
    out = ['<table><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr>']
    for r in rows:
        out.append('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>')
    return '\n'.join(out) + '</table>'


# ─── coleta ──────────────────────────────────────────────────────────────────

def yolo_info(cfg, class_name):
    cands = glob.glob(os.path.join(cfg['yolo_runs_dir'], 'yolo_seg', f'{class_name}_*', 'weights', 'best.pt'))
    if not cands:
        return None
    run = os.path.dirname(os.path.dirname(max(cands, key=os.path.getmtime)))
    rows = read_csv(os.path.join(run, 'results.csv'))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    best = None
    if rows:
        best = max(rows, key=lambda r: num(r.get('metrics/mAP50-95(M)')))
    meta = os.path.join(run, 'train_meta.json')
    return dict(run=run, rows=rows, best=best, meta=cm.read_json(meta) if os.path.isfile(meta) else {})


def masks_info(class_dir):
    acfg = os.path.join(class_dir, 'audit', 'audit_config.json')
    if not os.path.isfile(acfg):
        return None
    mdir = cm.read_json(acfg)['masks_dir']
    rows = read_csv(os.path.join(mdir, 'masks_report.csv'))
    return dict(dir=mdir, rows=rows, manifest=cm.read_json(os.path.join(mdir, 'manifest.json'))
                if os.path.isfile(os.path.join(mdir, 'manifest.json')) else {})


# ─── seções ──────────────────────────────────────────────────────────────────

def section_status(status, classes):
    stages = ['convert', 'yolo', 'masks', 'audit', 'select', 'frozen', 'retrain', 'report', 'image']
    rows = []
    for c in classes:
        st = status.get('classes', {}).get(c, {})
        cells = [c]
        for s in stages:
            e = st.get(s)
            if not e:
                cells.append('<span class="none">—</span>')
                continue
            label = {'ok': f'ok ({e.get("minutes", 0):.0f} min)', 'skipped': 'reaproveitado',
                     'failed': 'FALHOU'}.get(e['state'], e['state'])
            title = html.escape(e.get('reason') or e.get('error') or '')
            cells.append(f'<span class="{e["state"]}" title="{title}">{label}</span>')
        rows.append(cells)
    out = '<section id="status"><h2>Status do pipeline</h2>' + table(['classe'] + stages, rows)
    fails = [(c, s, e) for c in classes for s, e in status.get('classes', {}).get(c, {}).items()
             if e.get('state') == 'failed']
    if fails:
        out += '<h3>Falhas</h3>' + ''.join(
            f'<details><summary>{html.escape(c)} / {s}: {html.escape(e.get("error", ""))[:200]}</summary>'
            f'<pre>{html.escape(e.get("traceback", ""))}</pre><p>log: <code>{html.escape(e.get("log", ""))}</code></p>'
            '</details>' for c, s, e in fails)
    return out + '</section>'


def section_macro(run_dir, classes, pixel_rows, image_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parts = ['<section id="macro"><h2>Resumo macro</h2>',
             '<div class="note">Uma seed (D-13). O teste já foi inspecionado; o holdout não é independente (D-18). '
             'Seleção (critério pré-registrado) só na validação. Os encoders SEDifferNet de <code>final_models</code> '
             'escolheram a época no próprio teste: AUROCs de imagem absolutos são otimistas.</div>']

    # pixel: Δ val por classe × variante
    cand = [r for r in pixel_rows if r.get('arm') in ('A', 'B')]
    if cand:
        variants = list(dict.fromkeys(f'{r["arm"]}:{r["variant"]}' for r in cand))
        cls = [c for c in classes if any(r['class_name'] == c for r in cand)]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
        width = 0.8 / max(len(variants), 1)
        for ax, part in zip(axes, ('val', 'holdout')):
            for k, v in enumerate(variants):
                ys, lo, hi = [], [], []
                for c in cls:
                    r = next((r for r in cand if r['class_name'] == c and f'{r["arm"]}:{r["variant"]}' == v), None)
                    d = num(r[f'{part}_delta_pixel_auroc']) if r else float('nan')
                    ys.append(d)
                    lo.append(d - num(r[f'{part}_delta_ci_low']) if r else 0)
                    hi.append(num(r[f'{part}_delta_ci_high']) - d if r else 0)
                x = np.arange(len(cls)) + (k - (len(variants) - 1) / 2) * width
                ax.bar(x, ys, width, label=v, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], capsize=2)
            ax.axhline(0, c='k', lw=0.8)
            ax.set_xticks(np.arange(len(cls)))
            ax.set_xticklabels([c.split('-')[0] for c in cls])
            ax.set_title(f'Δ Pixel AUROC vs com fundo — {part}')
        axes[0].legend(fontsize=7)
        parts.append('<div class="imgs">' + fig_tag(fig, 'Barras = Δ (variante − head com fundo); erro = IC95 pareado.') + '</div>')

        rows = []
        for c in cls:
            for r in [r for r in pixel_rows if r['class_name'] == c]:
                crit = '★ recomendada' if r.get('recommended') == 'True' else (
                    'ok' if r.get('passes_criteria') == 'True' else (f'falhou: {r["failed_checks"]}' if r.get('failed_checks') else '—'))
                rows.append([c, r['arm'], f'<code>{r["variant"]}</code>', fmt(r['val_pixel_auroc']),
                             fmt(r['val_delta_pixel_auroc'], signed=True) + ci(r['val_delta_ci_low'], r['val_delta_ci_high']),
                             fmt(r['val_aupro']), fmt(r['val_delta_aupro'], signed=True), fmt(r['val_gt_lost_frac'], 3),
                             fmt(r['holdout_pixel_auroc']), fmt(r['holdout_delta_pixel_auroc'], signed=True), crit])
        parts.append('<h3>Nível de pixel — todas as classes</h3>' + table(
            ['classe', 'braço', 'variante', 'val AUROC', 'Δ val [IC95]', 'val AUPRO', 'ΔAUPRO val', 'GT perdido val',
             'holdout AUROC', 'Δ holdout', 'critério'], rows))

    # imagem: bg@orig × nobg@nobg
    if image_rows:
        cls = [c for c in classes if any(r['class_name'] == c for r in image_rows)]
        conds = list(dict.fromkeys(r['condition'] for r in image_rows))
        fig, ax = plt.subplots(figsize=(12, 4.2))
        width = 0.8 / len(conds)
        for k, cond in enumerate(conds):
            ys, lo, hi = [], [], []
            for c in cls:
                r = next((r for r in image_rows if r['class_name'] == c and r['condition'] == cond), None)
                y = num(r['all_auroc']) if r else float('nan')
                ys.append(y)
                lo.append(y - num(r['all_auroc_ci_low']) if r else 0)
                hi.append(num(r['all_auroc_ci_high']) - y if r else 0)
            x = np.arange(len(cls)) + (k - (len(conds) - 1) / 2) * width
            ax.bar(x, ys, width, label=cond, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], capsize=2)
        ax.set_xticks(np.arange(len(cls)))
        ax.set_xticklabels([c.split('-')[0] for c in cls])
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel('Image AUROC (interseção)')
        ax.legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        ax.set_title('Nível de imagem — SEDifferNet (encoder@imagens)')
        parts.append('<div class="imgs">' + fig_tag(fig, 'Erro = IC95 por bootstrap; mesmas imagens em todas as condições da classe.') + '</div>')
        rows = []
        for c in cls:
            for r in [r for r in image_rows if r['class_name'] == c]:
                rows.append([c, f'<code>{r["condition"]}</code>', fmt(r['ckpt_auroc']), r.get('ckpt_epoch', ''),
                             f'{r["all_n"]}/{r["all_n_anom"]}', fmt(r['all_auroc']) + ci(r['all_auroc_ci_low'], r['all_auroc_ci_high']),
                             fmt(r['all_delta'], signed=True) + ci(r['all_delta_ci_low'], r['all_delta_ci_high']),
                             fmt(r['val_delta'], signed=True), fmt(r['holdout_delta'], signed=True), fmt(r['all_ap'])])
        main = [r for r in image_rows if r['condition'] == 'nobg@nobg']
        parts.append('<h3>Nível de imagem — todas as classes</h3>' + table(
            ['classe', 'condição', 'AUROC ckpt', 'época', 'n/anômalas', 'AUROC [IC95]', 'Δ vs bg@orig [IC95]',
             'Δ val', 'Δ holdout', 'AP'], rows))
        if main:
            d = [num(r['all_delta']) for r in main]
            parts.append(f'<p><b>nobg@nobg − bg@orig</b>: Δ médio {np.nanmean(d):+.4f} em {len(main)} classes; '
                         f'IC95 inteiro &lt; 0 em {sum(num(r["all_delta_ci_high"]) < 0 for r in main)}, '
                         f'&gt; 0 em {sum(num(r["all_delta_ci_low"]) > 0 for r in main)}.</p>')
    return '\n'.join(parts) + '</section>'


def section_class(c, run_dir, cfg, pixel_rows, image_rows):
    cdir = os.path.join(run_dir, c)
    out = [f'<section id="{c}"><h2>{c}</h2>']

    y = yolo_info(cfg, c)
    out.append('<h3>1–2. YOLO26-seg</h3>')
    if y:
        b = y['best'] or {}
        conv = (y['meta'].get('conversion_summary') or {}).get('counts', {})
        out.append(table(['run', 'épocas', 'imagens treino/val', 'melhor mask mAP50-95', 'mask mAP50', 'mask P', 'mask R'],
                         [[f'<code>{html.escape(os.path.basename(y["run"]))}</code>', len(y['rows']),
                           f'{conv.get("train", "?")}/{conv.get("val", "?")}', fmt(b.get('metrics/mAP50-95(M)')),
                           fmt(b.get('metrics/mAP50(M)')), fmt(b.get('metrics/precision(M)')), fmt(b.get('metrics/recall(M)'))]]))
        out.append('<div class="imgs">' + img_tag(os.path.join(y['run'], 'results.png'), 'curvas de treino', 900)
                   + img_tag(os.path.join(y['run'], 'val_batch0_pred.jpg'), 'predições na validação YOLO', 500) + '</div>')
    else:
        out.append('<p class="none">sem treino YOLO.</p>')

    m = masks_info(cdir)
    out.append('<h3>3. Máscaras no dataset original</h3>')
    if m:
        groups = {}
        for r in m['rows']:
            g = groups.setdefault(f'{r["split"]}/{r["defect"]}', [0, 0, 0, []])
            g[0] += 1
            g[1] += r['status'] == 'no_detection'
            g[2] += int(r['n_instances'] or 0) > 1
            g[3].append(num(r['area_frac']))
        out.append(f'<p><code>{html.escape(m["dir"])}</code> — conf={m["manifest"].get("conf")}, '
                   f'união de instâncias, sem descarte.</p>')
        out.append(table(['split/defeito', 'imagens', 'sem detecção (máscara cheia)', 'multi-instância', 'área média'],
                         [[k, g[0], g[1], g[2], fmt(np.nanmean(g[3]), 3)] for k, g in groups.items()]))
        prev = sorted(glob.glob(os.path.join(m['dir'], 'previews', 'test', '*', '*.png')))
        prev = [p for p in prev if os.sep + 'good' + os.sep not in p][:cfg.get('html_mask_previews', 4)] + \
               [p for p in prev if os.sep + 'good' + os.sep in p][:2]
        out.append('<div class="imgs">' + ''.join(img_tag(p, os.path.basename(os.path.dirname(p)), 280) for p in prev) + '</div>')
    else:
        out.append('<p class="none">sem máscaras/audit nesta rodada.</p>')

    audit = read_csv(os.path.join(cdir, 'audit', 'audit_summary.csv'))
    out.append('<h3>4. Audit — fundo removido × defeito preservado</h3>')
    if audit:
        area = {r['strategy']: r['area_kept'] for r in audit if r['partition'] == 'train_good'}
        rows = [[f'<code>{r["strategy"]}</code>', fmt(area.get(r['strategy']), 3), fmt(r['gt_lost_frac'], 4),
                 r['n_lost50'], r['n_lost95'], r['n_full_mask']] for r in audit if r['partition'] == 'val']
        out.append(table(['estratégia', 'área mantida (train)', 'GT perdido (val)', 'imgs >50%', 'imgs >95%', 'sem detecção'], rows))
        out.append('<div class="imgs">' + img_tag(os.path.join(cdir, 'audit', 'audit_tradeoff.png'), 'trade-off', 480)
                   + img_tag(os.path.join(cdir, 'audit', 'audit_examples.png'), 'piores casos da validação (contornos por estratégia)', 1100)
                   + '</div>')
    fin = os.path.join(cdir, 'finalists.json')
    if os.path.isfile(fin):
        f = cm.read_json(fin)
        rule = 'override ' + ','.join(f['override']) if f.get('override') else 'regra pré-registrada'
        out.append(f'<h3>5. Finalistas</h3><p>{html.escape(rule)}; máscara confiável: {f.get("mask_reliable")}; '
                   f'variantes: ' + ', '.join(f'<code>{v["variant"]}</code> ({v["role"]})' for v in f['variants']) + '</p>')

    prow = [r for r in pixel_rows if r['class_name'] == c]
    out.append('<h3>6–8. Nível de pixel (NF head) × referência com fundo</h3>')
    if prow:
        rows = []
        for r in prow:
            crit = '★ recomendada' if r.get('recommended') == 'True' else (
                'ok' if r.get('passes_criteria') == 'True' else (f'falhou: {r["failed_checks"]}' if r.get('failed_checks') else '—'))
            rows.append([r['arm'], f'<code>{r["variant"]}</code>', r['role'], fmt(r['val_pixel_auroc']),
                         fmt(r['val_delta_pixel_auroc'], signed=True) + ci(r['val_delta_ci_low'], r['val_delta_ci_high']),
                         fmt(r['val_aupro']), fmt(r['val_delta_aupro'], signed=True), fmt(r['val_gt_lost_frac'], 3),
                         fmt(r['val_pixel_auroc_in_object']),
                         f'{fmt(r["val_contour_fpr"], 3)} / {fmt(r["val_offcontour_fpr"], 3)}',
                         fmt(r['holdout_pixel_auroc']),
                         fmt(r['holdout_delta_pixel_auroc'], signed=True) + ci(r['holdout_delta_ci_low'], r['holdout_delta_ci_high']),
                         fmt(r['holdout_aupro']), crit])
        out.append(table(['braço', 'variante', 'papel', 'val AUROC', 'Δ val [IC95]', 'val AUPRO', 'ΔAUPRO', 'GT perdido',
                          'AUROC no objeto', 'FPR contorno/fora', 'holdout AUROC', 'Δ holdout [IC95]', 'holdout AUPRO',
                          'critério'], rows))
        enc = [r for r in prow if math.isfinite(num(r.get('val_encoder_image_auroc')))]
        if enc:
            out.append('<p><b>SEDifferNet nas imagens da variante</b> (braço A = encoder com fundo recebendo a imagem '
                       'mascarada; braço B = encoder retreinado na mesma máscara). Δ contra o encoder com fundo nas imagens originais:</p>')
            out.append(table(['braço', 'variante', 'val Img AUROC', 'Δ val [IC95]', 'holdout Img AUROC', 'Δ holdout [IC95]'],
                             [[r['arm'], f'<code>{r["variant"]}</code>', fmt(r['val_encoder_image_auroc']),
                               fmt(r.get('val_delta_encoder_image_auroc'), signed=True)
                               + ci(r.get('val_delta_encoder_ci_low'), r.get('val_delta_encoder_ci_high')),
                               fmt(r['holdout_encoder_image_auroc']),
                               fmt(r.get('holdout_delta_encoder_image_auroc'), signed=True)
                               + ci(r.get('holdout_delta_encoder_ci_low'), r.get('holdout_delta_encoder_ci_high'))]
                              for r in enc]))
        out.append('<div class="imgs">' + img_tag(os.path.join(cdir, f'report_{c}.png'), 'val × holdout por variante', 1100) + '</div>')
    else:
        out.append('<p class="none">sem relatório de pixel.</p>')

    irow = [r for r in image_rows if r['class_name'] == c]
    out.append('<h3>9. Nível de imagem — SEDifferNet com fundo × sem fundo × retreinado na máscara (braço B)</h3>')
    if irow:
        out.append(table(['condição', 'época ckpt', 'AUROC ckpt', 'AUROC próprio teste (n)', 'AUROC interseção [IC95]',
                          'Δ vs bg@orig [IC95]', 'P(Δ≤0)', 'Δ val', 'Δ holdout', 'AP'],
                         [[f'<code>{r["condition"]}</code>', r['ckpt_epoch'], fmt(r['ckpt_auroc']),
                           f'{fmt(r["own_auroc"])} ({r["own_n"]})',
                           fmt(r['all_auroc']) + ci(r['all_auroc_ci_low'], r['all_auroc_ci_high']),
                           fmt(r['all_delta'], signed=True) + ci(r['all_delta_ci_low'], r['all_delta_ci_high']),
                           fmt(r['all_delta_p_le0'], 3), fmt(r['val_delta'], signed=True),
                           fmt(r['holdout_delta'], signed=True), fmt(r['all_ap'])] for r in irow]))
        out.append(f'<p>Interseção: {irow[0]["all_n"]} imagens ({irow[0]["all_n_anom"]} anômalas).</p>')
        out.append('<div class="imgs">' + img_tag(os.path.join(run_dir, 'image_level', c, f'roc_{c}.png'), 'ROC e distribuição dos scores', 1100) + '</div>')
    else:
        out.append('<p class="none">sem avaliação de imagem.</p>')
    return '\n'.join(out) + '</section>'


def build(run_dir, cfg=None):
    cfg = cfg or {}
    cfg_path = os.path.join(run_dir, 'pipeline_config.json')
    if os.path.isfile(cfg_path):
        cfg = {**cm.read_json(cfg_path), **{k: v for k, v in cfg.items() if k not in ('classes',)}}
    cfg.setdefault('yolo_runs_dir', cm.RUNS_DIR)
    status_path = os.path.join(run_dir, 'pipeline_status.json')
    status = cm.read_json(status_path) if os.path.isfile(status_path) else {'classes': {}}
    classes = [c for c in cm.ALL_CLASSES if c in (cfg.get('classes') or cm.ALL_CLASSES)
               or c in status.get('classes', {}) or os.path.isdir(os.path.join(run_dir, c))]
    classes = [c for c in classes if c in status.get('classes', {}) or os.path.isdir(os.path.join(run_dir, c))
               or os.path.isdir(os.path.join(run_dir, 'image_level', c))] or classes
    pixel_rows = read_csv(os.path.join(run_dir, 'summary.csv'))
    image_rows = []
    for c in classes:
        image_rows += read_csv(os.path.join(run_dir, 'image_level', c, 'summary.csv'))

    nav = ''.join(f'<a href="#{a}">{t}</a>' for a, t in [('status', 'Status'), ('macro', 'Resumo')] + [(c, c) for c in classes])
    cfg_html = html.escape(json.dumps({k: v for k, v in cfg.items()}, indent=2, ensure_ascii=False))
    body = [section_status(status, classes), section_macro(run_dir, classes, pixel_rows, image_rows)]
    body += [section_class(c, run_dir, cfg, pixel_rows, image_rows) for c in classes]
    body.append(f'<section id="config"><h2>Configuração</h2><details><summary>pipeline_config.json</summary>'
                f'<pre>{cfg_html}</pre></details></section>')
    page = (f'<!doctype html><html lang="pt-br"><head><meta charset="utf-8"><title>Ablação de fundo — '
            f'{html.escape(os.path.basename(run_dir))}</title><style>{CSS}</style></head><body>'
            f'<header><h1>Ablação de remoção de fundo (D-19…D-22)</h1><p>Rodada <code>{html.escape(run_dir)}</code> · '
            f'gerado em {cm.timestamp()}</p></header><nav>{nav}</nav><main>' + '\n'.join(body) + '</main></body></html>')
    path = os.path.join(run_dir, 'relatorio_pipeline.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(page)
    return path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('uso: python report_html.py <rodada>')
    print(build(os.path.abspath(sys.argv[1])))
