"""Etapa 9 — nível de imagem: SEDifferNet COM fundo × SEDifferNet SEM fundo, nas mesmas imagens.

Encoders (``final_models``):
  * ``bg``   = ``SEDiffernet/<classe>/*.pt``      (treinado com fundo);
  * ``nobg`` = ``SEDiffernet_noBg/<classe>_no_bg_best.pt`` (treinado com fundo preto).

Condições (encoder × imagens de teste):
  * ``bg@orig``     — referência: encoder com fundo, imagens originais;
  * ``nobg@nobg``   — encoder sem fundo, imagens do dataset sem fundo (como foi treinado);
  * ``--cross`` acrescenta ``bg@nobg`` e ``nobg@orig`` (cada encoder no domínio do outro);
  * ``--extra NOME=RAIZ`` acrescenta ``nobg@NOME`` (ex.: dataset mascarado pela YOLO26 com o
    inventário completo, ``data/masked/<máscaras>/raw__black``).
  * ``--encoder NOME=CKPT@DADOS`` acrescenta ``NOME@DADOS`` com um encoder qualquer (ex.: o
    SEDifferNet retreinado no braço B sobre o dataset mascarado ``DADOS``, que precisa ser
    ``orig``, ``nobg`` ou um nome de ``--extra``).

Comparação **pareada** na interseção das imagens (nomes normalizados: ``Fotos `` = ``Fotos-``),
com o split val/holdout do protocolo (``val_fraction`` 0.4, ``split_seed`` 42, calculado no
inventário original). O score é o DifferNet: média de ||z||² sobre ``--views`` rotações fixas,
igual ao ``08_report.py``.

Ressalva: os dois encoders tiveram a época escolhida pelo Image AUROC no **próprio teste** (o
noBg no teste sem fundo). Nenhuma partição é limpa para eles; ``val``/``holdout`` mostram só a
estabilidade do Δ. O valor gravado no checkpoint é reportado como ``ckpt_auroc``.

Saída: ``resultado_analise_final/imagem_bg_vs_nobg/run_<ts>/`` com ``per_image_scores.csv``,
``summary.csv``, ``summary.md``, ``roc_<classe>.png`` e cache de scores em ``cache/``.
"""
import argparse
import csv
import glob
import hashlib
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as cm  # noqa: E402

if cm.DIFFERNET_ROOT not in sys.path:
    sys.path.insert(0, cm.DIFFERNET_ROOT)

OUT_ROOT = os.path.join(cm.DIFFERNET_ROOT, 'resultado_analise_final', 'imagem_bg_vs_nobg')
SUBSETS = ('all', 'val', 'holdout')


def norm_key(rel_path):
    """``test/<defeito>/<arquivo>`` com nome normalizado (espaço = hífen, sem extensão)."""
    rel = rel_path.replace('\\', '/')
    d, f = os.path.dirname(rel), os.path.basename(rel)
    return f'{d}/{os.path.splitext(f)[0].replace(" ", "-").lower()}'


def find_encoder(kind, class_name):
    return cm.find_class_encoder(cm.BG_ENCODER_DIR if kind == 'bg' else cm.NOBG_ENCODER_DIR, class_name)


def checkpoint_meta(path):
    import torch
    d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    epoch = d.get('epoch') if isinstance(d, dict) else None
    if isinstance(d, dict) and d.get('image_aurocs'):
        auc = float(max(d['image_aurocs']))
    elif isinstance(d, dict) and 'image_auroc' in d:
        auc = float(d['image_auroc'])
    else:
        auc = float('nan')
    return dict(epoch=epoch, ckpt_auroc=auc)


def inventory(root, class_name):
    """{chave normalizada: (caminho absoluto, rótulo 0/1)} do ``test`` de ``root``."""
    test = os.path.join(root, class_name, 'test')
    out = {}
    for defect in sorted(os.listdir(test)):
        for f in cm.list_images(os.path.join(test, defect)):
            rel = f'test/{defect}/{f}'
            key = norm_key(rel)
            if key in out:
                raise ValueError(f'Chave duplicada após normalização: {key}')
            out[key] = (os.path.join(test, defect, f), int(defect != 'good'))
    return out


def score_images(se_ckpt, paths, views, cache_path, desc):
    """Score DifferNet por imagem (média ||z||² sobre ``views`` rotações fixas), com cache."""
    if os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        if list(data['paths']) == list(paths):
            return data['scores']
    import torch
    from PIL import Image
    from tqdm import tqdm
    import config as c
    from core.eval_pipeline import image_score, load_se_model
    from core.utils import get_fixed_transforms

    c.img_size = (448, 448)
    tfs = [get_fixed_transforms(i * 360.0 / views) for i in range(views)]
    model = load_se_model(se_ckpt)
    device = next(model.parameters()).device
    scores = np.zeros(len(paths))
    for i, p in enumerate(tqdm(paths, desc=desc, leave=False)):
        img = Image.open(p).convert('RGB')
        batch = torch.stack([t(img) for t in tfs]).to(device)
        scores[i] = image_score(model, batch)
    del model
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, paths=np.array(paths), scores=scores)
    return scores


def metrics(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) < 2:
        return dict(auroc=float('nan'), ap=float('nan'), n=len(labels), n_anom=int(labels.sum()))
    return dict(auroc=float(roc_auc_score(labels, scores)), ap=float(average_precision_score(labels, scores)),
                n=len(labels), n_anom=int(labels.sum()))


def auroc_ci(labels, scores, n_boot, seed):
    from sklearn.metrics import roc_auc_score
    labels, scores = np.asarray(labels), np.asarray(scores)
    groups = [np.flatnonzero(labels == g) for g in (0, 1)]
    if any(len(g) == 0 for g in groups):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    vals = [roc_auc_score(labels[idx], scores[idx]) for idx in
            (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n_boot))]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run_class(class_name, args, out_dir):
    import evaluate as ev
    import protocol as pt

    roots = {'orig': args.orig_root, 'nobg': args.nobg_root, **args.extra}
    encoders = {k: find_encoder(k, class_name) for k in ('bg', 'nobg')}
    conditions = [('bg', 'orig'), ('nobg', 'nobg')]
    if args.cross:
        conditions += [('bg', 'nobg'), ('nobg', 'orig')]
    conditions += [('nobg', name) for name in args.extra]
    for name, (ckpt, data) in args.encoders.items():
        if data not in roots:
            raise ValueError(f'--encoder {name}: dataset {data!r} desconhecido ({sorted(roots)})')
        encoders[name] = ckpt
        conditions.append((name, data))

    orig_inv = pt.test_inventory(args.orig_root, class_name)
    labels_orig = [lab for _, lab, _ in orig_inv]
    val, holdout = pt.split_indices(labels_orig, args.val_fraction, args.split_seed)
    part_of = {}
    for i in val:
        part_of[norm_key(orig_inv[i][0])] = 'val'
    for i in holdout:
        part_of[norm_key(orig_inv[i][0])] = 'holdout'

    invs = {name: inventory(root, class_name) for name, root in roots.items()
            if name in {d for _, d in conditions}}
    keys = sorted(set.intersection(*(set(inv) for inv in invs.values())))
    if args.limit:
        rng = np.random.default_rng(0)
        lab = np.array([invs['orig'][k][1] for k in keys])
        keep = np.concatenate([rng.permutation(np.flatnonzero(lab == g))[:args.limit // 2] for g in (0, 1)])
        keys = [keys[i] for i in sorted(keep)]
    for name, inv in invs.items():
        bad = [k for k in keys if inv[k][1] != invs['orig'][k][1]]
        if bad:
            raise ValueError(f'{class_name}: rótulos divergem em {name}: {bad[:3]}')
    labels = np.array([invs['orig'][k][1] for k in keys])
    parts = np.array([part_of[k] for k in keys])

    if args.dry_run:
        for e, d in conditions:
            own = len(invs[d]) if not args.limit else len(keys)
            print(f'[dry-run] {class_name}: {e}@{d} encoder={os.path.basename(encoders[e])} '
                  f'imagens próprias={own} interseção={len(keys)} ({labels.sum()} anômalas)')
        return []

    meta = {k: checkpoint_meta(p) for k, p in encoders.items()}
    scores, own = {}, {}
    for e, d in conditions:
        name = f'{e}@{d}'
        inv = invs[d]
        own_keys = sorted(inv) if not args.limit else keys
        paths = [inv[k][0] for k in own_keys]
        tag = hashlib.sha1(f'{os.path.abspath(encoders[e])}|{args.views}'.encode()).hexdigest()[:10]
        cache = os.path.join(args.cache_dir, class_name, f'{name.replace("@", "_at_")}_{tag}.npz')
        s = score_images(encoders[e], paths, args.views, cache, f'{class_name} {name}')
        by_key = dict(zip(own_keys, s))
        scores[name] = np.array([by_key[k] for k in keys])
        own[name] = metrics([inv[k][1] for k in own_keys], s)

    rows = []
    ref = 'bg@orig'
    for e, d in conditions:
        name = f'{e}@{d}'
        row = dict(class_name=class_name, condition=name, encoder=os.path.basename(encoders[e]),
                   images=d, ckpt_epoch=meta[e]['epoch'], ckpt_auroc=meta[e]['ckpt_auroc'],
                   own_n=own[name]['n'], own_n_anom=own[name]['n_anom'], own_auroc=own[name]['auroc'],
                   own_ap=own[name]['ap'])
        for sub in SUBSETS:
            idx = np.arange(len(keys)) if sub == 'all' else np.flatnonzero(parts == sub)
            m = metrics(labels[idx], scores[name][idx])
            lo, hi = auroc_ci(labels[idx], scores[name][idx], args.n_boot, args.seed)
            row.update({f'{sub}_n': m['n'], f'{sub}_n_anom': m['n_anom'], f'{sub}_auroc': m['auroc'],
                        f'{sub}_auroc_ci_low': lo, f'{sub}_auroc_ci_high': hi, f'{sub}_ap': m['ap']})
            if name == ref:
                row.update({f'{sub}_delta': 0.0, f'{sub}_delta_ci_low': 0.0, f'{sub}_delta_ci_high': 0.0,
                            f'{sub}_delta_p_le0': float('nan')})
            else:
                b = ev.image_auroc_bootstrap(labels[idx], scores[ref][idx], scores[name][idx],
                                             n_boot=args.n_boot, seed=args.seed)
                row.update({f'{sub}_delta': m['auroc'] - metrics(labels[idx], scores[ref][idx])['auroc'],
                            f'{sub}_delta_ci_low': b['ci_low'], f'{sub}_delta_ci_high': b['ci_high'],
                            f'{sub}_delta_p_le0': b['p_le0']})
        rows.append(row)
        print(f'{class_name:32s} {name:16s} própria={own[name]["auroc"]:.4f} (n={own[name]["n"]}) '
              f'interseção={row["all_auroc"]:.4f} Δ={row["all_delta"]:+.4f} '
              f'[{row["all_delta_ci_low"]:+.4f}, {row["all_delta_ci_high"]:+.4f}]')

    with open(os.path.join(out_dir, 'per_image_scores.csv'), 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(['class_name', 'key', 'label', 'partition', 'condition', 'score'])
        for name, s in scores.items():
            for k, lab, p, v in zip(keys, labels, parts, s):
                w.writerow([class_name, k, int(lab), p, name, f'{v:.6g}'])
    plot_roc(class_name, labels, scores, os.path.join(out_dir, f'roc_{class_name}.png'))
    return rows


def plot_roc(class_name, labels, scores, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(labels, s)
        axes[0].plot(fpr, tpr, label=f'{name} ({roc_auc_score(labels, s):.3f})')
    axes[0].plot([0, 1], [0, 1], ls='--', c='gray', lw=1)
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].legend(fontsize=8)
    axes[0].set_title(f'{class_name}: ROC na interseção')
    for name, s in list(scores.items())[:2]:
        ls = np.log10(np.maximum(s, 1e-12))
        med = np.median(ls[labels == 0])
        iqr = np.subtract(*np.percentile(ls[labels == 0], [75, 25])) + 1e-12
        z = np.clip((ls - med) / iqr, -5, 15)
        bins = np.linspace(-5, 15, 50)
        axes[1].hist(z[labels == 0], bins=bins, alpha=0.4, density=True, label=f'{name} normal')
        axes[1].hist(z[labels == 1], bins=bins, alpha=0.9, density=True, histtype='step', lw=2, label=f'{name} anômala')
    axes[1].set_xlabel('log10(score), centrado na mediana / IQR dos normais (cortado em [-5, 15])')
    axes[1].legend(fontsize=7)
    axes[1].set_title('Distribuição dos scores')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(v, nd=4):
    return 'NaN' if v is None or (isinstance(v, float) and not math.isfinite(v)) else (f'{v:.{nd}f}' if isinstance(v, float) else str(v))


def write_markdown(rows, out_dir, args):
    lines = ['# Nível de imagem — SEDifferNet com fundo × sem fundo', '',
             f'Gerado em {cm.timestamp()}. Score DifferNet com {args.views} rotações fixas. Δ = condição − `bg@orig` '
             f'nas **mesmas imagens** (interseção), IC95 por bootstrap pareado estratificado ({args.n_boot}).', '',
             '> Os dois encoders tiveram a época escolhida pelo Image AUROC no próprio teste (o noBg no teste sem '
             'fundo): nenhuma partição é limpa e os AUROCs absolutos são otimistas. `val`/`holdout` mostram só a '
             'estabilidade do Δ.', '']
    for class_name in dict.fromkeys(r['class_name'] for r in rows):
        rs = [r for r in rows if r['class_name'] == class_name]
        lines += [f'## {class_name}', '',
                  '| condição | época ckpt | AUROC ckpt | AUROC próprio teste (n) | AUROC interseção [IC95] | '
                  'Δ interseção [IC95] | P(Δ≤0) | Δ val | Δ holdout | AP interseção |',
                  '|---|---|---|---|---|---|---|---|---|---|']
        for r in rs:
            lines.append(
                f'| `{r["condition"]}` | {r["ckpt_epoch"]} | {fmt(r["ckpt_auroc"])} | {fmt(r["own_auroc"])} '
                f'({r["own_n"]}/{r["own_n_anom"]}) | {fmt(r["all_auroc"])} [{fmt(r["all_auroc_ci_low"])}, '
                f'{fmt(r["all_auroc_ci_high"])}] | {fmt(r["all_delta"])} [{fmt(r["all_delta_ci_low"])}, '
                f'{fmt(r["all_delta_ci_high"])}] | {fmt(r["all_delta_p_le0"], 3)} | {fmt(r["val_delta"])} | '
                f'{fmt(r["holdout_delta"])} | {fmt(r["all_ap"])} |')
        lines += ['', f'Interseção: {rs[0]["all_n"]} imagens ({rs[0]["all_n_anom"]} anômalas); '
                      f'val {rs[0]["val_n"]}/{rs[0]["val_n_anom"]}, holdout {rs[0]["holdout_n"]}/{rs[0]["holdout_n_anom"]}.',
                  '', f'![{class_name}](roc_{class_name}.png)', '']
    main = [r for r in rows if r['condition'] == 'nobg@nobg']
    if len(main) > 1:
        lines += ['## Média macro (`nobg@nobg` − `bg@orig`, interseção)', '',
                  f'ΔAUROC médio = {np.mean([r["all_delta"] for r in main]):+.4f} em {len(main)} classes; '
                  f'classes com IC95 inteiro abaixo de 0: {sum(r["all_delta_ci_high"] < 0 for r in main)}; '
                  f'acima de 0: {sum(r["all_delta_ci_low"] > 0 for r in main)}.', '']
    with open(os.path.join(out_dir, 'summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--classes', default='all')
    p.add_argument('--orig_root', default=cm.ORIG_ROOT)
    p.add_argument('--nobg_root', default=cm.NOBG_ROOT)
    p.add_argument('--extra', action='append', default=[],
                   help='NOME=RAIZ de um dataset extra avaliado com o encoder noBg (repetível).')
    p.add_argument('--encoder', action='append', default=[], dest='encoders',
                   help='NOME=CKPT@DADOS: encoder extra avaliado no dataset DADOS (repetível).')
    p.add_argument('--cross', action='store_true', help='Também bg@nobg e nobg@orig.')
    p.add_argument('--views', type=int, default=64)
    p.add_argument('--val_fraction', type=float, default=0.4)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--n_boot', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cache_dir', default=os.path.join(OUT_ROOT, 'cache'))
    p.add_argument('--out_dir', default=None)
    p.add_argument('--limit', type=int, default=None, help='Smoke test: N imagens balanceadas por classe.')
    p.add_argument('--dry_run', action='store_true')
    args = p.parse_args()
    extra = {}
    for item in args.extra:
        name, _, root = item.partition('=')
        if not root or name in ('orig', 'nobg'):
            p.error(f'--extra inválido: {item!r}')
        extra[name] = root
    args.extra = extra
    encoders = {}
    for item in args.encoders:
        name, _, rest = item.partition('=')
        ckpt, _, data = rest.rpartition('@')
        if not (name and ckpt and data) or name in ('bg', 'nobg') or not os.path.isfile(ckpt):
            p.error(f'--encoder inválido (NOME=CKPT@DADOS, CKPT existente): {item!r}')
        encoders[name] = (ckpt, data)
    args.encoders = encoders

    out_dir = args.out_dir or os.path.join(OUT_ROOT, f'run_{cm.timestamp()}' + (f'_limit{args.limit}' if args.limit else ''))
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=False)
        cm.write_json(os.path.join(out_dir, 'config.json'), {**vars(args), 'created_at': cm.timestamp()})
    rows = []
    for class_name in cm.parse_classes(args.classes):
        rows += run_class(class_name, args, out_dir)
    if args.dry_run or not rows:
        return
    with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)
    write_markdown(rows, out_dir, args)
    print(f'\nRelatório: {os.path.join(out_dir, "summary.md")}')


if __name__ == '__main__':
    main()
