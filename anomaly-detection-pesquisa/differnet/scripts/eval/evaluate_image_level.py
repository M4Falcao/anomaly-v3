"""Image-level evaluation of the frozen SEDifferNet classifier.

Answers one question only: *is this image anomalous?* The pixel head is never
loaded, so the numbers here are the ceiling the detection stage can deliver and
are unaffected by any localization choice.

Two axes matter at this level and both are exposed:

- ``--n_transforms``  DifferNet's native TTA. The score of an image is the mean
  squared latent norm over N fixed rotations; N=1 is deterministic and fast,
  larger N is what the original paper reports.
- ``--threshold_mode`` How the operating point is chosen. ``best_f1`` is
  optimistic (it peeks at the test labels); ``train_quantile`` calibrates on
  defect-free training images only and is what a deployment can actually do.

Outputs, per run, under ``resultado_analise_final/imagem/run_<ts>/``:
    results_per_class.csv   one row per class plus the mean
    scores_<class>.csv      per-image score and label (for post-hoc analysis)
    config.txt              every resolved argument
    <class>/histograma.png, roc.png, precision_recall.png

Examples:
    python scripts/eval/evaluate_image_level.py
    python scripts/eval/evaluate_image_level.py --classes vari-grip --n_transforms 8
    python scripts/eval/evaluate_image_level.py --threshold_mode train_quantile --quantile 0.98
"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
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
from sklearn.metrics import precision_recall_curve, roc_curve
from tqdm import tqdm

import config as c
from core.eval_pipeline import (
    ALL_CLASSES,
    DEVICE,
    binary_metrics,
    build_eval_datasets,
    confusion_at,
    eval_loader,
    find_se_checkpoint,
    image_score,
    load_se_model,
    unpack_batch,
)
from core.paths import project_path

THRESHOLD_MODES = ('best_f1', 'train_quantile')


@torch.no_grad()
def train_score_quantile(model, trainset, quantile, batch_size=8):
    """Operating point calibrated on defect-free training images only.

    Unlike ``best_f1`` this never looks at test labels, so it is the honest
    threshold to report for a deployed system.
    """
    loader = eval_loader(trainset, batch_size=batch_size)
    scores = []
    for data in tqdm(loader, desc='  calibrando limiar', leave=False):
        images = data[0] if isinstance(data, (list, tuple)) else data
        images = images.to(DEVICE)
        if images.dim() == 5:
            images = images.view(-1, *images.shape[-3:])
        z = model(images)
        scores.append(float(torch.mean(z ** 2).item()))
    return float(np.quantile(scores, quantile))


def plot_histogram(normal, anomaly, threshold, save_path, title):
    plt.figure(figsize=(9, 5))
    plt.hist(normal, bins=40, alpha=0.65, label=f'Normal (n={len(normal)})', color='tab:green')
    plt.hist(anomaly, bins=40, alpha=0.65, label=f'Anomalia (n={len(anomaly)})', color='tab:red')
    if threshold is not None:
        plt.axvline(threshold, color='black', ls='--', lw=1.5,
                    label=f'Limiar = {threshold:.4f}')
    plt.xlabel('Score de imagem (mean z^2)')
    plt.ylabel('Contagem')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_roc(labels, scores, auroc, save_path, title):
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(fpr, tpr, lw=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def plot_pr(labels, scores, ap, save_path, title):
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(5.5, 5.5))
    plt.plot(recall, precision, lw=2, label=f'AP = {ap:.4f}')
    plt.axhline(np.mean(labels), color='gray', ls=':', lw=1,
                label=f'Trivial = {np.mean(labels):.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=130)
    plt.close()


def evaluate_class(class_name, args, out_dir):
    """Image-level metrics for one class. Returns ``(metrics, scores_df)``."""
    print(f"\n{'=' * 70}\n  {class_name}\n{'=' * 70}")

    se_ckpt = args.se_checkpoint or find_se_checkpoint(class_name)
    if se_ckpt is None:
        print('  ERRO: SEDifferNet nao encontrado.')
        return None

    print(f'  Checkpoint: {os.path.basename(se_ckpt)}')
    model = load_se_model(se_ckpt)

    trainset, testset, labels = build_eval_datasets(
        class_name, args.dataset, n_transforms=args.n_transforms,
        limit=args.limit, seed=args.seed)
    loader = eval_loader(testset)
    print(f'  Teste: {len(testset)} imagens ({int(labels.sum())} anomalas) | '
          f'n_transforms={args.n_transforms}')

    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 3:
                break
            model(unpack_batch(data)[0])
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

    scores, times = [], []
    with torch.no_grad():
        for data in tqdm(loader, desc=f'  [{class_name}]', leave=False):
            images, _, _ = unpack_batch(data)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            s = image_score(model, images)
            if DEVICE == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            scores.append(s)

    scores = np.asarray(scores)
    metrics = {'class': class_name, 'n_images': len(scores),
               'n_anomaly': int(labels.sum())}
    metrics.update(binary_metrics(labels, scores, prefix='image_'))

    if args.threshold_mode == 'train_quantile':
        thr = train_score_quantile(model, trainset, args.quantile)
        metrics['threshold_source'] = f'train_q{args.quantile}'
    else:
        thr = metrics.get('image_threshold')
        metrics['threshold_source'] = 'best_f1'
    if thr is not None:
        metrics['threshold'] = thr
        metrics.update(confusion_at(labels, scores, thr))

    lat = float(np.mean(times)) * 1000
    metrics['latency_ms'] = lat
    metrics['fps'] = 1000.0 / lat if lat > 0 else 0.0

    os.makedirs(out_dir, exist_ok=True)
    normal, anomaly = scores[labels == 0], scores[labels == 1]
    if len(normal) and len(anomaly):
        plot_histogram(normal, anomaly, thr, os.path.join(out_dir, 'histograma.png'),
                       f'{class_name} - scores de imagem')
    if 'image_auroc' in metrics:
        plot_roc(labels, scores, metrics['image_auroc'], os.path.join(out_dir, 'roc.png'),
                 f'{class_name} - ROC (imagem)')
        plot_pr(labels, scores, metrics['image_ap'], os.path.join(out_dir, 'precision_recall.png'),
                f'{class_name} - Precision-Recall (imagem)')

    print(f"  AUROC={metrics.get('image_auroc', float('nan')):.4f}  "
          f"AP={metrics.get('image_ap', float('nan')):.4f}  "
          f"F1={metrics.get('image_f1', float('nan')):.4f}  "
          f"recall={metrics.get('recall', float('nan')):.3f}  "
          f"spec={metrics.get('specificity', float('nan')):.3f}  "
          f"{metrics['fps']:.1f} FPS")

    del model
    torch.cuda.empty_cache()
    return metrics, pd.DataFrame({'score': scores, 'label': labels})


def main():
    parser = argparse.ArgumentParser(
        description='Avaliacao nivel de IMAGEM (SEDifferNet isolado).')
    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--classes', type=str, default=None,
                        help='Lista separada por virgula; default = as 5 classes.')
    parser.add_argument('--se_checkpoint', type=str, default=None,
                        help='Forca um checkpoint (so faz sentido com uma classe).')
    parser.add_argument('--limit', type=int, default=None,
                        help='Teto balanceado de imagens de teste; preserva todas as anomalias.')
    parser.add_argument('--n_transforms', type=int, default=1,
                        help='Rotacoes fixas promediadas por imagem (TTA nativo do DifferNet). '
                             'Default 1 = deterministico.')
    parser.add_argument('--threshold_mode', type=str, default='best_f1',
                        choices=THRESHOLD_MODES,
                        help='best_f1 espia os rotulos de teste (otimista); train_quantile '
                             'calibra so com imagens boas de treino (realista).')
    parser.add_argument('--quantile', type=float, default=0.99,
                        help='Quantil usado por --threshold_mode train_quantile.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class_list = ([x.strip() for x in args.classes.split(',') if x.strip()]
                  if args.classes else ALL_CLASSES)

    run_dir = args.out_dir or project_path(
        'resultado_analise_final', 'imagem', f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}={v}\n')
        f.write(f'device={DEVICE}\nimg_size={c.img_size}\nn_scales={c.n_scales}\n')

    print('=' * 70)
    print('  AVALIACAO NIVEL DE IMAGEM - SEDifferNet')
    print('=' * 70)
    print(f'  Classes:   {class_list}')
    print(f'  TTA:       n_transforms={args.n_transforms}')
    print(f'  Limiar:    {args.threshold_mode}'
          + (f' (q={args.quantile})' if args.threshold_mode == 'train_quantile' else ''))
    print(f'  Saida:     {run_dir}')

    rows = []
    for class_name in class_list:
        try:
            result = evaluate_class(class_name, args, os.path.join(run_dir, class_name))
        except Exception as err:
            print(f'  ERRO em {class_name}: {err}')
            continue
        if result is None:
            continue
        metrics, scores_df = result
        rows.append(metrics)
        scores_df.to_csv(os.path.join(run_dir, f'scores_{class_name}.csv'), index=False)

    if not rows:
        print('\nNenhuma classe avaliada.')
        return

    df = pd.DataFrame(rows)
    num = df.select_dtypes(include=[np.number]).columns
    mean_row = {col: df[col].mean() for col in num}
    mean_row['class'] = 'MEDIA'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    front = ['class', 'n_images', 'n_anomaly', 'image_auroc', 'image_ap', 'image_f1',
             'precision', 'recall', 'specificity', 'balanced_acc', 'fps', 'latency_ms']
    df = df[[col for col in front if col in df.columns]
            + [col for col in df.columns if col not in front]]

    out_csv = os.path.join(run_dir, 'results_per_class.csv')
    df.to_csv(out_csv, index=False)

    print('\n' + '=' * 100)
    print('  RESULTADOS - NIVEL DE IMAGEM')
    print('=' * 100)
    show = [col for col in ['class', 'image_auroc', 'image_ap', 'image_f1', 'precision',
                            'recall', 'specificity', 'fps'] if col in df.columns]
    print(df[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nCSV: {out_csv}')


if __name__ == '__main__':
    main()
