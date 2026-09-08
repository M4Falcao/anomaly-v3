"""Train CFLOW NF heads for all (or selected) classes in the dataset.

This script automates the full training loop across the INSPLAD dataset classes,
using each class's frozen pretrained SEDifferNet backbone.

All training artifacts are stored inside a dedicated run directory, neatly
partitioned by class:

    resultado_analise_final/treino_nf_heads/run_<timestamp>/
    ├── config.txt                 (global training configuration)
    ├── training_summary.csv       (per-class metrics, epochs, times, returncodes)
    ├── training_summary.md        (clean markdown table)
    ├── best_models/               (centralized directory of all trained heads)
    │   ├── glass-insulator/best_pixel_auroc.pt
    │   ├── lightning-rod-suspension/best_pixel_auroc.pt
    │   └── ...
    ├── <class_name_1>/
    │   ├── best_models/           (best_pixel_auroc.pt, best_image_auroc.pt)
    │   ├── checkpoints/           (periodic checkpoints cflow_epoch_*.pt)
    │   ├── plots/                 (training_curves.png)
    │   └── run_<ts>/              (full raw output from pixel_train)
    └── <class_name_2>/
        └── ...

Recipes:
    --recipe unified (default):
        The universal recommendation established in Section 9.0 of docs/plano_ablation_nf_head.md:
        - Features: L1 + L3 (--levels 0,2)
        - In-training selection: --score_norm raw
        - Memory optimization: --disable_patchcore
        - Adaptive epoch budget: 40 epochs for smaller classes, 15 for yoke, 30 for polymer
          (unless overridden by --epochs).

    --recipe selective_l1:
        Same as unified, but standardizes L1 only (--feat_norm_levels 0), avoiding noise
        amplification in higher semantic levels.

    --recipe optimal:
        Best-performing per-class configurations identified in the multi-class study:
        - glass-insulator: --levels 0,2 --feat_norm --clamp_scale 1.9 --epochs 40
        - vari-grip: --levels 0,2 --feat_norm --clamp_scale 1.9 --epochs 40
        - yoke-suspension: --levels 0,2 --epochs 12 --eval_interval 2
        - lightning-rod-suspension: --levels 0,2 --epochs 40
        - polymer-insulator-upper-shackle: --levels 0,1,2 --epochs 30 --eval_interval 3

    --recipe baseline:
        Standard baseline across all classes (--levels 0,1,2, no feat_norm, clamp 0.5).

    Any explicit CLI flag (--epochs, --levels, --feat_norm, --feat_norm_levels,
    --clamp_scale, --eval_interval, etc.) overrides the recipe setting for all runs.

Examples:
    # Train all 5 classes with the unified recipe (dry-run to inspect commands):
    python scripts/train/train_all_nf_heads.py --dry_run

    # Train all 5 classes with real training:
    python scripts/train/train_all_nf_heads.py

    # Train only two classes with 20 epochs each:
    python scripts/train/train_all_nf_heads.py --classes vari-grip,glass-insulator --epochs 20

    # Train with selective L1 standardization:
    python scripts/train/train_all_nf_heads.py --recipe selective_l1 --epochs 80 --eval_interval 5

    # Resume an existing run directory without repeating completed classes:
    python scripts/train/train_all_nf_heads.py --out_dir resultado_analise_final/treino_nf_heads/run_20260907_120000 --skip_existing
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch

import config as c
from core.cflow import (
    DEFAULT_ATTENTION,
    PAD_MODES,
    SCORE_MODES,
    TRAIN_CLAMP_SCALE,
    TRAIN_SCORE_MODE,
)
from core.eval_pipeline import (
    ALL_CLASSES,
    DEVICE,
    find_se_checkpoint,
)
from core.paths import project_path

RECIPES = ('unified', 'selective_l1', 'optimal', 'baseline')

# Per-class budgets when --epochs is omitted
CLASS_BUDGETS = {
    'unified': {
        'glass-insulator': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2'},
        'lightning-rod-suspension': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2'},
        'polymer-insulator-upper-shackle': {'epochs': 30, 'eval_interval': 3, 'levels': '0,2'},
        'vari-grip': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2'},
        'yoke-suspension': {'epochs': 15, 'eval_interval': 3, 'levels': '0,2'},
    },
    'selective_l1': {
        'glass-insulator': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2', 'feat_norm_levels': '0'},
        'lightning-rod-suspension': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2', 'feat_norm_levels': '0'},
        'polymer-insulator-upper-shackle': {'epochs': 30, 'eval_interval': 3, 'levels': '0,2', 'feat_norm_levels': '0'},
        'vari-grip': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2', 'feat_norm_levels': '0'},
        'yoke-suspension': {'epochs': 15, 'eval_interval': 3, 'levels': '0,2', 'feat_norm_levels': '0'},
    },
    'optimal': {
        # Training levels must be a SUPERSET of the levels used at scoring time,
        # since a level that was never trained cannot be recovered post-hoc.
        # Measured on run_20260907_145952 (full test set, raw + pad112 + flips + sigma 8):
        #   vari-grip L1=0.9191, L3=0.8052, L1+L3=0.8561.  Training on '0,2' locked
        #   vari-grip's weakest level into the head and left no L2 to fall back on,
        #   which is why it landed at 0.856 instead of the 0.927 seen in the ablation
        #   (where the head had all three levels and scoring picked L1+L2).
        'glass-insulator': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2', 'feat_norm': True, 'clamp_scale': 1.9},
        'lightning-rod-suspension': {'epochs': 40, 'eval_interval': 5, 'levels': '0,2'},
        'polymer-insulator-upper-shackle': {'epochs': 30, 'eval_interval': 3, 'levels': '0,1,2'},
        'vari-grip': {'epochs': 40, 'eval_interval': 5, 'levels': '0,1', 'feat_norm': True, 'clamp_scale': 1.9},
        'yoke-suspension': {'epochs': 12, 'eval_interval': 2, 'levels': '0,2'},
    },
    'baseline': {
        'glass-insulator': {'epochs': 40, 'eval_interval': 5, 'levels': '0,1,2'},
        'lightning-rod-suspension': {'epochs': 40, 'eval_interval': 5, 'levels': '0,1,2'},
        'polymer-insulator-upper-shackle': {'epochs': 40, 'eval_interval': 5, 'levels': '0,1,2'},
        'vari-grip': {'epochs': 40, 'eval_interval': 5, 'levels': '0,1,2'},
        'yoke-suspension': {'epochs': 15, 'eval_interval': 3, 'levels': '0,1,2'},
    },
}


def build_class_cmd(class_name, se_ckpt, class_out_dir, args):
    """Build command argument list for training one class."""
    recipe_cfg = CLASS_BUDGETS.get(args.recipe, {}).get(class_name, {})

    epochs = args.epochs if args.epochs is not None else recipe_cfg.get('epochs', 40)
    if args.epochs is not None and recipe_cfg.get('epochs') not in (None, args.epochs):
        print(f"  AVISO: --epochs {args.epochs} sobrescreve o orcamento da receita "
              f"'{args.recipe}' para {class_name} ({recipe_cfg['epochs']} epocas). "
              f"Em run_20260907_145952 esse override custou ate 0.030 de AUROC "
              f"(yoke atingiu o pico na epoca 4 e degradou ate a 80).")
    eval_interval = args.eval_interval if args.eval_interval is not None else recipe_cfg.get('eval_interval', 5)
    levels = args.levels if args.levels is not None else recipe_cfg.get('levels', '0,2')
    clamp_scale = args.clamp_scale if args.clamp_scale is not None else recipe_cfg.get('clamp_scale', TRAIN_CLAMP_SCALE)

    feat_norm = args.feat_norm or recipe_cfg.get('feat_norm', False)
    feat_norm_levels = args.feat_norm_levels if args.feat_norm_levels is not None else recipe_cfg.get('feat_norm_levels', None)

    train_script = os.path.join(PROJECT_ROOT, 'scripts', 'train', 'pixel_train_from_pretrained.py')

    cmd = [
        sys.executable, train_script,
        '--checkpoint', se_ckpt,
        '--class_name', class_name,
        '--dataset', args.dataset,
        '--epochs', str(epochs),
        '--eval_interval', str(eval_interval),
        '--out_dir', class_out_dir,
        '--score_norm', args.score_norm,
        '--clamp_scale', str(clamp_scale),
        '--seed', str(args.seed),
        '--lr', str(args.lr),
        '--attention', args.attention,
    ]

    if levels:
        cmd.extend(['--levels', str(levels)])
    if args.patience:
        cmd.extend(['--patience', str(args.patience)])
    if args.disable_patchcore:
        cmd.append('--disable_patchcore')
    if args.no_rotation:
        cmd.append('--no_rotation')
    if args.reflect_pad > 0:
        cmd.extend(['--reflect_pad', str(args.reflect_pad)])
    if args.pad_mode:
        cmd.extend(['--pad_mode', str(args.pad_mode)])
    if args.feat_pool > 0:
        cmd.extend(['--feat_pool', str(args.feat_pool)])
    if feat_norm_levels:
        cmd.extend(['--feat_norm_levels', str(feat_norm_levels)])
    elif feat_norm:
        cmd.append('--feat_norm')

    return cmd, {
        'epochs': epochs,
        'eval_interval': eval_interval,
        'levels': levels,
        'clamp_scale': clamp_scale,
        'patience': args.patience,
        'feat_norm': feat_norm or (feat_norm_levels is not None),
        'feat_norm_levels': feat_norm_levels,
    }


def df_to_markdown(df):
    """Convert dataframe to a markdown table without requiring the tabulate package."""
    headers = [str(col) for col in df.columns]
    rows = [[str(val) if pd.notna(val) else '' for val in row] for row in df.values]
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep_line = "| " + " | ".join("-" * max(w, 3) for w in widths) + " |"
    data_lines = ["| " + " | ".join(r[i].ljust(w) for i, w in zip(range(len(headers)), widths)) + " |" for r in rows]
    return "\n".join([header_line, sep_line] + data_lines)


def organize_class_artifacts(class_dir, master_best_dir, class_name):
    """Find newest run in class_dir, organize best_models and plots into root of class_dir."""
    sub_runs = sorted(
        [os.path.join(class_dir, d) for d in os.listdir(class_dir)
         if d.startswith('run_') and os.path.isdir(os.path.join(class_dir, d))]
    )
    if not sub_runs:
        return None

    newest_run = sub_runs[-1]
    best_src = os.path.join(newest_run, 'best_models')
    plots_src = os.path.join(newest_run, 'plots')

    class_best_dir = os.path.join(class_dir, 'best_models')
    class_plots_dir = os.path.join(class_dir, 'plots')
    os.makedirs(class_best_dir, exist_ok=True)
    os.makedirs(class_plots_dir, exist_ok=True)

    best_pixel_path = os.path.join(best_src, 'best_pixel_auroc.pt')
    if os.path.exists(best_pixel_path):
        target_path = os.path.join(class_best_dir, 'best_pixel_auroc.pt')
        shutil.copy2(best_pixel_path, target_path)

        # Copy to master best_models/<class_name>/best_pixel_auroc.pt
        master_class_best = os.path.join(master_best_dir, class_name, 'best_models')
        os.makedirs(master_class_best, exist_ok=True)
        shutil.copy2(best_pixel_path, os.path.join(master_class_best, 'best_pixel_auroc.pt'))

    best_img_path = os.path.join(best_src, 'best_image_auroc.pt')
    if os.path.exists(best_img_path):
        shutil.copy2(best_img_path, os.path.join(class_best_dir, 'best_image_auroc.pt'))

    training_curves = os.path.join(plots_src, 'training_curves.png')
    if os.path.exists(training_curves):
        shutil.copy2(training_curves, os.path.join(class_plots_dir, 'training_curves.png'))

    # Read metrics metadata from checkpoint
    meta = {}
    if os.path.exists(best_pixel_path):
        try:
            ckpt_data = torch.load(best_pixel_path, map_location='cpu', weights_only=False)
            meta['epoch'] = ckpt_data.get('epoch')
            meta['pixel_auroc'] = ckpt_data.get('pixel_auroc')
            meta['image_auroc'] = ckpt_data.get('image_auroc')
            meta['hparams'] = ckpt_data.get('cflow_hparams', {})
        except Exception as err:
            meta['read_error'] = str(err)

    return {
        'newest_run': newest_run,
        'best_pixel_path': os.path.join(class_best_dir, 'best_pixel_auroc.pt'),
        'meta': meta,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Treinamento em lote de NF heads (CFLOW) para todas as classes do dataset.')

    # Class and dataset selection
    parser.add_argument('--classes', type=str, default=None,
                        help=f'Comma-separated classes to train (default: all 5: {", ".join(ALL_CLASSES)}).')
    parser.add_argument('--dataset', type=str, default=c.dataset_path,
                        help='Root dataset path containing INSPLAD classes.')
    parser.add_argument('--recipe', type=str, default='unified', choices=list(RECIPES),
                        help='Preconfigured training recipe: unified (default L1+L3 raw), selective_l1, optimal, baseline.')

    # Hyperparameter overrides (override recipe defaults when specified)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Epochs to train (overrides recipe per-class budgets if provided).')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Epochs between in-training evaluations (overrides recipe default).')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience in number of evaluations (0 = off). '
                             'Recommended when raising --epochs above the recipe budget.')
    parser.add_argument('--levels', type=str, default=None,
                        help='Comma-separated feature levels (0=L1, 1=L2, 2=L3). E.g. "0,2".')
    parser.add_argument('--score_norm', type=str, default=TRAIN_SCORE_MODE, choices=list(SCORE_MODES),
                        help='Score aggregation for in-training validation and checkpoint selection (default: raw).')
    parser.add_argument('--clamp_scale', type=float, default=None,
                        help='Affine coupling clamp scale (default: 0.5 or recipe setting).')
    parser.add_argument('--feat_norm', action='store_true',
                        help='Standardize features per channel on all levels.')
    parser.add_argument('--feat_norm_levels', type=str, default=None,
                        help='Selective per-channel feature standardization (e.g. "0" for L1 only).')
    parser.add_argument('--feat_pool', type=int, default=0)
    parser.add_argument('--reflect_pad', type=int, default=0)
    parser.add_argument('--pad_mode', type=str, default='reflect', choices=list(PAD_MODES))
    parser.add_argument('--no_rotation', action='store_true',
                        help='Disable rotation augmentation during training.')
    parser.add_argument('--attention', type=str, default=DEFAULT_ATTENTION, choices=['se', 'cbam', 'auto'])
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--disable_patchcore', action='store_true', default=True,
                        help='Disable PatchCore during training to conserve memory and speed up runs (default: True).')
    parser.add_argument('--seed', type=int, default=42)

    # Operational controls
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Master output directory. Defaults to resultado_analise_final/treino_nf_heads/run_<ts>/.')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip training if best_models/best_pixel_auroc.pt already exists in the class folder.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print the planned commands and paths without executing.')
    parser.add_argument('--evaluate_after', type=str, default='pixel', choices=['none', 'pixel', 'full'],
                        help='Run evaluation on the newly trained models after batch completes (default: pixel).')
    parser.add_argument('--export_to_final_models', action='store_true',
                        help='Copy successful best_models into final_models/NF Head/<class>/best_models/ with backup.')

    args = parser.parse_args()

    class_list = (
        [cls.strip() for cls in args.classes.split(',') if cls.strip()]
        if args.classes else ALL_CLASSES
    )
    for cls in class_list:
        if cls not in ALL_CLASSES:
            print(f"AVISO: classe '{cls}' desconhecida. Classes padrao: {ALL_CLASSES}")

    # Set up master run directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    master_run_dir = args.out_dir or project_path(
        'resultado_analise_final', 'treino_nf_heads', f"run_{timestamp}"
    )
    master_best_dir = os.path.join(master_run_dir, 'best_models')

    if not args.dry_run:
        os.makedirs(master_run_dir, exist_ok=True)
        os.makedirs(master_best_dir, exist_ok=True)
        # Write config.txt
        with open(os.path.join(master_run_dir, 'config.txt'), 'w', encoding='utf-8') as f:
            for k, v in vars(args).items():
                f.write(f"{k}={v}\n")
            f.write(f"timestamp={timestamp}\n")
            f.write(f"device={DEVICE}\n")

    print('=' * 80)
    print('  TREINAMENTO EM LOTE - NF HEADS (CFLOW)')
    print('=' * 80)
    print(f"  Diretorio do run: {master_run_dir}")
    print(f"  Receita base:     {args.recipe}")
    print(f"  Classes ({len(class_list)}):     {class_list}")
    print(f"  Score norm:       {args.score_norm}")
    print(f"  Device:           {DEVICE}")
    if args.dry_run:
        print("  MODO:             DRY-RUN (somente exibicao, sem execucao)")
    print('=' * 80)

    summary_rows = []

    for idx, class_name in enumerate(class_list, 1):
        print(f"\n[{idx}/{len(class_list)}] Classe: {class_name}")

        se_ckpt = find_se_checkpoint(class_name)
        if se_ckpt is None:
            print(f"  ERRO: Checkpoint SEDifferNet nao encontrado para '{class_name}' em final_models/SEDiffernet/!")
            summary_rows.append({
                'class': class_name,
                'status': 'FAILED (no SE ckpt)',
                'returncode': -1,
                'train_minutes': 0.0,
                'epochs': 0,
                'pixel_auroc': None,
                'image_auroc': None,
                'best_epoch': None,
                'checkpoint': None,
            })
            continue

        class_out_dir = os.path.join(master_run_dir, class_name)
        class_best_pt = os.path.join(class_out_dir, 'best_models', 'best_pixel_auroc.pt')

        if args.skip_existing and os.path.exists(class_best_pt):
            print(f"  -> Checkpoint ja existe ({class_best_pt}). Pulando conforme --skip_existing.")
            summary_rows.append({
                'class': class_name,
                'status': 'SKIPPED (already exists)',
                'returncode': 0,
                'train_minutes': 0.0,
                'epochs': 0,
                'pixel_auroc': None,
                'image_auroc': None,
                'best_epoch': None,
                'checkpoint': class_best_pt,
            })
            continue

        cmd, effective_cfg = build_class_cmd(class_name, se_ckpt, class_out_dir, args)

        print(f"  SE Backbone:   {os.path.basename(se_ckpt)}")
        print(f"  Configuracao:  epochs={effective_cfg['epochs']} | eval_interval={effective_cfg['eval_interval']} | levels={effective_cfg['levels']} | clamp={effective_cfg['clamp_scale']} | feat_norm={effective_cfg['feat_norm']}")
        print(f"  Subprocesso:   {' '.join(cmd)}")

        if args.dry_run:
            summary_rows.append({
                'class': class_name,
                'status': 'DRY-RUN',
                'returncode': 0,
                'train_minutes': 0.0,
                'epochs': effective_cfg['epochs'],
                'pixel_auroc': None,
                'image_auroc': None,
                'best_epoch': None,
                'checkpoint': class_best_pt,
            })
            continue

        try:
            t0 = time.time()
            ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
            returncode = ret.returncode
            elapsed_min = (time.time() - t0) / 60.0
        except KeyboardInterrupt:
            print(f"\n[Treino interrompido pelo usuario na classe {class_name}]")
            summary_rows.append({
                'class': class_name,
                'status': 'INTERRUPTED',
                'returncode': -1,
                'train_minutes': round((time.time() - t0) / 60.0, 2),
                'epochs': effective_cfg['epochs'],
                'pixel_auroc': None,
                'image_auroc': None,
                'best_epoch': None,
                'checkpoint': None,
            })
            break

        if returncode != 0:
            print(f"  AVISO: Treino falhou para {class_name} com exit code {returncode}.")
            summary_rows.append({
                'class': class_name,
                'status': f'FAILED (code {returncode})',
                'returncode': returncode,
                'train_minutes': round(elapsed_min, 2),
                'epochs': effective_cfg['epochs'],
                'pixel_auroc': None,
                'image_auroc': None,
                'best_epoch': None,
                'checkpoint': None,
            })
            continue

        # Organize artifacts into <class_name>/best_models and master best_models/
        info = organize_class_artifacts(class_out_dir, master_best_dir, class_name)
        meta = info['meta'] if info else {}

        pix_auc = meta.get('pixel_auroc')
        img_auc = meta.get('image_auroc')
        best_ep = meta.get('epoch')

        print(f"  -> Treino concluido em {elapsed_min:.1f} min | Best pix AUROC ({args.score_norm}): {pix_auc or 'n/a'} (epoch {best_ep or 'n/a'})")
        summary_rows.append({
            'class': class_name,
            'status': 'SUCCESS',
            'returncode': returncode,
            'train_minutes': round(elapsed_min, 2),
            'epochs': effective_cfg['epochs'],
            'pixel_auroc': round(pix_auc, 4) if pix_auc is not None else None,
            'image_auroc': round(img_auc, 4) if img_auc is not None else None,
            'best_epoch': best_ep,
            'checkpoint': info['best_pixel_path'] if info else None,
        })

    # Save summary table
    summary_df = pd.DataFrame(summary_rows)
    print('\n' + '=' * 80)
    print('  RESUMO DO TREINAMENTO DE TODAS AS CLASSES')
    print('=' * 80)
    print(summary_df.to_string(index=False))

    if not args.dry_run:
        csv_path = os.path.join(master_run_dir, 'training_summary.csv')
        md_path = os.path.join(master_run_dir, 'training_summary.md')
        summary_df.to_csv(csv_path, index=False)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Resumo do Treinamento de NF Heads (Run: {timestamp})\n\n")
            f.write(df_to_markdown(summary_df) + "\n")
        print(f"\nResumo salvo em:\n  CSV: {csv_path}\n  MD:  {md_path}")

    # Optional export to final_models/NF Head/
    if args.export_to_final_models and not args.dry_run:
        print("\n--- Exportando modelos treinados para final_models/NF Head/ ---")
        for row in summary_rows:
            if row.get('status') == 'SUCCESS' and row.get('checkpoint') and os.path.exists(row['checkpoint']):
                cls = row['class']
                dest_dir = project_path('final_models', 'NF Head', cls, 'best_models')
                os.makedirs(dest_dir, exist_ok=True)
                dest_file = os.path.join(dest_dir, 'best_pixel_auroc.pt')
                if os.path.exists(dest_file):
                    backup_file = os.path.join(dest_dir, f"best_pixel_auroc_backup_{timestamp}.pt")
                    shutil.copy2(dest_file, backup_file)
                    print(f"  [{cls}] Backup do modelo anterior: {os.path.basename(backup_file)}")
                shutil.copy2(row['checkpoint'], dest_file)
                print(f"  [{cls}] Atualizado: {dest_file}")

    # Optional automated post-training evaluation
    if args.evaluate_after != 'none' and not args.dry_run:
        successful_classes = [
            r['class'] for r in summary_rows
            if r.get('status') in ('SUCCESS', 'SKIPPED (already exists)')
            and r.get('checkpoint') and os.path.exists(r['checkpoint'])
        ]
        if successful_classes:
            classes_arg = ','.join(successful_classes)
            eval_script = (
                'scripts/eval/evaluate_pixel_level.py'
                if args.evaluate_after == 'pixel'
                else 'scripts/eval/evaluate_full_pipeline.py'
            )
            eval_out_dir = os.path.join(master_run_dir, 'avaliacao_pos_treino')
            eval_cmd = [
                sys.executable, os.path.join(PROJECT_ROOT, eval_script),
                '--classes', classes_arg,
                '--nf_dir', master_run_dir,
                '--out_dir', eval_out_dir,
            ]
            print(f"\n{'=' * 80}")
            print(f"  EXECUTANDO AVALIACAO POS-TREINO ({args.evaluate_after})")
            print(f"  Comando: {' '.join(eval_cmd)}")
            print('=' * 80)
            subprocess.run(eval_cmd, cwd=PROJECT_ROOT)
            print(f"\nAvaliacao concluida! Resultados salvos em:\n  {eval_out_dir}")


if __name__ == '__main__':
    main()
