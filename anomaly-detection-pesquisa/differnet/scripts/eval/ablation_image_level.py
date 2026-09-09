"""Ablation for the frozen SEDifferNet classifier (Image Level).

This script performs post-hoc ablation on the SEDifferNet model directly,
without loading the CFLOW NF head. It evaluates how different post-processing
techniques and scale subset selections affect the image-level anomaly detection.

Features explored:
- n_transforms: Number of test-time augmentations (rotations).
- levels: Which scales to include (0=full, 1=half, 2=quarter). 
          Unselected scales are masked out before passing to the NF.
- masking_mode: How to mask unselected scales ('zero' or 'mean_impute').
- z_score_mode: How to reduce the z**2 latent vector into a scalar score 
                ('mean', 'max', 'topk_0.1', 'topk_0.05').

Outputs, per run, under `resultado_analise_final/ablacao_image_level/run_<ts>/`:
    grid_val_image.csv        - Every grid configuration on validation subset
    holdout_test_image.csv    - Top configurations evaluated on held-out subset
    per_level_image_auroc.csv - Single-level performance isolated
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, SCRIPT_DIR)

import config as c
from core.model import SEDifferNet, load_weights
from core.eval_pipeline import (
    ALL_CLASSES, DEVICE, build_eval_datasets, eval_loader, unpack_batch,
    binary_metrics, find_se_checkpoint, compute_best_f1
)
from core.paths import project_path

# ==============================================================================
# Model Feature Extraction (Bypassing normal forward to inject masking)
# ==============================================================================

@torch.no_grad()
def extract_sediffernet_y_cat(model, x_input, pad=0, pad_mode='reflect'):
    """Extracts the pooled features for all scales before the NF head."""
    y_cat = []
    
    if pad > 0:
        x_input = F.pad(x_input, [pad]*4, mode=pad_mode)
        
    for s in range(c.n_scales):
        if s > 0:
            new_size = x_input.shape[-1] // (2 ** s)
            x_scaled = F.interpolate(x_input, size=new_size)
        else:
            x_scaled = x_input
            
        x = model.alexnet.features[0](x_scaled)
        x = model.alexnet.features[1](x)
        x = model.simsa1(x)
        x = model.alexnet.features[2](x)
        x = model.alexnet.features[3](x)
        x = model.alexnet.features[4](x)
        x = model.simsa2(x)
        x = model.alexnet.features[5](x)
        x = model.alexnet.features[6](x)
        x = model.alexnet.features[7](x)
        x = model.alexnet.features[8](x)
        x = model.alexnet.features[9](x)
        x = model.alexnet.features[10](x)
        x = model.alexnet.features[11](x)
        x = model.simsa4(x)
        feat_s = model.alexnet.features[12](x)
        y_cat.append(torch.mean(feat_s, dim=(2, 3)))
    return y_cat

@torch.no_grad()
def compute_train_means(model, train_loader):
    """Computes the mean vector of y_cat over the training set for mean_impute. 
    Note: Always computed with no padding for stable reference.
    """
    sums = [None] * c.n_scales
    count = 0
    for data in tqdm(train_loader, desc="  Computando médias de treino", leave=False):
        if len(data) == 3:
            images, _, _ = data
        else:
            images, _ = data
        images = images.to(DEVICE)
        
        # images may be (B, n_transforms, C, H, W)
        if images.dim() == 5:
            images = images.view(-1, *images.shape[-3:])
        
        y_cat = extract_sediffernet_y_cat(model, images, pad=0)
        for s in range(c.n_scales):
            if sums[s] is None:
                sums[s] = torch.zeros_like(y_cat[s][0])
            sums[s] += y_cat[s].sum(dim=0)
        count += images.size(0)
        
    means = [s / count for s in sums]
    return means

# ==============================================================================
# Scoring Methods
# ==============================================================================

def reduce_z_squared(z_sq, mode):
    """Reduces the z**2 vector (B, D) into a scalar score (B)."""
    if mode == 'mean':
        return z_sq.mean(dim=1)
    elif mode == 'max':
        return z_sq.max(dim=1).values
    elif mode.startswith('topk_'):
        percent = float(mode.split('_')[1])
        k = max(1, int(z_sq.size(1) * percent))
        topk_vals = torch.topk(z_sq, k, dim=1).values
        return topk_vals.mean(dim=1)
    else:
        raise ValueError(f"Unknown z_score_mode: {mode}")

def apply_masking(y_cat, levels_to_keep, masking_mode, train_means):
    """Masks unselected scales in y_cat."""
    masked_y_cat = []
    for s in range(c.n_scales):
        if str(s) in levels_to_keep:
            masked_y_cat.append(y_cat[s])
        else:
            if masking_mode == 'zero':
                masked_y_cat.append(torch.zeros_like(y_cat[s]))
            elif masking_mode == 'mean_impute':
                mean_vec = train_means[s].unsqueeze(0).expand_as(y_cat[s])
                masked_y_cat.append(mean_vec)
    return masked_y_cat

@torch.no_grad()
def collect_sediffernet_features(model, loader, pad=0, pad_mode='reflect', desc=""):
    """Collects y_cat for all items in loader to allow fast offline ablation. Returns (y_cat_acc, elapsed_time)."""
    y_cat_acc = [[] for _ in range(c.n_scales)]
    start_time = time.perf_counter()
    for data in tqdm(loader, desc=desc, leave=False):
        images, _, _ = unpack_batch(data)
        if images.dim() == 5:
            # Keep flat for now, we'll reshape when aggregating
            images = images.view(-1, *images.shape[-3:])
        y_cat = extract_sediffernet_y_cat(model, images, pad=pad, pad_mode=pad_mode)
        for s in range(c.n_scales):
            y_cat_acc[s].append(y_cat[s].cpu())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    return [torch.cat(acc, dim=0) for acc in y_cat_acc], elapsed_time

# ==============================================================================
# Plotting
# ==============================================================================

def plot_main_effects(df, metric, save_path):
    factors = [f for f in df.columns if df[f].nunique() > 1 and f not in [metric, 'name', 'role', 'seed', 'scores_full']]
    if not factors:
        return
    fig, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, factors):
        agg = df.groupby(f)[metric].agg(['mean', 'max']).sort_index()
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg['mean'], width=0.4, label='média')
        ax.bar(x + 0.2, agg['max'], width=0.4, label='máximo')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in agg.index], rotation=30, ha='right')
        ax.set_title(f)
        ax.grid(alpha=0.3, axis='y')
    axes[0].set_ylabel(f'{metric} (val)')
    axes[0].legend(fontsize=8)
    lo = df[metric].min()
    axes[0].set_ylim(max(0.0, lo - 0.05), 1.0)
    fig.suptitle('Efeitos marginais por fator (grade completa, validação)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ablation post-hoc para nível de imagem (SEDifferNet).')
    parser.add_argument('--class_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default=c.dataset_path)
    parser.add_argument('--se_checkpoint', type=str, default=None)
    parser.add_argument('--seeds', type=str, default='42', help='Lista de seeds separadas por vírgula (ex: 42,123,999)')
    parser.add_argument('--seed', type=int, default=None, help='Seed única (legado, sobrepõe --seeds se especificado)')
    parser.add_argument('--val_fraction', type=float, default=0.4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--top_n', type=int, default=10)
    parser.add_argument('--select_metric', type=str, default='image_auroc')
    
    # Grid parameters
    parser.add_argument('--n_transforms', type=str, default='1,4,8,16')
    parser.add_argument('--pads', type=str, default='0,64,112')
    parser.add_argument('--pad_modes', type=str, default='reflect,replicate')
    parser.add_argument('--level_sets', type=str, default='012,01,02,12,0,1,2')
    parser.add_argument('--masking_modes', type=str, default='zero,mean_impute')
    parser.add_argument('--z_score_modes', type=str, default='mean,max,topk_0.1,topk_0.05')
    
    args = parser.parse_args()
    
    # Resolve seeds list
    if args.seed is not None:
        seeds = [args.seed]
    elif args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    else:
        seeds = [42]
        
    run_dir = project_path('resultado_analise_final', 'ablacao_image_level', f"run_{time.strftime('%Y%m%d_%H%M%S')}_{args.class_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    cfg_dict = vars(args)
    cfg_dict['resolved_seeds'] = seeds
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(cfg_dict, f, indent=4)
        
    print(f"\n{'=' * 70}\n  ABLATION IMAGE LEVEL (SEDifferNet) - {args.class_name}\n{'=' * 70}")
    print(f"  Seeds a executar: {seeds}")
    
    se_ckpt = args.se_checkpoint or find_se_checkpoint(args.class_name)
    if not se_ckpt:
        print(f"ERRO: SEDifferNet não encontrado para a classe {args.class_name}")
        return
    print(f"  Checkpoint: {se_ckpt}")
    
    model = SEDifferNet()
    model, _ = load_weights(model, se_ckpt)
    model.to(DEVICE).eval()
    
    # Pre-parse grid parameters
    n_transforms_list = [int(x) for x in args.n_transforms.split(',') if x]
    pads_list = [int(x) for x in args.pads.split(',') if x]
    pad_modes = [x for x in args.pad_modes.split(',') if x]
    level_sets = [x for x in args.level_sets.split(',') if x]
    masking_modes = [x for x in args.masking_modes.split(',') if x]
    z_score_modes = [x for x in args.z_score_modes.split(',') if x]
    
    all_grid_rows = []
    all_single_level_rows = []
    
    # Extract training means for mean imputation once (trainset is deterministic)
    trainset, _, _ = build_eval_datasets(args.class_name, args.dataset, n_transforms=1, seed=seeds[0])
    train_loader = eval_loader(trainset, batch_size=4)
    train_means = compute_train_means(model, train_loader)
    
    iters_per_seed = len(n_transforms_list) * len(pads_list) * len(pad_modes) * len(level_sets) * len(masking_modes) * len(z_score_modes)
    total_iters = len(seeds) * iters_per_seed
    pbar = tqdm(total=total_iters, desc="  Ablation grid")
    
    for seed_idx, current_seed in enumerate(seeds):
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)
            torch.cuda.manual_seed_all(current_seed)
            
        # Pre-compute features for all (TTA, Pad, Pad_Mode) settings for current seed
        cache_y_cat = {}
        for nt, pad, pad_mode in itertools.product(n_transforms_list, pads_list, pad_modes):
            if pad == 0 and pad_mode != pad_modes[0]:
                continue # Pad 0 is the same regardless of mode
                
            _, testset, labels = build_eval_datasets(args.class_name, args.dataset, n_transforms=nt, limit=args.limit, seed=current_seed)
            loader = eval_loader(testset, batch_size=1)
            y_cat_all, dt = collect_sediffernet_features(
                model, loader, pad=pad, pad_mode=pad_mode, 
                desc=f"  [Seed {current_seed}] TTA={nt}, pad={pad}({pad_mode})"
            )
            cache_y_cat[(nt, pad, pad_mode)] = (y_cat_all, dt, labels)
            
        for nt, pad, pad_mode in itertools.product(n_transforms_list, pads_list, pad_modes):
            if pad == 0 and pad_mode != pad_modes[0]:
                pbar.update(len(level_sets) * len(masking_modes) * len(z_score_modes))
                continue
                
            y_cat_all, y_cat_time, labels = cache_y_cat[(nt, pad, pad_mode)]
            # B * nt
            num_items = y_cat_all[0].size(0)
            B = num_items // nt
            
            for levels, mask_mode in itertools.product(level_sets, masking_modes):
                # Skip duplicate work: if keeping all levels, mask_mode has no effect
                if levels == '012' and mask_mode != masking_modes[0]:
                    pbar.update(len(z_score_modes))
                    continue
                    
                # Apply masking
                train_means_cpu = [m.cpu() for m in train_means]
                masked_y = apply_masking(y_cat_all, levels, mask_mode, train_means_cpu)
                y = torch.cat(masked_y, dim=1).to(DEVICE)
                
                # Run through NF (chunked to avoid OOM if batch is large)
                z_list = []
                chunk_size = 1024
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                nf_start = time.perf_counter()
                with torch.no_grad():
                    for i in range(0, y.size(0), chunk_size):
                        z_chunk = model.nf(y[i:i+chunk_size])
                        z_list.append(z_chunk.cpu())
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                nf_time = time.perf_counter() - nf_start
                z = torch.cat(z_list, dim=0)
                z_sq = z ** 2
                
                # Total inference time
                total_time = y_cat_time + nf_time
                fps = B / max(total_time, 1e-6)
                
                # Shape back to (B, nt, D) and average over TTA
                z_sq_tta = z_sq.view(B, nt, -1).mean(dim=1)
                
                for z_mode in z_score_modes:
                    scores = reduce_z_squared(z_sq_tta, z_mode).numpy()
                    
                    metrics = binary_metrics(labels, scores, prefix='image_')
                    
                    cfg = {
                        'seed': current_seed,
                        'n_transforms': nt,
                        'pad': pad,
                        'pad_mode': pad_mode if pad > 0 else '-',
                        'levels': levels,
                        'masking_mode': mask_mode if levels != '012' else '-',
                        'z_score_mode': z_mode,
                        'fps': fps
                    }
                    
                    if len(levels) == 1:
                        all_single_level_rows.append({**cfg, **metrics})
                    
                    cfg['name'] = f"TTA={nt}|pad={pad}({cfg['pad_mode']})|L={levels}|mask={cfg['masking_mode']}|z={z_mode}"
                    all_grid_rows.append({**cfg, **metrics, 'scores_full': scores})
                    pbar.update(1)
                    
    pbar.close()
    
    # Save Single-level results
    if all_single_level_rows:
        df_single = pd.DataFrame(all_single_level_rows)
        df_single.drop(columns=['scores_full'], errors='ignore', inplace=True)
        df_single.to_csv(os.path.join(run_dir, 'per_level_image_auroc.csv'), index=False)
        print("\n  Single-level (L0, L1, L2) results:")
        print(df_single.pivot_table(index=['n_transforms', 'masking_mode', 'z_score_mode'], 
                                    columns='levels', values='image_auroc', aggfunc='mean').round(4).to_string())
    
    # Save Grid results
    df_grid = pd.DataFrame(all_grid_rows)
    df_grid_val = df_grid.drop(columns=['scores_full'], errors='ignore')
    df_grid_val = df_grid_val.sort_values(['seed', args.select_metric], ascending=[True, False]).reset_index(drop=True)
    df_grid_val.to_csv(os.path.join(run_dir, 'grid_test_image.csv'), index=False)
    
    # Aggregated table when running multiple seeds
    if len(seeds) > 1:
        group_cols = ['name', 'levels', 'n_transforms', 'pad', 'pad_mode', 'masking_mode', 'z_score_mode']
        metric_cols = [c for c in ['image_auroc', 'image_ap', 'image_f1', 'fps'] if c in df_grid_val.columns]
        
        agg_dict = {}
        for m in metric_cols:
            agg_dict[f'{m}_mean'] = (m, 'mean')
            agg_dict[f'{m}_std'] = (m, 'std')
            
        df_agg = df_grid_val.groupby(group_cols).agg(**agg_dict).reset_index()
        sort_col = f"{args.select_metric}_mean" if f"{args.select_metric}_mean" in df_agg.columns else 'image_auroc_mean'
        df_agg = df_agg.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df_agg.to_csv(os.path.join(run_dir, 'grid_test_image_aggregated.csv'), index=False)
        
        print(f"\n  Top {min(args.top_n, len(df_agg))} Agregado ({len(seeds)} seeds) por {sort_col}:")
        display_cols = ['name', 'levels', 'n_transforms', 'pad', 'z_score_mode', 'fps_mean', 
                        'image_auroc_mean', 'image_auroc_std', 'image_ap_mean', 'image_f1_mean']
        print(df_agg.head(args.top_n)[[c for c in display_cols if c in df_agg.columns]].round(4).to_string(index=False))
    else:
        print(f"\n  Top {min(args.top_n, len(df_grid_val))} on all test images by {args.select_metric}:")
        display_cols = ['name', 'levels', 'n_transforms', 'pad', 'z_score_mode', 'fps', 'image_auroc', 'image_ap', 'image_f1']
        print(df_grid_val.head(args.top_n)[[c for c in display_cols if c in df_grid_val.columns]].round(4).to_string(index=False))
        
    plot_main_effects(df_grid_val, args.select_metric, os.path.join(run_dir, 'main_effects.png'))
    
    print(f"\n  Done. Results saved to {run_dir}")


if __name__ == '__main__':
    main()

