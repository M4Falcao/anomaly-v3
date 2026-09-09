"""Ablation for the full pipeline fusing SEDifferNet (Image Level) and CFLOW NF Head (Pixel Level).

This script explores methods to unify the image-level score from SEDifferNet and the pixel-level 
score from CFLOW to improve the overall image classification accuracy (Anomaly vs Normal).
We test 3 fusion strategies:
1. Min-Max Normalized Weighted Sum
2. Rank Averaging
3. Z-Score Fusion (calibrated on train set)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, SCRIPT_DIR)

import config as c
from core.model import SEDifferNet, load_weights
from core.eval_pipeline import (
    ALL_CLASSES, DEVICE, find_se_checkpoint, find_nf_head_checkpoint,
    binary_metrics, compute_best_f1, resolve_score_levels
)
from core.utils import load_datasets, t2np
from core.cflow import (
    CFlowPixelHead, SEBackboneFeatureExtractor, read_cflow_hparams, resolve_hparam,
    TRAIN_OUT_SIZE, TRAIN_CLAMP_SCALE, infer_n_blocks, select_levels, nll_with_tta,
    image_score_from_map, border_mask, gaussian_smooth
)

def compute_z_score(scores, mean, std):
    return (scores - mean) / (std + 1e-8)

def compute_minmax(scores):
    s_min, s_max = scores.min(), scores.max()
    return (scores - s_min) / (s_max - s_min + 1e-8)

def compute_ranks(scores):
    # rankdata assigns rank 1 to smallest. 
    # normalize to 0-1
    ranks = rankdata(scores)
    return (ranks - 1) / (len(scores) - 1 + 1e-8)

def evaluate_class(class_name, args, class_output_dir):
    print(f"\n{'='*70}")
    print(f"  Ablation Full Pipeline: {class_name}")
    print(f"{'='*70}")

    # 1. Checkpoints
    se_ckpt_path = find_se_checkpoint(class_name)
    nf_ckpt_path = find_nf_head_checkpoint(class_name, base_dir=args.nf_dir)

    if not se_ckpt_path or not nf_ckpt_path:
        print(f"  [ERRO] Checkpoints missing for {class_name}")
        return None

    # 2. Models
    model = SEDifferNet().to(DEVICE).eval()
    model, _ = load_weights(model, se_ckpt_path)

    nf_data = torch.load(nf_ckpt_path, map_location=DEVICE, weights_only=False)
    cflow_sd = nf_data.get('cflow_state_dict') or nf_data.get('model_state_dict')
    hp = read_cflow_hparams(nf_data)
    
    out_size, _ = resolve_hparam('out_size', None, hp, TRAIN_OUT_SIZE)
    clamp, _ = resolve_hparam('clamp_scale', None, hp, TRAIN_CLAMP_SCALE)
    n_blocks, _ = resolve_hparam('n_blocks', None, hp, infer_n_blocks(cflow_sd), warn=False)
    
    trained_levels = list(hp.get('levels') or [0, 1, 2])
    feat_pool = int(hp.get('feat_pool') or 0)
    pad_mode = hp.get('pad_mode') or 'reflect'
    feat_stats = nf_data.get('feat_stats')
    channels = select_levels([64, 192, 256], trained_levels)
    score_idx, _ = resolve_score_levels('per_class', trained_levels, class_name)

    backbone = SEBackboneFeatureExtractor(
        model, out_size=out_size, reflect_pad=112,
        attention='se', pad_mode=pad_mode, feat_pool=feat_pool).to(DEVICE).eval()

    cflow = CFlowPixelHead(
        layer_channels=channels, cond_dim=64, n_blocks=n_blocks,
        hidden=256, clamp_scale=clamp).to(DEVICE)
    cflow.load_state_dict(cflow_sd)
    cflow.eval()

    def get_pixel_map(images_flat):
        nll = nll_with_tta(backbone, cflow, images_flat, levels=trained_levels,
                           feat_stats=feat_stats, tta='flips', per_channel=False)
        smap = CFlowPixelHead.aggregate_nll(nll, channels, c.img_size[0], mode='raw',
                                            levels=score_idx)
        smap_np = t2np(smap)
        return gaussian_smooth(smap_np, sigma=8.0)

    # 3. Dataset
    original_class = c.class_name
    c.class_name = class_name
    c.n_transforms_test = 1
    c.transf_rotations = False

    trainset, testset = load_datasets(c.dataset_path, class_name, aligned=True)
    testset.n_transforms = 1
    testset.get_fixed = False

    from torchvision import transforms
    eval_transform = transforms.Compose([
        transforms.Resize(c.img_size),
        transforms.ToTensor(),
        transforms.Normalize(c.norm_mean, c.norm_std),
    ])
    trainset.transform = eval_transform
    testset.transform = eval_transform

    train_loader = DataLoader(trainset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 4. Extract CFLOW scores from trainset to compute baseline distributions for Z-Score Fusion
    print("  Extraindo distribuição do CFLOW (Treinamento)...")
    train_pix_max = []
    train_pix_topk = []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="  Treino CFLOW"):
            images = images.to(DEVICE).view(-1, *images.shape[-3:])
            smaps = get_pixel_map(images)
            for smap in smaps:
                train_pix_max.append(image_score_from_map(smap[np.newaxis, ...], mode='max')[0])
                train_pix_topk.append(image_score_from_map(smap[np.newaxis, ...], mode='topk', top_percent=0.05)[0])

    train_pix_max = np.array(train_pix_max)
    train_pix_topk = np.array(train_pix_topk)
    stat_pmax_mean, stat_pmax_std = train_pix_max.mean(), train_pix_max.std()
    stat_ptopk_mean, stat_ptopk_std = train_pix_topk.mean(), train_pix_topk.std()

    # 5. Extract CFLOW scores from testset
    print("  Avaliando Test Set (CFLOW)...")
    test_labels = []
    test_pix_max = []
    test_pix_topk = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="  Teste CFLOW"):
            images = data[0].to(DEVICE).view(-1, *data[0].shape[-3:])
            label_val = 1 if int(data[1][0].item()) > 0 else 0
            test_labels.append(label_val)
            smap = get_pixel_map(images)
            test_pix_max.append(image_score_from_map(smap, mode='max')[0])
            test_pix_topk.append(image_score_from_map(smap, mode='topk', top_percent=0.05)[0])

    test_labels = np.array(test_labels)
    test_pix_max = np.array(test_pix_max)
    test_pix_topk = np.array(test_pix_topk)

    if len(np.unique(test_labels)) < 2:
        print("  [AVISO] Conjunto de teste tem apenas 1 classe.")
        c.class_name = original_class
        return None

    results = []
    def log_result(method, agg, scores, t_val):
        auroc = roc_auc_score(test_labels, scores)
        ap = average_precision_score(test_labels, scores)
        prec, rec, _ = precision_recall_curve(test_labels, scores)
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
        f1 = f1_scores.max()
        results.append({
            'class': class_name,
            'n_transforms': t_val,
            'method': method,
            'pixel_agg': agg,
            'auroc': auroc,
            'ap': ap,
            'f1': f1
        })

    # Loop over n_transforms for SEDifferNet
    n_transforms_list = [int(x.strip()) for x in args.n_transforms.split(',')]
    for t in n_transforms_list:
        print(f"  Avaliando SEDifferNet com n_transforms={t}...")
        
        # Train stats for this t
        trainset.n_transforms = t
        trainset.get_fixed = (t > 1)
        trainset.fixed_degrees = [i * 360.0 / t for i in range(t)]
        t_train_loader = DataLoader(trainset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
        train_img_scores = []
        with torch.no_grad():
            for images, labels in tqdm(t_train_loader, desc=f"  Treino SE (t={t})", leave=False):
                images = images.to(DEVICE).view(-1, *images.shape[-3:])
                z = model(images)
                z_sq = z**2
                z_sq = z_sq.view(-1, t, z_sq.shape[-1])
                scores_img = torch.mean(z_sq, dim=(1,2)).cpu().numpy()
                train_img_scores.extend(scores_img)
        train_img_scores = np.array(train_img_scores)
        stat_img_mean, stat_img_std = train_img_scores.mean(), train_img_scores.std()

        # Test scores for this t
        testset.n_transforms = t
        testset.get_fixed = (t > 1)
        testset.fixed_degrees = [i * 360.0 / t for i in range(t)]
        t_test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        test_img_scores = []
        with torch.no_grad():
            for data in tqdm(t_test_loader, desc=f"  Teste SE (t={t})", leave=False):
                images = data[0].to(DEVICE).view(-1, *data[0].shape[-3:])
                z = model(images)
                z_sq = z**2
                score_img = torch.mean(z_sq).item()
                test_img_scores.append(score_img)
        test_img_scores = np.array(test_img_scores)

        # Fusions
        log_result('Baseline Image (SEDifferNet)', 'N/A', test_img_scores, t)
        log_result('Baseline Pixel CFLOW', 'max', test_pix_max, t)
        log_result('Baseline Pixel CFLOW', 'topk0.05', test_pix_topk, t)

        for agg_name, pix_scores, mean_p, std_p in [
            ('max', test_pix_max, stat_pmax_mean, stat_pmax_std),
            ('topk0.05', test_pix_topk, stat_ptopk_mean, stat_ptopk_std)
        ]:
            # A) Z-Score Fusion
            z_img = compute_z_score(test_img_scores, stat_img_mean, stat_img_std)
            z_pix = compute_z_score(pix_scores, mean_p, std_p)
            fused_z = z_img + z_pix
            log_result('Z-Score Sum', agg_name, fused_z, t)

            # B) Rank Averaging
            rank_img = compute_ranks(test_img_scores)
            rank_pix = compute_ranks(pix_scores)
            fused_rank = rank_img + rank_pix
            log_result('Rank Average', agg_name, fused_rank, t)

            # C) Min-Max Weighted Sum
            mm_img = compute_minmax(test_img_scores)
            mm_pix = compute_minmax(pix_scores)
            
            for a in [0.25, 0.5, 0.75]:
                fused_mm = a * mm_img + (1 - a) * mm_pix
                log_result(f'MinMax Sum (a={a:.2f})', agg_name, fused_mm, t)

    c.class_name = original_class

    df = pd.DataFrame(results)
    os.makedirs(class_output_dir, exist_ok=True)
    df.to_csv(os.path.join(class_output_dir, 'ablation_fusion.csv'), index=False, float_format='%.4f')
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Ablation Full Pipeline (Fusion)")
    parser.add_argument('--classes', type=str, default=None,
                        help='Comma-separated class names. Default: all 5 classes.')
    parser.add_argument('--nf_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_transforms', type=str, default='32', 
                        help='Comma-separated list of n_transforms to evaluate for SEDifferNet.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.classes:
        class_list = [cl.strip() for cl in args.classes.split(',') if cl.strip()]
    else:
        class_list = ALL_CLASSES

    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(SCRIPT_DIR, '..', '..')
    run_dir = os.path.join(base_dir, 'result_ablation_full', f'run_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    tested_modes = [
        "Baseline Image (SEDifferNet)",
        "Baseline Pixel CFLOW",
        "Z-Score Sum",
        "Rank Average",
        "MinMax Sum (a=0.25)",
        "MinMax Sum (a=0.50)",
        "MinMax Sum (a=0.75)"
    ]
    with open(os.path.join(run_dir, 'config.txt'), 'w') as f:
        f.write("Tested Fusion Modes:\n")
        for mode in tested_modes:
            f.write(f"- {mode}\n")

    print("=" * 80)
    print("  ABLAÇÃO: Fusão Image Level (SEDifferNet) + Pixel Level (CFLOW)")
    print("  Modos Testados:", tested_modes)
    print("=" * 80)

    all_dfs = []
    for cls in class_list:
        cls_dir = os.path.join(run_dir, cls)
        df_cls = evaluate_class(cls, args, cls_dir)
        if df_cls is not None:
            all_dfs.append(df_cls)
            
    if not all_dfs:
        print("Nenhum dado processado.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    
    # Resumo agregado (Mean AUROC per method across classes)
    summary = df_all.groupby(['n_transforms', 'method', 'pixel_agg'])[['auroc', 'ap', 'f1']].mean().reset_index()
    summary = summary.sort_values(by=['n_transforms', 'auroc'], ascending=[True, False])
    
    csv_summary = os.path.join(run_dir, 'summary_fusion.csv')
    summary.to_csv(csv_summary, index=False, float_format='%.4f')

    print("\n" + "=" * 80)
    print("  MÉDIA FINAL (CROSS-CLASS)")
    print("=" * 80)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\nResultados salvos em: {run_dir}")

if __name__ == '__main__':
    main()
