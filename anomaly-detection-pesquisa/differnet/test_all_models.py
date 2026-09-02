import os
import sys
import glob
import gc
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score, average_precision_score
)
import skimage.transform
from torch.amp import autocast
from torch.autograd import Variable
from scipy.ndimage import rotate, gaussian_filter

import config as c
from utils import load_datasets, preprocess_batch, get_loss, t2np, make_dataloaders
from model import DifferNet, SEDifferNet, CBAMDifferNet, load_weights

device = c.device
c.set_seed(c.seed)

# --- Inference Optimizations ---
torch.backends.cudnn.benchmark = True        # optimize convs for fixed input size
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmuls (RTX 30/40 series)
torch.backends.cudnn.allow_tf32 = True        # TF32 for cuDNN convolutions

# --- Metric Helpers ---

def calculate_pg2(y_true, y_scores):
    """
    Calculates PG2 (Presorted Good at 2%).
    This means the True Negative Rate (correctly classified good parts)
    at a False Negative Rate of 2% (TPR = 98%).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    idx = np.where(tpr >= 0.98)[0]
    if len(idx) == 0:
        return 0.0
    idx = idx[0]
    tnr = 1.0 - fpr[idx]
    return tnr

def get_optimal_metrics(y_true, y_scores):
    if len(np.unique(y_true)) < 2:
        return {
            'Accuracy': np.nan, 'F1': np.nan, 'Precision': np.nan, 
            'Recall': np.nan, 'PR-AUC': np.nan, 'PG2': np.nan, 'AUROC': np.nan
        }
    
    auroc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    pg2 = calculate_pg2(y_true, y_scores)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    denominator = (precisions + recalls)
    f1_scores = np.divide(2 * (precisions * recalls), denominator, out=np.zeros_like(denominator), where=denominator!=0)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    y_pred = (y_scores >= best_thresh).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': prec,
        'Recall': rec,
        'PR-AUC': pr_auc,
        'PG2': pg2,
        'AUROC': auroc
    }

# --- System Metrics ---

def get_system_metrics():
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        vram_mb = 0.0
    return ram_mb, vram_mb

# --- Original Gradient Functions ---

def get_grad_maps(model, inputs, labels, optimizer):
    model.eval()
    inputs = Variable(inputs, requires_grad=True)
    
    with autocast('cuda'):
        z = model(inputs)
        loss = get_loss(z, model.nf.jacobian(run_forward=False))
    
    optimizer.zero_grad()
    loss.backward()

    if inputs.grad is None:
        return None

    grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    grad = grad[labels > 0]
    
    if grad.shape[0] == 0:
        return None

    grad = t2np(grad)
    degrees = -1 * np.arange(c.n_transforms_test) * 360.0 / c.n_transforms_test

    for i_item in range(c.n_transforms_test):
        old_shape = grad[:, i_item].shape
        img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
        img = np.transpose(img, [1, 2, 0])
        img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
        img = gaussian_filter(img, (0, 3, 3))
        grad[:, i_item] = np.reshape(img, old_shape)

    grad = np.reshape(grad, [grad.shape[0], -1, *grad.shape[-2:]])
    grad_img = np.mean(np.abs(grad), axis=1)
    grad_img_sq = grad_img ** 2
    return grad_img_sq

def calculate_pixel_level_metrics(predictions, ground_truth_masks, fg_mask=None):
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    ground_truth_masks_binary = (ground_truth_masks > 0).astype(int)

    if fg_mask is not None:
        # Resize foreground mask to match ground_truth shape and apply
        fg_resized = skimage.transform.resize(fg_mask.astype(float), ground_truth_masks.shape, mode='constant') > 0.5
        predictions_flat = predictions_resized[fg_resized].reshape(-1)
        ground_truth_flat = ground_truth_masks_binary[fg_resized].reshape(-1)
    else:
        predictions_flat = predictions_resized.reshape(-1)
        ground_truth_flat = ground_truth_masks_binary.reshape(-1)
    
    return get_optimal_metrics(ground_truth_flat, predictions_flat)

ARCH_MAP = {
    "cbam": CBAMDifferNet,
    "cbam differnet": CBAMDifferNet,
    "cbamdiffernet": CBAMDifferNet,
    "se": SEDifferNet,
    "sediffernet": SEDifferNet,
    "differnet": DifferNet,
}

def load_architecture_from_group(group_name):
    """Select architecture class based on the folder/group name."""
    key = group_name.strip().lower()
    for pattern, arch_cls in ARCH_MAP.items():
        if pattern in key:
            return arch_cls
    return None


def detect_architecture_from_state_dict(state_dict, folder_name=None):
    """Detect model architecture from state dict keys.

    - DifferNet uses 'feature_extractor.*' keys
    - CBAMDifferNet and SEDifferNet both use 'alexnet.*' + 'cbam*' + 'simsa*'
      (identical module definitions; they differ only in forward())
    - For CBAM vs SE disambiguation, fall back to folder_name
    """
    keys = set(state_dict.keys())
    has_feature_extractor = any(k.startswith('feature_extractor.') for k in keys)
    has_alexnet = any(k.startswith('alexnet.') for k in keys)

    if has_feature_extractor and not has_alexnet:
        return DifferNet

    if has_alexnet:
        # Both CBAMDifferNet and SEDifferNet have identical state dict structure;
        # use folder name to disambiguate.
        if folder_name:
            folder_cls = load_architecture_from_group(folder_name)
            if folder_cls in (SEDifferNet, CBAMDifferNet):
                return folder_cls
        return CBAMDifferNet  # default for alexnet-based checkpoints

    return None  # unknown structure

def run_evaluation(mode="fast", output_dir="results_aggregated", no_bg=False):
    # Create a timestamped run subdirectory
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    bg_suffix = "_nobg" if no_bg else ""
    run_dir = os.path.join(output_dir, f"run_{run_timestamp}_{mode}{bg_suffix}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}", flush=True)
    if no_bg:
        print("  [INFO] --no_bg enabled: black background pixels will be excluded from pixel-level metrics", flush=True)
    base_dir = r"c:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.", flush=True)
        return
        
    model_groups = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Pre-calculate the total list of models to evaluate for flat looping and clear progress logging
    eval_list = []
    for group in model_groups:
        group_path = os.path.join(base_dir, group)
        classes = [d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))]
        for cls in classes:
            class_path = os.path.join(group_path, cls)
            pt_files = glob.glob(os.path.join(class_path, "*.pt"))
            if len(pt_files) > 0:
                eval_list.append((group, cls, pt_files[0]))
                
    total_evals = len(eval_list)
    results = []

    # Print discovery summary before starting evaluation
    print(f"\nFound {total_evals} models to evaluate:", flush=True)
    print("-" * 80, flush=True)
    for i, (group, cls, pt_file) in enumerate(eval_list, 1):
        folder_cls = load_architecture_from_group(group)
        detected_arch = folder_cls.__name__ if folder_cls else "Unknown"
        print(f"  [{i:02d}] {detected_arch:<20} | {os.path.relpath(pt_file)}", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()

    for idx, (group, cls, pt_file) in enumerate(eval_list, 1):
        print(f"\n--- [{idx}/{total_evals}] Evaluating Group: {group} | Class: {cls} ---", flush=True)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        # Load file and inspect state dict
        loaded_content = torch.load(pt_file, map_location=device, weights_only=False)
        if isinstance(loaded_content, dict) and 'model_state_dict' in loaded_content:
            state_dict = loaded_content['model_state_dict']
        elif hasattr(loaded_content, 'state_dict'):
            state_dict = loaded_content.state_dict()
        else:
            state_dict = loaded_content
            
        arch_name = {
            DifferNet: "DifferNet",
            SEDifferNet: "SEDifferNet",
            CBAMDifferNet: "CBAMDifferNet",
        }

        # Detect architecture from state dict keys; folder name is a secondary signal
        folder_cls = load_architecture_from_group(group)
        detected_cls = detect_architecture_from_state_dict(state_dict, folder_name=group)

        if detected_cls is not None and folder_cls is not None and detected_cls != folder_cls:
            print(f"  [WARN] Folder '{group}' suggests {arch_name.get(folder_cls, folder_cls.__name__)}, "
                  f"but checkpoint matches {arch_name.get(detected_cls, detected_cls.__name__)}. "
                  f"Using detected architecture.", flush=True)

        chosen_cls = detected_cls or folder_cls or DifferNet
        base_model = chosen_cls()
        print(f"  Architecture: {arch_name.get(chosen_cls, chosen_cls.__name__)} (from folder '{group}')", flush=True)
        base_model.load_state_dict(state_dict)
        model = base_model
        model.to(device)
        model.eval()
        
        optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
        
        # Load using the identical logic as train/main, enforcing full config params
        # Use num_workers=0 on Windows to prevent worker processes from re-spawning the script
        _orig_workers = c.num_workers
        c.num_workers = 0
        try:
            train_set, test_set, ground_truth_set = load_datasets(c.dataset_path, cls)
            _, test_loader, ground_truth_loader = make_dataloaders(train_set, test_set, ground_truth_set)
        except Exception as e:
            print(f"Could not load dataset for class {cls}: {e}", flush=True)
            c.num_workers = _orig_workers
            continue
        finally:
            c.num_workers = _orig_workers

        test_labels = []
        test_z = []
        
        pixel_auroc_list = []
        pixel_acc_list = []
        pixel_f1_list = []
        pixel_prec_list = []
        pixel_rec_list = []
        pixel_prauc_list = []
        pixel_pg2_list = []
        
        total_images = 0
        start_time = time.time()
        
        test_loader.dataset.get_fixed = True
        gt_iter = None
        
        # tqdm writing to stdout and flushing for instant terminal updates
        pbar = tqdm(test_loader, desc=f"Eval {group}/{cls}", file=sys.stdout, dynamic_ncols=True)
        for i, data in enumerate(pbar):
            inputs, labels = preprocess_batch(data)
            
            # Replace black background pixels with neutral values (ImageNet mean)
            if no_bg:
                norm_mean_t = torch.tensor(c.norm_mean, device=device).view(1, 3, 1, 1)
                norm_std_t = torch.tensor(c.norm_std, device=device).view(1, 3, 1, 1)
                img_raw = inputs * norm_std_t + norm_mean_t  # un-normalize to [0, 1]
                bg_mask = img_raw.mean(dim=1, keepdim=True) < 0.02  # detect black pixels
                bg_mask = bg_mask.expand_as(inputs)
                inputs = inputs.clone()
                inputs[bg_mask] = 0.0  # 0 in normalized space = ImageNet mean in pixel space
            
            with torch.no_grad():
                z = model(inputs)
                test_labels.append(t2np(labels))
                test_z.append(z)
            total_images += inputs.size(0)

            if mode == "full":
                # Skip gradient computation for normal-only batches (no anomalies)
                if t2np(labels).max() > 0:
                    with torch.enable_grad():
                        grad_map = get_grad_maps(model, inputs, labels, optimizer)
                else:
                    grad_map = None
                
                if grad_map is not None:
                    if gt_iter is None:
                        gt_iter = iter(ground_truth_loader)
                    try:
                        gt_data = next(gt_iter)
                    except StopIteration:
                        gt_iter = iter(ground_truth_loader)
                        gt_data = next(gt_iter)
                        
                    gt_masks = gt_data[0].to(device)
                    gt_masks = gt_masks[:grad_map.shape[0]]
                    
                    if len(gt_masks) == len(grad_map):
                        # Build foreground mask from input images when --no_bg is enabled
                        fg_mask = None
                        if no_bg:
                            # Reshape inputs to (batch, n_transforms, C, H, W) and select anomalous samples
                            inputs_grouped = inputs.view(-1, c.n_transforms_test, *inputs.shape[-3:])
                            anom_inputs = inputs_grouped[labels > 0]  # same filter as get_grad_maps
                            if anom_inputs.shape[0] > 0:
                                # Take the first transform (0° rotation = original orientation)
                                first_tf = anom_inputs[:, 0]  # (N_anom, 3, H, W)
                                # Un-normalize: pixel = tensor * std + mean → back to [0, 1]
                                norm_mean_t = torch.tensor(c.norm_mean, device=device).view(1, 3, 1, 1)
                                norm_std_t = torch.tensor(c.norm_std, device=device).view(1, 3, 1, 1)
                                img_raw = first_tf * norm_std_t + norm_mean_t
                                # Foreground = mean RGB > threshold (black pixels ≈ 0)
                                fg_mask = t2np(img_raw.mean(dim=1) > 0.02)  # (N_anom, H, W)
                        
                        p_metrics = calculate_pixel_level_metrics(grad_map, t2np(gt_masks), fg_mask=fg_mask)
                        if not np.isnan(p_metrics['AUROC']):
                            pixel_auroc_list.append(p_metrics['AUROC'])
                            pixel_acc_list.append(p_metrics['Accuracy'])
                            pixel_f1_list.append(p_metrics['F1'])
                            pixel_prec_list.append(p_metrics['Precision'])
                            pixel_rec_list.append(p_metrics['Recall'])
                            pixel_prauc_list.append(p_metrics['PR-AUC'])
                            pixel_pg2_list.append(p_metrics['PG2'])

            if i % 20 == 0 or i == len(test_loader) - 1:
                ram_mb, vram_mb = get_system_metrics()
                pbar.set_postfix({"images": total_images, "RAM_MB": f"{ram_mb:.0f}", "VRAM_MB": f"{vram_mb:.0f}"})
                sys.stdout.flush()

        test_loader.dataset.get_fixed = False
        
        elapsed_time = time.time() - start_time
        latency_ms = (elapsed_time / total_images) * 1000 if total_images > 0 else 0
        
        is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        
        img_metrics = get_optimal_metrics(is_anomaly, anomaly_score)
        ram_mb, vram_mb = get_system_metrics()
        
        res_dict = {
            "Model Group": group,
            "Class": cls,
            
            # Image Level
            "Img AUROC": img_metrics['AUROC'],
            "Img Accuracy": img_metrics['Accuracy'],
            "Img F1": img_metrics['F1'],
            "Img Precision": img_metrics['Precision'],
            "Img Recall": img_metrics['Recall'],
            "Img PR-AUC": img_metrics['PR-AUC'],
            "Img PG2": img_metrics['PG2'],
            
            # Pixel Level
            "Pix AUROC": np.mean(pixel_auroc_list) if pixel_auroc_list else np.nan,
            "Pix Accuracy": np.mean(pixel_acc_list) if pixel_acc_list else np.nan,
            "Pix F1": np.mean(pixel_f1_list) if pixel_f1_list else np.nan,
            "Pix Precision": np.mean(pixel_prec_list) if pixel_prec_list else np.nan,
            "Pix Recall": np.mean(pixel_rec_list) if pixel_rec_list else np.nan,
            "Pix PR-AUC": np.mean(pixel_prauc_list) if pixel_prauc_list else np.nan,
            "Pix PG2": np.mean(pixel_pg2_list) if pixel_pg2_list else np.nan,
            
            # System
            "Latency (ms/img)": latency_ms,
            "Peak RAM (MB)": ram_mb,
            "Peak VRAM (MB)": vram_mb
        }
        results.append(res_dict)
        
        print(f"Img AUROC: {res_dict['Img AUROC']:.4f} | Img F1: {res_dict['Img F1']:.4f} | Img PG2: {res_dict['Img PG2']:.4f}", flush=True)
        print(f"System: Latency {latency_ms:.2f}ms | RAM {ram_mb:.1f}MB | VRAM {vram_mb:.1f}MB", flush=True)
        sys.stdout.flush()

        # Free GPU memory between models to avoid VRAM accumulation
        del model, base_model, optimizer
        torch.cuda.empty_cache()
        gc.collect()
            
    df = pd.DataFrame(results)
    
    # Save as CSV
    csv_path = os.path.join(run_dir, f"final_report_{mode}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save as Excel
    excel_path = os.path.join(run_dir, f"final_report_{mode}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nFinal report saved to:", flush=True)
    print(f"  CSV  -> {csv_path}", flush=True)
    print(f"  XLSX -> {excel_path}", flush=True)
    sys.stdout.flush()
    
    # ================================================================
    # PLOTTING SECTION
    # ================================================================
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("Set2", n_colors=df["Model Group"].nunique())
    model_colors = {g: palette[i] for i, g in enumerate(df["Model Group"].unique())}

    # --- Helper: basic grouped bar ---
    def plot_metric(metric_name, filename):
        if not df[metric_name].isnull().all():
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=df, x="Class", y=metric_name, hue="Model Group", palette=model_colors, ax=ax)
            ax.set_title(f"{metric_name} Comparison")
            plt.xticks(rotation=45)
            is_ratio = "Latency" not in metric_name and "RAM" not in metric_name and "VRAM" not in metric_name
            if is_ratio:
                ax.set_ylim(0, 1.15)
            fmt = "%.3f" if is_ratio else "%.1f"
            for container in ax.containers:
                ax.bar_label(container, fmt=fmt, fontsize=7, rotation=90, padding=3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, filename), dpi=150)
            plt.close()

    # 1. Original simple bar charts
    plot_metric("Img AUROC", f"img_auroc_comparison_{mode}.png")
    plot_metric("Img F1", f"img_f1_comparison_{mode}.png")
    plot_metric("Img PG2", f"img_pg2_comparison_{mode}.png")
    plot_metric("Latency (ms/img)", f"latency_comparison_{mode}.png")
    plot_metric("Peak VRAM (MB)", f"vram_comparison_{mode}.png")

    if mode == "full":
        plot_metric("Pix AUROC", f"pix_auroc_comparison_{mode}.png")
        plot_metric("Pix F1", f"pix_f1_comparison_{mode}.png")
        plot_metric("Pix PG2", f"pix_pg2_comparison_{mode}.png")

    # 2a. Grouped bar chart: Img AUROC por Classe e Modelo
    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=df, x="Class", y="Img AUROC", hue="Model Group", palette=model_colors, ax=ax)
        ax.set_ylim(0, 1.20)
        ax.set_title("Img AUROC por Classe e Modelo", fontsize=14, fontweight='bold')
        ax.set_ylabel("Img AUROC")
        ax.set_xlabel("Classe")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"grouped_auroc_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate grouped AUROC chart: {e}", flush=True)

    # 2b. Grouped bar chart: Img Accuracy por Classe e Modelo
    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(data=df, x="Class", y="Img Accuracy", hue="Model Group", palette=model_colors, ax=ax)
        ax.set_ylim(0, 1.20)
        ax.set_title("Img Accuracy por Classe e Modelo", fontsize=14, fontweight='bold')
        ax.set_ylabel("Img Accuracy")
        ax.set_xlabel("Classe")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"grouped_accuracy_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate grouped Accuracy chart: {e}", flush=True)

    # 3. Scatter plot: Img AUROC vs Latency (ms/img) — efficiency frontier
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        for group, group_df in df.groupby("Model Group"):
            ax.scatter(
                group_df["Latency (ms/img)"], group_df["Img AUROC"],
                label=group, color=model_colors[group], s=120, edgecolors='black', linewidths=0.5, zorder=3
            )
            for _, row in group_df.iterrows():
                ax.annotate(
                    row["Class"], (row["Latency (ms/img)"], row["Img AUROC"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, alpha=0.8
                )
        ax.set_xlabel("Latência (ms/img)")
        ax.set_ylabel("Img AUROC")
        ax.set_title("Img AUROC vs Tempo de Inferência", fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"auroc_vs_latency_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate AUROC vs Latency chart: {e}", flush=True)

    # 4. Radar / Spider chart — one per Model Group (mean metrics across classes)
    try:
        from math import pi
        radar_metrics = ["Img AUROC", "Img Accuracy", "Img F1", "Img Precision", "Img Recall", "Img PR-AUC", "Img PG2"]
        mean_df = df.groupby("Model Group")[radar_metrics].mean()

        angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        for group in mean_df.index:
            values = mean_df.loc[group].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=group, color=model_colors[group])
            ax.fill(angles, values, alpha=0.15, color=model_colors[group])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace("Img ", "") for m in radar_metrics], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title("Perfil de Métricas por Arquitetura (Média)", fontsize=13, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"radar_chart_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate radar chart: {e}", flush=True)

    # 5. Heatmap — all image-level metrics per Model+Class
    try:
        heatmap_cols = ["Img AUROC", "Img Accuracy", "Img F1", "Img Precision", "Img Recall", "Img PR-AUC", "Img PG2"]
        heatmap_df = df.set_index(["Model Group", "Class"])[heatmap_cols]
        heatmap_df.index = [f"{g} / {c}" for g, c in heatmap_df.index]

        fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_df) * 0.7)))
        sns.heatmap(
            heatmap_df, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Valor"}
        )
        ax.set_title("Heatmap de Métricas Image-Level", fontsize=14, fontweight='bold')
        ax.set_ylabel("")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"heatmap_metrics_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate heatmap: {e}", flush=True)

    # 6. Horizontal bar — Mean Img AUROC por Model Group (ranking)
    try:
        mean_auroc = df.groupby("Model Group")["Img AUROC"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(mean_auroc.index, mean_auroc.values, color=[model_colors[g] for g in mean_auroc.index], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, mean_auroc.values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va='center', fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Img AUROC Médio")
        ax.set_title("Ranking de Arquiteturas por AUROC Médio", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"ranking_auroc_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate ranking chart: {e}", flush=True)

    # 7. Box plot — distribution of each metric across classes per Model Group
    try:
        box_metrics = ["Img AUROC", "Img F1", "Img Precision", "Img Recall", "Img PR-AUC"]
        melted_box = df.melt(
            id_vars=["Model Group"], value_vars=box_metrics,
            var_name="Metric", value_name="Value"
        )
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.boxplot(data=melted_box, x="Metric", y="Value", hue="Model Group", palette=model_colors, ax=ax)
        sns.stripplot(data=melted_box, x="Metric", y="Value", hue="Model Group", palette=model_colors,
                      dodge=True, size=5, alpha=0.6, ax=ax, legend=False)
        ax.set_ylim(0, 1.05)
        ax.set_title("Distribuição de Métricas por Arquitetura (Box Plot)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Valor")
        ax.set_xlabel("")
        handles, labels = ax.get_legend_handles_labels()
        n_groups = df["Model Group"].nunique()
        ax.legend(handles[:n_groups], labels[:n_groups], bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"boxplot_metrics_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate boxplot chart: {e}", flush=True)

    # 8. Efficiency quadrant — Img AUROC vs Latency, annotated quadrants
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        for group, group_df in df.groupby("Model Group"):
            ax.scatter(
                group_df["Latency (ms/img)"], group_df["Img AUROC"],
                label=group, color=model_colors[group], s=140, edgecolors='black', linewidths=0.5, zorder=3, marker='D'
            )
        median_lat = df["Latency (ms/img)"].median()
        median_auroc = df["Img AUROC"].median()
        ax.axvline(median_lat, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(median_auroc, color='gray', linestyle='--', alpha=0.5)
        ax.text(df["Latency (ms/img)"].min(), 1.01, "Rápido + Alto AUROC ★", fontsize=8, color='green', fontweight='bold')
        ax.text(df["Latency (ms/img)"].max() * 0.65, df["Img AUROC"].min() - 0.02, "Lento + Baixo AUROC", fontsize=8, color='red', fontweight='bold')
        ax.set_xlabel("Latência (ms/img)")
        ax.set_ylabel("Img AUROC")
        ax.set_title("Quadrante de Eficiência: AUROC vs Latência", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"efficiency_quadrant_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate efficiency quadrant chart: {e}", flush=True)

    # 9. Correlation matrix — numeric columns
    try:
        numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.int64, float, int] and not df[c].isnull().all()]
        if len(numeric_cols) >= 3:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                        linewidths=0.5, ax=ax, square=True, cbar_kws={"shrink": 0.8})
            ax.set_title("Matriz de Correlação entre Métricas", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"correlation_matrix_{mode}.png"), dpi=150)
            plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate correlation matrix: {e}", flush=True)

    # 10. PR-AUC vs F1 scatter — precision-recall trade-off view
    try:
        fig, ax = plt.subplots(figsize=(9, 7))
        for group, group_df in df.groupby("Model Group"):
            ax.scatter(
                group_df["Img PR-AUC"], group_df["Img F1"],
                label=group, color=model_colors[group], s=120, edgecolors='black', linewidths=0.5
            )
            for _, row in group_df.iterrows():
                ax.annotate(row["Class"], (row["Img PR-AUC"], row["Img F1"]),
                            textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.8)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, label='y=x')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Img PR-AUC")
        ax.set_ylabel("Img F1")
        ax.set_title("PR-AUC vs F1-Score", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"prauc_vs_f1_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate PR-AUC vs F1 chart: {e}", flush=True)

    # 11. Lollipop chart — PG2 per Model+Class
    try:
        lollipop_df = df[["Model Group", "Class", "Img PG2"]].copy()
        lollipop_df["Label"] = lollipop_df["Model Group"] + " / " + lollipop_df["Class"]
        lollipop_df = lollipop_df.sort_values("Img PG2", ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, max(6, len(lollipop_df) * 0.5)))
        colors = [model_colors[g] for g in lollipop_df["Model Group"]]
        ax.hlines(y=lollipop_df["Label"], xmin=0, xmax=lollipop_df["Img PG2"], color=colors, alpha=0.7, linewidth=2)
        ax.scatter(lollipop_df["Img PG2"], lollipop_df["Label"], color=colors, s=80, zorder=3, edgecolors='black', linewidths=0.5)
        for _, row in lollipop_df.iterrows():
            ax.text(row["Img PG2"] + 0.01, row["Label"], f"{row['Img PG2']:.3f}", va='center', fontsize=8)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Img PG2")
        ax.set_title("PG2 (Presorted Good at 2%) por Modelo e Classe", fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"lollipop_pg2_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate PG2 lollipop chart: {e}", flush=True)

    # 12. Grouped bar — Mean Latency + Mean VRAM per architecture (resource comparison)
    try:
        resource_mean = df.groupby("Model Group")[["Latency (ms/img)", "Peak VRAM (MB)"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        bars1 = axes[0].bar(resource_mean.index, resource_mean["Latency (ms/img)"],
                            color=[model_colors[g] for g in resource_mean.index], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars1, resource_mean["Latency (ms/img)"]):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}", ha='center', fontsize=9)
        axes[0].set_title("Latência Média (ms/img)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("ms/img")
        axes[0].grid(axis='y', alpha=0.3)

        bars2 = axes[1].bar(resource_mean.index, resource_mean["Peak VRAM (MB)"],
                            color=[model_colors[g] for g in resource_mean.index], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars2, resource_mean["Peak VRAM (MB)"]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                         f"{val:.0f}", ha='center', fontsize=9)
        axes[1].set_title("VRAM Média (MB)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("MB")
        axes[1].grid(axis='y', alpha=0.3)

        plt.suptitle("Comparação de Recursos por Arquitetura", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"resource_comparison_{mode}.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate resource comparison chart: {e}", flush=True)

    # 13. Per-class performance comparison — small multiples (one subplot per class)
    try:
        classes = df["Class"].unique()
        n_classes = len(classes)
        compare_metrics = ["Img AUROC", "Img F1", "Img Precision", "Img Recall"]
        fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6), sharey=True)
        if n_classes == 1:
            axes = [axes]

        for ax, cls_name in zip(axes, classes):
            cls_df = df[df["Class"] == cls_name]
            x = np.arange(len(compare_metrics))
            width = 0.8 / len(cls_df)
            for j, (_, row) in enumerate(cls_df.iterrows()):
                vals = [row[m] for m in compare_metrics]
                bars = ax.bar(x + j * width, vals, width, label=row["Model Group"],
                              color=model_colors[row["Model Group"]], edgecolor='black', linewidth=0.3)
                ax.bar_label(bars, fmt="%.3f", fontsize=6, rotation=90, padding=2)
            ax.set_xticks(x + width * (len(cls_df) - 1) / 2)
            ax.set_xticklabels([m.replace("Img ", "") for m in compare_metrics], fontsize=8, rotation=30, ha='right')
            ax.set_ylim(0, 1.20)
            ax.set_title(cls_name, fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        axes[0].set_ylabel("Valor")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=len(df["Model Group"].unique()), fontsize=9, bbox_to_anchor=(0.5, 1.02))
        plt.suptitle("Comparação por Classe (Small Multiples)", fontsize=14, fontweight='bold', y=1.06)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"small_multiples_{mode}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [WARN] Could not generate small multiples chart: {e}", flush=True)

    print(f"\nAll charts saved to: {run_dir}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Evaluation mode")
    parser.add_argument("--no_bg", action="store_true", help="Exclude black background pixels from pixel-level metrics (for background-removed images)")
    args = parser.parse_args()
    
    print(f"Starting aggregated evaluation in '{args.mode}' mode with full metrics...", flush=True)
    sys.stdout.flush()
    run_evaluation(mode=args.mode, no_bg=args.no_bg)
