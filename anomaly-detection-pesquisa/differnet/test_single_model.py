import os
import sys
import glob
import gc
import argparse
import time
import numpy as np
import pandas as pd
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
    """Detect model architecture from state dict keys."""
    keys = set(state_dict.keys())
    has_feature_extractor = any(k.startswith('feature_extractor.') for k in keys)
    has_alexnet = any(k.startswith('alexnet.') for k in keys)

    if has_feature_extractor and not has_alexnet:
        return DifferNet

    if has_alexnet:
        if folder_name:
            folder_cls = load_architecture_from_group(folder_name)
            if folder_cls in (SEDifferNet, CBAMDifferNet):
                return folder_cls
        return CBAMDifferNet

    return None

def run_evaluation(model_group, class_name, mode="fast", output_dir="results_single", no_bg=False):
    # Create a timestamped run subdirectory
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    bg_suffix = "_nobg" if no_bg else ""
    run_dir = os.path.join(output_dir, f"run_{run_timestamp}_{mode}{bg_suffix}_{model_group}_{class_name}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Output directory: {run_dir}", flush=True)
    if no_bg:
        print("  [INFO] --no_bg enabled: black background pixels will be excluded from pixel-level metrics", flush=True)
    base_dir = r"c:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models"
    
    group_path = os.path.join(base_dir, model_group)
    class_path = os.path.join(group_path, class_name)
    
    if not os.path.exists(class_path):
        print(f"Error: {class_path} does not exist.", flush=True)
        return
        
    pt_files = glob.glob(os.path.join(class_path, "*.pt"))
    if len(pt_files) == 0:
        print(f"Error: No .pt files found in {class_path}.", flush=True)
        return
        
    pt_file = pt_files[0]
    
    print(f"\nEvaluating Group: {model_group} | Class: {class_name} | File: {os.path.basename(pt_file)}", flush=True)
    print("-" * 80, flush=True)
    sys.stdout.flush()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
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

    folder_cls = load_architecture_from_group(model_group)
    detected_cls = detect_architecture_from_state_dict(state_dict, folder_name=model_group)

    if detected_cls is not None and folder_cls is not None and detected_cls != folder_cls:
        print(f"  [WARN] Folder '{model_group}' suggests {arch_name.get(folder_cls, folder_cls.__name__)}, "
              f"but checkpoint matches {arch_name.get(detected_cls, detected_cls.__name__)}. "
              f"Using detected architecture.", flush=True)

    chosen_cls = detected_cls or folder_cls or DifferNet
    base_model = DifferNet()
    print(f"  Architecture: {arch_name.get(chosen_cls, chosen_cls.__name__)} (from folder '{model_group}')", flush=True)
    base_model.load_state_dict(state_dict)
    model = base_model
    model.to(device)
    model.eval()
    
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    
    _orig_workers = c.num_workers
    c.num_workers = 0
    try:
        train_set, test_set, ground_truth_set = load_datasets(c.dataset_path, class_name)
        _, test_loader, ground_truth_loader = make_dataloaders(train_set, test_set, ground_truth_set)
    except Exception as e:
        print(f"Could not load dataset for class {class_name}: {e}", flush=True)
        c.num_workers = _orig_workers
        return
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
    
    pbar = tqdm(test_loader, desc=f"Eval {model_group}/{class_name}", file=sys.stdout, dynamic_ncols=True)
    for i, data in enumerate(pbar):
        inputs, labels = preprocess_batch(data)
        
        if no_bg:
            norm_mean_t = torch.tensor(c.norm_mean, device=device).view(1, 3, 1, 1)
            norm_std_t = torch.tensor(c.norm_std, device=device).view(1, 3, 1, 1)
            img_raw = inputs * norm_std_t + norm_mean_t
            bg_mask = img_raw.mean(dim=1, keepdim=True) < 0.02
            bg_mask = bg_mask.expand_as(inputs)
            inputs = inputs.clone()
            inputs[bg_mask] = 0.0
        
        with torch.no_grad():
            z = model(inputs)
            test_labels.append(t2np(labels))
            test_z.append(z)
        total_images += inputs.size(0)

        if mode == "full":
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
                    fg_mask = None
                    if no_bg:
                        inputs_grouped = inputs.view(-1, c.n_transforms_test, *inputs.shape[-3:])
                        anom_inputs = inputs_grouped[labels > 0]
                        if anom_inputs.shape[0] > 0:
                            first_tf = anom_inputs[:, 0]
                            norm_mean_t = torch.tensor(c.norm_mean, device=device).view(1, 3, 1, 1)
                            norm_std_t = torch.tensor(c.norm_std, device=device).view(1, 3, 1, 1)
                            img_raw = first_tf * norm_std_t + norm_mean_t
                            fg_mask = t2np(img_raw.mean(dim=1) > 0.02)
                    
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
        "Model Group": model_group,
        "Class": class_name,
        
        "Img AUROC": img_metrics['AUROC'],
        "Img Accuracy": img_metrics['Accuracy'],
        "Img F1": img_metrics['F1'],
        "Img Precision": img_metrics['Precision'],
        "Img Recall": img_metrics['Recall'],
        "Img PR-AUC": img_metrics['PR-AUC'],
        "Img PG2": img_metrics['PG2'],
        
        "Pix AUROC": np.mean(pixel_auroc_list) if pixel_auroc_list else np.nan,
        "Pix Accuracy": np.mean(pixel_acc_list) if pixel_acc_list else np.nan,
        "Pix F1": np.mean(pixel_f1_list) if pixel_f1_list else np.nan,
        "Pix Precision": np.mean(pixel_prec_list) if pixel_prec_list else np.nan,
        "Pix Recall": np.mean(pixel_rec_list) if pixel_rec_list else np.nan,
        "Pix PR-AUC": np.mean(pixel_prauc_list) if pixel_prauc_list else np.nan,
        "Pix PG2": np.mean(pixel_pg2_list) if pixel_pg2_list else np.nan,
        
        "Latency (ms/img)": latency_ms,
        "Peak RAM (MB)": ram_mb,
        "Peak VRAM (MB)": vram_mb
    }
    
    print(f"\n--- Results ---", flush=True)
    print(f"Img AUROC: {res_dict['Img AUROC']:.4f} | Img F1: {res_dict['Img F1']:.4f} | Img PG2: {res_dict['Img PG2']:.4f}", flush=True)
    print(f"System: Latency {latency_ms:.2f}ms | RAM {ram_mb:.1f}MB | VRAM {vram_mb:.1f}MB", flush=True)
    sys.stdout.flush()

    del model, base_model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
        
    df = pd.DataFrame([res_dict])
    
    csv_path = os.path.join(run_dir, f"report_{mode}.csv")
    df.to_csv(csv_path, index=False)
    
    excel_path = os.path.join(run_dir, f"report_{mode}.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\nReport saved to:", flush=True)
    print(f"  CSV  -> {csv_path}", flush=True)
    print(f"  XLSX -> {excel_path}", flush=True)
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_group", type=str, required=True, help="Name of the model group (e.g. cbamdiffernet)")
    parser.add_argument("--class_name", type=str, required=True, help="Name of the class (e.g. transistor)")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Evaluation mode")
    parser.add_argument("--no_bg", action="store_true", help="Exclude black background pixels from pixel-level metrics (for background-removed images)")
    args = parser.parse_args()
    
    print(f"Starting single evaluation for {args.model_group}/{args.class_name} in '{args.mode}' mode...", flush=True)
    sys.stdout.flush()
    run_evaluation(model_group=args.model_group, class_name=args.class_name, mode=args.mode, no_bg=args.no_bg)

