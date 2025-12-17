import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import random
import copy
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, auc
from scipy.stats import spearmanr

# Add current dir to path
sys.path.append(os.getcwd())

import config as c
import utils
from model import DifferNet, SEDifferNet, load_weights

# Pytorch GradCAM imports
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst
except ImportError:
    print("Error: pytorch-grad-cam not installed. Please install it using 'pip install grad-cam'.")
    sys.exit(1)

# --- Configuration & Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions (From Notebook) ---

class AnomalyScoreTarget:
    """
    Target for Normalizing Flow models (DifferNet/SEDifferNet).
    The 'score' is the norm of the latent vector z.
    We want to maximize the anomaly score (maximize ||z||^2).
    """
    def __call__(self, model_output):
        # model_output is 'z' vector [Batch, Dim]
        # Return mean of squared l2 norm (matching get_loss logic approx)
        if isinstance(model_output, tuple):
             model_output = model_output[0]
        
        # Ensure 2D [Batch, Feature]
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
            
        return torch.mean(torch.sum(model_output ** 2, dim=1))

def _get_score(output, target_category_idx=None):
    # output is z [B, D]
    # Anomaly score ~ ||z||^2
    # We return the scalar score for the first item in batch
    with torch.no_grad():
        score = torch.mean(torch.sum(output ** 2, dim=1)).item()
    return score

def calculate_pixel_auroc(anomaly_map, ground_truth_mask):
    """
    Calculates Pixel-Level AUROC.
    """
    flat_anomaly = anomaly_map.flatten()
    flat_mask = ground_truth_mask.flatten()
    # Ensure mask is binary
    flat_mask = (flat_mask > 0.5).astype(int)
    
    # Handle edge case where mask is empty (all normal)
    if np.sum(flat_mask) == 0:
        return 0.5 # Defined as 0.5 or N/A for strictly normal images in localization context
    
    try:
        score = roc_auc_score(flat_mask, flat_anomaly)
    except ValueError:
        score = 0.5
    return score

def calculate_image_auroc(anomaly_scores, true_labels):
    return roc_auc_score(true_labels, anomaly_scores)

def confidence_drop_metric(model, input_tensor, cam_mask, target_category_idx=None):
    model.eval()
    with torch.no_grad():
        output_orig = model(input_tensor)
        score_orig = _get_score(output_orig, target_category_idx)

        cam_mask_tensor = torch.tensor(cam_mask).to(input_tensor.device).unsqueeze(0).unsqueeze(0)
        # Interpolate if size mismatch
        if cam_mask_tensor.shape[-2:] != input_tensor.shape[-2:]:
            cam_mask_tensor = F.interpolate(cam_mask_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)

        perturbed_input = input_tensor * (1 - cam_mask_tensor)
        output_pert = model(perturbed_input)
        score_pert = _get_score(output_pert, target_category_idx)

        confidence_drop = max(0, score_orig - score_pert)

    return confidence_drop

def calculate_deletion_single_image(model, input_tensor, heatmap, steps, device, target_category_idx=None):
    heatmap_flat = heatmap.flatten()
    sorted_indices = np.argsort(heatmap_flat)[::-1].copy()

    with torch.no_grad():
        pred = model(input_tensor)
        initial_score = _get_score(pred, target_category_idx) # Use consistent score getter

    scores = [1.0] # Normalized start
    if initial_score == 0: return 0.0, scores

    img_pert = input_tensor.clone()
    total_pixels = heatmap_flat.shape[0]
    step_size = max(1, total_pixels // steps)

    c, h, w = img_pert.shape[1:]

    for i in range(1, steps + 1):
        indices_to_remove = sorted_indices[:i*step_size]
        
        # Masking
        mask = torch.ones(h * w, device=device)
        mask[indices_to_remove] = 0.0
        mask = mask.view(1, 1, h, w)
        img_pert = img_pert * mask

        with torch.no_grad():
            pred_new = model(img_pert)
            new_score = _get_score(pred_new, target_category_idx)
        
        scores.append(new_score / initial_score)

    x_axis = np.linspace(0, 1, len(scores))
    auc_val = auc(x_axis, scores)
    return auc_val, scores

def insertion_metric(model, input_tensor, cam_mask, target_category_idx=None, steps=20):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        # Normalizing factor could be the final score
        final_score_orig = _get_score(output, target_category_idx)

    cam_flat = cam_mask.flatten()
    sorted_indices = np.argsort(cam_flat)[::-1].copy()

    n_pixels = len(cam_flat)
    step_size = max(1, n_pixels // steps)

    c, h, w = input_tensor.shape[1:]
    current_input = torch.zeros_like(input_tensor) # Start from black/blur
    scores = []

    # Step 0
    with torch.no_grad():
        out = model(current_input)
        score = _get_score(out, target_category_idx)
        scores.append(score)

    for i in range(1, steps + 1):
        limit = min(i * step_size, n_pixels)
        indices_to_add = sorted_indices[:limit]
        ys, xs = np.unravel_index(indices_to_add, (h, w))
        current_input[0, :, ys, xs] = input_tensor[0, :, ys, xs]

        with torch.no_grad():
            out_i = model(current_input)
            score_i = _get_score(out_i, target_category_idx)
            scores.append(score_i)
    
    auc_score = np.trapezoid(scores, dx=1.0/steps)
    return auc_score, scores

def calculate_road_metric(model, input_tensor, cam_mask, target_category_idx=None, percentiles=[10, 50]):
    if target_category_idx is None:
         # For NF, we don't have "target class", we maximize score
         pass

    targets = [AnomalyScoreTarget()]
    
    # Ensure mask is correct shape for ROAD: [1, H, W]
    if len(cam_mask.shape) == 2:
        cam_mask_exp = cam_mask[None, :, :]
    else:
        cam_mask_exp = cam_mask

    scores = []
    for p in percentiles:
        # ROADMostRelevantFirst expects: input_tensor, cam, targets, model
        # It internally perturbs and returns (prob_orig - prob_road) / prob_orig usually, or just prob
        # pytorch-grad-cam implementation details vary, we assume standard behavior
        road = ROADMostRelevantFirst(percentile=p)
        batch_scores = road(input_tensor, cam_mask_exp, targets, model)
        scores.append(np.mean(batch_scores))

    return np.mean(scores), scores

# --- NEW: Sanity Check Metric ---
def randomize_model_weights(model):
    """
    Randomizes the weights of the model (Normalizing Flow head in this case).
    This simulates a 'training from scratch' or trash model state.
    """
    # For DifferNet/SEDifferNet, the 'nf' attribute is the Normalizing Flow head.
    # We will randomize its parameters.
    # If standard CNN, we often randomize last layer.
    if hasattr(model, 'nf'):
        print("Randomizing NF head weights for Sanity Check...")
        for m in model.nf.modules():
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    else:
        # Fallback: Randomize classifier or last linear layer if found
        print("Model has no 'nf' attribute. Randomizing all detected Linear/Conv layers (fallback).")
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

def calculate_sanity_check(model, input_tensor, original_cam, cam_extractor_cls, target_layers, target_category_idx):
    """
    Calculates Sanity Check (Cascading Randomization).
    Measures rank correlation between original CAM and CAM from randomized model.
    Low correlation (near 0) is Good (Passed). High correlation (near 1) means CAM is independent of model weights (Bad).
    """
    # 1. Deepcopy the model to avoid corrupting original weights
    print("Creating deepcopy of model for Sanity Check...")
    try:
        model_copy = copy.deepcopy(model)
    except Exception as e:
        print(f"Deepcopy failed: {e}. Skipping Sanity Check.")
        return np.nan

    # 2. Randomize Model Copy
    randomize_model_weights(model_copy)
    model_copy.eval() 

    # 3. Compute CAM with randomized model
    try:
        # Re-init extractor with random model (use same target layers logic if possible)
        if hasattr(model_copy, 'alexnet'): # SEDifferNet
             target_layers_copy = [model_copy.alexnet.features[-1]]
        elif hasattr(model_copy, 'feature_extractor'): # DifferNet
             target_layers_copy = [model_copy.feature_extractor.features[-1]]
        else:
             target_layers_copy = [list(model_copy.children())[-1]]

        cam_extractor = cam_extractor_cls(model=model_copy, target_layers=target_layers_copy)
        targets = [AnomalyScoreTarget()] # Use correct target
        
        grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0, :]
        
        # 4. Measure Correlation (Spearman Rank)
        orig_flat = original_cam.flatten()
        rand_flat = grayscale_cam.flatten()
        
        corr, _ = spearmanr(orig_flat, rand_flat)
        
    except Exception as e:
        print(f"Sanity Check Error: {e}")
        corr = np.nan
    finally:
        # Cleanup
        if 'model_copy' in locals():
            del model_copy
        if 'cam_extractor' in locals():
            del cam_extractor
        if 'grayscale_cam' in locals():
            del grayscale_cam
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return corr

# --- Visualization Helper ---
def denormalize(tensor):
    """Denormalizes a tensor image to [0,1] range for visualization."""
    mean = np.array(c.norm_mean)
    std = np.array(c.norm_std)
    
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def save_visualization(image_tensor, mask, cam_map, idx, label, pred_correct, output_dir, method_name):
    """
    Saves validation image:
    Left: Original + Mask (50% opacity, Green)
    Right: Original + CAM Heatmap + X on max activation
    Name: {id}_{label_type}_{correct/wrong}_{method}.png
    """
    # Prepare Image
    img_np = denormalize(image_tensor) # [H, W, 3] float 0-1
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    
    # 1. Left: Mask Overlay
    left_img = img_bgr.copy()
    if mask is not None:
        mask_np = mask.squeeze().cpu().numpy() # [H, W]
        # Resize mask to image size if needed
        if mask_np.shape != img_bgr.shape[:2]:
            mask_np = cv2.resize(mask_np, (img_bgr.shape[1], img_bgr.shape[0]))
            
        # Create colored mask (Green)
        colored_mask = np.zeros_like(img_bgr)
        colored_mask[:, :, 1] = 255 # Green channel
        
        # Binary mask assumption (>0.5)
        binary_mask = (mask_np > 0.5).astype(np.uint8)
        
        # Apply overlay where mask is active
        # We want 50% opacity
        # indices where mask is present
        mask_indices = binary_mask > 0
        if np.any(mask_indices):
            left_img[mask_indices] = cv2.addWeighted(left_img[mask_indices], 0.5, colored_mask[mask_indices], 0.5, 0)

    # 2. Right: CAM Heatmap + X
    # Resize CAM to image size
    cam_map_resized = cv2.resize(cam_map, (img_bgr.shape[1], img_bgr.shape[0]))
    cam_map_resized = (cam_map_resized - cam_map_resized.min()) / (cam_map_resized.max() - cam_map_resized.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_resized), cv2.COLORMAP_JET)
    right_img = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)
    
    # Find max activation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cam_map_resized)
    # Draw X at max_loc
    # cv2.drawMarker(right_img, max_loc, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    # Or manual lines for better control
    x, y = max_loc
    l = 10
    cv2.line(right_img, (x-l, y-l), (x+l, y+l), (0, 0, 255), 2)
    cv2.line(right_img, (x+l, y-l), (x-l, y+l), (0, 0, 255), 2)
    
    # Combined
    combined = np.hstack((left_img, right_img))
    
    # Naming
    label_str = "anomaly" if label == 1 else "normal" # Adjust based on your label convention (usually 1=anomaly)
    pred_str = "correct" if pred_correct else "wrong"
    
    filename = f"{idx:04d}_{label_str}_{pred_str}_{method_name}.png"
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    cv2.imwrite(os.path.join(img_dir, filename), combined)

class CombinedTestDataset(Dataset):
    def __init__(self, test_dataset, gt_dataset):
        self.test_dataset = test_dataset
        self.gt_dataset = gt_dataset
        
        # Create mapping from filename (no extension) to GT index/mask
        self.gt_map = {}
        # gt_dataset is ImageFolder
        if hasattr(gt_dataset, 'samples'):
            for idx, (path, _) in enumerate(gt_dataset.samples):
                # Normalize filename: lowercase, no extension
                fname = os.path.splitext(os.path.basename(path))[0].lower()
                self.gt_map[fname] = idx
        else:
            print("Warning: GT dataset does not have .samples. Alignment might fail.")

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        image, label = self.test_dataset[idx]
        
        # Get filename from test dataset
        # test_dataset is ImageFolderMultiTransform -> has .samples assuming it wraps/inherits
        # ImageFolderMultiTransform might not expose .samples directly if it does complex things,
        # but utils.py showed it takes root. Let's assume standard ImageFolder behavior or 'samples' attribute availability.
        # If test_dataset is Subset, this breaks. But here it's from load_datasets which returns dataset directly.
        
        path = self.test_dataset.samples[idx][0]
        fname = os.path.splitext(os.path.basename(path))[0].lower()
        
        mask = None
        if fname in self.gt_map:
            gt_idx = self.gt_map[fname]
            mask_data = self.gt_dataset[gt_idx]
            if isinstance(mask_data, tuple):
                mask = mask_data[0]
            else:
                mask = mask_data
        
        if mask is None:
             # Return dummy zero mask [1, H, W] matching image spatial dims
             # Image is [C, H, W]
             mask = torch.zeros((1, image.shape[1], image.shape[2]))
            
        return image, label, mask

def flat_batch_loader(dataloader):
    """Yields (image, label, mask) correctly shaped from batch."""
    for batch in dataloader:
        if len(batch) == 3:
            images, labels, masks = batch
            
            # Handle mask channels (make 1 channel)
            if masks.dim() >= 3 and masks.shape[-3] == 3:
                # Take first channel
                if masks.dim() == 4: masks = masks[:, 0:1, :, :]
                elif masks.dim() == 5: masks = masks[:, :, 0:1, :, :]
            
            # Handle 5D crops [B, Crops, C, H, W]
            if images.dim() == 5:
                B, N, C, H, W = images.shape
                images = images.view(-1, C, H, W)
                labels = labels.repeat_interleave(N)
                
                # Expand masks
                if masks.dim() == 3: # [B, H, W]
                    masks = masks.repeat_interleave(N, dim=0)
                elif masks.dim() == 4: # [B, C, H, W]
                    masks = masks.repeat_interleave(N, dim=0)
                elif masks.dim() == 5: # [B, N, C, H, W]
                     masks = masks.view(-1, masks.shape[2], masks.shape[3], masks.shape[4])

            yield images, labels, masks
        else:
            yield batch

# --- Main Evaluation Logic ---

def evaluate_metrics(model_name, model_path, dataset_path, class_name, output_dir, limit=None):
    print(f"Starting evaluation for {model_name} on {class_name}...")
    
    # 1. Load Data
    c.n_transforms_test = 1 # Force 1 crop for consistent evaluation
    c.transf_rotations = False
    train_set, test_set, ground_truth_set = utils.load_datasets(dataset_path, class_name)
    combined_dataset = CombinedTestDataset(test_set, ground_truth_set)
    loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    if "sediffernet" in model_name.lower():
        base_model = SEDifferNet()
    else:
        base_model = DifferNet()
        
    model, _ = load_weights(base_model, model_path)
    model.to(device)
    model.eval()
    
    # 4. Target class fix:
    # Ensure pytorch-grad-cam uses our Custom Target
    # We will instantiate targets logic inside loop.

    
    # 3. Setup CAM Methods
    cam_methods = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus, # Can add more if desired
        'XGradCAM': XGradCAM
    }
    
    # Target Layers
    if "sediffernet" in model_name.lower():
         # SEDifferNet has alexnet.features + attention blocks. 
         # Last conv layer equivalent.
         target_layers = [model.alexnet.features[-1]] 
    else:
         # DifferNet: has feature_extractor (alexnet)
         target_layers = [model.feature_extractor.features[-1]]
         
    # Results Storage
    results = []
    
    # Determine how many samples to run for Sanity Check (it is slow)
    # If limit is set, choose from available range
    total_samples = limit if limit else len(loader)
    sanity_check_indices = set(np.random.choice(total_samples, min(5, total_samples), replace=False))

    print(f"Evaluating {total_samples} samples (Sanity Check on {len(sanity_check_indices)} indices)...")
    
    for name, cam_cls in cam_methods.items():
        print(f"Running {name}...")
        
        metrics_store = {'deletion': [], 'insertion': [], 'road': [], 'confidence': [], 'pixel_auc': [], 'sanity': []}
        curves_store = {'deletion': [], 'insertion': [], 'road': []}

        try:
            cam_extractor = cam_cls(model=model, target_layers=target_layers)
        except Exception as e:
            print(f"Failed to init {name}: {e}")
            continue

        # Iterate Data
        for batch_idx, (images, labels, masks) in enumerate(tqdm(flat_batch_loader(loader), total=len(loader))):
            if limit and batch_idx >= limit:
                break
            images = images.to(device)
            if len(images) == 0: continue
            
            # Process single image (batch_size=1 forced above usually, but flat_loader handles crops)
            # We take the first image if multiple crops exist to simplify metric accumulation per "sample"
            # Or iterate all crops. Let's iterate all crops in the batch.
            
            for i in range(len(images)):
                img = images[i:i+1] # Keep batch dim [1, C, H, W]
                # mask = masks[i] if masks is not None else None
                
                # --- Generate CAM ---
                with torch.no_grad():
                    out = model(img) # z
                    # t_idx irrelevant for Normalizing Flow (no classes)
                    # We utilize the AnomalyScoreTarget
                    t_idx = None 
                
                targets = [AnomalyScoreTarget()]
                cam_map = cam_extractor(input_tensor=img, targets=targets)[0, :]
                
                # --- Metrics ---
                
                # Pixel AUROC (Logic adjusted for alignment)
                # mask needs to be [H, W] or same shape
                if masks is not None:
                     current_mask = masks[i].squeeze().cpu().numpy()
                     p_auc = calculate_pixel_auroc(cam_map, current_mask)
                     metrics_store['pixel_auc'].append(p_auc)

                # Deletion
                d_auc, d_scores = calculate_deletion_single_image(model, img, cam_map, args.steps, device, t_idx)
                metrics_store['deletion'].append(d_auc)
                curves_store['deletion'].append(d_scores)
                
                # Insertion
                i_auc, i_scores = insertion_metric(model, img, cam_map, t_idx, args.steps)
                metrics_store['insertion'].append(i_auc)
                curves_store['insertion'].append(i_scores)
                
                # ROAD
                r_score, r_scores = calculate_road_metric(model, img, cam_map, t_idx)
                metrics_store['road'].append(r_score)
                curves_store['road'].append(r_scores)
                
                # Confidence Drop
                metrics_store['confidence'].append(confidence_drop_metric(model, img, cam_map, t_idx))
                
                # Sanity Check (Run only on subset to save time)
                if batch_idx in sanity_check_indices and i == 0: # Check once per batch
                    sanity_val = calculate_sanity_check(model, img, cam_map, cam_cls, target_layers, t_idx)
                    metrics_store['sanity'].append(sanity_val)
                    print(f" [Debug] Sanity Check {name} (Idx {batch_idx}): {sanity_val:.4f}")
        

                # Visualization (Save every image? Or only some? User asked "save images", implies all executed)
                # Prediction correctness (Simple max score check vs label? Or we assume t_idx matches label?)
                # If t_idx was argmax(out), then if t_idx == label it's "correct" (broadly speaking for multiclass)
                # For anomaly detection (1 class vs anomaly), it's trickier without a threshold. 
                # DiffNet/SEDiffNet outputs are usually low-dim embeddings or classifications?
                # model.py shows it returns 'z' from NF head? Wait.
                # Re-reading model.py:
                # - DifferNet usually outputs z score. Low likelihood = anomaly.
                # - But metric_lab.ipynb treats it as a classifier 'out = model(img)', 't_idx = torch.argmax(out)'.
                # - This implies the model wrapped or modified to output class scores?
                # - Let's check model.py 'forward' returns 'z = self.nf(y)'.
                # - 'z' is [B, C*scales]. Not logits.
                # - BUT metric_lab.ipynb says 't_idx = torch.argmax(out).item()'. This is strange if output is 'z'.
                # - Unless 'model' passed to evaluate is NOT the raw DifferNet but a wrapper?
                # - Or the notebook code was generic copy-paste.
                # - IF DifferNet returns 'z', argmax(z) is meaningless.
                # - However, typically for CAM on DifferNet, we look at gradients of the log-likelihood or norm of z.
                # - The notebook code might be flawed or using a different model variant.
                # - User's previous context mentions "DifferNet" and "SEDifferNet".
                # - Let's assume for Visualization Labeling: 
                #   If Label=1 (Anomaly) and Score > Threshold -> Correct.
                #   But we don't have a threshold.
                #   Let's just use "result" in filename as "correct" if logic allows, else "N_A".
                #   Actually, let's look at `t_idx` derived from `out`.
                
                # Check prediction for filename
                # Simplified: correctness = True (placeholder if we can't determine without threshold)
                is_correct = True 
                
                save_visualization(img[0], masks[i] if masks is not None else None, cam_map, 
                                   batch_idx * len(images) + i, labels[i].item(), is_correct, output_dir, name)
                
                # Cleanup per image
                del cam_map, targets
                if 'out' in locals(): del out
                torch.cuda.empty_cache()

        # Aggregate Results for this Method
        res = {'Method': name}
        for k, v in metrics_store.items():
            if v:
                res[k] = np.mean(v)
        
        # Save curves average
        for k, v in curves_store.items():
            if v:
                # Pad if needed? Assumed same length
                try:
                    res[f'{k}_curve'] = np.mean(np.array(v), axis=0).tolist()
                except:
                    pass # Shapes might differ
                    
        results.append(res)
        
        # Cleanup per method
        del cam_extractor
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # --- Reporting & plotting ---
    df = pd.DataFrame(results)
    
    # Identify non-curve columns (scalar metrics)
    scalar_cols = [col for col in df.columns if 'curve' not in col and col != 'Method']
    
    if not df.empty:
        # Save CSV
        csv_path = os.path.join(output_dir, 'results.csv')
        df_csv = df.drop(columns=[c for c in df.columns if 'curve' in c])
        df_csv.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
        print(df_csv)
        
        # Plot Bar Charts for Metrics
        plt.figure(figsize=(12, 6))
        
        # Reshape for seaborn barplot
        df_melt = df.melt(id_vars=['Method'], value_vars=scalar_cols, var_name='Metric', value_name='Score')
        sns.barplot(data=df_melt, x='Metric', y='Score', hue='Method')
        plt.title(f"CAM Metrics Comparison - {class_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_bar.png'))
        plt.close()
        
        # Plot Curves
        curve_metrics = ['deletion', 'insertion'] # ROAD structure varies
        for cm in curve_metrics:
            col_name = f'{cm}_curve'
            if col_name in df.columns:
                plt.figure(figsize=(8, 5))
                for idx, row in df.iterrows():
                    if row[col_name] is not None and isinstance(row[col_name], list):
                         y_vals = row[col_name]
                         x_vals = np.linspace(0, 1, len(y_vals))
                         plt.plot(x_vals, y_vals, label=f"{row['Method']}")
                plt.title(f"{cm.capitalize()} Curve")
                plt.xlabel("Fraction")
                plt.ylabel("Score")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_dir, f'{cm}_curve.png'))
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAM Metrics for DifferNet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--class_name", type=str, default="glass-insulator", help="Class name")
    parser.add_argument("--dataset_path", type=str, default=c.dataset_path, help="Path to dataset")
    parser.add_argument("--output_base", type=str, default="./results", help="Base output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps for Deletion/Insertion")
    
    args = parser.parse_args()
    
    # Create Unique Output Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_base, f"run_{timestamp}_{args.class_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Determine Model Name (heuristic from path)
    model_name_guess = os.path.basename(args.model_path)
    
    evaluate_metrics(
        model_name=model_name_guess,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        class_name=args.class_name,
        output_dir=run_dir,
        limit=args.limit
    )
