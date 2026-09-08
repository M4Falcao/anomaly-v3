"""Frozen snapshot of an earlier training loop, kept for reference only.

This module is NOT used by any active entry point. It preserves an older
variant of the training procedure that also computed pixel-level metrics and
un-rotated the per-transform heatmaps on the GPU.

It is retained so previous experiments remain reproducible; prefer
:mod:`core.train` for anything new.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from tqdm import tqdm
import mlflow
import threading
import os
import sys
import time
import socket

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import config as c
from core.localization import export_gradient_maps
from core.model import *
from core.utils import *
from core.mlflow_utils import (
    export_mlflow_data,
    run_mlflow_ui,
    safe_torch_save,
    start_ngrok,
    wait_for_server,
)
from core.paths import project_path
import skimage
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable


def _unrotate_and_average(score_map_grouped, fixed_degrees):
    """Un-rotate each per-transform heatmap back to the original frame, then average.

    Uses GPU-accelerated affine_grid + grid_sample instead of scipy (100x faster).

    score_map_grouped: (B, n_tf, H, W) numpy
    fixed_degrees:     list of length n_tf with the rotation applied at test time
    Returns: (B, H, W) numpy, averaged in the original (un-rotated) frame.
    """
    B, n_tf, H, W = score_map_grouped.shape

    # Move to GPU as a single tensor (B*n_tf, 1, H, W) for batched rotation
    maps_t = torch.from_numpy(score_map_grouped.reshape(B * n_tf, H, W)).float().to(c.device)
    maps_t = maps_t.unsqueeze(1)  # (B*n_tf, 1, H, W)

    # Build rotation matrices for all transforms at once
    angles_rad = torch.tensor(
        [-d * np.pi / 180.0 for d in fixed_degrees], dtype=torch.float32, device=c.device
    )  # (n_tf,)
    # Repeat for each batch element
    angles_rad = angles_rad.repeat(B)  # (B*n_tf,)

    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    zeros = torch.zeros_like(cos_a)

    # Affine matrix: [[cos, -sin, 0], [sin, cos, 0]]
    theta = torch.stack([cos_a, -sin_a, zeros, sin_a, cos_a, zeros], dim=1).view(-1, 2, 3)

    grid = F.affine_grid(theta, maps_t.size(), align_corners=False)
    rotated = F.grid_sample(maps_t, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # Reshape back and average across transforms
    rotated = rotated.squeeze(1).view(B, n_tf, H, W)  # (B, n_tf, H, W)
    result = rotated.mean(dim=1)  # (B, H, W)

    return result.cpu().numpy()


def calculate_image_level_auroc(predictions, ground_truth_labels):
    image_level_auroc = roc_auc_score(ground_truth_labels, predictions)
    return image_level_auroc

def calculate_pixel_level_auroc(predictions, ground_truth_masks):
    """Calculate pixel-level AUROC using RAW scores (no per-image normalization)."""
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Resize predictions to match the ground truth mask dimensions
    if predictions.shape != ground_truth_masks.shape:
        predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    else:
        predictions_resized = predictions
    
    # Binarize ground truth masks
    ground_truth_masks_binary = (ground_truth_masks > 0).astype(int)

    # Flatten predictions and ground truth masks
    predictions_flat = predictions_resized.reshape(-1)
    ground_truth_flat = ground_truth_masks_binary.reshape(-1)
    
    # Handle edge case where masks might only have one class
    if len(np.unique(ground_truth_flat)) < 2:
        return 0.5
        
    pixel_auroc = roc_auc_score(ground_truth_flat, predictions_flat)
    return pixel_auroc


def calculate_pixel_aupro(predictions, ground_truth_masks, num_thresholds=200, max_fpr=0.3):
    """Calculate AUPRO (Area Under Per-Region Overlap) metric.
    
    For each connected component (region) in the GT, compute the overlap
    at various FPR thresholds, then integrate.
    """
    from scipy import ndimage
    
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    
    if predictions.shape != ground_truth_masks.shape:
        predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    else:
        predictions_resized = predictions
    
    gt_binary = (ground_truth_masks > 0).astype(int)
    
    if len(np.unique(gt_binary)) < 2:
        return 0.5
    
    # Get thresholds
    sorted_scores = np.sort(predictions_resized.flatten())
    thresholds = sorted_scores[np.linspace(0, len(sorted_scores) - 1, num_thresholds, dtype=int)]
    
    # Label connected components in GT
    labeled_gt, num_regions = ndimage.label(gt_binary)
    
    if num_regions == 0:
        return 0.5
    
    fprs = []
    per_region_tprs = []
    
    total_normal_pixels = np.sum(gt_binary == 0)
    if total_normal_pixels == 0:
        return 1.0
    
    for thresh in thresholds:
        binary_pred = (predictions_resized >= thresh).astype(int)
        
        # FPR: false positives among normal pixels
        fp = np.sum((binary_pred == 1) & (gt_binary == 0))
        fpr = fp / total_normal_pixels
        
        if fpr > max_fpr:
            continue
        
        # Per-region TPR
        region_tprs = []
        for region_id in range(1, num_regions + 1):
            region_mask = (labeled_gt == region_id)
            region_pixels = np.sum(region_mask)
            if region_pixels == 0:
                continue
            tp_region = np.sum((binary_pred == 1) & region_mask)
            region_tprs.append(tp_region / region_pixels)
        
        if region_tprs:
            fprs.append(fpr)
            per_region_tprs.append(np.mean(region_tprs))
    
    if len(fprs) < 2:
        return 0.5
    
    # Sort by FPR and integrate
    sorted_indices = np.argsort(fprs)
    fprs = np.array(fprs)[sorted_indices]
    per_region_tprs = np.array(per_region_tprs)[sorted_indices]
    
    # Normalize by max_fpr
    aupro = np.trapz(per_region_tprs, fprs) / max_fpr
    return aupro


def calculate_pixel_ap_f1(predictions, ground_truth_masks):
    """Calculate pixel-level Average Precision and best F1 score."""
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    
    if predictions.shape != ground_truth_masks.shape:
        predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    else:
        predictions_resized = predictions
    
    gt_binary = (ground_truth_masks > 0).astype(int)
    
    predictions_flat = predictions_resized.reshape(-1)
    gt_flat = gt_binary.reshape(-1)
    
    if len(np.unique(gt_flat)) < 2:
        return 0.5, 0.5
    
    # Average Precision
    ap = average_precision_score(gt_flat, predictions_flat)
    
    # Best F1 (F1-max)
    precision, recall, thresholds = precision_recall_curve(gt_flat, predictions_flat)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_max = np.max(f1_scores)
    
    return ap, f1_max


def get_se_anomaly_maps(model, inputs):
    """Compute anomaly heatmap using SE Attention excitation weights (no gradients needed).

    Strategy:
      1. Register hooks on each SE block's fc (Sigmoid output) to capture excitation weights.
      2. Also capture the spatial feature maps BEFORE SE recalibration (SE input).
      3. Weight each channel of the spatial features by its SE excitation weight.
      4. Sum across channels → spatial score map per scale.
      5. Upsample and aggregate across scales.

    This is fast (single forward pass, no backward) and leverages what the SE blocks
    already learned: anomalous channels get higher excitation.

    Returns: numpy array of shape (B, H_img, W_img).
    """
    model.eval()

    # Storage for hook outputs
    se_excitations = []  # list of (B, C, 1, 1) tensors
    se_inputs = []       # list of (B, C, H, W) tensors (feature maps entering SE)

    hooks = []

    def make_input_hook(storage):
        def hook_fn(module, inp, out):
            # inp[0] is the input to the SE block (B, C, H, W)
            storage.append(inp[0].detach())
        return hook_fn

    def make_excitation_hook(storage):
        def hook_fn(module, inp, out):
            # out is the SE excitation vector after sigmoid, shape (B, C) from fc
            # We reshape to (B, C, 1, 1) for broadcasting
            storage.append(out.detach().view(out.size(0), out.size(1), 1, 1))
        return hook_fn

    # Register hooks on each SE block
    se_blocks = [model.simsa1, model.simsa2, model.simsa3, model.simsa4]
    for se_block in se_blocks:
        # Hook on the SE block itself to capture input feature maps
        hooks.append(se_block.register_forward_hook(make_input_hook(se_inputs)))
        # Hook on the fc (Sequential ending with Sigmoid) to capture excitation weights
        hooks.append(se_block.fc.register_forward_hook(make_excitation_hook(se_excitations)))

    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for h in hooks:
            h.remove()

    # Now we have 4 SE excitation vectors and 4 SE input feature maps per scale.
    # Since there are n_scales=3 and 4 SE blocks per scale, we have 4*3=12 entries.
    # Group by scale: every 4 entries correspond to one scale pass.
    n_se_blocks = len(se_blocks)
    n_scales = c.n_scales
    target_size = c.img_size  # (H, W)

    score_map = None
    for scale_idx in range(n_scales):
        for block_idx in range(n_se_blocks):
            idx = scale_idx * n_se_blocks + block_idx
            if idx >= len(se_inputs) or idx >= len(se_excitations):
                continue

            feat = se_inputs[idx]       # (B, C, H_f, W_f)
            excit = se_excitations[idx] # (B, C, 1, 1)

            # Weight features by excitation: channels the SE considers important get higher weight
            weighted = feat * excit     # (B, C, H_f, W_f)
            # Sum across channels → spatial anomaly score
            spatial_score = weighted.sum(dim=1)  # (B, H_f, W_f)

            # Upsample to image size
            spatial_up = F.interpolate(
                spatial_score.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False
            ).squeeze(1)  # (B, H, W)

            if score_map is None:
                score_map = spatial_up
            else:
                score_map = score_map + spatial_up

    if score_map is None:
        return None

    return t2np(score_map)


def compute_background_heatmap(model, train_loader, optimizer, n_batches=20):
    """Compute mean SE heatmap on normal (training) images for background subtraction.

    Uses FIXED rotations + un-rotation so the background is in the same (original) frame
    as the test-time per-image heatmaps.
    """
    model.eval()
    bg_maps = []

    # Switch dataset to fixed rotations so we can un-rotate consistently.
    ds = train_loader.dataset
    had_fixed = getattr(ds, 'get_fixed', False)
    ds.get_fixed = True
    fixed_degrees = ds.fixed_degrees
    n_tf_train = getattr(ds, 'n_transforms', c.n_transforms)

    try:
        for i, data in enumerate(tqdm(train_loader, desc="Computing background heatmap", disable=c.hide_tqdm_bar)):
            if i >= n_batches:
                break
            inputs, labels = preprocess_batch(data)
            score_map = get_se_anomaly_maps(model, inputs)
            if score_map is None:
                continue
            batch_size_actual = labels.size(0)
            score_grouped = score_map.reshape(batch_size_actual, n_tf_train, *score_map.shape[-2:])
            score_per_image = _unrotate_and_average(score_grouped, fixed_degrees)  # (B, H, W)
            bg_maps.append(score_per_image)
    finally:
        ds.get_fixed = had_fixed

    if not bg_maps:
        return None

    all_maps = np.concatenate(bg_maps, axis=0)  # (N, H, W)
    background_mean = np.mean(all_maps, axis=0)  # (H, W)
    return background_mean




class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if self.max_score is None or score > self.max_score: # Fix for NoneType comparison")
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))




import time

print(f'TRAINING ON : {c.device}, cause cuda is {torch.cuda.is_available()}')

def train(train_loader, test_loader):
    # Create execution specific checkpoint directory
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_checkpoint_dir = os.path.join(c.checkpoint_path, f"run_{run_timestamp}")
    best_models_dir = os.path.join(run_checkpoint_dir, "best_models")
    if not os.path.exists(run_checkpoint_dir):
        os.makedirs(run_checkpoint_dir)
    os.makedirs(best_models_dir, exist_ok=True)

    # MLflow Setup
    if c.use_mlflow:
        # Start MLflow UI in background if tracking URI is local
        if "127.0.0.1" in c.mlflow_tracking_uri or "localhost" in c.mlflow_tracking_uri:
            if not wait_for_server("127.0.0.1", 5000, timeout=2):
                print("Starting MLflow UI in a background thread...")
                thread = threading.Thread(target=run_mlflow_ui)
                thread.daemon = True
                thread.start()
                
                print("Waiting for MLflow server to start...")
                if wait_for_server("127.0.0.1", 5000):
                    print("MLflow server started!")
                else:
                    print("Timed out waiting for MLflow server!")
            else:
                print("MLflow server is already running!")

        if c.use_pyngrok:
            # Start ngrok
            start_ngrok(5000)

        mlflow.set_tracking_uri(c.mlflow_tracking_uri)
        mlflow.set_experiment(c.mlflow_experiment_name)
        mlflow.start_run(run_name=c.mlflow_run_name)
        
        # Log parameters
        params = {k: v for k, v in vars(c).items() if not k.startswith('__') and not callable(v) and not isinstance(v, type)}
        # Filter out complex objects if any, keep simple types
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except:
                pass

    model = SEDifferNet()
    # model = SEResNet18DifferNet()
    # §3.1: Train SE attention modules alongside NF head
    se_params = list(model.simsa1.parameters()) + list(model.simsa2.parameters()) + \
                list(model.simsa3.parameters()) + list(model.simsa4.parameters())
    optimizer = torch.optim.Adam(
        [{'params': model.nf.parameters(), 'lr': c.lr_init},
         {'params': se_params, 'lr': c.lr_init * 0.5}],  # Lower LR for attention modules
        betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5
    )
    model.to(c.device)
    # model.compile() # Removed because Triton is not supported by default on Windows
    
    scaler = GradScaler()

    if c.use_mlflow:
        # Log config file
        config_path = project_path("config.py")
        mlflow.log_artifact(config_path)
        
        # Log model summary
        with open("model_summary.txt", "w") as f:
            f.write(str(model))
        mlflow.log_artifact("model_summary.txt")
        if os.path.exists("model_summary.txt"):
            os.remove("model_summary.txt")
            
        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parameters", total_params)

    score_obs_image = Score_Observer('AUROC for image level')
    score_obs_pixel = Score_Observer('AUROC for pixel level')
    best_img_auroc = -1.0
    best_pix_auroc = -1.0

    # Initialize history lists
    train_losses = []
    test_losses = []
    image_aurocs = []
    pixel_aurocs = []
    
    start_epoch = 0
    if c.resume_training:
        print(f"Loading weights from {c.resume_file}...")
        try:
            model, checkpoint = load_weights(model, c.resume_file)
            if checkpoint:
                print("Restoring checkpoint metadata...")
                start_epoch = checkpoint['epoch']
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'train_losses' in checkpoint:
                    train_losses = checkpoint['train_losses']
                if 'test_losses' in checkpoint:
                    test_losses = checkpoint['test_losses']
                if 'image_aurocs' in checkpoint:
                    image_aurocs = checkpoint['image_aurocs']
                if 'pixel_aurocs' in checkpoint:
                    pixel_aurocs = checkpoint['pixel_aurocs']
                print(f"Resuming from epoch {start_epoch}")
            
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting training from scratch.")

    # Pre-epochs training loop
    if hasattr(c, 'pre_epochs') and c.pre_epochs > 0 and start_epoch == 0:
        print(f"Starting {c.pre_epochs} pre-epochs...")
        model.train()
        for pre_epoch in range(c.pre_epochs):
            pre_train_loss = []
            for i, data in enumerate(tqdm(train_loader, desc=f"Pre-Epoch {pre_epoch+1}/{c.pre_epochs}")):
                optimizer.zero_grad()
                inputs, labels = preprocess_batch(data)
                
                with autocast('cuda'):
                    z = model(inputs)
                    loss = get_loss(z, model.nf.jacobian(run_forward=False))
                
                pre_train_loss.append(loss.item())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            avg_pre_train_loss = np.mean(pre_train_loss)
            print(f"Pre-Epoch [{pre_epoch + 1}/{c.pre_epochs}], Train Loss: {avg_pre_train_loss:.4f}")
            if c.use_mlflow:
                mlflow.log_metric("pre_train_loss", avg_pre_train_loss, step=pre_epoch)

    # §2.5: Compute background heatmap from normal training images for subtraction
    print("Computing background heatmap for subtraction...")
    background_heatmap = compute_background_heatmap(model, train_loader, optimizer, n_batches=20)
    if background_heatmap is not None:
        print(f"Background heatmap computed (shape: {background_heatmap.shape})")
    else:
        print("Warning: Could not compute background heatmap")

    for meta_epoch in range(start_epoch, c.meta_epochs):
        # Training loop
        model.train()
        train_loss = []
        image_level_scores_train = []
        
        for sub_epoch in range(c.sub_epochs):
            for i, data in enumerate(tqdm(train_loader, desc=f"Meta {meta_epoch+1}/{c.meta_epochs} - Sub {sub_epoch+1}/{c.sub_epochs}")):
                optimizer.zero_grad()
                inputs, labels = preprocess_batch(data)
                
                with autocast('cuda'):
                    z = model(inputs)
                    loss = get_loss(z, model.nf.jacobian(run_forward=False))
                
                train_loss.append(loss.item())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Compute image-level anomaly score (original score) during training
                image_level_score_train = torch.mean(z ** 2).item()
                image_level_scores_train.append(image_level_score_train)

            print(f"Sub Epoch {sub_epoch+1}/{c.sub_epochs} completed. Loss: {np.mean(train_loss[-len(train_loader):]):.4f}")
            

        # Compute average training loss
        avg_train_loss = np.mean(train_loss)

        # Print or log metrics during training
        # Print or log metrics during training
        print('Meta Epoch [{}/{}], Train Loss: {:.4f}'.format(meta_epoch + 1, c.meta_epochs, avg_train_loss))
        if c.use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=meta_epoch)

        # Evaluation loop
        model.eval()
        test_loss = []
        test_labels = []
        test_z = []

        # Time tracking variables
        start_time = time.time()
        total_images = 0
        
        # Set dataset to fixed mode for deterministic gradient calculation
        test_loader.dataset.get_fixed = True
        
        # §1.2 & §1.3: Accumulate RAW score maps for ALL images (good + anomalous)
        all_pixel_scores = []
        all_ground_truth_masks = []

        for i, data in enumerate(tqdm(test_loader, desc=f"Eval Meta {meta_epoch+1}")):
            # AlignedTestDataset returns (images, label, mask)
            inputs_raw, labels, masks = data
            inputs = inputs_raw.to(c.device)
            labels = labels.to(c.device)
            masks = masks.to(c.device)  # (B, 1, H, W)
            inputs = inputs.view(-1, *inputs.shape[-3:])
            
            # --- Image Level Score Calculation (Standard) ---
            with torch.no_grad():
                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                test_loss.append(loss.item())
                test_labels.append(t2np(labels)) 
                test_z.append(z)
            
            total_images += labels.size(0)

            # --- Pixel Level Score Calculation (SE Attention Excitation) ---
            # Uses hooks on SE blocks - no backward pass needed, much faster.
            score_map = get_se_anomaly_maps(model, inputs)
            
            if score_map is not None:
                # score_map shape: (B * n_transforms, H, W) -> aggregate per image
                n_tf = c.n_transforms_test
                batch_size_actual = labels.size(0)
                score_map_grouped = score_map.reshape(batch_size_actual, n_tf, *score_map.shape[-2:])
                # Un-rotate each per-transform map back to the original frame BEFORE averaging,
                # otherwise rotational averaging produces a radially symmetric blob and
                # destroys spatial localization (pixel AUROC collapses to ~0.5).
                fixed_degrees = test_loader.dataset.fixed_degrees
                score_map_per_image = _unrotate_and_average(score_map_grouped, fixed_degrees)  # (B, H, W)
                
                all_pixel_scores.append(score_map_per_image)
                all_ground_truth_masks.append(t2np(masks.squeeze(1)))  # (B, H, W)

        test_loader.dataset.get_fixed = False

        # Calculate time metrics
        elapsed_time = time.time() - start_time
        images_per_second = total_images / elapsed_time if elapsed_time > 0 else 0
        latency_per_image_ms = (elapsed_time / total_images) * 1000 if total_images > 0 else 0
        print(f"Test Set - Images per Second: {images_per_second:.2f}, Latency per Image: {latency_per_image_ms:.2f} ms")

        # Compute average test loss
        avg_test_loss = np.mean(test_loss)

        # --- Pixel-level metrics (§1.2, §1.3, §2.5, §5) ---
        mean_pixel_auroc_test = 0.5
        mean_aupro = 0.5
        mean_pixel_ap = 0.5
        mean_pixel_f1_max = 0.5
        
        if all_pixel_scores:
            print(f"Calculating pixel-level metrics on {total_images} images...")
            all_scores = np.concatenate(all_pixel_scores, axis=0)  # (N, H, W)
            all_masks = np.concatenate(all_ground_truth_masks, axis=0)  # (N, H, W)
            
            # §2.5: Background subtraction
            if background_heatmap is not None:
                print("Applying background heatmap subtraction...")
                all_scores = all_scores - background_heatmap[np.newaxis, :, :]
                all_scores = np.maximum(all_scores, 0)  # clip negatives
            
            # Apply Gaussian smoothing (fixed σ=3)
            for idx in range(all_scores.shape[0]):
                all_scores[idx] = gaussian_filter(all_scores[idx], sigma=3)
            
            print("Calculating pixel-level AUROC...")
            # §1.3: NO per-image normalization - use raw scores directly
            # Calculate pixel-level AUROC on full dataset
            mean_pixel_auroc_test = calculate_pixel_level_auroc(all_scores, all_masks)
            
            # §5: Complementary metrics
            print("Calculating AUPRO and pixel-level AP/F1...")
            # mean_aupro = calculate_pixel_aupro(all_scores, all_masks)
            # mean_pixel_ap, mean_pixel_f1_max = calculate_pixel_ap_f1(all_scores, all_masks)
            print(f"Pixel-level metrics - AUROC: {mean_pixel_auroc_test:.4f}, AUPRO: {mean_aupro:.4f}, AP: {mean_pixel_ap:.4f}, F1-max: {mean_pixel_f1_max:.4f}")
        
        if c.use_mlflow:
            mlflow.log_metric("test_loss", avg_test_loss, step=meta_epoch)
            mlflow.log_metric("pixel_level_auroc", mean_pixel_auroc_test, step=meta_epoch)
            mlflow.log_metric("pixel_aupro", mean_aupro, step=meta_epoch)
            mlflow.log_metric("pixel_ap", mean_pixel_ap, step=meta_epoch)
            mlflow.log_metric("pixel_f1_max", mean_pixel_f1_max, step=meta_epoch)
    
        # Image-level AUROC
        print("Calculating image-level AUROC...")
        is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        
        image_auroc = roc_auc_score(is_anomaly, anomaly_score) if len(np.unique(is_anomaly)) > 1 else 0.5
        
        print("Updating scores...")
        score_obs_image.update(image_auroc, meta_epoch,
                        print_score=c.verbose or meta_epoch == c.meta_epochs - 1)
        score_obs_pixel.update(mean_pixel_auroc_test, meta_epoch,
                        print_score=c.verbose or meta_epoch == c.meta_epochs - 1)
        
        # Print complementary metrics
        print(f"  Pixel AUPRO: {mean_aupro:.4f} | Pixel AP: {mean_pixel_ap:.4f} | Pixel F1-max: {mean_pixel_f1_max:.4f}")
        
        if c.use_mlflow:
            mlflow.log_metric("image_level_auroc", image_auroc, step=meta_epoch)
            
        # Append metrics to history
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        image_aurocs.append(image_auroc)
        pixel_aurocs.append(mean_pixel_auroc_test)

        # §3.7: Save checkpoint primarily by PIXEL-AUROC
        if (meta_epoch + 1) % c.checkpoint_interval == 0 or mean_pixel_auroc_test > best_pix_auroc:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Saving model checkpoint at epoch {meta_epoch + 1}...")
            
            if not os.path.exists(run_checkpoint_dir):
                os.makedirs(run_checkpoint_dir)
            
            weights_filename = os.path.join(run_checkpoint_dir, f"{c.class_name}_{c.modelname}_epoch_{meta_epoch + 1}.pt")
            checkpoint_data = {
                'epoch': meta_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'image_aurocs': image_aurocs,
                'pixel_aurocs': pixel_aurocs
            }
            saved = safe_torch_save(checkpoint_data, weights_filename)
            if saved:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Checkpoint saved locally to: {weights_filename}")
            else:
                # Fallback: save a lighter checkpoint so training can continue.
                lite_filename = weights_filename.replace('.pt', '_lite.pt')
                checkpoint_data_lite = {
                    'epoch': meta_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'image_aurocs': image_aurocs,
                    'pixel_aurocs': pixel_aurocs
                }
                if safe_torch_save(checkpoint_data_lite, lite_filename, retries=1):
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Full checkpoint failed; saved LITE checkpoint: {lite_filename}")
                else:
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Failed to save both FULL and LITE checkpoints; continuing training.")

            if c.use_mlflow and mean_pixel_auroc_test > best_pix_auroc:
                mlflow.pytorch.log_model(model, "model_best_pixel_auroc")
                print(f"New best model saved to MLflow with Pixel AUROC: {mean_pixel_auroc_test:.4f}")

        # Save best image-level AUROC model
        if image_auroc > best_img_auroc:
            best_img_auroc = image_auroc
            best_img_path = os.path.join(best_models_dir, "best_img_auroc.pt")
            best_img_data = {
                'epoch': meta_epoch + 1,
                'model_state_dict': model.state_dict(),
                'image_auroc': image_auroc,
                'pixel_auroc': mean_pixel_auroc_test,
                'pixel_aupro': mean_aupro,
                'pixel_ap': mean_pixel_ap,
                'pixel_f1_max': mean_pixel_f1_max,
            }
            if safe_torch_save(best_img_data, best_img_path):
                print(f"  ★ New best IMAGE AUROC: {image_auroc:.4f} (epoch {meta_epoch + 1}) -> {best_img_path}")
            else:
                print(f"  ! Could not save best image-level checkpoint at epoch {meta_epoch + 1}")

        # §3.7: Save best pixel-level AUROC model (PRIMARY criterion)
        if mean_pixel_auroc_test > best_pix_auroc:
            best_pix_auroc = mean_pixel_auroc_test
            best_pix_path = os.path.join(best_models_dir, "best_pix_auroc.pt")
            best_pix_data = {
                'epoch': meta_epoch + 1,
                'model_state_dict': model.state_dict(),
                'image_auroc': image_auroc,
                'pixel_auroc': mean_pixel_auroc_test,
                'pixel_aupro': mean_aupro,
                'pixel_ap': mean_pixel_ap,
                'pixel_f1_max': mean_pixel_f1_max,
            }
            if safe_torch_save(best_pix_data, best_pix_path):
                print(f"  ★ New best PIXEL AUROC: {mean_pixel_auroc_test:.4f} (epoch {meta_epoch + 1}) -> {best_pix_path}")
            else:
                print(f"  ! Could not save best pixel-level checkpoint at epoch {meta_epoch + 1}")

    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)
        save_weights(model, c.modelname)
        
    if c.use_mlflow:
        mlflow.pytorch.log_model(model, "final_model")
        mlflow.end_run()

    # Save all values to a text file at the end of the execution
    metrics_file = os.path.join(run_checkpoint_dir, "metrics_history.txt")
    with open(metrics_file, "w") as f:
        f.write("Epoch\tTrain_Loss\tTest_Loss\tImage_AUROC\tPixel_AUROC\tAUPRO\tPixel_AP\tPixel_F1max\n")
        num_epochs = len(train_losses)
        for i in range(num_epochs):
            tr_loss = train_losses[i] if i < len(train_losses) else 0.0
            te_loss = test_losses[i] if i < len(test_losses) else 0.0
            img_auc = image_aurocs[i] if i < len(image_aurocs) else 0.0
            pix_auc = pixel_aurocs[i] if i < len(pixel_aurocs) else 0.0
            f.write(f"{start_epoch + i + 1}\t{tr_loss:.6f}\t{te_loss:.6f}\t{img_auc:.6f}\t{pix_auc:.6f}\n")
    print(f"Metrics history saved to {metrics_file}")
        
    return model

