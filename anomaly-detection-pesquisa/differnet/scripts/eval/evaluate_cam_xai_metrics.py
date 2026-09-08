"""GradCAM/XAI faithfulness metrics for DifferNet-style models.

Computes Deletion, Insertion and ROAD scores over GradCAM explanations and
optionally evaluates the CFLOW pixel head and PatchCore in the same run.

Usage:
    python scripts/eval/evaluate_cam_xai_metrics.py --model_path <model.pt> --class_name <class>
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import random
import copy
import datetime
import time
import math
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
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.transform import resize as sk_resize

# Make the project root importable regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import config as c
from core import utils
from core.model import DifferNet, SEDifferNet, load_weights
from core.xai import AnomalyScoreTarget

# Pytorch GradCAM imports
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst
except ImportError:
    print("Error: pytorch-grad-cam not installed. Please install it using 'pip install grad-cam'.")
    sys.exit(1)

# --- Configuration & Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Helper Functions (From Notebook) ---

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
    Resizes anomaly_map to match ground_truth_mask dimensions.
    Returns np.nan for normal images (mask all zeros) so they are excluded from averaging.
    """
    # Resize CAM to match GT mask shape if needed
    if anomaly_map.shape != ground_truth_mask.shape:
        anomaly_map = sk_resize(anomaly_map, ground_truth_mask.shape, mode='constant', preserve_range=True)
    
    flat_anomaly = anomaly_map.flatten()
    flat_mask = ground_truth_mask.flatten()
    # Ensure mask is binary
    flat_mask = (flat_mask > 0.5).astype(int)
    
    # Skip normal images — localization only makes sense for anomalous ones
    if np.sum(flat_mask) == 0:
        return np.nan
    
    # Also skip if mask is all anomaly (no normal pixels to contrast)
    if np.sum(flat_mask) == len(flat_mask):
        return np.nan
    
    try:
        score = roc_auc_score(flat_mask, flat_anomaly)
    except ValueError:
        score = np.nan
    return score

def calculate_iou_dice(anomaly_map, ground_truth_mask):
    """
    Calculates IoU and Dice score between binarized CAM and GT mask.
    Uses Otsu thresholding to binarize the anomaly map.
    Returns np.nan for normal images.
    """
    # Resize CAM to match GT mask shape if needed
    if anomaly_map.shape != ground_truth_mask.shape:
        anomaly_map = sk_resize(anomaly_map, ground_truth_mask.shape, mode='constant', preserve_range=True)
    
    gt_binary = (ground_truth_mask > 0.5).astype(int)
    
    # Skip normal images
    if np.sum(gt_binary) == 0:
        return np.nan, np.nan
    
    # Binarize CAM using Otsu threshold
    try:
        thresh = threshold_otsu(anomaly_map)
        cam_binary = (anomaly_map >= thresh).astype(int)
    except ValueError:
        # Otsu fails if CAM is constant
        return np.nan, np.nan
    
    intersection = np.sum(cam_binary * gt_binary)
    union = np.sum(cam_binary) + np.sum(gt_binary) - intersection
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (np.sum(cam_binary) + np.sum(gt_binary)) if (np.sum(cam_binary) + np.sum(gt_binary)) > 0 else 0.0
    
    return iou, dice

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

def calculate_road_metrics(model, input_tensor, cam_mask, target_category_idx=None, percentiles=[20, 40, 60, 80]):
    """
    Calculates both ROAD Most Relevant First and ROAD Least Relevant First.
    
    - ROAD_Most: removes the MOST relevant pixels. A good CAM should cause a LARGE score drop.
      => Higher ROAD_Most (normalized) = better CAM (it found the important pixels).
    - ROAD_Least: removes the LEAST relevant pixels. A good CAM should cause a SMALL score drop.
      => Lower ROAD_Least (normalized) = better CAM (irrelevant pixels are truly irrelevant).
    
    Values are normalized by the original model score to be in [0, 1] range.
    """
    targets = [AnomalyScoreTarget()]
    
    # Get original score for normalization
    with torch.no_grad():
        orig_output = model(input_tensor)
        orig_score = _get_score(orig_output)
    
    if orig_score == 0:
        return 0.0, [], 0.0, []
    
    # Ensure mask is correct shape for ROAD: [1, H, W]
    if len(cam_mask.shape) == 2:
        cam_mask_exp = cam_mask[None, :, :]
    else:
        cam_mask_exp = cam_mask

    scores_most = []
    scores_least = []
    for p in percentiles:
        # ROAD Most Relevant First
        road_most = ROADMostRelevantFirst(percentile=p)
        with torch.no_grad():
            batch_most = road_most(input_tensor, cam_mask_exp, targets, model)
        # Normalize: raw_score / original_score
        scores_most.append(np.abs(np.mean(batch_most)) / orig_score)
        # print(f"road_most: {batch_most} / {orig_score} = {np.mean(batch_most) / orig_score}")
        
        # ROAD Least Relevant First
        road_least = ROADLeastRelevantFirst(percentile=p)
        with torch.no_grad():
            batch_least = road_least(input_tensor, cam_mask_exp, targets, model)
        scores_least.append(np.abs(np.mean(batch_least)) / orig_score)

    return np.mean(scores_most), scores_most, np.mean(scores_least), scores_least

# --- NEW: Sanity Check Metric ---
def randomize_model_weights(model):
    """
    Randomizes ALL weights of the model (backbone + NF head).
    This performs a full Sanity Check to simulate an untrained model.
    If an attribution method truly depends on what the model learned,
    its output must change drastically (low correlation) when ALL 
    weights are randomized.
    """
    print("Randomizing ALL model weights for Sanity Check...")
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)

def calculate_sanity_check(model, input_tensor, original_cam, cam_extractor_cls, target_layers, target_category_idx):
    """
    Calculates Sanity Check (Cascading Randomization).
    Measures rank correlation between original CAM and CAM from randomized model.
    Low correlation (near 0) is Good (Passed). High correlation (near 1) means CAM is independent of model weights (Bad).
    """
    # 1. Save original weights to CPU (RAM) to save GPU
    # Clone properly
    original_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    try:
        # 2. Randomize ORIGINAL Model in-place
        print("Randomizing model (in-place) for Sanity Check...")
        randomize_model_weights(model)
        model.eval() 

        # 3. Compute CAM with randomized model
        # Use 'model' (it is randomized now)
        cam_extractor = cam_extractor_cls(model=model, target_layers=target_layers)
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
        # 5. Restore original weights
        print("Restoring original model weights...")
        model.load_state_dict(original_state_dict)
        
        # Cleanup
        if 'cam_extractor' in locals():
            del cam_extractor
        if 'grayscale_cam' in locals():
            del grayscale_cam
        del original_state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return corr

def calculate_sanity_check_custom(model, input_tensor, original_cam, custom_method):
    """
    Calculates Sanity Check (Cascading Randomization) for custom attribution methods.
    """
    original_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    try:
        print("Randomizing model (in-place) for Sanity Check...")
        randomize_model_weights(model)
        model.eval() 

        grayscale_cam = custom_method(input_tensor)
        
        orig_flat = original_cam.flatten()
        rand_flat = grayscale_cam.flatten()
        
        corr, _ = spearmanr(orig_flat, rand_flat)
        
    except Exception as e:
        print(f"Sanity Check Error: {e}")
        corr = np.nan
    finally:
        print("Restoring original model weights...")
        model.load_state_dict(original_state_dict)
        
        if 'grayscale_cam' in locals():
            del grayscale_cam
        del original_state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return corr

# --- Custom Attribution Methods (Non-GradCAM) ---

class InputGradientMethod:
    """
    Generates anomaly maps by computing the gradient of the anomaly score (||z||²)
    directly with respect to input image pixels.
    
    Unlike GradCAM (designed for classifiers), this method works natively with
    any differentiable model, including Normalizing Flows like DifferNet/SEDifferNet.
    
    The gradient magnitude at each pixel indicates how much that pixel contributes
    to the anomaly score — high gradient = high influence on anomaly detection.
    """
    def __init__(self, model, sigma=4):
        self.model = model
        self.sigma = sigma  # Gaussian smoothing to reduce noise
    
    def __call__(self, input_tensor):
        """
        Args:
            input_tensor: [1, C, H, W] tensor on device
        Returns:
            anomaly_map: [H, W] numpy array, normalized to [0, 1]
        """
        # Clone to avoid modifying the original, enable gradients
        img = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(img)
        
        # Anomaly score = ||z||² (same as model training loss)
        if isinstance(output, tuple):
            output = output[0]
        if output.dim() == 1:
            output = output.unsqueeze(0)
        score = torch.mean(torch.sum(output ** 2, dim=1))
        
        # Backward pass to get gradients w.r.t. input pixels
        self.model.zero_grad()
        score.backward()
        
        # Saliency = absolute gradient magnitude, averaged across color channels
        saliency = img.grad.abs().squeeze(0).mean(dim=0).cpu().numpy()  # [H, W]
        
        # Gaussian smoothing to reduce pixel-level noise
        if self.sigma > 0:
            saliency = gaussian_filter(saliency, sigma=self.sigma)
        
        # Normalize to [0, 1]
        smin, smax = saliency.min(), saliency.max()
        if smax - smin > 1e-8:
            saliency = (saliency - smin) / (smax - smin)
        else:
            saliency = np.zeros_like(saliency)
        
        return saliency
    
    def cleanup(self):
        pass


class SEAttentionMethod:
    """
    Generates anomaly maps by extracting Squeeze-and-Excitation attention weights
    from SEDifferNet's SE blocks and projecting them into image space.
    
    How it works:
    1. Register forward hooks on each SE block (simsa1–simsa4) to capture:
       - The channel attention weights (output of the SE's fc → sigmoid)
       - The feature maps after attention weighting
    2. For each SE block, create a spatial activation map by weighting the
       feature map channels by their SE attention scores, then sum.
    3. Combine maps from all SE blocks (multi-scale fusion) into a single map.
    
    This is model-native: the SE weights were TRAINED with the model, so they
    reflect what the model actually learned, unlike post-hoc GradCAM.
    
    Only works with SEDifferNet (requires simsa1-simsa4 attributes).
    """
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.se_outputs = {}  # {name: (attention_weights, output_features)}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all SE blocks to capture attention weights."""
        se_blocks = []
        for attr_name in ['simsa1', 'simsa2', 'simsa3', 'simsa4']:
            if hasattr(self.model, attr_name):
                se_block = getattr(self.model, attr_name)
                se_blocks.append((attr_name, se_block))
        
        if not se_blocks:
            print("Warning: No SE blocks found in model. SEAttention method will produce empty maps.")
            return
        
        for attr_name, se_block in se_blocks:
            # Hook on the fc (the sigmoid output = channel attention weights)
            def make_hook(name, block):
                def hook_fn(module, input, output):
                    # input[0] is the feature map BEFORE SE weighting: [B, C, H, W]
                    # output is the feature map AFTER SE weighting: [B, C, H, W]
                    feat_before = input[0].detach()
                    feat_after = output.detach()
                    
                    # Recover attention weights: output / input (element-wise)
                    # SE does: output = input * weights.expand_as(input)
                    # weights = avg_pool(input) -> fc -> sigmoid -> [B, C, 1, 1]
                    with torch.no_grad():
                        pooled = block.avg_pool(feat_before)  # [B, C, 1, 1]
                        attn_weights = block.fc(pooled.view(pooled.size(0), -1))  # [B, C]
                    
                    self.se_outputs[name] = {
                        'attn_weights': attn_weights.detach(),  # [B, C]
                        'features': feat_before.detach(),       # [B, C, H, W]
                    }
                return hook_fn
            
            hook = se_block.register_forward_hook(make_hook(attr_name, se_block))
            self.hooks.append(hook)
    
    def __call__(self, input_tensor):
        """
        Args:
            input_tensor: [1, C, H, W] tensor on device
        Returns:
            anomaly_map: [H, W] numpy array, normalized to [0, 1]
        """
        # Clear previous captures
        self.se_outputs.clear()
        
        # Forward pass (hooks will capture SE data)
        with torch.no_grad():
            self.model(input_tensor)
        
        if not self.se_outputs:
            # Fallback: return zeros if no SE data captured
            H, W = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((H, W))
        
        target_H, target_W = input_tensor.shape[2], input_tensor.shape[3]
        combined_map = np.zeros((target_H, target_W))
        
        for name, data in self.se_outputs.items():
            attn_weights = data['attn_weights']  # [B, C]
            features = data['features']          # [B, C, H, W]
            
            # Weighted spatial activation: sum over channels of (weight_c * |feature_c|)
            # attn_weights: [1, C] -> [1, C, 1, 1]
            weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(-1)  # [1, C, 1, 1]
            
            # Weighted activation map
            weighted_feat = (weights_expanded * features.abs()).sum(dim=1).squeeze(0)  # [H, W]
            spatial_map = weighted_feat.cpu().numpy()
            
            # Resize to input image size
            spatial_map = sk_resize(spatial_map, (target_H, target_W), mode='constant', preserve_range=True)
            
            # Normalize this scale's map to [0, 1] before combining
            smin, smax = spatial_map.min(), spatial_map.max()
            if smax - smin > 1e-8:
                spatial_map = (spatial_map - smin) / (smax - smin)
            
            combined_map += spatial_map
        
        # Normalize final combined map
        cmin, cmax = combined_map.min(), combined_map.max()
        if cmax - cmin > 1e-8:
            combined_map = (combined_map - cmin) / (cmax - cmin)
        else:
            combined_map = np.zeros_like(combined_map)
        
        # Light Gaussian smoothing
        combined_map = gaussian_filter(combined_map, sigma=2)
        cmin, cmax = combined_map.min(), combined_map.max()
        if cmax - cmin > 1e-8:
            combined_map = (combined_map - cmin) / (cmax - cmin)
        
        return combined_map
    
    def cleanup(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.se_outputs.clear()


class PixelBackboneFeatureExtractor(torch.nn.Module):
    """Extracts multi-level spatial features from the loaded DifferNet backbone."""

    def __init__(self, model, out_size=56):
        super().__init__()
        self.out_size = out_size
        if hasattr(model, 'alexnet'):
            self.alexnet = model.alexnet
            self.attn1 = getattr(model, 'cbam1', getattr(model, 'simsa1', None))
            self.attn2 = getattr(model, 'cbam2', getattr(model, 'simsa2', None))
            self.attn3 = getattr(model, 'cbam3', getattr(model, 'simsa3', None))
            self.attn4 = getattr(model, 'cbam4', getattr(model, 'simsa4', None))
        else:
            self.alexnet = model.feature_extractor
            self.attn1 = None
            self.attn2 = None
            self.attn3 = None
            self.attn4 = None

        self.layer_channels = [64, 192, 256]

        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x_input):
        feats = []

        x = self.alexnet.features[0](x_input)
        x = self.alexnet.features[1](x)
        if self.attn1 is not None:
            x = self.attn1(x)
        feats.append(x)

        x = self.alexnet.features[2](x)
        x = self.alexnet.features[3](x)
        x = self.alexnet.features[4](x)
        if self.attn2 is not None:
            x = self.attn2(x)
        feats.append(x)

        x = self.alexnet.features[5](x)
        x = self.alexnet.features[6](x)
        x = self.alexnet.features[7](x)
        if self.attn3 is not None:
            x = self.attn3(x)
        x = self.alexnet.features[8](x)
        x = self.alexnet.features[9](x)
        x = self.alexnet.features[10](x)
        x = self.alexnet.features[11](x)
        if self.attn4 is not None:
            x = self.attn4(x)
        feats.append(x)

        resized = [
            F.interpolate(f, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
            for f in feats
        ]
        concat = torch.cat(resized, dim=1)
        return concat, resized


def _pixel_pos_encoding(h, w, dim=64, target_device=device):
    assert dim % 4 == 0
    quarter_dim = dim // 4
    y_pos = torch.arange(h, device=target_device).float()
    x_pos = torch.arange(w, device=target_device).float()
    div = torch.exp(torch.arange(0, quarter_dim, device=target_device).float() * (-math.log(10000.0) / quarter_dim))
    yy = y_pos.unsqueeze(1) * div.unsqueeze(0)
    xx = x_pos.unsqueeze(1) * div.unsqueeze(0)
    pe_y = torch.cat([yy.sin(), yy.cos()], dim=1)
    pe_x = torch.cat([xx.sin(), xx.cos()], dim=1)
    pe_y = pe_y.unsqueeze(1).expand(h, w, dim // 2)
    pe_x = pe_x.unsqueeze(0).expand(h, w, dim // 2)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)


class PixelCondCouplingBlock(torch.nn.Module):
    def __init__(self, channels, cond_dim, hidden=256):
        super().__init__()
        self.c_split = channels // 2
        c_pass = channels - self.c_split
        self.net = torch.nn.Sequential(
            torch.nn.Linear(c_pass + cond_dim, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden, 2 * self.c_split),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, cond):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        st = self.net(torch.cat([x2, cond], dim=1))
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 2.0
        y1 = x1 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([y1, x2], dim=1), log_det


class PixelCondFlow(torch.nn.Module):
    def __init__(self, channels, cond_dim, n_blocks=4, hidden=256):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            PixelCondCouplingBlock(channels, cond_dim, hidden) for _ in range(n_blocks)
        ])
        self.register_buffer(
            'perm_idx',
            torch.tensor([(i + channels // 2) % channels for i in range(channels)])
        )

    def forward(self, x, cond):
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        for block in self.blocks:
            x, log_det = block(x, cond)
            log_det_sum = log_det_sum + log_det
            x = x.index_select(1, self.perm_idx)
        return x, log_det_sum


class CFlowPixelHead(torch.nn.Module):
    def __init__(self, layer_channels, cond_dim=64, n_blocks=4, hidden=256):
        super().__init__()
        self.cond_dim = cond_dim
        self.flows = torch.nn.ModuleList([
            PixelCondFlow(channels, cond_dim, n_blocks=n_blocks, hidden=hidden)
            for channels in layer_channels
        ])

    def nll_per_level(self, feats_per_level):
        maps = []
        for feat, flow in zip(feats_per_level, self.flows):
            batch_size, channels, height, width = feat.shape
            pe = _pixel_pos_encoding(height, width, dim=self.cond_dim, target_device=feat.device)
            pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
            x = feat.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
            cond = pe.reshape(batch_size * height * width, self.cond_dim)
            z, log_det = flow(x, cond)
            log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * channels * math.log(2 * math.pi)
            log_p = log_prob + log_det
            maps.append((-log_p).view(batch_size, height, width))
        return maps

    def score_map(self, feats_per_level, img_size):
        nll_maps = self.nll_per_level(feats_per_level)
        upsampled = [
            F.interpolate(level.unsqueeze(1), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(1)
            for level in nll_maps
        ]
        return sum(upsampled)


class PatchCoreBank:
    def __init__(self, coreset_ratio=0.1, k=3, seed=42):
        self.coreset_ratio = coreset_ratio
        self.k = k
        self.seed = seed
        self.bank = None

    @torch.no_grad()
    def fit(self, backbone, train_loader):
        features = []
        for batch in tqdm(train_loader, desc='[PatchCore] Building memory bank'):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device).view(-1, *images.shape[-3:])
            concat, _ = backbone(images)
            feat = concat.permute(0, 2, 3, 1).reshape(-1, concat.shape[1])
            features.append(feat.cpu())

        feat_all = torch.cat(features, dim=0)
        n_select = max(int(len(feat_all) * self.coreset_ratio), 1024)
        n_select = min(n_select, len(feat_all))
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(len(feat_all), n_select, replace=False)
        self.bank = feat_all[indices].to(device)
        print(f"[PatchCore] Memory bank: {self.bank.shape}")

    @torch.no_grad()
    def score_map(self, concat, img_size):
        batch_size, channels, height, width = concat.shape
        feat = concat.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        chunk_size = 2048
        min_distances = torch.empty(batch_size * height * width, device=device)

        for start in range(0, feat.shape[0], chunk_size):
            distances = torch.cdist(feat[start:start + chunk_size], self.bank)
            topk = torch.topk(distances, k=self.k, dim=1, largest=False).values
            min_distances[start:start + chunk_size] = topk.mean(dim=1)

        score = min_distances.view(batch_size, height, width)
        return F.interpolate(score.unsqueeze(1), size=(img_size, img_size), mode='bilinear', align_corners=False).squeeze(1)


def normalize_map(anomaly_map):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    if max_val - min_val <= 1e-8:
        return np.zeros_like(anomaly_map)
    return (anomaly_map - min_val) / (max_val - min_val)


def load_metric_datasets(dataset_path, class_name):
    loaded = utils.load_datasets(dataset_path, class_name)
    if not isinstance(loaded, tuple):
        raise RuntimeError("utils.load_datasets returned an unexpected value")

    if len(loaded) == 2:
        train_set, combined_dataset = loaded
    elif len(loaded) == 3:
        train_set, test_set, ground_truth_set = loaded
        combined_dataset = CombinedTestDataset(test_set, ground_truth_set)
    else:
        raise RuntimeError(f"Unsupported dataset tuple length: {len(loaded)}")

    return train_set, combined_dataset


def select_balanced_subset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset

    normal_idx = []
    anomaly_idx = []
    for idx in range(len(dataset)):
        _, label, _ = dataset[idx]
        if int(label) == 0:
            normal_idx.append(idx)
        else:
            anomaly_idx.append(idx)

    k_anom = min(limit // 2, len(anomaly_idx))
    k_norm = min(limit - k_anom, len(normal_idx))
    selected_indices = sorted(normal_idx[:k_norm] + anomaly_idx[:k_anom])
    return torch.utils.data.Subset(dataset, selected_indices)


def maybe_load_cflow_weights(cflow, candidate_paths, checkpoint=None):
    if checkpoint is not None and isinstance(checkpoint, dict) and 'cflow_state_dict' in checkpoint:
        cflow.load_state_dict(checkpoint['cflow_state_dict'])
        return 'model checkpoint'

    for path in candidate_paths:
        if not path or not os.path.exists(path):
            continue
        data = torch.load(path, map_location=device, weights_only=False)
        if isinstance(data, dict) and 'cflow_state_dict' in data:
            cflow.load_state_dict(data['cflow_state_dict'])
            return path
    return None


def train_cflow_head(cflow, backbone, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(cflow.parameters(), lr=lr)
    cflow.train()
    for epoch in range(epochs):
        losses = []
        for batch in tqdm(train_loader, desc=f'[CFLOW] Epoch {epoch + 1}/{epochs}', leave=False):
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device).view(-1, *images.shape[-3:])
            with torch.no_grad():
                _, feats = backbone(images)

            optimizer.zero_grad()
            nll_maps = cflow.nll_per_level(feats)
            loss = sum(level.mean() for level in nll_maps)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = float(np.mean(losses)) if losses else float('nan')
        print(f'[CFLOW] Epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}')
    cflow.eval()


def evaluate_pixel_models(model, train_set, eval_loader, output_dir, cflow_checkpoint=None,
                          loaded_checkpoint=None, cflow_epochs=10, cflow_cond_dim=64,
                          cflow_n_blocks=6, cflow_hidden=256, cflow_lr=1e-3,
                          patchcore_coreset=0.02, patchcore_k=3, pixel_sigma=4.0,
                          disable_patchcore=False, run_sanity_check=False, sanity_check_indices=None):
    print("\nPreparing CFLOW and PatchCore evaluation...")
    out_size = max(1, c.img_size[0] // 8)
    backbone = PixelBackboneFeatureExtractor(model, out_size=out_size).to(device).eval()
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    patchcore = None
    if disable_patchcore:
        print("PatchCore disabled for this run.")
    else:
        patchcore = PatchCoreBank(coreset_ratio=patchcore_coreset, k=patchcore_k)
        patchcore.fit(backbone, train_loader)

    cflow = CFlowPixelHead(
        layer_channels=backbone.layer_channels,
        cond_dim=cflow_cond_dim,
        n_blocks=cflow_n_blocks,
        hidden=cflow_hidden,
    ).to(device)
    cflow.eval()

    cflow_source = maybe_load_cflow_weights(
        cflow,
        candidate_paths=[cflow_checkpoint],
        checkpoint=loaded_checkpoint,
    )
    if cflow_source:
        print(f"CFLOW weights loaded from {cflow_source}.")
    else:
        print(f"CFLOW weights not found. Training head for {cflow_epochs} epochs without validation...")
        train_cflow_head(cflow, backbone, train_loader, cflow_epochs, cflow_lr)
        torch.save(
            {'cflow_state_dict': cflow.state_dict()},
            os.path.join(output_dir, 'cflow_trained_for_evaluation.pt')
        )

    metrics_by_method = {
        'CFLOW': {'pixel_auc': [], 'iou': [], 'dice': [], 'time': [], 'image_scores': [], 'road_most': [], 'road_least': [], 'sanity': []},
    }
    curves_store = {
        'CFLOW': {'road_most_curve': [], 'road_least_curve': []}
    }
    if not disable_patchcore:
        metrics_by_method['PatchCore'] = {'pixel_auc': [], 'iou': [], 'dice': [], 'time': [], 'image_scores': [], 'road_most': [], 'road_least': [], 'sanity': []}
        curves_store['PatchCore'] = {'road_most_curve': [], 'road_least_curve': []}
    image_labels = []

    print("Running CFLOW and PatchCore on the evaluation split...")
    for batch_idx, (images, labels, masks) in enumerate(tqdm(flat_batch_loader(eval_loader), total=len(eval_loader))):
        images = images.to(device)
        if len(images) == 0:
            continue

        for index in range(len(images)):
            img = images[index:index + 1]
            mask_tensor = masks[index] if masks is not None else None
            label_value = int(labels[index].item())
            image_labels.append(1 if label_value > 0 else 0)

            with torch.no_grad():
                concat, feats = backbone(img)

                start_time = time.time()
                cflow_map = cflow.score_map(feats, img.shape[-1]).squeeze(0).detach().cpu().numpy()
                cflow_map = normalize_map(gaussian_filter(cflow_map, sigma=pixel_sigma))
                metrics_by_method['CFLOW']['time'].append(time.time() - start_time)
                metrics_by_method['CFLOW']['image_scores'].append(float(cflow_map.max()))

                if not disable_patchcore:
                    start_time = time.time()
                    patchcore_map = patchcore.score_map(concat, img.shape[-1]).squeeze(0).detach().cpu().numpy()
                    patchcore_map = normalize_map(gaussian_filter(patchcore_map, sigma=pixel_sigma))
                    metrics_by_method['PatchCore']['time'].append(time.time() - start_time)
                    metrics_by_method['PatchCore']['image_scores'].append(float(patchcore_map.max()))

            current_mask = None
            if mask_tensor is not None:
                current_mask = mask_tensor.squeeze().cpu().numpy()
                if current_mask.ndim == 3:
                    current_mask = current_mask[0]

            method_maps = [('CFLOW', cflow_map)]
            if not disable_patchcore:
                method_maps.append(('PatchCore', patchcore_map))

            for method_name, anomaly_map in method_maps:
                if current_mask is not None:
                    is_anomalous = np.sum((current_mask > 0.5).astype(int)) > 0
                    if is_anomalous:
                        pixel_auc = calculate_pixel_auroc(anomaly_map, current_mask)
                        if not np.isnan(pixel_auc):
                            metrics_by_method[method_name]['pixel_auc'].append(pixel_auc)

                        iou_val, dice_val = calculate_iou_dice(anomaly_map, current_mask)
                        if not np.isnan(iou_val):
                            metrics_by_method[method_name]['iou'].append(iou_val)
                        if not np.isnan(dice_val):
                            metrics_by_method[method_name]['dice'].append(dice_val)

                # Compute ROAD
                r_most, r_most_scores, r_least, r_least_scores = calculate_road_metrics(model, img, anomaly_map)
                metrics_by_method[method_name]['road_most'].append(r_most)
                metrics_by_method[method_name]['road_least'].append(r_least)
                curves_store[method_name]['road_most_curve'].append(r_most_scores)
                curves_store[method_name]['road_least_curve'].append(r_least_scores)
                
                # Compute Sanity Check
                if run_sanity_check and sanity_check_indices and (batch_idx in sanity_check_indices) and (index == 0):
                    if method_name == 'CFLOW':
                        orig_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        orig_cflow_state = {k: v.cpu().clone() for k, v in cflow.state_dict().items()}
                        
                        randomize_model_weights(model)
                        randomize_model_weights(cflow)
                        model.eval()
                        cflow.eval()
                        
                        with torch.no_grad():
                            _, feats_rand = backbone(img)
                            cflow_map_rand = cflow.score_map(feats_rand, img.shape[-1]).squeeze(0).cpu().numpy()
                            cflow_map_rand = normalize_map(gaussian_filter(cflow_map_rand, sigma=pixel_sigma))
                            
                        corr, _ = spearmanr(anomaly_map.flatten(), cflow_map_rand.flatten())
                        metrics_by_method[method_name]['sanity'].append(corr)
                        print(f" [Debug] Sanity Check {method_name} (Idx {batch_idx}): {corr:.4f}")
                        
                        model.load_state_dict(orig_model_state)
                        cflow.load_state_dict(orig_cflow_state)
                        
                    elif method_name == 'PatchCore':
                        orig_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        randomize_model_weights(model)
                        model.eval()
                        
                        with torch.no_grad():
                            concat_rand, _ = backbone(img)
                            pc_map_rand = patchcore.score_map(concat_rand, img.shape[-1]).squeeze(0).cpu().numpy()
                            pc_map_rand = normalize_map(gaussian_filter(pc_map_rand, sigma=pixel_sigma))
                            
                        corr, _ = spearmanr(anomaly_map.flatten(), pc_map_rand.flatten())
                        metrics_by_method[method_name]['sanity'].append(corr)
                        print(f" [Debug] Sanity Check {method_name} (Idx {batch_idx}): {corr:.4f}")
                        
                        model.load_state_dict(orig_model_state)

                save_visualization(
                    img[0],
                    mask_tensor if mask_tensor is not None else None,
                    anomaly_map,
                    batch_idx * len(images) + index,
                    label_value,
                    True,
                    output_dir,
                    method_name,
                )

            torch.cuda.empty_cache()

    results = []
    labels_np = np.array(image_labels)
    for method_name, metric_store in metrics_by_method.items():
        result = {'Method': method_name}
        if len(np.unique(labels_np)) > 1 and metric_store['image_scores']:
            result['image_auc'] = calculate_image_auroc(metric_store['image_scores'], labels_np)
        for key in ['pixel_auc', 'iou', 'dice', 'time', 'road_most', 'road_least', 'sanity']:
            if metric_store[key]:
                result[key] = float(np.mean(metric_store[key]))
                
        # Save curves average
        if method_name in curves_store:
            for k, v in curves_store[method_name].items():
                if v:
                    try:
                        result[k] = np.mean(np.array(v), axis=0).tolist()
                    except:
                        pass
                        
        results.append(result)

    return results


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
    label_str = "anomaly" if label != 0 else "normal"
    pred_str = "correct" if pred_correct else "wrong"
    
    filename = f"{idx:04d}_{label_str}_{pred_str}_{method_name}.png"
    img_dir = os.path.join(output_dir, "images", method_name)
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
                if fname.endswith('_mask'):
                    fname = fname[:-5]
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

def evaluate_cam_xai_metrics(model_name, model_path, dataset_path, class_name, output_dir, limit=None,
                     run_sanity_check=True, cflow_checkpoint=None, cflow_epochs=10,
                     cflow_cond_dim=64, cflow_n_blocks=6, cflow_hidden=256,
                     cflow_lr=1e-3, patchcore_coreset=0.1, patchcore_k=3,
                     pixel_sigma=4.0, disable_patchcore=False):
    print(f"Starting evaluation for {model_name} on {class_name}...")
    
    # 1. Load Data
    c.n_transforms_test = 1 # Force 1 crop for consistent evaluation
    c.transf_rotations = False
    train_set, combined_dataset = load_metric_datasets(dataset_path, class_name)
    combined_dataset = select_balanced_subset(combined_dataset, limit)

    loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    if "se_differnet" in model_name.lower():
        base_model = SEDifferNet()
    else:
        base_model = DifferNet()
        
    model, checkpoint = load_weights(base_model, model_path)
    model.to(device)
    model.eval()
    
    # 3. Setup ALL Attribution Methods
    # -- GradCAM-based methods (from pytorch-grad-cam) --
    gradcam_methods = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'XGradCAM': XGradCAM
    }
    
    # Target Layers for GradCAM methods
    is_se_model = "se_differnet" in model_name.lower()
    if is_se_model:
         target_layers = [model.alexnet.features[-1]] 
    else:
         target_layers = [model.feature_extractor.features[-1]]
    
    # -- Custom methods (model-native, no GradCAM dependency) --
    custom_methods = {
        'InputGradient': InputGradientMethod(model, sigma=4),
    }
    
    # SEAttention only for SEDifferNet (requires SE blocks)
    if is_se_model:
        custom_methods['SEAttention'] = SEAttentionMethod(model)
    else:
        print("Note: SEAttention method skipped (model is not SEDifferNet).")
    
    # Build unified method list: (name, type, method_or_class)
    # type = 'gradcam' or 'custom'
    all_methods = []
    for name, cam_cls in gradcam_methods.items():
        all_methods.append((name, 'gradcam', cam_cls))
    for name, method_instance in custom_methods.items():
        all_methods.append((name, 'custom', method_instance))
    
    print(f"Methods to evaluate: {[m[0] for m in all_methods]}")
    
    # Results Storage
    results = []
    
    # Determine how many samples to run for Sanity Check (it is slow)
    total_samples = limit if limit else len(loader)
    
    sanity_check_indices = set()
    norm_count = 0
    anom_count = 0
    
    for i in range(len(combined_dataset)):
        _, label_value, _ = combined_dataset[i]
        if int(label_value) == 0:
            if norm_count < 3:
                sanity_check_indices.add(i)
                norm_count += 1
        else:
            if anom_count < 3:
                sanity_check_indices.add(i)
                anom_count += 1
                
        if norm_count == 3 and anom_count == 3:
            break

    print(f"Evaluating {total_samples} samples (Sanity Check on {len(sanity_check_indices)} indices - {norm_count} normal, {anom_count} anom)...")
    
    for name, method_type, method_ref in all_methods:
        print(f"\nRunning {name} ({'GradCAM-based' if method_type == 'gradcam' else 'Model-native'})...")
        
        metrics_store = {
            'road_most': [], 'road_least': [],
            'pixel_auc': [], 'iou': [], 'dice': [],
            'sanity': [], 'time': []
        }
        curves_store = {'road_most': [], 'road_least': []}

        # Initialize method
        cam_extractor = None
        custom_method = None
        
        if method_type == 'gradcam':
            try:
                cam_extractor = method_ref(model=model, target_layers=target_layers)
            except Exception as e:
                print(f"Failed to init {name}: {e}")
                continue
        else:
            custom_method = method_ref  # Already instantiated

        # Iterate Data
        for batch_idx, (images, labels, masks) in enumerate(tqdm(flat_batch_loader(loader), total=len(loader))):
            images = images.to(device)
            if len(images) == 0: continue
            
            for i in range(len(images)):
                img = images[i:i+1] # Keep batch dim [1, C, H, W]
                start_time = time.time()
                
                # --- Generate Attribution Map ---
                t_idx = None
                
                if method_type == 'gradcam':
                    # GradCAM-based: uses pytorch-grad-cam interface
                    targets = [AnomalyScoreTarget()]
                    cam_map = cam_extractor(input_tensor=img, targets=targets)[0, :]
                else:
                    # Custom method: direct call
                    cam_map = custom_method(img)
                
                # --- Metrics ---
                
                # Pixel-level metrics (ONLY for anomalous images where GT mask has defects)
                if masks is not None:
                     current_mask = masks[i].squeeze().cpu().numpy()
                     if current_mask.ndim == 3:
                         current_mask = current_mask[0] # Take first channel if RGB
                         
                     is_anomalous = np.sum((current_mask > 0.5).astype(int)) > 0
                     if is_anomalous:
                         # Pixel AUROC
                         p_auc = calculate_pixel_auroc(cam_map, current_mask)
                         if not np.isnan(p_auc):
                             metrics_store['pixel_auc'].append(p_auc)
                         
                         # IoU & Dice
                         iou_val, dice_val = calculate_iou_dice(cam_map, current_mask)
                         if not np.isnan(iou_val):
                             metrics_store['iou'].append(iou_val)
                         if not np.isnan(dice_val):
                             metrics_store['dice'].append(dice_val)
                
                # ROAD (faithfulness metrics — computed for all images)
                r_most, r_most_scores, r_least, r_least_scores = calculate_road_metrics(model, img, cam_map, t_idx)
                metrics_store['road_most'].append(r_most)
                metrics_store['road_least'].append(r_least)
                curves_store['road_most'].append(r_most_scores)
                curves_store['road_least'].append(r_least_scores)
                
                # Sanity Check (Run only on subset to save time)
                if run_sanity_check and batch_idx in sanity_check_indices and i == 0:
                    if method_type == 'gradcam':
                        # Standard sanity check: randomize model, recompute GradCAM, compare
                        sanity_val = calculate_sanity_check(model, img, cam_map, method_ref, target_layers, t_idx)
                    else:
                        # Custom method sanity check: randomize model, recompute with same custom method, compare
                        sanity_val = calculate_sanity_check_custom(model, img, cam_map, custom_method)
                    metrics_store['sanity'].append(sanity_val)
                    print(f" [Debug] Sanity Check {name} (Idx {batch_idx}): {sanity_val:.4f}")
        
                # Save visualization
                is_correct = True 
                save_visualization(img[0], masks[i] if masks is not None else None, cam_map, 
                                   batch_idx * len(images) + i, labels[i].item(), is_correct, output_dir, name)
                
                elapsed = time.time() - start_time
                metrics_store['time'].append(elapsed)

                # Cleanup per image
                del cam_map
                torch.cuda.empty_cache()

        # Aggregate Results for this Method
        res = {'Method': name}
        for k, v in metrics_store.items():
            if v:
                res[k] = np.mean(v)
        
        # Save curves average
        for k, v in curves_store.items():
            if v:
                try:
                    res[f'{k}_curve'] = np.mean(np.array(v), axis=0).tolist()
                except:
                    pass
                    
        results.append(res)
        
        # Cleanup per method
        if cam_extractor is not None:
            del cam_extractor
        if method_type == 'custom' and hasattr(custom_method, 'cleanup'):
            custom_method.cleanup()
        del metrics_store, curves_store, res
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    pixel_model_results = evaluate_pixel_models(
        model=model,
        train_set=train_set,
        eval_loader=loader,
        output_dir=output_dir,
        cflow_checkpoint=cflow_checkpoint,
        loaded_checkpoint=checkpoint,
        cflow_epochs=cflow_epochs,
        cflow_cond_dim=cflow_cond_dim,
        cflow_n_blocks=cflow_n_blocks,
        cflow_hidden=cflow_hidden,
        cflow_lr=cflow_lr,
        patchcore_coreset=patchcore_coreset,
        patchcore_k=patchcore_k,
        pixel_sigma=pixel_sigma,
        disable_patchcore=disable_patchcore,
        run_sanity_check=run_sanity_check,
        sanity_check_indices=sanity_check_indices,
    )
    results.extend(pixel_model_results)
    
    # --- Reporting & plotting ---
    df = pd.DataFrame(results)
    
    # Identify non-curve columns (scalar metrics)
    scalar_cols = [col for col in df.columns if 'curve' not in col and col != 'Method']
    
    if not df.empty:
        # Save CSV
        csv_path = os.path.join(output_dir, 'results.csv')
        df_csv = df.drop(columns=[c for c in df.columns if 'curve' in c])
        
        # Add road_delta for interpretation BEFORE printing and saving
        if 'road_most' in df_csv.columns and 'road_least' in df_csv.columns:
            df_csv['road_delta'] = df_csv['road_most'] - df_csv['road_least']
            
        # Re-order columns to make sure they are in a nice, consistent order
        ordered_cols = ['Method']
        for col in ['image_auc', 'pixel_auc', 'iou', 'dice', 'road_most', 'road_least', 'road_delta', 'sanity', 'time']:
            if col in df_csv.columns:
                ordered_cols.append(col)
        # Any other columns that might exist
        for col in df_csv.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)
        df_csv = df_csv[ordered_cols]
        
        df_csv.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Create a display dataframe with arrows in column names for the terminal print
        df_display = df_csv.copy()
        column_rename = {
            'image_auc': 'image_auc (↑)',
            'pixel_auc': 'pixel_auc (↑)',
            'iou': 'iou (↑)',
            'dice': 'dice (↑)',
            'road_most': 'road_most (↑)',
            'road_least': 'road_least (↓)',
            'road_delta': 'road_delta (↑)',
            'sanity': 'sanity (↓)',
            'time': 'time (↓)'
        }
        df_display = df_display.rename(columns=column_rename)
        
        print("\n" + "=" * 100)
        print("ANOMALY MAP EVALUATION RESULTS")
        print("=" * 100)
        print(df_display.to_string(index=False))
        print("\n--- Interpretation Guide ---")
        print("  image_auc  ↑ : Higher is better (separates normal vs anomalous images).")
        print("  pixel_auc  ↑ : Higher is better (1.0 = perfect localization). Only anomalous images.")
        print("  iou        ↑ : Higher is better (overlap between binarized CAM and GT mask).")
        print("  dice       ↑ : Higher is better (similar to IoU but emphasizes overlap).")
        print("  road_most  ↑ : Higher is better (removing important pixels hurts the model).")
        print("  road_least ↓ : Lower is better (removing unimportant pixels doesn't hurt).")
        print("  road_delta ↑ : road_most - road_least. Higher is better (good separation).")
        print("  sanity     ↓ : Lower is better (Spearman correlation with random-weights CAM).")
        print("  time       ↓ : Lower is better (average time per image).")
        print("=" * 100)
        
        # --- Plot 1: Localization Metrics (pixel_auc, iou, dice) ---
        loc_cols = [c for c in ['pixel_auc', 'iou', 'dice'] if c in df_csv.columns]
        # Arrow labels for charts
        METRIC_ARROWS = {
            'image_auc': 'image_auc ↑',
            'pixel_auc': 'pixel_auc ↑',
            'iou': 'iou ↑',
            'dice': 'dice ↑',
            'road_most': 'road_most ↑',
            'road_least': 'road_least ↓',
            'road_delta': 'road_delta ↑',
        }
        if loc_cols:
            plt.figure(figsize=(10, 5))
            df_loc = df_csv.melt(id_vars=['Method'], value_vars=loc_cols, var_name='Metric', value_name='Score')
            df_loc['Metric'] = df_loc['Metric'].map(lambda m: METRIC_ARROWS.get(m, m))
            sns.barplot(data=df_loc, x='Metric', y='Score', hue='Method')
            plt.title(f"Localization Metrics — {class_name}")
            plt.ylim(0, 1)
            plt.ylabel("Score")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'localization_metrics.png'), dpi=150)
            plt.close()
        
        # --- Plot 2: Faithfulness Metrics (road_most, road_least, road_delta) ---
        faith_cols = [c for c in ['road_most', 'road_least', 'road_delta'] if c in df_csv.columns]
        if faith_cols:
            plt.figure(figsize=(10, 5))
            df_faith = df_csv.melt(id_vars=['Method'], value_vars=faith_cols, var_name='Metric', value_name='Score')
            df_faith['Metric'] = df_faith['Metric'].map(lambda m: METRIC_ARROWS.get(m, m))
            sns.barplot(data=df_faith, x='Metric', y='Score', hue='Method')
            plt.title(f"Faithfulness Metrics (ROAD) — {class_name}")
            plt.ylabel("Normalized Score")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'faithfulness_metrics.png'), dpi=150)
            plt.close()
        
        # --- Plot 3: ROAD Curves per percentile ---
        curve_metrics = ['road_most', 'road_least']
        for cm in curve_metrics:
            col_name = f'{cm}_curve'
            if col_name in df.columns:
                plt.figure(figsize=(8, 5))
                percentiles = [20, 40, 60, 80]
                for idx, row in df.iterrows():
                    if row[col_name] is not None and isinstance(row[col_name], list):
                         y_vals = row[col_name]
                         plt.plot(percentiles[:len(y_vals)], y_vals, marker='o', label=f"{row['Method']}")
                plt.title(f"{cm.replace('_', ' ').title()} per Percentile")
                plt.xlabel("Percentile of pixels removed (%)")
                plt.ylabel("Normalized score drop")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{cm}_curve.png'), dpi=150)
                plt.close()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CAM Metrics for DifferNet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--class_name", type=str, default="glass-insulator", help="Class name")
    parser.add_argument("--dataset_path", type=str, default=c.dataset_path, help="Path to dataset")
    parser.add_argument("--output_base", type=str, default="./results_anomaly_map", help="Base output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--steps", type=int, default=20, help="Number of steps for Deletion/Insertion")
    parser.add_argument("--run_sanity_check", type=str2bool, default=False, help="Run sanity check")
    parser.add_argument("--cflow_checkpoint", type=str, default=None, help="Optional checkpoint containing cflow_state_dict")
    parser.add_argument("--cflow_epochs", type=int, default=10, help="Train CFLOW for this many epochs if no checkpoint is provided")
    parser.add_argument("--cflow_cond_dim", type=int, default=64, help="CFLOW positional encoding dimension")
    parser.add_argument("--cflow_n_blocks", type=int, default=6, help="Number of conditional flow blocks per level")
    parser.add_argument("--cflow_hidden", type=int, default=256, help="Hidden size of CFLOW coupling MLPs")
    parser.add_argument("--cflow_lr", type=float, default=1e-3, help="Learning rate used when CFLOW needs quick training")
    parser.add_argument("--patchcore_coreset", type=float, default=0.1, help="PatchCore coreset ratio")
    parser.add_argument("--patchcore_k", type=int, default=3, help="PatchCore kNN neighbors")
    parser.add_argument("--pixel_sigma", type=float, default=4.0, help="Gaussian smoothing sigma for CFLOW and PatchCore maps")
    parser.add_argument("--disable_patchcore", action='store_true', help="Disable PatchCore evaluation")
    # python evaluate_cam_xai_metrics.py --model_path anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\glass-insulator\glass-insulator_se_differnet_glass_insulator_200_1_epoch_170.pt --limit 10
    args = parser.parse_args()
    
    # Create Unique Output Dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_base, f"run_{timestamp}_{args.class_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Determine Model Name (heuristic from path)
    model_name_guess = os.path.basename(args.model_path)
    
    evaluate_cam_xai_metrics(
        model_name=model_name_guess,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        class_name=args.class_name,
        output_dir=run_dir,
        limit=args.limit,
        run_sanity_check=args.run_sanity_check,
        cflow_checkpoint=args.cflow_checkpoint,
        cflow_epochs=args.cflow_epochs,
        cflow_cond_dim=args.cflow_cond_dim,
        cflow_n_blocks=args.cflow_n_blocks,
        cflow_hidden=args.cflow_hidden,
        cflow_lr=args.cflow_lr,
        patchcore_coreset=args.patchcore_coreset,
        patchcore_k=args.patchcore_k,
        pixel_sigma=args.pixel_sigma,
        disable_patchcore=args.disable_patchcore,
    )
