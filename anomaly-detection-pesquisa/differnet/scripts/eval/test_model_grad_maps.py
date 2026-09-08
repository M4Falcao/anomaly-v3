"""Single-model test run reporting image- and pixel-level metrics.

Evaluates one checkpoint on the configured class, deriving pixel-level anomaly
maps from input gradients and comparing them against the ground-truth masks.

Usage:
    python scripts/eval/test_model_grad_maps.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, confusion_matrix
import config as c
from core import utils
from core.model import DifferNet, SEDifferNet, load_weights
from scipy.ndimage import rotate, gaussian_filter
from torch.autograd import Variable
from torch.amp import autocast
import skimage
import skimage.transform


def get_grad_maps(model, inputs, labels, optimizer):
    model.eval()
    inputs = Variable(inputs, requires_grad=True)

    with autocast('cuda'):
        z = model(inputs)
        loss = utils.get_loss(z, model.nf.jacobian(run_forward=False))
    
    optimizer.zero_grad()
    loss.backward()

    if inputs.grad is None:
        return None

    grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Filter only anomalies (labels > 0)
    grad = grad[labels > 0]
    
    if grad.shape[0] == 0:
        return None

    grad = utils.t2np(grad)
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

def calculate_pixel_level_auroc(predictions, ground_truth_masks):
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    ground_truth_masks_binary = (ground_truth_masks > 0).astype(int)
    predictions_flat = predictions_resized.reshape(-1)
    ground_truth_flat = ground_truth_masks_binary.reshape(-1)
    
    if len(np.unique(ground_truth_flat)) < 2:
        return 0.5
        
    pixel_auroc = roc_auc_score(ground_truth_flat, predictions_flat)
    return pixel_auroc

def calculate_pg2(y_true, y_scores, fixed_fnr=0.02):
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 1000)
    best_tnr = 0.0
    best_threshold = thresholds[0]
    
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    
    if P == 0 or N == 0:
        return 0.0, best_threshold

    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        fn = np.sum((preds == 0) & (y_true == 1))
        fnr = fn / P
        
        if fnr <= fixed_fnr:
            tn = np.sum((preds == 0) & (y_true == 0))
            tnr = tn / N
            if tnr > best_tnr:
                best_tnr = tnr
                best_threshold = thresh

    return best_tnr, best_threshold

def find_best_f1_and_acc(y_true, y_scores):
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 1000)
    best_f1 = 0.0
    best_acc = 0.0
    
    for thresh in thresholds:
        preds = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        
        if f1 > best_f1:
            best_f1 = f1
        if acc > best_acc:
            best_acc = acc
            
    return best_f1, best_acc

def evaluate_differnet_model(model=None, model_path=None, class_name=None, dataset_path=None, device=None):
    if device is None:
        device = c.device

    if dataset_path is None:
        dataset_path = c.dataset_path

    if class_name is None:
        class_name = c.class_name

    print(f"Preparing evaluation for class: {class_name}")

    if model is None:
        if model_path is None:
            raise ValueError("Either 'model' or 'model_path' must be provided.")
        
        if "sediffernet" in model_path.lower() or "se_differnet" in model_path.lower():
            base_model = SEDifferNet()
        else:
            base_model = DifferNet()
            
        model, _ = load_weights(base_model, model_path)
    
    model.to(device)
    model.eval()

    # Need optimizer for the gradient calculation
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init)

    # Prepare Data
    train_set, test_set, ground_truth_set = utils.load_datasets(dataset_path, class_name)
    train_loader, test_loader, ground_truth_loader = utils.make_dataloaders(train_set, test_set, ground_truth_set)

    test_loader.dataset.get_fixed = True

    pixel_level_auroc_scores_test = []
    test_labels = []
    test_z = []

    print("Extracting features and scoring...")
    for i, data in enumerate(tqdm(test_loader)):
        optimizer.zero_grad()
        inputs, labels = utils.preprocess_batch(data)
        
        # --- Image Level Score Calculation (Standard) ---
        with torch.no_grad():
            z = model(inputs)
            test_labels.append(utils.t2np(labels))
            test_z.append(z)

        # --- Pixel Level Score Calculation (Gradient-based) ---
        grad_map = get_grad_maps(model, inputs, labels, optimizer)
        
        if grad_map is not None and ground_truth_loader is not None:
            if 'ground_truth_iter' not in locals():
                ground_truth_iter = iter(ground_truth_loader)
            try:
                gt_dat = next(ground_truth_iter)
            except StopIteration:
                ground_truth_iter = iter(ground_truth_loader)
                gt_dat = next(ground_truth_iter)
            
            gt_masks = gt_dat[0].to(device)
            gt_masks = gt_masks[:grad_map.shape[0]]
            
            if len(gt_masks) == len(grad_map):
                pixel_auroc_temp = calculate_pixel_level_auroc(grad_map, utils.t2np(gt_masks))
                pixel_level_auroc_scores_test.append(pixel_auroc_temp)

    test_loader.dataset.get_fixed = False

    is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = utils.t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))

    all_scores = anomaly_score
    all_labels = is_anomaly

    metrics = {}

    try:
        image_auroc = roc_auc_score(all_labels, all_scores)
        metrics['Image_AUROC'] = image_auroc
    except ValueError:
        metrics['Image_AUROC'] = 0.5
        print("Warning: Could not compute Image AUROC")

    best_f1, best_acc = find_best_f1_and_acc(all_labels, all_scores)
    metrics['Best_F1_Score'] = best_f1
    metrics['Best_Accuracy'] = best_acc

    pg2_score, _ = calculate_pg2(all_labels, all_scores, fixed_fnr=0.02)
    metrics['PG2'] = pg2_score

    if len(pixel_level_auroc_scores_test) > 0:
        metrics['Pixel_AUROC'] = np.mean(pixel_level_auroc_scores_test)
    else:
        metrics['Pixel_AUROC'] = None
        print("Warning: Pixel AUROC could not be calculated.")

    return metrics

if __name__ == '__main__':
    try:
        results = evaluate_differnet_model(
            model_path=r"C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\models\lightning-rod-suspension_se_differnet_lightning_rod_suspension_100_1_epoch_40.pt",
            class_name="lightning-rod-suspension"
        )
        print("\n--- Evaluation Results ---")
        for k, v in results.items():
            print(f"{k}: {v:.4f}" if v is not None else f"{k}: None")
    except Exception as e:
        print(f"Could not run default evaluation: {e}")
