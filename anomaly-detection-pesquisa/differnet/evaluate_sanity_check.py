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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.stats import spearmanr

# Add current dir to path
sys.path.append(os.getcwd())

import config as c
import utils
from model import DifferNet, SEDifferNet, load_weights

# Pytorch GradCAM imports
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
except ImportError:
    print("Error: pytorch-grad-cam not installed. Please install it using 'pip install grad-cam'.")
    sys.exit(1)

# --- Configuration & Setup ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AnomalyScoreTarget:
    """
    Target for Normalizing Flow models (DifferNet/SEDifferNet).
    The 'score' is the norm of the latent vector z.
    We want to maximize the anomaly score (maximize ||z||^2).
    """
    def __call__(self, model_output):
        if isinstance(model_output, tuple):
             model_output = model_output[0]
        
        # Ensure 2D [Batch, Feature]
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
            
        return torch.mean(torch.sum(model_output ** 2, dim=1))

def randomize_model_weights(model):
    """
    Randomizes the weights of the model (Normalizing Flow head in this case).
    This simulates a 'training from scratch' or trash model state.
    """
    if hasattr(model, 'nf'):
        print("Randomizing NF head weights for Sanity Check...")
        for m in model.nf.modules():
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    else:
        # Fallback
        print("Model has no 'nf' attribute. Randomizing all detected Linear/Conv layers (fallback).")
        for m in model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.0)

def calculate_sanity_check(model, input_tensor, original_cam, cam_extractor_cls, target_layers):
    """
    Calculates Sanity Check (Cascading Randomization).
    Measures rank correlation between original CAM and CAM from randomized model.
    """
    # 1. Save original weights to CPU (RAM) to save GPU
    original_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    try:
        # 2. Randomize ORIGINAL Model in-place
        randomize_model_weights(model)
        model.eval() 

        # 3. Compute CAM with randomized model
        cam_extractor = cam_extractor_cls(model=model, target_layers=target_layers)
        targets = [AnomalyScoreTarget()]
        
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
        model.load_state_dict(original_state_dict)
        
        # Cleanup
        if 'cam_extractor' in locals():
            del cam_extractor
        del original_state_dict
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    return corr

class CombinedTestDataset(Dataset):
    def __init__(self, test_dataset):
        self.test_dataset = test_dataset

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        image, label = self.test_dataset[idx]
        return image, label

def flat_batch_loader(dataloader):
    """Yields (image, label) correctly shaped from batch."""
    for batch in dataloader:
        if len(batch) == 2:
            images, labels = batch
            if images.dim() == 5:
                B, N, C, H, W = images.shape
                images = images.view(-1, C, H, W)
                labels = labels.repeat_interleave(N)
            yield images, labels
        else:
            yield batch

def evaluate_sanity_check(model_name, model_path, dataset_path, class_name, output_dir, limit=None):
    print(f"Starting Sanity Check Evaluation for {model_name} on {class_name}...")
    
    # 1. Load Data
    c.n_transforms_test = 1 
    c.transf_rotations = False
    # We only need test set, no ground truth needed for Sanity Check
    _, test_set, _ = utils.load_datasets(dataset_path, class_name)
    combined_dataset = CombinedTestDataset(test_set)
    loader = DataLoader(combined_dataset, batch_size=1, shuffle=False)
    
    # 2. Load Model
    if "sediffernet" in model_name.lower():
        base_model = SEDifferNet()
        target_layers = [base_model.simsa4]
    else:
        base_model = DifferNet()
        target_layers = [base_model.feature_extractor.features[-1]]
        
    model, _ = load_weights(base_model, model_path)
    model.to(device)
    model.eval()
    
    cam_methods = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'XGradCAM': XGradCAM
    }
    
    results = []
    
    total_samples = limit if limit else len(loader)
    print(f"Evaluating {total_samples} samples...")

    for name, cam_cls in cam_methods.items():
        print(f"Running {name}...")
        sanity_scores = []
        
        try:
            cam_extractor = cam_cls(model=model, target_layers=target_layers)
        except Exception as e:
            print(f"Failed to init {name}: {e}")
            continue

        for batch_idx, (images, labels) in enumerate(tqdm(flat_batch_loader(loader), total=len(loader))):
            if limit and batch_idx >= limit:
                break
            images = images.to(device)
            if len(images) == 0: continue
            
            for i in range(len(images)):
                img = images[i:i+1]
                
                # Original CAM
                targets = [AnomalyScoreTarget()]
                cam_map = cam_extractor(input_tensor=img, targets=targets)[0, :]
                
                # Sanity Check
                sanity_val = calculate_sanity_check(model, img, cam_map, cam_cls, target_layers)
                sanity_scores.append(sanity_val)
                
                # Cleanup
                del cam_map, targets
                torch.cuda.empty_cache()

        # Save Result
        mean_sanity = np.nanmean(sanity_scores)
        print(f" > {name} Mean Sanity Score: {mean_sanity:.4f}")
        results.append({'Method': name, 'Sanity Check': mean_sanity})
        
        # Cleanup Method
        if 'cam_extractor' in locals():
            del cam_extractor
        del sanity_scores
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'sanity_check_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sanity Check for DifferNet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pt model weights")
    parser.add_argument("--class_name", type=str, default="glass-insulator", help="Class name")
    parser.add_argument("--dataset_path", type=str, default=c.dataset_path, help="Path to dataset")
    parser.add_argument("--output_base", type=str, default="./results_sanity", help="Base output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_base, f"run_{timestamp}_{args.class_name}")
    os.makedirs(run_dir, exist_ok=True)
    
    model_name_guess = os.path.basename(args.model_path)
    
    evaluate_sanity_check(
        model_name=model_name_guess,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        class_name=args.class_name,
        output_dir=run_dir,
        limit=args.limit
    )
