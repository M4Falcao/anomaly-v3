import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Add current dir to path
sys.path.append(os.getcwd())

import config as c
import utils
from model import DifferNet, SEDifferNet, load_weights

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
except ImportError:
    print("Error: pytorch-grad-cam not installed.")
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Utils ---
class AnomalyScoreTarget:
    def __call__(self, model_output):
        if isinstance(model_output, tuple):
             model_output = model_output[0]
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)
        return torch.mean(torch.sum(model_output ** 2, dim=1))

def _get_score(output):
    with torch.no_grad():
        if isinstance(output, tuple):
             output = output[0]
        if output.dim() == 1:
            output = output.unsqueeze(0)
        score = torch.mean(torch.sum(output ** 2, dim=1)).item()
    return score

def denormalize(tensor):
    mean = np.array(c.norm_mean)
    std = np.array(c.norm_std)
    img = tensor.permute(1, 2, 0).cpu().clone().numpy()
    img = img * std + mean
    return np.clip(img, 0, 1)

def tensor_to_cv2(tensor):
    img = denormalize(tensor[0])
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# --- Visualization Generators ---

def visualize_deletion(model, input_tensor, cam_mask, steps=10):
    """
    Returns list of (image, score) tuples for Deletion metric.
    Deletion: Remove most relevant pixels (black out).
    """
    cam_flat = cam_mask.flatten()
    sorted_indices = np.argsort(cam_flat)[::-1].copy() # Descending
    
    total_pixels = cam_flat.shape[0]
    step_size = max(1, total_pixels // steps)
    
    img_pert = input_tensor.clone()
    c, h, w = img_pert.shape[1:]
    
    results = []
    
    # Step 0 (Original)
    with torch.no_grad():
        score = _get_score(model(img_pert))
    results.append((tensor_to_cv2(img_pert), score))
    
    for i in range(1, steps + 1):
        indices_to_remove = sorted_indices[:i*step_size]
        
        mask = torch.ones(h * w, device=device)
        mask[indices_to_remove] = 0.0 # Black out
        mask = mask.view(1, 1, h, w)
        img_pert = img_pert * mask
        
        with torch.no_grad():
            score = _get_score(model(img_pert))
        results.append((tensor_to_cv2(img_pert), score))
        
    return results

def visualize_insertion(model, input_tensor, cam_mask, steps=10):
    """
    Returns list of (image, score) tuples for Insertion metric.
    Insertion: Start from blur/black, add most relevant pixels.
    """
    cam_flat = cam_mask.flatten()
    sorted_indices = np.argsort(cam_flat)[::-1].copy() # Descending
    
    total_pixels = cam_flat.shape[0]
    step_size = max(1, total_pixels // steps)
    
    c, h, w = input_tensor.shape[1:]
    
    # Base: Blurred image
    # Simple Gaussian Blur simulation via interpolation
    img_blur = F.interpolate(input_tensor, scale_factor=0.1, mode='bilinear')
    img_blur = F.interpolate(img_blur, size=(h, w), mode='bilinear')
    
    # Or black start:
    # current_input = torch.zeros_like(input_tensor)
    current_input = img_blur.clone()
    
    results = []
    
    # Step 0
    with torch.no_grad():
        score = _get_score(model(current_input))
    results.append((tensor_to_cv2(current_input), score))
    
    for i in range(1, steps + 1):
        limit = min(i * step_size, total_pixels)
        indices_to_add = sorted_indices[:limit]
        ys, xs = np.unravel_index(indices_to_add, (h, w))
        
        # Restore pixels from original
        current_input[0, :, ys, xs] = input_tensor[0, :, ys, xs]
        
        with torch.no_grad():
             score = _get_score(model(current_input))
        results.append((tensor_to_cv2(current_input), score))
        
    return results

def visualize_road_lerf(model, input_tensor, cam_mask, steps=10):
    """
    Visualizes ROAD (Least Relevant First).
    Removes Least Relevant pixels first.
    Imputation: Blur (approximated).
    """
    cam_flat = cam_mask.flatten()
    sorted_indices = np.argsort(cam_flat) # Ascending (Least relevant first)
    
    total_pixels = cam_flat.shape[0]
    step_size = max(1, total_pixels // steps)
    
    c, h, w = input_tensor.shape[1:]
    
    # Blur source
    img_blur = F.interpolate(input_tensor, scale_factor=0.1, mode='bilinear')
    img_blur = F.interpolate(img_blur, size=(h, w), mode='bilinear')
    
    current_input = input_tensor.clone()
    
    results = []
    
    # Step 0 (Original)
    with torch.no_grad():
        score = _get_score(model(current_input))
    results.append((tensor_to_cv2(current_input), score))
    
    for i in range(1, steps + 1):
        indices_to_replace = sorted_indices[:i*step_size]
        ys, xs = np.unravel_index(indices_to_replace, (h, w))
        
        # Replace with blur
        current_input[0, :, ys, xs] = img_blur[0, :, ys, xs]
        
        with torch.no_grad():
            score = _get_score(model(current_input))
        results.append((tensor_to_cv2(current_input), score))
            
    return results

def stitch_steps(results, title, step_names):
    """
    Tiles images side-by-side with scores.
    """
    images = [r[0] for r in results]
    scores = [r[1] for r in results]
    
    h, w, c = images[0].shape
    
    # Compose logic
    # Add text to each image
    labeled_images = []
    
    # Calculate footer height based on likely image width to keep proportion, 
    # but fixed increase is requested. Let's make it bigger.
    footer_h = 60 
    font_scale = 0.8
    thickness = 2
    
    for i, img in enumerate(images):
        canvas = img.copy()
        # Add white strip at bottom
        footer = np.ones((footer_h, w, 3), dtype=np.uint8) * 255
        
        label = f"Step {i}" if i > 0 else "Orig"
        score_txt = f"{scores[i]:.2f}"
        
        # Center text roughly
        cv2.putText(footer, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)
        cv2.putText(footer, score_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness)
        
        # Stack
        full = np.vstack((canvas, footer))
        labeled_images.append(full)
        
    final_img = np.hstack(labeled_images)
    return final_img

def plot_metric_curve(results, title, output_path):
    """
    Plots the metric curve and saves it clearly.
    results: list of (image, score)
    """
    scores = [r[1] for r in results]
    steps = len(scores)
    x = np.linspace(0, 1, steps)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, scores, marker='o', linestyle='-', color='blue', linewidth=2, markersize=5)
    
    # Shade area under curve
    plt.fill_between(x, scores, alpha=0.2, color='blue')
    
    plt.title(f"{title} Curve", fontsize=14)
    plt.xlabel("Fraction of Pixels Modified", fontsize=12)
    plt.ylabel("Anomaly Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Annotate AUC
    auc_val = np.trapezoid(scores, dx=1.0/(steps-1)) if steps > 1 else scores[0]
    plt.text(0.5, 0.9, f"AUC: {auc_val:.3f}", transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

# --- Main ---
def run_visualization(model_path, image_path=None, dataset_path=None, class_name=None, index=0, steps=5, output_dir="./viz_steps"):
    
    print(f"Vizualizing steps... Model: {os.path.basename(model_path)}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Image
    c.n_transforms_test = 1
    c.transf_rotations = False
    
    if image_path:
        # Load single image
        import torchvision.transforms as T
        from PIL import Image
        img_pil = Image.open(image_path).convert('RGB')
        t = T.Compose([
            T.Resize(c.img_size),
            T.CenterCrop(c.img_size),
            T.ToTensor(),
            T.Normalize(c.norm_mean, c.norm_std)
        ])
        img = t(img_pil).unsqueeze(0)
        label = "Custom"
    else:
        # Load from dataset
        _, test_set, _ = utils.load_datasets(dataset_path, class_name)
        img, lbl = test_set[index]
        # Dataset might return [N_crops, C, H, W]
        if img.dim() == 3: # [C, H, W]
             img = img.unsqueeze(0)
        elif img.dim() == 4: # [N, C, H, W] or [1, C, H, W]
             pass # Already has batch/crop dim
        
        label = f"Idx_{index}"
        
    img = img.to(device)
    # Ensure it is [1, C, H, W] if we want single image
    if img.shape[0] > 1:
         img = img[0:1]
    
    # Load Model
    if "sediffernet" in os.path.basename(model_path).lower():
        base_model = SEDifferNet()
        target_layers = [base_model.alexnet.features[-1]]
    else:
        base_model = DifferNet()
        target_layers = [base_model.feature_extractor.features[-1]]
        
    model, _ = load_weights(base_model, model_path)
    model.to(device)
    model.eval()
    
    # Generate CAM
    cam_extractor = GradCAM(model=model, target_layers=target_layers)
    targets = [AnomalyScoreTarget()]
    cam_map = cam_extractor(input_tensor=img, targets=targets)[0, :]
    
    # Run Visualizations
    
    # 1. Deletion
    print("Generating Deletion visualization...")
    res_del = visualize_deletion(model, img, cam_map, steps=steps)
    img_del = stitch_steps(res_del, "Deletion", [])
    cv2.imwrite(os.path.join(output_dir, f"{label}_deletion_steps.png"), img_del)
    plot_path = os.path.join(output_dir, f"{label}_deletion_curve.png")
    plot_metric_curve(res_del, "Deletion", plot_path)
    
    # 2. Insertion
    print("Generating Insertion visualization...")
    res_ins = visualize_insertion(model, img, cam_map, steps=steps)
    img_ins = stitch_steps(res_ins, "Insertion", [])
    cv2.imwrite(os.path.join(output_dir, f"{label}_insertion_steps.png"), img_ins)
    plot_path = os.path.join(output_dir, f"{label}_insertion_curve.png")
    plot_metric_curve(res_ins, "Insertion", plot_path)
    
    # 3. ROAD (LeRF)
    print("Generating ROAD (LeRF) visualization...")
    res_road = visualize_road_lerf(model, img, cam_map, steps=steps)
    img_road = stitch_steps(res_road, "ROAD_LeRF", [])
    cv2.imwrite(os.path.join(output_dir, f"{label}_road_lerf_steps.png"), img_road)
    plot_path = os.path.join(output_dir, f"{label}_road_lerf_curve.png")
    plot_metric_curve(res_road, "ROAD (LeRF)", plot_path)
    
    print(f"Done. Images and plots saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", default=c.dataset_path)
    parser.add_argument("--class_name", default="glass-insulator")
    parser.add_argument("--index", type=int, default=0, help="Index of image in test set")
    parser.add_argument("--steps", type=int, default=5)
    
    args = parser.parse_args()
    
    run_visualization(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        class_name=args.class_name,
        index=args.index,
        steps=args.steps
    )
