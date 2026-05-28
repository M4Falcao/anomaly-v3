"""Complete training pipeline for all pixel-level anomaly detection methods.

Trains each method fully (like train.py), evaluating both:
- Image-level AUROC (anomaly detection)
- Pixel-level AUROC (anomaly localization)

Saves best models for both metrics.

Usage:
    python -m pixel_pipeline.train_full
    python -m pixel_pipeline.train_full --methods "F_FastFlow,G_CFLOW" --epochs 50
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from .data import make_loaders, TrainNormalDataset, PixelTestDataset
from .methods import METHODS
from .postproc import gaussian_smooth


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def safe_torch_save(obj, target_path, retries=2):
    """Safely save checkpoint with temp file + atomic replace."""
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tmp_path = target_path + ".tmp"
    for attempt in range(retries + 1):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, target_path)
            return True
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            try:
                torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
                os.replace(tmp_path, target_path)
                return True
            except:
                pass
            time.sleep(0.5)
    print(f"Warning: failed to save checkpoint to {target_path}")
    return False


class ScoreObserver:
    """Track best score and epoch."""
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = -1.0
        self.last = 0.0

    def update(self, score, epoch):
        self.last = score
        improved = False
        if score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
            improved = True
        return improved

    def __str__(self):
        return f"{self.name}: last={self.last:.4f} max={self.max_score:.4f} (ep {self.max_epoch})"


# =============================================================================
# Evaluation functions
# =============================================================================

def compute_pixel_auroc(score_maps, masks, sigma=4.0):
    """Compute pixel-level AUROC from score maps and ground truth masks.
    
    Args:
        score_maps: list of numpy arrays (B, H, W)
        masks: list of numpy arrays (B, H, W) binary
        sigma: Gaussian smoothing sigma
    Returns:
        auroc: float
    """
    all_scores = []
    all_labels = []
    
    for smap, mask in zip(score_maps, masks):
        # Gaussian smooth
        smoothed = gaussian_smooth(smap, sigma=sigma)
        all_scores.append(smoothed.reshape(-1))
        all_labels.append(mask.reshape(-1))
    
    scores_flat = np.concatenate(all_scores)
    labels_flat = np.concatenate(all_labels).astype(np.uint8)
    
    if labels_flat.sum() == 0 or labels_flat.sum() == len(labels_flat):
        return 0.5
    
    return roc_auc_score(labels_flat, scores_flat)


def compute_image_auroc(image_scores, image_labels):
    """Compute image-level AUROC."""
    labels = np.array(image_labels)
    scores = np.array(image_scores)
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


# =============================================================================
# Method-specific training and evaluation
# =============================================================================

def train_padim(train_loader, test_loader, args, run_dir):
    """Train PaDiM (no training needed, just fit statistics)."""
    from .methods import PaDiM
    
    print("\n" + "="*60)
    print("Training D_PaDiM")
    print("="*60)
    
    method = PaDiM(img_size=args.img_size, out_size=args.out_size, 
                   d_keep=args.padim_d_keep, seed=args.seed)
    
    # Fit (compute statistics)
    t0 = time.time()
    method.fit(train_loader)
    fit_time = time.time() - t0
    print(f"Fit time: {fit_time:.1f}s")
    
    # Evaluate
    return evaluate_and_save(method, "D_PaDiM", test_loader, args, run_dir)


def train_patchcore(train_loader, test_loader, args, run_dir):
    """Train PatchCore (no training needed, just build memory bank)."""
    from .methods import PatchCore
    
    print("\n" + "="*60)
    print("Training E_PatchCore")
    print("="*60)
    
    method = PatchCore(img_size=args.img_size, out_size=args.out_size,
                       coreset_ratio=args.patchcore_coreset, k=args.patchcore_k,
                       seed=args.seed)
    
    t0 = time.time()
    method.fit(train_loader)
    fit_time = time.time() - t0
    print(f"Fit time: {fit_time:.1f}s")
    
    return evaluate_and_save(method, "E_PatchCore", test_loader, args, run_dir)


def train_fastflow(train_loader, test_loader, args, run_dir):
    """Train FastFlow with full training loop."""
    from .methods import FastFlow, _to_device
    
    print("\n" + "="*60)
    print("Training F_FastFlow")
    print("="*60)
    
    method = FastFlow(img_size=args.img_size, out_size=args.out_size,
                      n_blocks=args.nf_blocks, hidden=args.nf_hidden,
                      lr=args.lr, epochs=1)  # We'll handle epochs manually
    
    best_dir = os.path.join(run_dir, "F_FastFlow", "best_models")
    os.makedirs(best_dir, exist_ok=True)
    
    img_obs = ScoreObserver("Image AUROC")
    pix_obs = ScoreObserver("Pixel AUROC")
    
    opt = torch.optim.Adam(method.flows.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'image_auroc': [], 'pixel_auroc': []}
    
    for epoch in range(args.epochs):
        # Train
        method.flows.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[FastFlow] ep {epoch+1}/{args.epochs}"):
            x = _to_device(batch)
            with torch.no_grad():
                _, feats = method.backbone(x)
            opt.zero_grad()
            with autocast('cuda'):
                nll_maps = method._nll_per_level(feats)
                loss = sum(m.mean() for m in nll_maps)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        history['train_loss'].append(avg_loss)
        
        # Evaluate every eval_interval epochs
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            method.flows.eval()
            img_auroc, pix_auroc, _ = full_evaluate(method, test_loader, args)
            
            history['image_auroc'].append(img_auroc)
            history['pixel_auroc'].append(pix_auroc)
            
            img_improved = img_obs.update(img_auroc, epoch + 1)
            pix_improved = pix_obs.update(pix_auroc, epoch + 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} img={img_auroc:.4f} pix={pix_auroc:.4f}")
            
            # Save best models
            if img_improved:
                save_method_checkpoint(method.flows, best_dir, "best_image_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc)
                print(f"    -> New best IMAGE AUROC: {img_auroc:.4f}")
            
            if pix_improved:
                save_method_checkpoint(method.flows, best_dir, "best_pixel_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc)
                print(f"    -> New best PIXEL AUROC: {pix_auroc:.4f}")
        
        # Checkpoint every checkpoint_interval
        if (epoch + 1) % args.checkpoint_interval == 0:
            ckpt_path = os.path.join(run_dir, "F_FastFlow", f"epoch_{epoch+1}.pt")
            save_method_checkpoint(method.flows, run_dir, f"F_FastFlow/epoch_{epoch+1}.pt",
                                   epoch + 1, img_obs.last, pix_obs.last)
    
    save_history(history, run_dir, "F_FastFlow")
    return "F_FastFlow", img_obs.max_score, pix_obs.max_score, img_obs.max_epoch, pix_obs.max_epoch


def train_cflow(train_loader, test_loader, args, run_dir):
    """Train CFLOW-AD with full training loop."""
    from .methods import CFLOW, _to_device, _pos_encoding
    
    print("\n" + "="*60)
    print("Training G_CFLOW")
    print("="*60)
    
    method = CFLOW(img_size=args.img_size, out_size=args.out_size,
                   cond_dim=args.cflow_cond_dim, n_blocks=args.nf_blocks,
                   hidden=args.nf_hidden, lr=args.lr, epochs=1)
    
    best_dir = os.path.join(run_dir, "G_CFLOW", "best_models")
    os.makedirs(best_dir, exist_ok=True)
    
    img_obs = ScoreObserver("Image AUROC")
    pix_obs = ScoreObserver("Pixel AUROC")
    
    opt = torch.optim.Adam(method.flows.parameters(), lr=args.lr)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'image_auroc': [], 'pixel_auroc': []}
    
    for epoch in range(args.epochs):
        method.flows.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[CFLOW] ep {epoch+1}/{args.epochs}"):
            x = _to_device(batch)
            with torch.no_grad():
                _, feats = method.backbone(x)
            opt.zero_grad()
            with autocast('cuda'):
                nll_maps = method._nll_per_level(feats)
                loss = sum(m.mean() for m in nll_maps)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        history['train_loss'].append(avg_loss)
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            method.flows.eval()
            img_auroc, pix_auroc, _ = full_evaluate(method, test_loader, args)
            
            history['image_auroc'].append(img_auroc)
            history['pixel_auroc'].append(pix_auroc)
            
            img_improved = img_obs.update(img_auroc, epoch + 1)
            pix_improved = pix_obs.update(pix_auroc, epoch + 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} img={img_auroc:.4f} pix={pix_auroc:.4f}")
            
            if img_improved:
                save_method_checkpoint(method.flows, best_dir, "best_image_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc)
                print(f"    -> New best IMAGE AUROC: {img_auroc:.4f}")
            
            if pix_improved:
                save_method_checkpoint(method.flows, best_dir, "best_pixel_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc)
                print(f"    -> New best PIXEL AUROC: {pix_auroc:.4f}")
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_method_checkpoint(method.flows, run_dir, f"G_CFLOW/epoch_{epoch+1}.pt",
                                   epoch + 1, img_obs.last, pix_obs.last)
    
    save_history(history, run_dir, "G_CFLOW")
    return "G_CFLOW", img_obs.max_score, pix_obs.max_score, img_obs.max_epoch, pix_obs.max_epoch


def train_differnet(train_loader, test_loader, args, run_dir):
    """Train standard DifferNet with full training loop."""
    from .methods import DifferNetMethod, _to_device
    
    print("\n" + "="*60)
    print("Training H_DifferNet")
    print("="*60)
    
    method = DifferNetMethod(img_size=args.img_size, n_scales=args.n_scales,
                             n_coupling_blocks=args.nf_blocks,
                             fc_internal=args.nf_hidden, lr=args.lr, epochs=1)
    
    best_dir = os.path.join(run_dir, "H_DifferNet", "best_models")
    os.makedirs(best_dir, exist_ok=True)
    
    img_obs = ScoreObserver("Image AUROC")
    pix_obs = ScoreObserver("Pixel AUROC")
    
    opt = torch.optim.Adam(method.nf.parameters(), lr=args.lr,
                           betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'image_auroc': [], 'pixel_auroc': []}
    
    for epoch in range(args.epochs):
        method.nf.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"[DifferNet] ep {epoch+1}/{args.epochs}"):
            x = _to_device(batch)
            opt.zero_grad()
            with autocast('cuda'):
                nll = method._nll(x)
                loss = nll.mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        history['train_loss'].append(avg_loss)
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            method.nf.eval()
            img_auroc, pix_auroc, _ = full_evaluate(method, test_loader, args)
            
            history['image_auroc'].append(img_auroc)
            history['pixel_auroc'].append(pix_auroc)
            
            img_improved = img_obs.update(img_auroc, epoch + 1)
            pix_improved = pix_obs.update(pix_auroc, epoch + 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} img={img_auroc:.4f} pix={pix_auroc:.4f}")
            
            if img_improved:
                save_method_checkpoint(method.nf, best_dir, "best_image_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc)
                print(f"    -> New best IMAGE AUROC: {img_auroc:.4f}")
            
            if pix_improved:
                save_method_checkpoint(method.nf, best_dir, "best_pixel_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc)
                print(f"    -> New best PIXEL AUROC: {pix_auroc:.4f}")
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            save_method_checkpoint(method.nf, run_dir, f"H_DifferNet/epoch_{epoch+1}.pt",
                                   epoch + 1, img_obs.last, pix_obs.last)
    
    save_history(history, run_dir, "H_DifferNet")
    return "H_DifferNet", img_obs.max_score, pix_obs.max_score, img_obs.max_epoch, pix_obs.max_epoch


def train_sediffernet(train_loader, test_loader, args, run_dir):
    """Train SEDifferNet with full training loop (SE blocks trained jointly)."""
    from .methods import SEDifferNetMethod, _to_device
    
    print("\n" + "="*60)
    print("Training I_SEDifferNet")
    print("="*60)
    
    method = SEDifferNetMethod(img_size=args.img_size, n_scales=args.n_scales,
                               n_coupling_blocks=args.nf_blocks,
                               fc_internal=args.nf_hidden, lr=args.lr, epochs=1)
    
    best_dir = os.path.join(run_dir, "I_SEDifferNet", "best_models")
    os.makedirs(best_dir, exist_ok=True)
    
    img_obs = ScoreObserver("Image AUROC")
    pix_obs = ScoreObserver("Pixel AUROC")
    
    # Separate optimizer groups: NF at full LR, SE at half LR
    se_params = (list(method.se1.parameters()) + list(method.se2.parameters()) +
                 list(method.se3.parameters()) + list(method.se4.parameters()))
    opt = torch.optim.Adam([
        {'params': method.nf.parameters(), 'lr': args.lr},
        {'params': se_params, 'lr': args.lr * 0.5},
    ], betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    
    history = {'train_loss': [], 'image_auroc': [], 'pixel_auroc': []}
    
    for epoch in range(args.epochs):
        method.nf.train()
        method.se1.train(); method.se2.train()
        method.se3.train(); method.se4.train()
        
        losses = []
        for batch in tqdm(train_loader, desc=f"[SEDifferNet] ep {epoch+1}/{args.epochs}"):
            x = _to_device(batch)
            opt.zero_grad()
            with autocast('cuda'):
                nll = method._nll(x)
                loss = nll.mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        history['train_loss'].append(avg_loss)
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            method.nf.eval()
            method.se1.eval(); method.se2.eval()
            method.se3.eval(); method.se4.eval()
            
            img_auroc, pix_auroc, _ = full_evaluate(method, test_loader, args)
            
            history['image_auroc'].append(img_auroc)
            history['pixel_auroc'].append(pix_auroc)
            
            img_improved = img_obs.update(img_auroc, epoch + 1)
            pix_improved = pix_obs.update(pix_auroc, epoch + 1)
            
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f} img={img_auroc:.4f} pix={pix_auroc:.4f}")
            
            if img_improved:
                state = {
                    'nf': method.nf.state_dict(),
                    'se1': method.se1.state_dict(),
                    'se2': method.se2.state_dict(),
                    'se3': method.se3.state_dict(),
                    'se4': method.se4.state_dict(),
                }
                save_method_checkpoint(state, best_dir, "best_image_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc, is_state_dict=True)
                print(f"    -> New best IMAGE AUROC: {img_auroc:.4f}")
            
            if pix_improved:
                state = {
                    'nf': method.nf.state_dict(),
                    'se1': method.se1.state_dict(),
                    'se2': method.se2.state_dict(),
                    'se3': method.se3.state_dict(),
                    'se4': method.se4.state_dict(),
                }
                save_method_checkpoint(state, best_dir, "best_pixel_auroc.pt",
                                       epoch + 1, img_auroc, pix_auroc, is_state_dict=True)
                print(f"    -> New best PIXEL AUROC: {pix_auroc:.4f}")
        
        if (epoch + 1) % args.checkpoint_interval == 0:
            state = {
                'nf': method.nf.state_dict(),
                'se1': method.se1.state_dict(),
                'se2': method.se2.state_dict(),
                'se3': method.se3.state_dict(),
                'se4': method.se4.state_dict(),
            }
            save_method_checkpoint(state, run_dir, f"I_SEDifferNet/epoch_{epoch+1}.pt",
                                   epoch + 1, img_obs.last, pix_obs.last, is_state_dict=True)
    
    save_history(history, run_dir, "I_SEDifferNet")
    return "I_SEDifferNet", img_obs.max_score, pix_obs.max_score, img_obs.max_epoch, pix_obs.max_epoch


# =============================================================================
# Helper functions
# =============================================================================

def full_evaluate(method, test_loader, args):
    """Evaluate method on both image-level and pixel-level AUROC."""
    score_maps = []
    masks = []
    image_scores = []
    image_labels = []
    
    with torch.no_grad():
        for inputs, label, mask in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = inputs.to(DEVICE)
            
            # Pixel-level score
            smap = method.score(inputs)  # (B, H, W)
            score_maps.append(smap)
            masks.append(mask.squeeze(1).numpy())
            
            # Image-level score (max of smoothed pixel scores)
            smoothed = gaussian_smooth(smap, sigma=args.sigma)
            img_score = smoothed.reshape(smoothed.shape[0], -1).max(axis=1)
            image_scores.extend(img_score.tolist())
            image_labels.extend([1 if l > 0 else 0 for l in label.numpy()])
    
    pix_auroc = compute_pixel_auroc(score_maps, masks, sigma=args.sigma)
    img_auroc = compute_image_auroc(image_scores, image_labels)
    
    return img_auroc, pix_auroc, (score_maps, masks)


def evaluate_and_save(method, name, test_loader, args, run_dir):
    """Evaluate non-trainable method and save results."""
    best_dir = os.path.join(run_dir, name, "best_models")
    os.makedirs(best_dir, exist_ok=True)
    
    img_auroc, pix_auroc, _ = full_evaluate(method, test_loader, args)
    
    print(f"  Image AUROC: {img_auroc:.4f}")
    print(f"  Pixel AUROC: {pix_auroc:.4f}")
    
    # Save method state (statistics/memory bank)
    state = {}
    if hasattr(method, 'mean') and method.mean is not None:
        state['mean'] = method.mean.cpu()
        state['cov_inv'] = method.cov_inv.cpu()
    if hasattr(method, 'bank') and method.bank is not None:
        state['bank'] = method.bank.cpu()
    
    if state:
        save_method_checkpoint(state, best_dir, "model.pt", 0, img_auroc, pix_auroc, is_state_dict=True)
    
    return name, img_auroc, pix_auroc, 0, 0


def save_method_checkpoint(model_or_state, base_dir, filename, epoch, img_auroc, pix_auroc, is_state_dict=False):
    """Save checkpoint with metrics."""
    path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if is_state_dict:
        state_dict = model_or_state
    else:
        state_dict = model_or_state.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict,
        'image_auroc': img_auroc,
        'pixel_auroc': pix_auroc,
    }
    safe_torch_save(checkpoint, path)


def save_history(history, run_dir, method_name):
    """Save training history to file."""
    path = os.path.join(run_dir, method_name, "history.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        f.write("epoch\ttrain_loss\timage_auroc\tpixel_auroc\n")
        n_loss = len(history.get('train_loss', []))
        n_img = len(history.get('image_auroc', []))
        n_pix = len(history.get('pixel_auroc', []))
        
        for i in range(n_loss):
            loss = history['train_loss'][i] if i < n_loss else 0
            # Image/pixel auroc are logged less frequently
            img_idx = i // max(1, n_loss // max(n_img, 1)) if n_img > 0 else 0
            img = history['image_auroc'][min(img_idx, n_img-1)] if n_img > 0 else 0
            pix = history['pixel_auroc'][min(img_idx, n_pix-1)] if n_pix > 0 else 0
            f.write(f"{i+1}\t{loss:.6f}\t{img:.6f}\t{pix:.6f}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full training pipeline for pixel-level AD methods")
    
    # Data
    parser.add_argument('--dataset', type=str,
                        default=r"C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg")
    parser.add_argument('--class_name', type=str, default='lightning-rod-suspension')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--out_size', type=int, default=56)
    parser.add_argument('--n_train', type=int, default=-1, help='-1 for all training images')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=4.0)
    
    # Model-specific
    parser.add_argument('--n_scales', type=int, default=3)
    parser.add_argument('--nf_blocks', type=int, default=8)
    parser.add_argument('--nf_hidden', type=int, default=512)
    parser.add_argument('--padim_d_keep', type=int, default=100)
    parser.add_argument('--patchcore_coreset', type=float, default=0.1)
    parser.add_argument('--patchcore_k', type=int, default=3)
    parser.add_argument('--cflow_cond_dim', type=int, default=64)
    
    # Methods
    parser.add_argument('--methods', type=str, 
                        default='D_PaDiM,E_PatchCore,F_FastFlow,G_CFLOW,H_DifferNet,I_SEDifferNet')
    
    # Output
    parser.add_argument('--out_dir', type=str, default='./pixel_pipeline_runs')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create run directory
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.out_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Device: {DEVICE}")
    print(f"Config: img_size={args.img_size}, epochs={args.epochs}, lr={args.lr}")
    
    # Save config
    with open(os.path.join(run_dir, "config.txt"), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")
    
    # Load data
    train_loader, test_loader = make_loaders(
        args.dataset, args.class_name, img_size=args.img_size,
        n_train=args.n_train if args.n_train > 0 else 9999,  # Large number = all
        batch_size=args.batch_size, seed=args.seed,
    )
    
    # Parse methods
    method_names = [m.strip() for m in args.methods.split(',') if m.strip()]
    
    # Training dispatch
    trainers = {
        'D_PaDiM': train_padim,
        'E_PatchCore': train_patchcore,
        'F_FastFlow': train_fastflow,
        'G_CFLOW': train_cflow,
        'H_DifferNet': train_differnet,
        'I_SEDifferNet': train_sediffernet,
    }
    
    results = []
    
    for name in method_names:
        if name not in trainers:
            print(f"Skipping unknown method: {name}")
            continue
        
        try:
            result = trainers[name](train_loader, test_loader, args, run_dir)
            results.append(result)
        except Exception as e:
            print(f"Error training {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, 0.0, 0.0, 0, 0))
        
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Method':15s} {'Best Img AUROC':>15s} {'Best Pix AUROC':>15s} {'Img Epoch':>10s} {'Pix Epoch':>10s}")
    print("-"*80)
    
    best_img = ('', 0.0)
    best_pix = ('', 0.0)
    
    for name, img_auroc, pix_auroc, img_ep, pix_ep in results:
        print(f"{name:15s} {img_auroc:>15.4f} {pix_auroc:>15.4f} {img_ep:>10d} {pix_ep:>10d}")
        if img_auroc > best_img[1]:
            best_img = (name, img_auroc)
        if pix_auroc > best_pix[1]:
            best_pix = (name, pix_auroc)
    
    print("-"*80)
    print(f"BEST IMAGE-LEVEL:  {best_img[0]} with AUROC = {best_img[1]:.4f}")
    print(f"BEST PIXEL-LEVEL:  {best_pix[0]} with AUROC = {best_pix[1]:.4f}")
    
    # Save results CSV
    csv_path = os.path.join(run_dir, "results.csv")
    with open(csv_path, 'w') as f:
        f.write("method,best_image_auroc,best_pixel_auroc,best_image_epoch,best_pixel_epoch\n")
        for name, img_auroc, pix_auroc, img_ep, pix_ep in results:
            f.write(f"{name},{img_auroc:.6f},{pix_auroc:.6f},{img_ep},{pix_ep}\n")
    
    print(f"\nResults saved to {csv_path}")
    print(f"Run directory: {run_dir}")


if __name__ == '__main__':
    main()
