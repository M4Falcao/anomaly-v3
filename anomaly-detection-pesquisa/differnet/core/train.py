"""Image-level training loop for DifferNet / SEDifferNet.

Trains the normalizing-flow head (and, for SEDifferNet, the SE attention
blocks) on defect-free images, evaluating image-level AUROC after every
sub-epoch. Progress, parameters and artifacts are logged to MLflow when
``config.use_mlflow`` is enabled.

This module is a library: use ``main.py`` as the entry point.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mlflow
import threading
import os
import time
import socket

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
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
# SE Attention anomaly map extraction (post-hoc, for pixel-level evaluation)
# ─────────────────────────────────────────────────────────────────────────────

def get_se_anomaly_maps(model, inputs):
    """Compute anomaly heatmap using SE Attention excitation weights.

    Strategy:
      1. Register hooks on each SE block to capture excitation weights and input features.
      2. Weight each channel by its SE excitation weight.
      3. Sum across channels -> spatial score map per SE block per scale.
      4. Upsample and aggregate across all blocks and scales.

    Returns: numpy array of shape (B, H_img, W_img).
    """
    model.eval()

    se_excitations = []
    se_inputs = []
    hooks = []

    def make_input_hook(storage):
        def hook_fn(module, inp, out):
            storage.append(inp[0].detach())
        return hook_fn

    def make_excitation_hook(storage):
        def hook_fn(module, inp, out):
            storage.append(out.detach().view(out.size(0), out.size(1), 1, 1))
        return hook_fn

    se_blocks = [model.simsa1, model.simsa2, model.simsa3, model.simsa4]
    for se_block in se_blocks:
        hooks.append(se_block.register_forward_hook(make_input_hook(se_inputs)))
        hooks.append(se_block.fc.register_forward_hook(make_excitation_hook(se_excitations)))

    try:
        with torch.no_grad():
            model(inputs)
    finally:
        for h in hooks:
            h.remove()

    n_se_blocks = len(se_blocks)
    n_scales = c.n_scales
    target_size = c.img_size

    score_map = None
    for scale_idx in range(n_scales):
        for block_idx in range(n_se_blocks):
            idx = scale_idx * n_se_blocks + block_idx
            if idx >= len(se_inputs) or idx >= len(se_excitations):
                continue

            feat = se_inputs[idx]
            excit = se_excitations[idx]
            weighted = feat * excit
            spatial_score = weighted.sum(dim=1)

            spatial_up = F.interpolate(
                spatial_score.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False
            ).squeeze(1)

            if score_map is None:
                score_map = spatial_up
            else:
                score_map = score_map + spatial_up

    if score_map is None:
        return None

    return t2np(score_map)


# ─────────────────────────────────────────────────────────────────────────────
# Score observer
# ─────────────────────────────────────────────────────────────────────────────

class Score_Observer:
    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if self.max_score is None or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(
            self.name, self.last, self.max_score, self.max_epoch))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

print(f'TRAINING ON : {c.device}, cause cuda is {torch.cuda.is_available()}')


def train(train_loader, test_loader):
    # Create execution specific checkpoint directory
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_checkpoint_dir = os.path.join(c.checkpoint_path, f"run_{run_timestamp}")
    best_models_dir = os.path.join(run_checkpoint_dir, "best_models")
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(best_models_dir, exist_ok=True)

    # MLflow Setup
    if c.use_mlflow:
        if "127.0.0.1" in c.mlflow_tracking_uri or "localhost" in c.mlflow_tracking_uri:
            if not wait_for_server("127.0.0.1", 5000, timeout=2):
                print("Starting MLflow UI in a background thread...")
                thread = threading.Thread(target=run_mlflow_ui)
                thread.daemon = True
                thread.start()
                if wait_for_server("127.0.0.1", 5000):
                    print("MLflow server started!")
                else:
                    print("Timed out waiting for MLflow server!")
            else:
                print("MLflow server is already running!")

        if c.use_pyngrok:
            start_ngrok(5000)

        mlflow.set_tracking_uri(c.mlflow_tracking_uri)
        mlflow.set_experiment(c.mlflow_experiment_name)
        mlflow.start_run(run_name=c.mlflow_run_name)

        params = {k: v for k, v in vars(c).items()
                  if not k.startswith('__') and not callable(v) and not isinstance(v, type)}
        for k, v in params.items():
            try:
                mlflow.log_param(k, v)
            except:
                pass

    # Model and optimizer
    model = SEDifferNet()

    # Train SE attention modules jointly with NF head (lower LR for attention)
    se_params = list(model.simsa1.parameters()) + list(model.simsa2.parameters()) + \
                list(model.simsa3.parameters()) + list(model.simsa4.parameters())
    optimizer = torch.optim.Adam(
        [{'params': model.nf.parameters(), 'lr': c.lr_init},
         {'params': se_params, 'lr': c.lr_init * 0.5}],
        betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5
    )
    model.to(c.device)

    scaler = GradScaler()

    if c.use_mlflow:
        config_path = project_path("config.py")
        mlflow.log_artifact(config_path)
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parameters", total_params)

    score_obs = Score_Observer('Image AUROC')
    best_img_auroc = -1.0

    train_losses = []
    test_losses = []
    image_aurocs = []

    start_epoch = 0
    if c.resume_training:
        print(f"Loading weights from {c.resume_file}...")
        try:
            model, checkpoint = load_weights(model, c.resume_file)
            if checkpoint:
                start_epoch = checkpoint['epoch']
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'train_losses' in checkpoint:
                    train_losses = checkpoint['train_losses']
                if 'test_losses' in checkpoint:
                    test_losses = checkpoint['test_losses']
                if 'image_aurocs' in checkpoint:
                    image_aurocs = checkpoint['image_aurocs']
                print(f"Resuming from epoch {start_epoch}")
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting training from scratch.")

    # Pre-epochs (warm-up)
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

            avg = np.mean(pre_train_loss)
            print(f"Pre-Epoch [{pre_epoch + 1}/{c.pre_epochs}], Train Loss: {avg:.4f}")
            if c.use_mlflow:
                mlflow.log_metric("pre_train_loss", avg, step=pre_epoch)

    # ─────────────────────────────────────────────────────────────────────────
    # Main training loop
    # ─────────────────────────────────────────────────────────────────────────
    for meta_epoch in range(start_epoch, c.meta_epochs):
        # --- Train ---
        model.train()
        train_loss = []

        for sub_epoch in range(c.sub_epochs):
            for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {meta_epoch+1}/{c.meta_epochs}")):
                optimizer.zero_grad()
                inputs, labels = preprocess_batch(data)

                with autocast('cuda'):
                    z = model(inputs)
                    loss = get_loss(z, model.nf.jacobian(run_forward=False))

                train_loss.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        avg_train_loss = np.mean(train_loss)
        print(f'Epoch [{meta_epoch + 1}/{c.meta_epochs}], Train Loss: {avg_train_loss:.4f}')
        if c.use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=meta_epoch)

        # --- Eval (image-level only) ---
        model.eval()
        test_loss = []
        test_labels = []
        test_z = []

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, desc=f"Eval {meta_epoch+1}")):
                # Handle both (images, label) and (images, label, mask)
                if len(data) == 3:
                    inputs_raw, labels, masks = data
                else:
                    inputs_raw, labels = data

                inputs = inputs_raw.to(c.device).view(-1, *inputs_raw.shape[-3:])
                labels = labels.to(c.device)

                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                test_loss.append(loss.item())
                test_labels.append(t2np(labels))
                test_z.append(z)

        avg_test_loss = np.mean(test_loss)

        # Image-level AUROC
        is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))

        image_auroc = roc_auc_score(is_anomaly, anomaly_score) if len(np.unique(is_anomaly)) > 1 else 0.5

        score_obs.update(image_auroc, meta_epoch,
                         print_score=c.verbose or meta_epoch == c.meta_epochs - 1)

        if c.use_mlflow:
            mlflow.log_metric("test_loss", avg_test_loss, step=meta_epoch)
            mlflow.log_metric("image_level_auroc", image_auroc, step=meta_epoch)

        # Append metrics to history
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        image_aurocs.append(image_auroc)

        # --- Save checkpoint ---
        if (meta_epoch + 1) % c.checkpoint_interval == 0 or image_auroc > best_img_auroc:
            weights_filename = os.path.join(
                run_checkpoint_dir, f"{c.class_name}_{c.modelname}_epoch_{meta_epoch + 1}.pt")
            checkpoint_data = {
                'epoch': meta_epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_losses': test_losses,
                'image_aurocs': image_aurocs,
            }
            saved = safe_torch_save(checkpoint_data, weights_filename)
            if saved:
                print(f"  Checkpoint saved: {weights_filename}")
            else:
                lite_fn = weights_filename.replace('.pt', '_lite.pt')
                lite_data = {
                    'epoch': meta_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'image_aurocs': image_aurocs,
                }
                if safe_torch_save(lite_data, lite_fn, retries=1):
                    print(f"  Full checkpoint failed; saved LITE: {lite_fn}")
                else:
                    print(f"  Checkpoint save failed; continuing training.")

        # Save best model
        if image_auroc > best_img_auroc:
            best_img_auroc = image_auroc
            best_path = os.path.join(best_models_dir, "best_img_auroc.pt")
            best_data = {
                'epoch': meta_epoch + 1,
                'model_state_dict': model.state_dict(),
                'image_auroc': image_auroc,
            }
            if safe_torch_save(best_data, best_path):
                print(f"  ★ New best IMAGE AUROC: {image_auroc:.4f} (epoch {meta_epoch + 1})")

            if c.use_mlflow:
                mlflow.pytorch.log_model(model, "model_best_img_auroc")

    # ─────────────────────────────────────────────────────────────────────────
    # End of training
    # ─────────────────────────────────────────────────────────────────────────
    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)
        save_weights(model, c.modelname)

    if c.use_mlflow:
        mlflow.pytorch.log_model(model, "final_model")
        mlflow.end_run()

    # Save metrics history
    metrics_file = os.path.join(run_checkpoint_dir, "metrics_history.txt")
    with open(metrics_file, "w") as f:
        f.write("Epoch\tTrain_Loss\tTest_Loss\tImage_AUROC\n")
        for i in range(len(train_losses)):
            tr = train_losses[i] if i < len(train_losses) else 0.0
            te = test_losses[i] if i < len(test_losses) else 0.0
            ia = image_aurocs[i] if i < len(image_aurocs) else 0.0
            f.write(f"{start_epoch + i + 1}\t{tr:.6f}\t{te:.6f}\t{ia:.6f}\n")
    print(f"Metrics history saved to {metrics_file}")

    return model
