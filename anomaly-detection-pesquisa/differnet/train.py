import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mlflow
import threading
import os

import config as c
from localization import export_gradient_maps
from model import *
from utils import *
import skimage

from sklearn.metrics import roc_auc_score

def start_ngrok(port):
    from pyngrok import ngrok
    # Set the authtoken if provided in environment or config (optional)
    ngrok.set_auth_token("365IEbpEEZzJJLPQeWyXrBkgrVS_ukKPM8QLsvZen1xpZsD3") 
    public_url = ngrok.connect(port).public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

def run_mlflow_ui():
    os.system(f"mlflow ui --backend-store-uri {c.mlflow_backend_store_uri} --port 5000 --host 0.0.0.0 &")

def calculate_image_level_auroc(predictions, ground_truth_labels):
    # Calculate image-level AUROC
    image_level_auroc = roc_auc_score(ground_truth_labels, predictions)
    return image_level_auroc

def calculate_pixel_level_auroc(predictions, ground_truth_masks):
    # Resize predictions to match the ground truth mask dimensions
    predictions_resized = skimage.transform.resize(predictions, ground_truth_masks.shape, mode='constant')
    
    # Binarize ground truth masks
    ground_truth_masks_binary = (ground_truth_masks > 0).astype(int)

    # Flatten predictions and ground truth masks
    predictions_flat = predictions_resized.reshape(-1)
    ground_truth_flat = ground_truth_masks_binary.reshape(-1)
    
    # Calculate pixel-level AUROC
    pixel_auroc = roc_auc_score(ground_truth_flat, predictions_flat)
    return pixel_auroc




class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.last = None

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d}'.format(self.name, self.last, self.max_score,
                                                                               self.max_epoch))




import time

print(f'TRAINING ON : {c.device}, cause cuda is {torch.cuda.is_available()}')

def train(train_loader, test_loader, ground_truth_loader):
    # MLflow Setup
    if c.use_mlflow:
        mlflow.set_tracking_uri(c.mlflow_tracking_uri)
        mlflow.set_experiment(c.mlflow_experiment_name)
        
        if c.use_pyngrok:
            # Start MLflow UI in a background thread
            thread = threading.Thread(target=run_mlflow_ui)
            thread.daemon = True
            thread.start()
            
            # Start ngrok
            start_ngrok(5000)

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
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)

    score_obs_image = Score_Observer('AUROC for image level')
    score_obs_pixel = Score_Observer('AUROC for pixel level')

    for epoch in range(c.meta_epochs):
        # Training loop
        model.train()
        train_loss = []
        image_level_scores_train = []

        for i, data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            inputs, labels = preprocess_batch(data)
            z = model(inputs)
            loss = get_loss(z, model.nf.jacobian(run_forward=False))
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            # Compute image-level anomaly score (original score) during training
            image_level_score_train = torch.mean(z ** 2).item()
            image_level_scores_train.append(image_level_score_train)

        # Compute average training loss
        avg_train_loss = np.mean(train_loss)

        # Print or log metrics during training
        # Print or log metrics during training
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, c.meta_epochs, avg_train_loss))
        if c.use_mlflow:
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

        # Evaluation loop
        model.eval()
        test_loss = []
        pixel_level_auroc_scores_test = []
        image_level_scores_test = []
        test_labels = []
        test_z = []

        # Time tracking variables
        start_time = time.time()
        total_images = 0

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader)):
                inputs, labels = preprocess_batch(data)
                z = model(inputs)
                loss = get_loss(z, model.nf.jacobian(run_forward=False))
                test_loss.append(loss.item())
                test_labels.append(t2np(labels)) 
                test_z.append(z)

                # Count images processed
                total_images += inputs.size(0)

                # Load ground truth masks
                ground_truth_data = next(iter(ground_truth_loader))
                ground_truth_masks = ground_truth_data[0].to(c.device)

                # Compute pixel-level AUROC during evaluation
                pixel_auroc_test = calculate_pixel_level_auroc(t2np(z), t2np(ground_truth_masks))
                pixel_level_auroc_scores_test.append(pixel_auroc_test)

        # Calculate time metrics
        elapsed_time = time.time() - start_time
        images_per_second = total_images / elapsed_time
        latency_per_image_ms = (elapsed_time / total_images) * 1000

        # Print metrics
        print(f"Test Set - Images per Second: {images_per_second:.2f}, Latency per Image: {latency_per_image_ms:.2f} ms")

        # Compute average test loss
        avg_test_loss = np.mean(test_loss)

        # Aggregate pixel-level AUROC scores during evaluation
        mean_pixel_auroc_test = np.mean(np.array(pixel_level_auroc_scores_test))
        
        if c.use_mlflow:
            mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
            mlflow.log_metric("pixel_level_auroc", mean_pixel_auroc_test, step=epoch)
    
        # Update score observer
        is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        score_obs_image.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
                        print_score=c.verbose or epoch == c.meta_epochs - 1)
        
        score_obs_pixel.update(mean_pixel_auroc_test, epoch,
                        print_score=c.verbose or epoch == c.meta_epochs - 1)
        
        if c.use_mlflow:
            image_auroc = roc_auc_score(is_anomaly, anomaly_score)
            mlflow.log_metric("image_level_auroc", image_auroc, step=epoch)
            
            # Checkpoint best model based on Image Level AUROC
            if image_auroc >= score_obs_image.max_score:
                 mlflow.pytorch.log_model(model, "model_best_auroc")
                 print(f"New best model saved to MLflow with AUROC: {image_auroc:.4f}")

    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)
        save_weights(model, c.modelname)
        
    if c.use_mlflow:
        mlflow.pytorch.log_model(model, "final_model")
        mlflow.end_run()
        
    return model

