import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mlflow
import threading
import os
import time
import socket

import config as c
from localization import export_gradient_maps
from model import *
from utils import *
from export_mlflow import export_mlflow_data
import skimage

from sklearn.metrics import roc_auc_score

def start_ngrok(port):
    from pyngrok import ngrok
    # Set the authtoken if provided in environment or config (optional)
    ngrok.set_auth_token(c.ngrok_auth_token) 
    public_url = ngrok.connect(port, host_header="rewrite").public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")

def run_mlflow_ui():
    os.system(f"mlflow ui --backend-store-uri {c.mlflow_backend_store_uri} --port 5000 --host 0.0.0.0")

def wait_for_server(host, port, timeout=30):
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)

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
        if c.use_pyngrok:
            # Start MLflow UI in a background thread
            thread = threading.Thread(target=run_mlflow_ui)
            thread.daemon = True
            thread.start()
            
            # Wait for MLflow server to start
            print("Waiting for MLflow server to start...")
            if wait_for_server("127.0.0.1", 5000):
                print("MLflow server started!")
            else:
                print("Timed out waiting for MLflow server!")

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
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)

    if c.use_mlflow:
        # Log config file
        mlflow.log_artifact("config.py")
        
        # Log model summary
        with open("model_summary.txt", "w") as f:
            f.write(str(model))
        mlflow.log_artifact("model_summary.txt")
        if os.path.exists("model_summary.txt"):
            os.remove("model_summary.txt")
            
        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("total_parameters", total_params)

    if c.resume_training:
        print(f"Loading weights from {c.resume_file}...")
        try:
            load_weights(model, c.resume_file)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Starting training from scratch.")

    score_obs_image = Score_Observer('AUROC for image level')
    score_obs_pixel = Score_Observer('AUROC for pixel level')

    # Initialize history lists
    train_losses = []
    test_losses = []
    image_aurocs = []
    pixel_aurocs = []

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
            
            # Append metrics to history
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            image_aurocs.append(image_auroc)
            pixel_aurocs.append(mean_pixel_auroc_test)

            # Save model every c.checkpoint_interval epochs
            if (epoch + 1) % c.checkpoint_interval == 0 or image_auroc >= score_obs_image.max_score:

                # Checkpoint best model based on Image Level AUROC
                if image_auroc >= score_obs_image.max_score:
                    # mlflow.pytorch.log_model(model, "model_best_auroc")
                    print(f"New best model saved to MLflow with AUROC: {image_auroc:.4f} in epoch {epoch + 1}")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Saving model checkpoint at epoch {epoch + 1}...")
                
                # Ensure checkpoint directory exists
                if not os.path.exists(c.checkpoint_path):
                    os.makedirs(c.checkpoint_path)
                
                # Save full model to MLflow
                # mlflow.pytorch.log_model(model, f"model_epoch_{epoch + 1}")
                
                # Save weights explicitly to local path
                weights_filename = os.path.join(c.checkpoint_path, f"{c.class_name}_{c.modelname}_epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'image_aurocs': image_aurocs,
                    'pixel_aurocs': pixel_aurocs
                }, weights_filename)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Checkpoint saved locally to: {weights_filename}")
                
                # Log the local file as an artifact to MLflow
                # mlflow.log_artifact(weights_filename, artifact_path="checkpoints")
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Checkpoint saved to MLflow")

            # if c.export_mlflow:
            #     if (epoch + 1) % (c.checkpoint_interval * 2) == 0:
            #         export_mlflow_data()

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

