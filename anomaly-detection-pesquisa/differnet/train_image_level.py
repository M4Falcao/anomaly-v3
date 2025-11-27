import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mlflow
import os

import config as c
from localization import export_gradient_maps
from model import *
from utils import *
import skimage

from sklearn.metrics import roc_auc_score

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

def train(train_loader, test_loader):
    model = SEDifferNet()
    optimizer = torch.optim.Adam(model.nf.parameters(), lr=c.lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=1e-5)
    model.to(c.device)

    score_obs_image = Score_Observer('AUROC for image level')

    mlflow.set_tracking_uri(c.mlflow_tracking_uri)
    mlflow.set_experiment(c.mlflow_experiment_name)
    mlflow.start_run(run_name=c.mlflow_run_name)

    mlflow.log_params({
        "dataset_path": c.dataset_path,
        "class_name": c.class_name,
        "modelname": c.modelname,
        "img_size": c.img_size,
        "n_scales": c.n_scales,
        "clamp_alpha": c.clamp_alpha,
        "n_coupling_blocks": c.n_coupling_blocks,
        "fc_internal": c.fc_internal,
        "dropout": c.dropout,
        "lr_init": c.lr_init,
        "n_feat": c.n_feat,
        "n_transforms": c.n_transforms,
        "n_transforms_test": c.n_transforms_test,
        "batch_size": c.batch_size,
        "meta_epochs": c.meta_epochs,
        "sub_epochs": c.sub_epochs,
    })

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
            
            mlflow.log_metric("train_loss_step", loss.item(), step=epoch * len(train_loader) + i)
            mlflow.log_metric("train_score_step", image_level_score_train, step=epoch * len(train_loader) + i)

        # Compute average training loss
        avg_train_loss = np.mean(train_loss)

        # Print or log metrics during training
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, c.meta_epochs, avg_train_loss))
        mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)

        # Evaluation loop
        model.eval()
        test_loss = []
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

        # Calculate time metrics
        elapsed_time = time.time() - start_time
        images_per_second = total_images / elapsed_time
        latency_per_image_ms = (elapsed_time / total_images) * 1000

        # Print metrics
        print(f"Test Set - Images per Second: {images_per_second:.2f}, Latency per Image: {latency_per_image_ms:.2f} ms")

        # Compute average test loss
        avg_test_loss = np.mean(test_loss)

        # Update score observer
        is_anomaly = np.array([0 if l == 0 else 1 for l in np.concatenate(test_labels)])
        z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
        anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
        image_level_auroc = roc_auc_score(is_anomaly, anomaly_score)
        score_obs_image.update(image_level_auroc, epoch,
                        print_score=c.verbose or epoch == c.meta_epochs - 1)
        
        mlflow.log_metric("avg_test_loss", avg_test_loss, step=epoch)
        mlflow.log_metric("image_level_auroc", image_level_auroc, step=epoch)

    if c.grad_map_viz:
        export_gradient_maps(model, test_loader, optimizer, -1)

    if c.save_model:
        model.to('cpu')
        save_model(model, c.modelname)
        save_weights(model, c.modelname)
        mlflow.pytorch.log_model(model, "model")

    mlflow.end_run()
    return model

if __name__ == "__main__":
    # --- Adicione estas duas linhas no topo absoluto do seu arquivo ---
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    setup_seed(42)
    train_set, test_set, _ = load_datasets_image_level(c.dataset_path, c.class_name)
    train_loader, test_loader, _ = make_dataloaders(train_set, test_set)
    model = train(train_loader, test_loader)
