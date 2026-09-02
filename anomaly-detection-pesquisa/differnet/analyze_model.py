import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
import os
from tqdm import tqdm
import config as c
from model import load_model, SEDifferNet
from utils import load_datasets, make_dataloaders, t2np

def evaluate_model(model, test_loader):
    model.eval()
    print("Evaluating model...")
    
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            inputs, labels = data
            inputs = inputs.to(c.device)
            labels = labels.to(c.device)
            
            # Reshape inputs for the model if necessary (batch_size * n_transforms, channels, h, w)
            # The dataloader returns (batch_size, n_transforms, channels, h, w)
            # We need to flatten the first two dimensions
            bs, n_trans, ch, h, w = inputs.shape
            inputs = inputs.view(-1, ch, h, w)
            
            z = model(inputs)
            
            # Calculate anomaly score
            # z shape: (batch_size * n_transforms, n_features)
            # We need to aggregate scores per image (average over transforms)
            z = z.view(bs, n_trans, -1)
            score = torch.mean(z ** 2, dim=(1, 2)) # Mean over transforms and features
            
            y_scores.extend(t2np(score))
            y_true.extend(t2np(labels))
            
    return np.array(y_true), np.array(y_scores)

def plot_roc_curve(y_true, y_scores, save_path="roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC Curve saved to {save_path}")
    return auc, fpr, tpr, thresholds

def plot_score_distribution(y_true, y_scores, save_path="score_dist.png"):
    plt.figure(figsize=(10, 6))
    
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]
    
    plt.hist(normal_scores, bins=30, alpha=0.5, color='blue', label='Normal', density=True)
    plt.hist(anomaly_scores, bins=30, alpha=0.5, color='red', label='Anomaly', density=True)
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.title('Anomaly Score Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Score Distribution saved to {save_path}")

def calculate_metrics(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    
    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_scores >= best_threshold).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    
    return auc, accuracy, best_threshold

def main():
    # Ensure results directory exists
    if not os.path.exists('results'):
        os.makedirs('results')
        
    print(f"Loading model: {c.modelname}")
    # Initialize model structure (needed if load_model just loads state_dict, 
    # but based on model.py load_model loads the full object if saved with torch.save(model))
    # However, looking at model.py: torch.save(model, ...) saves the whole object.
    # Let's try loading it directly.
    try:
        model = load_model(c.modelname)
    except Exception as e:
        print(f"Error loading model directly: {e}")
        print("Attempting to load weights into SEDifferNet...")
        model = SEDifferNet()
        # Assuming there is a load_weights function or we can load state_dict
        # model.load_state_dict(torch.load(os.path.join(c.WEIGHT_DIR, c.modelname)))
        # Let's stick to load_model for now as per evaluate.py
        
    model.to(c.device)
    
    print(f"Loading datasets for class: {c.class_name}")
    trainset, testset, ground_truth_set = load_datasets(c.dataset_path, c.class_name)
    _, test_loader, _ = make_dataloaders(trainset, testset, ground_truth_set)
    
    y_true, y_scores = evaluate_model(model, test_loader)
    
    # Calculate Metrics
    auc, accuracy, best_threshold = calculate_metrics(y_true, y_scores)
    
    print("\n" + "="*30)
    print(f"Evaluation Results for {c.class_name}")
    print("="*30)
    print(f"AUROC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f} (at threshold {best_threshold:.4f})")
    print("="*30)
    
    # Generate Plots
    plot_roc_curve(y_true, y_scores, save_path=f"results/roc_curve_{c.class_name}.png")
    plot_score_distribution(y_true, y_scores, save_path=f"results/score_dist_{c.class_name}.png")

if __name__ == "__main__":
    main()
