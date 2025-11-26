import os
import config as c
from train import train
from utils import load_datasets, make_dataloaders, setup_seed
import torch
import gc

def train_all_classes():
    setup_seed(42)
    # Get all classes from the dataset path
    dataset_path = c.dataset_path
    try:
        classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    except FileNotFoundError:
        print(f"Error: Dataset path not found: {dataset_path}")
        return

    if not classes:
        print(f"No classes found in {dataset_path}")
        return

    print(f"Found {len(classes)} classes: {classes}")
    
    original_modelname = c.modelname

    for class_name in classes:
        print(f"\n{'='*40}")
        print(f"Starting training for class: {class_name}")
        print(f"{'='*40}\n")

        # Update config for the current class
        c.class_name = class_name
        c.modelname = f"{original_modelname}_{class_name}"
        
        # Load datasets and create dataloaders
        try:
            train_set, test_set, ground_truth_set = load_datasets(c.dataset_path, c.class_name)
            train_loader, test_loader, ground_truth_loader = make_dataloaders(train_set, test_set, ground_truth_set)
        except Exception as e:
            print(f"Error loading data for class {class_name}: {e}")
            continue

        # Run training
        try:
            train(train_loader, test_loader, ground_truth_loader)
        except Exception as e:
            print(f"Error training class {class_name}: {e}")
        
        # Clean up to avoid memory leaks
        gc.collect()
        torch.cuda.empty_cache()

    print("\nTraining completed for all classes.")

if __name__ == "__main__":
    train_all_classes()
