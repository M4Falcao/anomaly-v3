import os
import config as c
from train import train
from utils import load_datasets, make_dataloaders, setup_seed
import torch
from torch.utils.data import ConcatDataset

def train_unified_model():
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
    print("Preparing unified dataset...")

    all_train_sets = []
    all_test_sets = []
    all_ground_truth_sets = []

    for class_name in classes:
        print(f"Loading data for class: {class_name}")
        try:
            train_set, test_set, ground_truth_set = load_datasets(c.dataset_path, class_name)
            all_train_sets.append(train_set)
            all_test_sets.append(test_set)
            all_ground_truth_sets.append(ground_truth_set)
        except Exception as e:
            print(f"Error loading data for class {class_name}: {e}")
            continue

    if not all_train_sets:
        print("No datasets loaded. Exiting.")
        return

    # Concatenate all datasets
    full_train_set = ConcatDataset(all_train_sets)
    full_test_set = ConcatDataset(all_test_sets)
    full_ground_truth_set = ConcatDataset(all_ground_truth_sets)

    print(f"Total training samples: {len(full_train_set)}")
    print(f"Total test samples: {len(full_test_set)}")

    # Create dataloaders
    train_loader, test_loader, ground_truth_loader = make_dataloaders(full_train_set, full_test_set, full_ground_truth_set)

    # Update model name for the unified model
    c.modelname = "unified_model"
    
    print(f"\n{'='*40}")
    print(f"Starting training for unified model: {c.modelname}")
    print(f"{'='*40}\n")

    # Run training
    try:
        train(train_loader, test_loader, ground_truth_loader)
    except Exception as e:
        print(f"Error training unified model: {e}")

    print("\nTraining completed for unified model.")

if __name__ == "__main__":
    train_unified_model()
