"""Entry point for image-level DifferNet / SEDifferNet training.

Loads the dataset described by ``config.dataset_path`` / ``config.class_name``,
builds the dataloaders and runs the training loop from :mod:`core.train`.

Based on the WACV 2021 paper "Same Same But DifferNet: Semi-Supervised Defect
Detection with Normalizing Flows" by Marco Rudolph, Bastian Wandt and Bodo
Rosenhahn.

Usage:
    python main.py

All behaviour is driven by ``config.py``; this script takes no arguments.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import config as c
c.set_seed(c.seed)
from core.train import train
from core.utils import load_datasets, make_dataloaders


def main():
    """Load the configured dataset and run a full training session."""
    train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    train_loader, test_loader = make_dataloaders(train_set, test_set)
    model = train(train_loader, test_loader)

if __name__ == '__main__':
    main()
