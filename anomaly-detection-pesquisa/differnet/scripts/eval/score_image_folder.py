"""Score a folder of images with a trained DifferNet model.

Loads a saved model by name from ``models/`` and prints one anomaly score per
image found in the target folder. When ``fixed_transforms`` is enabled the
image is scored under ``config.n_transforms_test`` evenly spaced rotations and
the scores are averaged, matching the test-time augmentation used in training.

Usage:
    python scripts/eval/score_image_folder.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from core.model import load_model
from os import listdir
from os.path import join
from core.utils import get_random_transforms, get_fixed_transforms
from PIL import Image
import config as c
import torch

def get_anomaly_score(model, image_path, transforms):
    """Return the mean squared latent norm of one image under a transform set.

    Args:
        model: A trained DifferNet-style model in eval mode.
        image_path: Path to the image to score.
        transforms: Iterable of torchvision transforms applied to the image.

    Returns:
        Scalar tensor holding the anomaly score (higher means more anomalous).
    """
    img = Image.open(image_path).convert('RGB')
    transformed_imgs = torch.stack([tf(img) for tf in transforms])
    z = model(transformed_imgs)
    anomaly_score = torch.mean(z ** 2)
    print("image: %s, score: %.2f" % (image_path, anomaly_score))
    return anomaly_score

def evaluate(model_name, image_folder, fixed_transforms=True):
    model = load_model(model_name)
    files = listdir(image_folder)

    if fixed_transforms:
        fixed_degrees = [i * 360.0 / c.n_transforms_test for i in range(c.n_transforms_test)]
        transforms = [get_fixed_transforms(fd) for fd in fixed_degrees]
    else:
        transforms = [get_random_transforms()] * c.n_transforms_test

    for f in files:
        get_anomaly_score(model, join(image_folder, f), transforms)

image_folder = 'dummy_dataset/dummy_class/train/good'
evaluate(c.modelname, image_folder, fixed_transforms=True)