"""Protocolo compartilhado com o pipeline principal (split, inventário de teste).

O split val/holdout é calculado com a **mesma função** do treinador da NF head
(``stratified_test_split`` em ``scripts/train/pixel_train_from_pretrained.py``) sobre a
**mesma ordem** de amostras de ``AlignedTestDataset``. Assim, o audit e o relatório usam
exatamente as imagens que o treino usou para escolher a época.
"""
import importlib.util
import os
import sys

import numpy as np

import common as cm

if cm.DIFFERNET_ROOT not in sys.path:
    sys.path.insert(0, cm.DIFFERNET_ROOT)

_TRAINER_PATH = os.path.join(cm.DIFFERNET_ROOT, 'scripts', 'train', 'pixel_train_from_pretrained.py')
_trainer = None


def trainer_module():
    global _trainer
    if _trainer is None:
        spec = importlib.util.spec_from_file_location('_pixel_trainer', _TRAINER_PATH)
        _trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_trainer)
    return _trainer


def test_inventory(dataset_root, class_name):
    """``[(rel_path, label01, gt_path_or_None)]`` na ordem de ``AlignedTestDataset``."""
    from core.utils import load_datasets
    _, testset = load_datasets(dataset_root, class_name, aligned=True)
    class_dir = os.path.join(dataset_root, class_name)
    return [(os.path.relpath(p, class_dir), int(label > 0), gt) for p, label, gt in testset.samples]


def split_indices(labels, val_fraction=0.4, split_seed=42):
    val, holdout = trainer_module().stratified_test_split(np.asarray(labels), val_fraction, split_seed)
    return val.tolist(), holdout.tolist()


def partition_of(n, val, holdout):
    part = np.empty(n, dtype=object)
    part[list(val)] = 'val'
    part[list(holdout)] = 'holdout'
    return part


def training_recipe(class_name, recipe='general'):
    return trainer_module().resolve_training_recipe(class_name, recipe)
