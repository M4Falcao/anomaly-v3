"""Limited-sample dataset loader for the pixel pipeline.

Loads `lightning-rod-suspension` train (normal) and test (with masks).
- Train: returns plain tensors (no rotations) — small subset (N_TRAIN).
- Test: returns (image, label, mask) — full test set, single transform.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import random


IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def _list_images(directory):
    files = []
    if not os.path.isdir(directory):
        return files
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in IMG_EXTS:
            files.append(os.path.join(directory, fname))
    return files


class TrainNormalDataset(Dataset):
    """Loads N normal training images, one tensor per sample (no rotations)."""

    def __init__(self, train_dir, img_size=256, n_samples=30, seed=42,
                 norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
        all_paths = []
        # train_dir contains class subdirs (e.g., 'good')
        for sub in sorted(os.listdir(train_dir)):
            sub_dir = os.path.join(train_dir, sub)
            if os.path.isdir(sub_dir):
                all_paths += _list_images(sub_dir)

        rng = random.Random(seed)
        rng.shuffle(all_paths)
        self.paths = all_paths[:n_samples]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


class PixelTestDataset(Dataset):
    """Returns (image, label, mask) — label 0=good, 1=anomaly."""

    def __init__(self, test_dir, gt_dir, img_size=256,
                 norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
        self.img_size = img_size
        self.samples = []  # (img_path, label, mask_path_or_None)

        for cls_name in sorted(os.listdir(test_dir)):
            cls_dir = os.path.join(test_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = 0 if cls_name == 'good' else 1
            gt_cls_dir = os.path.join(gt_dir, cls_name) if gt_dir else None

            for fpath in _list_images(cls_dir):
                fname = os.path.basename(fpath)
                stem = os.path.splitext(fname)[0]

                mask_path = None
                if label == 1 and gt_cls_dir and os.path.isdir(gt_cls_dir):
                    for mf in os.listdir(gt_cls_dir):
                        mf_stem = os.path.splitext(mf)[0]
                        if mf_stem in (stem, stem + '_mask', stem + '_gt'):
                            mask_path = os.path.join(gt_cls_dir, mf)
                            break

                self.samples.append((fpath, label, mask_path))

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, mask_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
            mask_t = (self.mask_tf(mask) > 0.5).float()
        else:
            mask_t = torch.zeros(1, self.img_size, self.img_size)
        return self.img_tf(img), label, mask_t


def make_loaders(dataset_root, class_name='lightning-rod-suspension',
                 img_size=256, n_train=30, batch_size=4, seed=42, num_workers=0):
    train_dir = os.path.join(dataset_root, class_name, 'train')
    test_dir = os.path.join(dataset_root, class_name, 'test')
    gt_dir = os.path.join(dataset_root, class_name, 'ground_truth')
    if not os.path.isdir(gt_dir):
        gt_dir = None

    train_ds = TrainNormalDataset(train_dir, img_size=img_size, n_samples=n_train, seed=seed)
    test_ds = PixelTestDataset(test_dir, gt_dir, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"[data] train: {len(train_ds)} normal images (subset of {class_name})")
    print(f"[data] test:  {len(test_ds)} images "
          f"({sum(1 for s in test_ds.samples if s[1]==0)} good / "
          f"{sum(1 for s in test_ds.samples if s[1]==1)} anomaly)")
    return train_loader, test_loader
