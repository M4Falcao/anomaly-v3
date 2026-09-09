"""Datasets, transforms, losses and metric tracking shared across the project.

Groups four concerns:

- Transform builders (:func:`get_random_transforms`, :func:`get_fixed_transforms`)
  that implement the multi-view augmentation DifferNet scores over.
- Dataset construction (:func:`load_datasets`, :func:`make_dataloaders`) plus
  :class:`AlignedTestDataset`, which keeps images and ground-truth masks aligned
  for pixel-level evaluation.
- The normalizing-flow objective (:func:`get_loss`) and tensor helpers.
- ``Score_Observer``, which tracks the best value of a metric across epochs.
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
from PIL import Image
import numpy as np
import config as c
from core.multi_transform_loader import ImageFolderMultiTransform
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS



def get_random_transforms():
    """Build the randomized training transform pipeline from ``config``.

    Returns:
        A ``Compose`` applying resize, optional rotation/colour jitter,
        tensor conversion and ImageNet normalization.
    """
    augmentative_transforms = []
    if c.transf_rotations:
        augmentative_transforms += [transforms.RandomRotation(180)]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                                           saturation=c.transf_saturation)]

    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean, c.norm_std)]

    transform_train = transforms.Compose(tfs)
    return transform_train


def get_fixed_transforms(degrees):
    """Build a deterministic transform pipeline for one rotation angle.

    Used at test time so every image is scored under the same fixed set of
    rotations, making evaluation reproducible.

    Args:
        degrees: Rotation angle in degrees.

    Returns:
        A ``Compose`` applying resize, the fixed rotation, optional colour
        jitter, tensor conversion and ImageNet normalization.
    """
    cust_rot = lambda x: rotate(x, degrees, False, False, None)
    augmentative_transforms = [cust_rot]
    if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
        augmentative_transforms += [
            transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                   saturation=c.transf_saturation)]
    tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                       transforms.Normalize(c.norm_mean,
                                                                                            c.norm_std)]
    return transforms.Compose(tfs)


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def get_loss(z, jac):
    '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


class TargetTransform:
    def __init__(self, class_perm):
        self.class_perm = class_perm

    def __call__(self, target):
        return self.class_perm[target]


class AlignedTestDataset(Dataset):
    """Test dataset that returns (images, label, mask) with proper alignment.
    
    For 'good' images: mask is all zeros.
    For anomaly images: mask is loaded from ground_truth directory.
    """

    def __init__(self, test_dir, ground_truth_dir, transform, mask_transform, 
                 class_perm, n_transforms=c.n_transforms_test):
        self.test_dir = test_dir
        self.ground_truth_dir = ground_truth_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.class_perm = class_perm
        self.n_transforms = n_transforms
        self.get_fixed = False
        self.fixed_degrees = [i * 360.0 / n_transforms for i in range(n_transforms)]

        # Build list of (image_path, label, mask_path_or_None)
        self.samples = []
        classes = sorted(os.listdir(test_dir))
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = os.path.join(test_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label = class_perm[cls_idx]
            
            for fname in sorted(os.listdir(cls_dir)):
                fpath = os.path.join(cls_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
                    continue

                mask_path = None
                if label > 0 and ground_truth_dir is not None:
                    # Look for mask in ground_truth/class_name/filename
                    gt_cls_dir = os.path.join(ground_truth_dir, cls_name)
                    if os.path.isdir(gt_cls_dir):
                        candidate = os.path.join(gt_cls_dir, fname)
                        if os.path.exists(candidate):
                            mask_path = candidate
                        else:
                            # Try matching by stem (different extension or _mask suffix)
                            stem = os.path.splitext(fname)[0]
                            for mf in os.listdir(gt_cls_dir):
                                mf_stem = os.path.splitext(mf)[0]
                                # Exact stem match or mask has _mask suffix
                                if mf_stem == stem or mf_stem == stem + '_mask' or mf_stem == stem + '_gt':
                                    mask_path = os.path.join(gt_cls_dir, mf)
                                    break

                self.samples.append((fpath, label, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, mask_path = self.samples[index]
        image = Image.open(img_path).convert('RGB')

        # Load mask
        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
        else:
            # Zero mask for 'good' images
            mask = Image.new('L', (c.img_size[1], c.img_size[0]), 0)

        # Apply transforms to image (n_transforms times)
        if self.get_fixed and len(self.fixed_degrees) != self.n_transforms:
            self.fixed_degrees = [i * 360.0 / self.n_transforms for i in range(self.n_transforms)]

        images = []
        for i in range(self.n_transforms):
            if self.get_fixed:
                images.append(self._fixed_rotation(image, self.fixed_degrees[i]))
            else:
                images.append(self.transform(image))
        images = torch.stack(images, dim=0)

        # Apply mask transform (resize only, no augmentation)
        mask = self.mask_transform(mask)

        return images, label, mask

    def _fixed_rotation(self, sample, degrees):
        cust_rot = lambda x: rotate(x, degrees, False, False, None)
        augmentative_transforms = [cust_rot]
        if c.transf_brightness > 0.0 or c.transf_contrast > 0.0 or c.transf_saturation > 0.0:
            augmentative_transforms += [
                transforms.ColorJitter(brightness=c.transf_brightness, contrast=c.transf_contrast,
                                       saturation=c.transf_saturation)]
        tfs = [transforms.Resize(c.img_size)] + augmentative_transforms + [transforms.ToTensor(),
                                                                           transforms.Normalize(c.norm_mean,
                                                                                                c.norm_std)]
        return transforms.Compose(tfs)(sample)


def load_datasets(dataset_path, class_name, aligned=True):
    """Load the train and test splits for one dataset class.

    The layout must be ``<dataset_path>/<class_name>/{train,test}`` where the
    test split contains a ``good`` subdirectory plus one directory per defect
    type. Every non-``good`` directory is mapped to label 1, ``good`` to 0.

    Args:
        dataset_path: Root directory holding the dataset classes.
        class_name: Class (object category) to load.
        aligned: When ``True``, return an :class:`AlignedTestDataset` that also
            yields ground-truth masks for pixel-level evaluation.

    Returns:
        ``(train_set, test_set)`` when ``aligned`` is ``True``; otherwise the
        legacy ``(train_set, test_set, ground_truth_set)`` triple.
    """

    data_dir_train = os.path.join(dataset_path, class_name, 'train')
    data_dir_test = os.path.join(dataset_path, class_name, 'test')

    classes = os.listdir(data_dir_test)
    if 'good' not in classes:
        print('There should exist a subdirectory "good". Read the doc of this function for further information.')
        exit()
    classes.sort()
    class_perm = list()
    class_idx = 1
    for cl in classes:
        if cl == 'good':
            class_perm.append(0)
        else:
            class_perm.append(class_idx)
            class_idx += 1

    transform_train = get_random_transforms()

    # Mask transform: resize to img_size, convert to tensor (no normalization)
    mask_transform = transforms.Compose([
        transforms.Resize(c.img_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Load ground truth directory path
    ground_truth_dir = os.path.join(dataset_path, class_name, 'ground_truth')
    if not os.path.isdir(ground_truth_dir):
        ground_truth_dir = None

    trainset = ImageFolderMultiTransform(data_dir_train, transform=transform_train, n_transforms=c.n_transforms)
    
    if aligned:
        # New: Aligned test dataset that returns (images, label, mask)
        testset = AlignedTestDataset(
            data_dir_test, ground_truth_dir, transform=transform_train,
            mask_transform=mask_transform, class_perm=class_perm,
            n_transforms=c.n_transforms_test
        )
        return trainset, testset
    else:
        # Legacy: separate test set and ground truth set
        from torchvision.datasets import ImageFolder as IF
        testset = ImageFolderMultiTransform(data_dir_test, transform=transform_train, 
                                            target_transform=TargetTransform(class_perm),
                                            n_transforms=c.n_transforms_test)
        ground_truth_set = None
        if ground_truth_dir:
            ground_truth_set = IF(ground_truth_dir, transform=transform_train)
        return trainset, testset, ground_truth_set


def make_dataloaders(trainset, testset, ground_truth_set=None):
    """Wrap datasets in dataloaders configured from ``config``.

    Shuffling is disabled for :class:`AlignedTestDataset` so that predictions
    stay aligned with their ground-truth masks.

    Args:
        trainset: Training dataset.
        testset: Test dataset.
        ground_truth_set: Optional separate mask dataset.

    Returns:
        ``(trainloader, testloader)``, or ``(trainloader, testloader,
        ground_truth_loader)`` when ``ground_truth_set`` is given.
    """
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False, num_workers=c.num_workers)
    # Use shuffle=False for aligned evaluation (AlignedTestDataset)
    is_aligned = isinstance(testset, AlignedTestDataset)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size_test, 
                                             shuffle=not is_aligned,
                                             drop_last=False, num_workers=c.num_workers)
    if ground_truth_set is not None:
        ground_truth_loader = torch.utils.data.DataLoader(ground_truth_set, pin_memory=True, batch_size=c.batch_size,
                                                          shuffle=False, drop_last=False, num_workers=c.num_workers)
        return trainloader, testloader, ground_truth_loader
    return trainloader, testloader



def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels
