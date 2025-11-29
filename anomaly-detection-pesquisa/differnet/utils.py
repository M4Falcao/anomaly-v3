import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms.functional import rotate
import config as c
from multi_transform_loader import ImageFolderMultiTransform
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def get_random_transforms():
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

def load_datasets(dataset_path, class_name):
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

    target_transform = TargetTransform(class_perm)
    transform_train = get_random_transforms()

    # Load ground truth masks
    ground_truth_dir = os.path.join(dataset_path, class_name, 'ground_truth')
    ground_truth_set = ImageFolder(ground_truth_dir, transform=transform_train)

    trainset = ImageFolderMultiTransform(data_dir_train, transform=transform_train, n_transforms=c.n_transforms)
    testset = ImageFolderMultiTransform(data_dir_test, transform=transform_train, target_transform=target_transform,
                                        n_transforms=c.n_transforms_test)

    return trainset, testset, ground_truth_set


def load_datasets_image_level(dataset_path, class_name):
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

    target_transform = TargetTransform(class_perm)
    transform_train = get_random_transforms()

    # Skip ground truth loading
    
    trainset = ImageFolderMultiTransform(data_dir_train, transform=transform_train, n_transforms=c.n_transforms)
    testset = ImageFolderMultiTransform(data_dir_test, transform=transform_train, target_transform=target_transform,
                                        n_transforms=c.n_transforms_test)

    return trainset, testset


def make_dataloaders(trainset, testset, ground_truth_set=None):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False, num_workers=c.num_workers)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size_test, shuffle=False,
                                             drop_last=False, num_workers=c.num_workers)
    ground_truth_loader = None
    if ground_truth_set:
        ground_truth_loader = torch.utils.data.DataLoader(ground_truth_set, pin_memory=True, batch_size=c.batch_size,
                                                          shuffle=False, drop_last=False, num_workers=c.num_workers)
    return trainloader, testloader, ground_truth_loader



def preprocess_batch(data):
    '''move data to device and reshape image'''
    inputs, labels = data
    inputs, labels = inputs.to(c.device), labels.to(c.device)
    inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels


from scipy.ndimage import rotate, gaussian_filter

def generate_gradient_map(model, inputs, labels):
    '''
    Generates anomaly map using gradients of the loss w.r.t inputs.
    Adapted from localization.py but for all images (not just anomalies).
    '''
    model.eval()
    model.zero_grad()
    
    inputs.requires_grad = True
    
    z = model(inputs)
    loss = get_loss(z, model.nf.jacobian(run_forward=False))
    loss.backward()
    
    grad = inputs.grad.view(-1, c.n_transforms_test, *inputs.shape[-3:])
    grad = t2np(grad)
    
    degrees = -1 * np.arange(c.n_transforms_test) * 360.0 / c.n_transforms_test
    
    for i_item in range(c.n_transforms_test):
        old_shape = grad[:, i_item].shape
        # Flatten for rotation
        img = np.reshape(grad[:, i_item], [-1, *grad.shape[-2:]])
        # Transpose to (H, W, C) for rotation
        img = np.transpose(img, [1, 2, 0])
        # Rotate back
        img = np.transpose(rotate(img, degrees[i_item], reshape=False), [2, 0, 1])
        # Apply gaussian filter
        img = gaussian_filter(img, (0, 3, 3))
        # Reshape back
        grad[:, i_item] = np.reshape(img, old_shape)
        
    # Average over transforms
    grad_img = np.mean(np.abs(grad), axis=1)
    
    # Square the gradients (common practice for energy/anomaly score)
    grad_img_sq = grad_img ** 2
    
    # Average over channels to get 2D map (H, W)
    # Input grad is (Batch, C, H, W), we want (Batch, H, W)
    # But wait, grad_img is (Batch, C, H, W).
    # Usually we take max or mean over channels.
    # localization.py does: grad_img = np.mean(np.abs(grad), axis=1) -> this is mean over TRANSFORMS.
    # So grad_img is (Batch, C, H, W).
    # Then grad_img_sq = grad_img ** 2.
    # We need a single map per image.
    # Let's take mean over channels.
    anomaly_map = np.mean(grad_img_sq, axis=1)
    
    return anomaly_map
