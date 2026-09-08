"""Model definitions for DifferNet and its attention-augmented variants.

All three architectures share the same idea: a frozen ImageNet backbone
produces multi-scale channel descriptors, which a normalizing-flow head maps to
a Gaussian latent space. Anomalies are samples with a large latent norm.

- :class:`DifferNet`     - the original AlexNet backbone.
- :class:`SEDifferNet`   - adds Squeeze-and-Excitation channel attention.
- :class:`CBAMDifferNet` - adds CBAM (channel + spatial) attention.

``SEDifferNet`` and ``CBAMDifferNet`` declare identical submodules and differ
only in ``forward()``, so a checkpoint alone cannot distinguish them; the folder
name is used as a tiebreaker (see :mod:`core.eval_common`).

Saved artifacts live in ``weights/`` (state dicts) and ``models/`` (full models).
"""

import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet, resnet18

from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.SEAttention import SEAttention


import config as c
from core.freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node
from core.paths import project_path

WEIGHT_DIR = project_path('weights')
MODEL_DIR = project_path('models')


def nf_head(input_dim=c.n_feat):
    """Build the normalizing-flow head.

    Stacks ``config.n_coupling_blocks`` pairs of a fixed random permutation and
    a GLOW-style affine coupling block.

    Args:
        input_dim: Width of the concatenated multi-scale feature vector.

    Returns:
        A ``ReversibleGraphNet`` mapping features to the latent space.
    """
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class DifferNet(nn.Module):
    """Original DifferNet: AlexNet features pooled per scale, then a flow head."""

    def __init__(self):
        super(DifferNet, self).__init__()
        self.feature_extractor = alexnet(pretrained=True)
        self.nf = nf_head()

    def forward(self, x):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x, size=c.img_size[0] // (2 ** s)) if s > 0 else x
            feat_s = self.feature_extractor.features(x_scaled)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z
    

class SEDifferNet(nn.Module):
    """DifferNet with Squeeze-and-Excitation attention inside the AlexNet stack.

    ``forward()`` applies ``simsa1``, ``simsa2`` and ``simsa4``. The ``cbam*``
    blocks and ``simsa3`` are still constructed so that checkpoints stay
    loadable, but they are not part of the forward pass.
    """

    def __init__(self):
        super(SEDifferNet, self).__init__()
        self.alexnet = alexnet(pretrained=True)
        # self.cbam1 = CBAMBlock(channel=384,reduction=16,kernel_size=49)
        # self.cbam2 = CBAMBlock(channel=256,reduction=16,kernel_size=49)
        self.cbam1 = CBAMBlock(channel=64,reduction=16,kernel_size=49)
        self.cbam2 = CBAMBlock(channel=192,reduction=16,kernel_size=49)        
        self.cbam3 = CBAMBlock(channel=384,reduction=16,kernel_size=49)
        self.cbam4 = CBAMBlock(channel=256,reduction=16,kernel_size=49)
        self.simsa1 = SEAttention(channel=64, reduction=2)
        self.simsa2 = SEAttention(channel=192, reduction=2)
        self.simsa3 = SEAttention(channel=384, reduction=2)
        self.simsa4 = SEAttention(channel=256, reduction=2)
        
        self.nf = nf_head()

    def forward(self, x_input):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x_input, size=c.img_size[0] // (2 ** s)) if s > 0 else x_input
            x = self.alexnet.features[0](x_scaled)
            x = self.alexnet.features[1](x)
            x = self.simsa1(x)
            x = self.alexnet.features[2](x)
            x = self.alexnet.features[3](x)
            x = self.alexnet.features[4](x)
            x = self.simsa2(x)
            x = self.alexnet.features[5](x)
            x = self.alexnet.features[6](x)
            x = self.alexnet.features[7](x)
            x = self.alexnet.features[8](x)
            x = self.alexnet.features[9](x)
            x = self.alexnet.features[10](x)
            x = self.alexnet.features[11](x)
            x = self.simsa4(x)
            feat_s = self.alexnet.features[12](x)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z


class CBAMDifferNet(nn.Module):
    """DifferNet with CBAM attention inside the AlexNet stack.

    Mirrors :class:`SEDifferNet` but applies ``cbam1``, ``cbam2`` and ``cbam4``
    in ``forward()``. The ``simsa*`` blocks and ``cbam3`` are constructed for
    checkpoint compatibility only.
    """

    def __init__(self):
        super(CBAMDifferNet, self).__init__()
        self.alexnet = alexnet(pretrained=True)
        self.cbam1 = CBAMBlock(channel=64, reduction=16, kernel_size=49)
        self.cbam2 = CBAMBlock(channel=192, reduction=16, kernel_size=49)
        self.cbam3 = CBAMBlock(channel=384, reduction=16, kernel_size=49)
        self.cbam4 = CBAMBlock(channel=256, reduction=16, kernel_size=49)
        self.simsa1 = SEAttention(channel=64, reduction=2)
        self.simsa2 = SEAttention(channel=192, reduction=2)
        self.simsa3 = SEAttention(channel=384, reduction=2)
        self.simsa4 = SEAttention(channel=256, reduction=2)

        self.nf = nf_head()

    def forward(self, x_input):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x_input, size=c.img_size[0] // (2 ** s)) if s > 0 else x_input
            x = self.alexnet.features[0](x_scaled)
            x = self.alexnet.features[1](x)
            x = self.cbam1(x)
            x = self.alexnet.features[2](x)
            x = self.alexnet.features[3](x)
            x = self.alexnet.features[4](x)
            x = self.cbam2(x)
            x = self.alexnet.features[5](x)
            x = self.alexnet.features[6](x)
            x = self.alexnet.features[7](x)
            x = self.alexnet.features[8](x)
            x = self.alexnet.features[9](x)
            x = self.alexnet.features[10](x)
            x = self.alexnet.features[11](x)
            x = self.cbam4(x)
            feat_s = self.alexnet.features[12](x)
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z

class SEResNet18DifferNet(nn.Module):
    def __init__(self):
        super(SEResNet18DifferNet, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.simsa1 = SEAttention(channel=64, reduction=2)
        self.simsa2 = SEAttention(channel=128, reduction=2)
        self.simsa3 = SEAttention(channel=256, reduction=2)
        
        self.nf = nf_head()

    def forward(self, x_input):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x_input, size=c.img_size[0] // (2 ** s)) if s > 0 else x_input
            
            x = self.resnet18.conv1(x_scaled)
            x = self.resnet18.bn1(x)
            x = self.resnet18.relu(x)
            x = self.resnet18.maxpool(x)
            
            x = self.resnet18.layer1(x)
            x = self.simsa1(x)
            
            x = self.resnet18.layer2(x)
            x = self.simsa2(x)
            
            x = self.resnet18.layer3(x)
            feat_s = self.simsa3(x)
            
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    if os.path.exists(filename):
        path = filename
    else:
        path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path, weights_only=False)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    # Try finding the file directly
    if os.path.exists(filename):
        path = filename
    elif os.path.exists(os.path.join(WEIGHT_DIR, filename)):
        path = os.path.join(WEIGHT_DIR, filename)
    else:
        # Try to resolve path if user passed a path relative to repo root but is inside a subdirectory
        normalized_filename = filename.replace('\\', '/')
        normalized_cwd = os.getcwd().replace('\\', '/')
        
        filename_parts = normalized_filename.split('/')
        resolved_path = None
        
        # Check if we can find the file by stripping leading parts of the filename that match CWD suffixes
        for i in range(len(filename_parts)):
            test_path = os.path.join(os.getcwd(), *filename_parts[i:])
            if os.path.exists(test_path):
                resolved_path = test_path
                break
                
        if resolved_path:
            path = resolved_path
        else:
            # Try to search in parent directories (up to 4 levels up)
            curr_dir = os.getcwd()
            found = False
            for _ in range(4):
                test_path = os.path.join(curr_dir, filename)
                if os.path.exists(test_path):
                    path = test_path
                    found = True
                    break
                
                # Also try matching suffix parts of the filename in parent directories
                for i in range(1, len(filename_parts)):
                    test_subpath = os.path.join(curr_dir, *filename_parts[i:])
                    if os.path.exists(test_subpath):
                        path = test_subpath
                        found = True
                        break
                if found:
                    break
                parent = os.path.dirname(curr_dir)
                if parent == curr_dir:
                    break
                curr_dir = parent
            
            if not found:
                # Fallback to original default behavior
                path = os.path.join(WEIGHT_DIR, filename)
    
    loaded_content = torch.load(path, weights_only=False)
    
    checkpoint = None
    
    # Check if we loaded a full model, a state_dict, or a checkpoint dict
    if isinstance(loaded_content, nn.Module):
        model.load_state_dict(loaded_content.state_dict())
    elif isinstance(loaded_content, dict) and 'model_state_dict' in loaded_content:
        model.load_state_dict(loaded_content['model_state_dict'])
        checkpoint = loaded_content
    else:
        model.load_state_dict(loaded_content)
        
    return model, checkpoint
