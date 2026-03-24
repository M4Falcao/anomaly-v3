import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet, resnet18

from fightingcv_attention.attention.CBAM import CBAMBlock
from fightingcv_attention.attention.SEAttention import SEAttention


import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head(input_dim=c.n_feat):
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

    print("SEDifferNet")
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
    if os.path.exists(filename):
        path = filename
    else:
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
