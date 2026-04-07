import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet

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


# -----------------------------------------------------------------------------
# DifferNet with Integrated SEResNet18 Architecture
# -----------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # We ensure channel // reduction is at least 1
        reduced_channels = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Squeeze-and-Excitation embutido no bloco (integrated)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNet18(nn.Module):
    def __init__(self):
        super(SEResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18 usa [2, 2, 2, 2] blocos por camada
        self.layer1 = self._make_layer(SEBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(SEBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, 512, 2, stride=2)

        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization para novos blocos SE e camadas sem pretrained match
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Baixar e injetar os pesos pré-treinados da ResNet-18 padrão
        # para que o extrator de características não forneça features aleatórias
        from torchvision.models import resnet18
        pretrained_model = resnet18(pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        # Filtra apenas os pesos que batem perfeitamente (ignora camadas do SENet)
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(matched_dict)
        self.load_state_dict(model_dict)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Convert RGB to BGR (optional, para alinhar caso pretenda restaurar certos pesos)
        # x = x[:, [2, 1, 0], :, :]
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x, x1, x2, x3, x4


# Esta é a nova SEResNet18DifferNet 100% pronta para substituir a do treinamento
class SEResNet18DifferNet(nn.Module):
    def __init__(self):
        super(SEResNet18DifferNet, self).__init__()
        self.resnet = SEResNet18()
        self.nf = nf_head()

    def forward(self, x_input):
        y_cat = list()

        for s in range(c.n_scales):
            x_scaled = F.interpolate(x_input, size=c.img_size[0] // (2 ** s)) if s > 0 else x_input
            
            # Pega as características da layer3. Na ResNet-18 isto gera exatos 256 canais!
            x, x1, x2, x3, x4 = self.resnet(x_scaled)
            feat_s = x3
            
            y_cat.append(torch.mean(feat_s, dim=(2, 3)))

        # Se n_scales = 3, (256 * 3) = 768 posições exatas
        y = torch.cat(y_cat, dim=1)
        z = self.nf(y)
        return z


# -----------------------------------------------------------------------------
# Existing Architectures from model.py
# -----------------------------------------------------------------------------

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
    
    if isinstance(loaded_content, nn.Module):
        model.load_state_dict(loaded_content.state_dict())
    elif isinstance(loaded_content, dict) and 'model_state_dict' in loaded_content:
        model.load_state_dict(loaded_content['model_state_dict'])
        checkpoint = loaded_content
    else:
        model.load_state_dict(loaded_content)
        
    return model, checkpoint
