import torch
import torch.nn as nn
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
import geomloss
from fastprogress import progress_bar
from argparse import ArgumentParser
from model.resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from model.de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50, de_wide_resnet50_2_senet, de_wide_resnet50_2_cbam
from utils.utils_test import evaluation_multi_proj
from utils.utils_train import MultiProjectionLayer, Revisit_RDLoss, loss_fucntion
from dataset.dataset import MVTecDataset_test, MVTecDataset_train, get_data_transforms

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--save_folder', default = './RD++_checkpoint_result', type=str)
    parser.add_argument('--batch_size', default = 16, type=int)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--detail_training', default='note', type = str)
    parser.add_argument('--proj_lr', default = 0.001, type=float)
    parser.add_argument('--distill_lr', default = 0.005, type=float)
    parser.add_argument('--weight_proj', default = 0.2, type=float) 
    parser.add_argument('--classes', nargs="+", default=["glass-insulator"])
    pars = parser.parse_args()
    return pars



def train(_class_, pars):
    print(_class_)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"DEVICE: {device}")

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    train_path = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\insplad-seg\class_name\train".replace("class_name", _class_)
    test_path = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\insplad-seg\class_name".replace("class_name", _class_)
    
    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path  = pars.save_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    train_data = MVTecDataset_train(root=train_path, transform=data_transform)
    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    print(f"train_data: {len(train_data)}")
    print(f"test_data: {len(test_data)}")

    # Use pretrained ImageNet for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2_cbam(pretrained=False)
    decoder = decoder.to(device)
    
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    proj_loss = Revisit_RDLoss()
    optimizer_proj = torch.optim.Adam(list(proj_layer.parameters()), lr=pars.proj_lr, betas=(0.5,0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=pars.distill_lr, betas=(0.5,0.999))

    
    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0
    
    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []
    
    loss_proj = []
    loss_distill = []
    total_loss = []
    
    history_infor = {}


    # set appropriate epochs for specific classes (Some classes converge faster than others)
    
    from tqdm import tqdm
    import time

    # Set appropriate epochs for specific classes (Some classes converge faster than others)
    num_epoch = 10

    print(f'With class {_class_}, Training with {num_epoch} Epochs')

    for epoch in tqdm(range(1, num_epoch + 1), desc="Epochs", unit="epoch"):
        bn.train()
        proj_layer.train()
        decoder.train()
        loss_proj_running = 0
        loss_distill_running = 0
        total_loss_running = 0

        # Gradient accumulation
        accumulation_steps = 2

        # Add a second progress bar for the data loader
        with tqdm(total=len(train_dataloader), desc=f"Training Epoch {epoch}", unit="batch") as batch_bar:
            for i, (img, img_noise, _) in enumerate(train_dataloader):
                img = img.to(device)
                img_noise = img_noise.to(device)
                inputs = encoder(img)
                inputs_noise = encoder(img_noise)

                (feature_space_noise, feature_space) = proj_layer(inputs, features_noise=inputs_noise)

                L_proj = proj_loss(inputs_noise, feature_space_noise, feature_space)

                outputs = decoder(bn(feature_space))
                L_distill = loss_fucntion(inputs, outputs)
                loss = L_distill + pars.weight_proj * L_proj
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer_proj.step()
                    optimizer_distill.step()
                    # Clear gradients
                    optimizer_proj.zero_grad()
                    optimizer_distill.zero_grad()

                total_loss_running += loss.detach().cpu().item()
                loss_proj_running += L_proj.detach().cpu().item()
                loss_distill_running += L_distill.detach().cpu().item()

                # Update the batch progress bar
                batch_bar.update(1)

        # Start timing for evaluation
        start_time = time.time()
        total_images = 0

        # Evaluation phase
        auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device)

        # End timing and calculate metrics
        elapsed_time = time.time() - start_time
        total_images = len(test_dataloader.dataset)  # Number of images in the test dataset
        images_per_second = total_images / elapsed_time
        latency_per_image_ms = (elapsed_time / total_images) * 1000

        print(f"Evaluation Metrics: Images per Second = {images_per_second:.2f}, Latency per Image = {latency_per_image_ms:.2f} ms")

        # Append evaluation results for monitoring
        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)
        loss_proj.append(loss_proj_running)
        loss_distill.append(loss_distill_running)
        total_loss.append(total_loss_running)

        # Save monitoring plots
        figure = plt.gcf()  # Get current figure
        figure.set_size_inches(8, 12)
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        ax[0][0].plot(auroc_px_list)
        ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_sp_list)
        ax[0][1].set_title('auroc_sp')
        ax[1][0].plot(aupro_px_list)
        ax[1][0].set_title('aupro_px')
        ax[1][1].plot(loss_proj)
        ax[1][1].set_title('loss_proj')
        ax[2][0].plot(loss_distill)
        ax[2][0].set_title('loss_distill')
        ax[2][1].plot(total_loss)
        ax[2][1].set_title('total_loss')
        plt.savefig(pars.save_folder + '/' + _class_ + '/monitor_training.png', dpi=100)

        print(f'Epoch {epoch}, Sample AUROC: {auroc_sp:.4f}, Pixel AUROC: {auroc_px:.4f}, Pixel AUPRO: {aupro_px:.4f}')

        # Save the best model
        if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
            best_score = (auroc_px + auroc_sp + aupro_px) / 3

            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch

            torch.save({
                'proj': proj_layer.state_dict(),
                'decoder': decoder.state_dict(),
                'bn': bn.state_dict()
            }, save_model_path)

            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch
            with open(os.path.join(pars.save_folder + '/' + _class_, 'history.json'), 'w') as f:
                json.dump(history_infor, f)

    return best_auroc_sp, best_auroc_px, best_aupro_px





if __name__ == '__main__':
    pars = get_args()
    print('Training with classes: ', pars.classes)
    all_classes = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    # train all_classes
    # for c in all_classes
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print('Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp, auroc_px, aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)
