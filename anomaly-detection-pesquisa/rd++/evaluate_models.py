import os
import glob
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, auc
from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser

# Models
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_train import MultiProjectionLayer
from utils.utils_test import cal_anomaly_map

from torchvision import transforms

class Normalize(object):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image

def get_data_transforms(size):
    data_transforms = transforms.Compose([Normalize(), ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms


class InsPLADDatasetTest(torch.utils.data.Dataset):
    def __init__(self, class_name, dataset_base_path, transform, gt_transform):
        self.class_name = class_name
        self.dataset_base_path = dataset_base_path
        self.transform = transform
        self.gt_transform = gt_transform
        
        # Paths
        self.img_path = os.path.join(dataset_base_path, class_name, 'test')
        self.gt_path = os.path.join(dataset_base_path, class_name, 'ground_truth')
        
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []

        if not os.path.exists(self.img_path):
            print(f"Dataset path not found: {self.img_path}")
            return [], [], []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg")
            img_paths.sort()
            img_tot_paths.extend(img_paths)
            
            if defect_type == 'good':
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
            else:
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.jpg")
                gt_paths.sort()
                
                # Handling mismatched lengths between test and ground truth
                if len(gt_paths) != len(img_paths):
                    print(f"Warning: Mismatch between images and GTs for {defect_type} in {self.class_name}. Img: {len(img_paths)}, GT: {len(gt_paths)}")
                
                # Match by index or just zip up to minimum length, assuming ordered identically
                gt_tot_paths.extend(gt_paths[:len(img_paths)])
                tot_labels.extend([1] * len(img_paths))

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt = self.gt_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path)
        if img is None:
            # Fallback for unreadable images
            img = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img/255., (256, 256))
            
        img = self.transform(img)
        
        if gt == 0:
            gt = torch.zeros([1, 256, 256])
        else:
            try:
                gt_img = Image.open(gt)
                gt = self.gt_transform(gt_img)
            except Exception as e:
                gt = torch.zeros([1, 256, 256])

        return img, gt, label, img_path

def remove_orig_mod_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def evaluate_model(encoder, proj, bn, decoder, dataloader, device, is_pixel_level):
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    
    img_paths_list = []
    anomaly_maps_list = []
    gts_list = []
    
    with torch.no_grad():
        for (img, gt, label, img_path) in tqdm(dataloader, desc="Evaluating", leave=False):
            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            
            if is_pixel_level:
                gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
                pr_list_px.extend(anomaly_map.ravel())
                
            gt_list_sp.append(label.item())
            pr_list_sp.append(np.max(anomaly_map))
            
            img_paths_list.extend(img_path)
            anomaly_maps_list.append(anomaly_map)
            gts_list.append(gt.cpu().numpy().astype(int).squeeze(0))

    return {
        'gt_sp': np.array(gt_list_sp),
        'pr_sp': np.array(pr_list_sp),
        'gt_px': np.array(gt_list_px) if is_pixel_level else None,
        'pr_px': np.array(pr_list_px) if is_pixel_level else None,
        'img_paths': img_paths_list,
        'anomaly_maps': anomaly_maps_list,
        'gts': gts_list
    }


def compute_metrics_and_threshold(y_true, y_scores):
    if len(y_true) == 0:
         return 0, 0, 0, 0
    
    # Check if there are only normal or only anomalous samples
    if len(np.unique(y_true)) == 1:
        print("Warning: Only one class present in true labels. ROC AUC not defined.")
        return np.nan, np.nan, np.nan, 0.5

    # Calculate AUROC
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = np.nan
        
    # Find optimal threshold using F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Avoid division by zero
    f1_scores = np.divide(2 * (precisions * recalls), (precisions + recalls), out=np.zeros_like(precisions), where=(precisions + recalls) != 0)
    
    if len(thresholds) > 0:
        optimal_idx = np.argmax(f1_scores)
        # precision_recall_curve returns thresholds len - 1 compared to precisions/recalls
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        best_f1 = f1_scores[optimal_idx]
    else:
        optimal_threshold = 0.5
        best_f1 = 0
        
    y_pred = (y_scores >= optimal_threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    
    return auroc, best_f1, acc, optimal_threshold

def plot_roc(y_true, y_scores, title, save_path):
    if len(y_true) == 0 or len(np.unique(y_true)) == 1:
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def save_misclassified(gt_sp, pr_sp, threshold, img_paths, anomaly_maps, gts, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    pred_sp = (pr_sp >= threshold).astype(int)
    
    for i in range(len(gt_sp)):
        if gt_sp[i] != pred_sp[i]: # Misclassified
            img_path = img_paths[i]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
            else:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            
            amap = anomaly_maps[i]
            mask = gts[i]
            
            # Normalize anomaly map for visualization
            amap_norm = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * amap_norm), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Overlay
            overlay = np.float32(heatmap) * 0.5 + np.float32(img) * 0.5
            overlay = np.uint8(overlay)
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img)
            ax[0].set_title(f'Original\nGT:{gt_sp[i]}, Pred:{pred_sp[i]}')
            ax[0].axis('off')
            
            ax[1].imshow(overlay)
            ax[1].set_title('Anomaly Map')
            ax[1].axis('off')
            
            # Mask might be 3D like (1, 256, 256)
            mask_2d = mask.squeeze() if mask.ndim == 3 else mask
            ax[2].imshow(mask_2d, cmap='gray')
            ax[2].set_title('Ground Truth Mask')
            ax[2].axis('off')
            
            filename = os.path.basename(img_path)
            # handle cases where multiple images have same name in different defect folders
            defect_type = os.path.basename(os.path.dirname(img_path))
            save_name = f'error_{defect_type}_{filename}'
            
            plt.savefig(os.path.join(save_dir, save_name))
            plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument('--dataset_base_path', type=str, default=r'C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg')
    parser.add_argument('--checkpoint_folder', type=str, default='./best_models')
    parser.add_argument('--output_folder', type=str, default='./analysis')
    parser.add_argument('--image_size', type=int, default=256)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    classes = [d for d in os.listdir(args.checkpoint_folder) if os.path.isdir(os.path.join(args.checkpoint_folder, d))]
    print(f"Found classes: {classes}")
    
    data_transform, gt_transform = get_data_transforms(args.image_size)
    
    all_metrics = []

    for c in classes:
        print(f"\n--- Processing class: {c} ---")
        class_output_dir = os.path.join(args.output_folder, c)
        os.makedirs(class_output_dir, exist_ok=True)
        
        test_data = InsPLADDatasetTest(c, args.dataset_base_path, data_transform, gt_transform)
        if len(test_data) == 0:
            print(f"Skipping {c} as no test data was found.")
            continue
            
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        
        # Shared setup
        encoder, bn = wide_resnet50_2(pretrained=True)
        encoder = encoder.to(device)
        bn = bn.to(device)
        decoder = de_wide_resnet50_2(pretrained=False)
        decoder = decoder.to(device)
        proj_layer = MultiProjectionLayer(base=64).to(device)
        
        class_metrics = {'class': c}
        
        # 1. Image Level Evaluation (using _best_sp.pth)
        sp_ckpt_path = os.path.join(args.checkpoint_folder, c, f'wres50_{c}_best_sp.pth')
        if os.path.exists(sp_ckpt_path):
            print(f"Evaluating Image level using {sp_ckpt_path}")
            ckp = torch.load(sp_ckpt_path, map_location='cpu')
            proj_layer.load_state_dict(remove_orig_mod_prefix(ckp['proj']))
            bn.load_state_dict(remove_orig_mod_prefix(ckp['bn']))
            decoder.load_state_dict(remove_orig_mod_prefix(ckp['decoder']))
            
            res_sp = evaluate_model(encoder, proj_layer, bn, decoder, test_dataloader, device, is_pixel_level=False)
            
            auroc_sp, f1_sp, acc_sp, th_sp = compute_metrics_and_threshold(res_sp['gt_sp'], res_sp['pr_sp'])
            
            class_metrics['AUROC_sample'] = auroc_sp
            class_metrics['F1_sample'] = f1_sp
            class_metrics['Accuracy_sample'] = acc_sp
            
            print(f"  [Image] AUROC: {auroc_sp:.4f} | F1: {f1_sp:.4f} | Accuracy: {acc_sp:.4f} | Threshold: {th_sp:.4f}")
            
            plot_roc(res_sp['gt_sp'], res_sp['pr_sp'], f'Image ROC - {c}', os.path.join(class_output_dir, 'roc_image.png'))
            
            errors_dir = os.path.join(class_output_dir, 'errors')
            save_misclassified(res_sp['gt_sp'], res_sp['pr_sp'], th_sp, res_sp['img_paths'], res_sp['anomaly_maps'], res_sp['gts'], errors_dir)
        else:
            print(f"Warning: {sp_ckpt_path} not found.")
            
        # 2. Pixel Level Evaluation (using _best_px.pth)
        px_ckpt_path = os.path.join(args.checkpoint_folder, c, f'wres50_{c}_best_px.pth')
        if os.path.exists(px_ckpt_path):
            print(f"Evaluating Pixel level using {px_ckpt_path}")
            ckp = torch.load(px_ckpt_path, map_location='cpu')
            proj_layer.load_state_dict(remove_orig_mod_prefix(ckp['proj']))
            bn.load_state_dict(remove_orig_mod_prefix(ckp['bn']))
            decoder.load_state_dict(remove_orig_mod_prefix(ckp['decoder']))
            
            res_px = evaluate_model(encoder, proj_layer, bn, decoder, test_dataloader, device, is_pixel_level=True)
            
            auroc_px, f1_px, acc_px, th_px = compute_metrics_and_threshold(res_px['gt_px'], res_px['pr_px'])
            
            class_metrics['AUROC_pixel'] = auroc_px
            class_metrics['F1_pixel'] = f1_px
            class_metrics['Accuracy_pixel'] = acc_px
            
            print(f"  [Pixel] AUROC: {auroc_px:.4f} | F1: {f1_px:.4f} | Accuracy: {acc_px:.4f} | Threshold: {th_px:.4f}")
            
            # Downsample pixel metrics to avoid memory issues when plotting ROC
            subset_size = min(len(res_px['gt_px']), 5000000)
            if subset_size < len(res_px['gt_px']):
                idx = np.random.choice(len(res_px['gt_px']), subset_size, replace=False)
                plot_roc(res_px['gt_px'][idx], res_px['pr_px'][idx], f'Pixel ROC - {c}', os.path.join(class_output_dir, 'roc_pixel.png'))
            else:
                plot_roc(res_px['gt_px'], res_px['pr_px'], f'Pixel ROC - {c}', os.path.join(class_output_dir, 'roc_pixel.png'))
        else:
            print(f"Warning: {px_ckpt_path} not found.")
            
        all_metrics.append(class_metrics)
        
    df = pd.DataFrame(all_metrics)
    df.to_csv(os.path.join(args.output_folder, 'validation_metrics.csv'), index=False)
    print("\nValidation complete. Results saved in", args.output_folder)

if __name__ == '__main__':
    main()
