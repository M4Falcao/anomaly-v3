'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''

# device settings
device = 'cuda' # or 'cpu'
import torch
torch.cuda.empty_cache()
torch.cuda.set_device(0)

# data settings
dataset_path = r"C:/Users/teo-s/OneDrive/Documentos/GitHub/anomaly-v3/insplad-seg/insplad-seg"
class_name = "glass-insulator"
modelname = "image_level_glass-insulator_resnet18"
# dataset_path = r"C:/Users/teo-s/OneDrive/Documentos/GitHub/anomaly-v3/anomaly-detection-pesquisa/differnet/dummy_dataset"
# class_name = "dummy_class"
# modelname = "dummy_model"

img_size = (448, 448)
img_dims = [3] + list(img_size)

# transformation settings
transf_rotations = True
transf_brightness = 0.0
transf_contrast = 0.0
transf_saturation = 0.0
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3 # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp_alpha = 3 # see paper equation 2 for explanation
n_coupling_blocks = 8
fc_internal = 2048 # number of neurons in hidden layers of s-t-networks
dropout = 0.0 # dropout in s-t-networks
lr_init = 2e-4
n_feat = 256 * n_scales # do not change except you change the feature extractor

# dataloader parameters
n_transforms = 4 # number of transformations per sample in training
n_transforms_test = 64 # number of transformations per sample in testing
batch_size = 1 # actual batch size is this value multiplied by n_transforms(_test)
batch_size_test = batch_size

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 20
sub_epochs = 5

# output settings
verbose = True
grad_map_viz = False
hide_tqdm_bar = True
save_model = True

# mlflow settings
mlflow_tracking_uri = 'sqlite:///mlflow.db'
mlflow_experiment_name = 'DifferNet_Experiment_v2'
mlflow_run_name = 'DifferNet_Run_v4_resnet18'
