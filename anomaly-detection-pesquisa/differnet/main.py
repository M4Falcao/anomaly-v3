'''This is the repo which contains the original code to the WACV 2021 paper
"Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows"
by Marco Rudolph, Bastian Wandt and Bodo Rosenhahn.
For further information contact Marco Rudolph (rudolph@tnt.uni-hannover.de)'''
import os

# --- Adicione estas duas linhas no topo absoluto do seu arquivo ---
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import config as c
from train import train
from train_v2 import train_v2
from utils import load_datasets, make_dataloaders, setup_seed, load_datasets_image_level

if __name__ == "__main__":
    setup_seed(42)

    # c.dataset_path = r"C:/Users/teo-s/OneDrive/Documentos/GitHub/anomaly-v3/anomaly-detection-pesquisa/differnet/dummy_dataset"
    # c.class_name = "dummy_class"
    # c.modelname = "dummy_model"
    # c.meta_epochs = 5
    # c.sub_epochs = 2

    train_set, test_set, ground_truth_set = load_datasets(c.dataset_path, c.class_name)
    train_loader, test_loader, ground_truth_loader = make_dataloaders(train_set, test_set, ground_truth_set)
    model = train(train_loader, test_loader, ground_truth_loader)
    # train_set, test_set = load_datasets_image_level(c.dataset_path, c.class_name)
    # train_loader, test_loader, ground_truth_loader = make_dataloaders(train_set, test_set)
    # model = train(train_loader, test_loader)
