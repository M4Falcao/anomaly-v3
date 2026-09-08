"""Run ``evaluate_cam_xai_metrics.py`` back-to-back for every InSPLAD-Seg class.

Each block below pins the final SEDifferNet checkpoint and the matching CFLOW
NF-head checkpoint for one class. Edit the paths to point at your own runs.

Usage:
    python scripts/tools/batch_evaluate_cam_metrics.py

Must be launched from the project root so the relative script paths resolve.
"""

import os

# os.system(
#     r'python .\scripts\eval\evaluate_cam_xai_metrics.py '
#     r'--cflow_checkpoint "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\NF Head\glass-insulator\best_models\best_pixel_auroc.pt" '
#     r'--class_name glass-insulator '
#     r'--dataset_path "C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg" '
#     r'--limit 100 '
#     r'--run_sanity_check true --disable_patchcore '
#     r'--model_path "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\glass-insulator\glass-insulator_se_differnet_glass_insulator_200_1_epoch_170.pt"'
# )

os.system(
    r'python .\scripts\eval\evaluate_cam_xai_metrics.py '
    r'--cflow_checkpoint "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\NF Head\lightning-rod-suspension\best_models\best_pixel_auroc.pt" '
    r'--class_name lightning-rod-suspension '
    r'--dataset_path "C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg" '
    r'--limit 100 '
    r'--run_sanity_check true --disable_patchcore '
    r'--model_path "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\lightning-rod-suspension\lightning-rod-suspension_se_differnet_lightning_rod_suspension_100_1_epoch_41.pt"'
)

os.system(
    r'python .\scripts\eval\evaluate_cam_xai_metrics.py '
    r'--cflow_checkpoint "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\NF Head\polymer-insulator-upper-shackle\best_models\best_pixel_auroc.pt" '
    r'--class_name polymer-insulator-upper-shackle '
    r'--dataset_path "C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg" '
    r'--limit 100 '
    r'--run_sanity_check true --disable_patchcore '
    r'--model_path "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\polymer-insulator-upper-shackle\polymer-insulator-upper-shackle_se_differnet_polymer_insulator_upper_shackle_200_1_epoch_146.pt"'
)

os.system(
    r'python .\scripts\eval\evaluate_cam_xai_metrics.py '
    r'--cflow_checkpoint "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\NF Head\vari-grip\best_models\best_pixel_auroc.pt" '
    r'--class_name vari-grip '
    r'--dataset_path "C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg" '
    r'--limit 100 '
    r'--run_sanity_check true --disable_patchcore '
    r'--model_path "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\vari-grip\vari-grip_se_differnet_vari_grip_200_1_epoch_163.pt"'
)

os.system(
    r'python .\scripts\eval\evaluate_cam_xai_metrics.py '
    r'--cflow_checkpoint "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\NF Head\yoke-suspension\best_models\best_pixel_auroc.pt" '
    r'--class_name yoke-suspension '
    r'--dataset_path "C:\Users\teo-s\Documents\GitHub\anomaly-detection-dataset\insplad-seg\insplad-seg" '
    r'--limit 100 '
    r'--run_sanity_check true --disable_patchcore '
    r'--model_path "C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet\final_models\SEDiffernet\yoke-suspension\yoke-suspension_se_differnet_yoke_suspension_200_1_epoch_70.pt"'
)