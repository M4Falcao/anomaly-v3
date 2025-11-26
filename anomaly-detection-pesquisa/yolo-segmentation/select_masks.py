import os
import shutil
from tqdm import tqdm

def copy_matching_masks(mask_folder, annotated_folder, output_folder):
    """
    Copy annotated mask images to an output folder if they have a match
    in the provided mask folder based on a shared prefix (up to '_jpg' or '.jpg').
    
    :param mask_folder: Path to the folder containing regular mask images.
    :param annotated_folder: Path to the folder with manually annotated masks.
    :param output_folder: Path to the output folder where matched annotated masks will be copied.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the list of regular mask filenames, extracting their unique parts up to '_jpg' or '.jpg'
    mask_files = os.listdir(mask_folder)
    mask_prefixes = {
        file_name.split('_jpg')[0] if '_jpg' in file_name else file_name.split('.jpg')[0]
        for file_name in mask_files
    }


    # Get the list of annotated files
    annotated_files = os.listdir(annotated_folder)
    
    # Loop through annotated files and check for prefix matches
    for file_name in tqdm(annotated_files, desc="Checking matches"):
        annotated_prefix = (
            file_name.split('_jpg')[0]
        )
        if annotated_prefix in mask_prefixes:
            # Copy the matching file to the output folder
            source_path = os.path.join(annotated_folder, file_name)
            destination_path = os.path.join(output_folder, file_name)
            shutil.copy(source_path, destination_path)
            tqdm.write(f"Copied: {file_name}")
    
    print("Completed copying matching files.")

# Example usage

annotated_folder = r"C:\Users\Pichau\Pesquisa\pesquisa\yolo-segmentation\extracted_objects\glass-insulator-masks"


mask_folder = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\insplad-seg-no-back-treino\glass-insulator-no-background-treino\test\missing-cap"
output_folder = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\insplad-seg-test-manual-annotation\glass-insulator\test\missing-cap"

copy_matching_masks(mask_folder, annotated_folder, output_folder)