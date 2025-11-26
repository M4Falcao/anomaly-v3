import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import shutil

# Load the trained YOLOv8 segmentation model
segmentor = YOLO(r"C:\Users\Pichau\Pesquisa\pesquisa\yolo-segmentation\runs\segment\yoke-suspension-treino\weights\best.pt")

# Define the input dataset path and output dataset path
input_dataset_path = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\yoke-suspension"
output_dataset_path = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\yoke-suspension-no-background-treino"

# Function to apply segmentation and remove background
def remove_background(image, model):
    # Run segmentation model
    results = model(image)
    
    if len(results) > 0:
        result = results[0]
        if hasattr(result, 'masks') and result.masks is not None:
            mask = result.masks.data[0].cpu().numpy()  # Get the first mask (640x640)

            # Resize mask back to the original image size
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            resized_mask = cv2.resize(mask, original_size)  # Resize mask to match original image dimensions
            
            # Expand the mask to match the 3-channel shape of the image
            resized_mask = resized_mask[:, :, None]  # Add an extra dimension for broadcasting
            
            # Apply the mask to keep only the object (object in mask has value 1)
            object_only = np.where(resized_mask, image, 0)  # Zero out the background

            return object_only
    
    return None  # If no object is found, return None

# Function to process a dataset (train or test)
def process_dataset(input_folder, output_folder, model):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Count total images for the progress bar
    total_images = sum([len(files) for r, d, files in os.walk(input_folder)])
    
    # Progress bar setup
    with tqdm(total=total_images, desc=f"Processing {input_folder}", unit="image") as pbar:
        # Iterate through the classes in the dataset
        for class_name in os.listdir(input_folder):
            class_input_path = os.path.join(input_folder, class_name)
            class_output_path = os.path.join(output_folder, class_name)
            
            # Ensure the class output directory exists
            os.makedirs(class_output_path, exist_ok=True)
            
            # Process each image in the class folder
            for image_name in os.listdir(class_input_path):
                image_path = os.path.join(class_input_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    # Remove background from the image
                    object_only = remove_background(image, model)
                    if object_only is not None:
                        # Save the new image without background
                        output_image_path = os.path.join(class_output_path, image_name)
                        cv2.imwrite(output_image_path, object_only)
                pbar.update(1)  # Update the progress bar after processing each image

# Process train and test folders with progress bars
train_input_path = os.path.join(input_dataset_path, "train")
train_output_path = os.path.join(output_dataset_path, "train")
process_dataset(train_input_path, train_output_path, segmentor)

test_input_path = os.path.join(input_dataset_path, "test")
test_output_path = os.path.join(output_dataset_path, "test")
process_dataset(test_input_path, test_output_path, segmentor)

print("Dataset creation with objects only (no background) is complete.")


# yolo segment train data=vari-grip-object-seg\data.yaml model=yolo11n-seg.pt epochs=100 imgsz=640

