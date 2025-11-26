import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_first_annotated_object(image_folder, label_folder, output_folder):
    """
    Extract the first object from images based on YOLO segmentation annotations.
    Save the extracted object with the same name as the original image.
    """
    os.makedirs(output_folder, exist_ok=True)

    label_files = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

    for label_file in tqdm(label_files, desc=f"Processing annotations in {os.path.basename(image_folder)}"):
        label_path = os.path.join(label_folder, label_file)
        image_path = os.path.join(image_folder, label_file.replace('.txt', '.jpg'))

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Skipping.")
            continue

        # Read the image
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) > 0:  # Process only the first label if it exists
            data = lines[0].strip().split()

            if len(data) > 1:  # Ensure the line is not empty or malformed
                points = [
                    (float(data[i]) * w, float(data[i+1]) * h)  # Denormalize coordinates
                    for i in range(1, len(data), 2)
                ]

                # Create a mask for the object
                mask = np.zeros((h, w), dtype=np.uint8)
                polygon = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [polygon], color=255)

                # Extract the object using the mask
                object_image = cv2.bitwise_and(image, image, mask=mask)

                # Crop the object bounding box
                x, y, w_bbox, h_bbox = cv2.boundingRect(polygon)
                cropped_object = object_image[y:y+h_bbox, x:x+w_bbox]

                # Save the cropped object with the same name as the original image
                output_path = os.path.join(output_folder, os.path.basename(image_path))
                cv2.imwrite(output_path, cropped_object)

def process_all_modes(base_path, class_name, output_base_path):
    """
    Automate the processing for train, test, and valid modes.
    """
    modes = ['train', 'test', 'valid']
    
    for mode in modes:
        print(f"\nProcessing mode: {mode}")
        
        # Generate paths dynamically
        image_folder = os.path.join(base_path, class_name, mode, "images")
        label_folder = os.path.join(base_path, class_name, mode, "labels")
        output_folder = os.path.join(output_base_path, mode)
        
        # Call the extraction function
        extract_first_annotated_object(image_folder, label_folder, output_folder)

if __name__ == "__main__":
    # Variables for class, base path, and output base path
    base_path = r"C:\Users\Pichau\Pesquisa\pesquisa\yolo-segmentation\object-seg-enviesado"
    class_name = "glass-insulator-object-seg"  # Change this to the desired class
    output_base_path = r"C:\Users\Pichau\Pesquisa\pesquisa\yolo-segmentation\extracted_objects"  # Set your custom output path here

    # Process all modes
    process_all_modes(base_path, class_name, output_base_path)
