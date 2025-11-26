from PIL import Image
import os

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)

        # Skip directories
        if os.path.isdir(input_path):
            continue

        # Open the image
        with Image.open(input_path) as img:
            # Convert the image to grayscale
            img = img.convert("L")

            # Add "_mask" to the file name
            output_filename = os.path.splitext(filename)[0] + "_mask" + os.path.splitext(filename)[1]
            output_path = os.path.join(output_folder, output_filename)

            # Save the modified image
            img.save(output_path)

if __name__ == "__main__":
    input_folder = "data/insplad/glass-insulator/ground_truth/missingcap"
    output_folder = "data/insplad/glass-insulator/ground_truth/missingcap_mask"

    process_images(input_folder, output_folder)
