import os

def keep_one_per_prefix(folder_path):
    # Dictionary to store one file per unique prefix
    prefix_map = {}

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Full path of the file
        file_path = os.path.join(folder_path, file_name)

        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        # Extract the prefix (before the last underscore)
        prefix = "_".join(file_name.split("_")[:-1])
        
        # If the prefix is not already added, keep this file
        if prefix not in prefix_map:
            prefix_map[prefix] = file_name
        else:
            # Delete the file if the prefix already exists
            os.remove(file_path)

    print(f"Processed {len(prefix_map)} unique prefixes. Files cleaned.")

if __name__ == "__main__":
    # Set the folder path here
    folder_path = r"C:\Users\Pichau\Pesquisa\pesquisa\yolo-segmentation\extracted_objects\glass-insulator-masks"

    if not os.path.exists(folder_path):
        print("Error: The specified folder does not exist.")
    else:
        keep_one_per_prefix(folder_path)
