import os

def rename_files_in_folder1(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        for file_name in files:
            # Create the full path to the file
            old_file_path = os.path.join(folder_path, file_name)

            # Ensure we're only renaming files (not directories)
            if os.path.isfile(old_file_path):
                # Replace spaces with hyphens
                new_file_name = file_name.replace(" ", "-")
                new_file_path = os.path.join(folder_path, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {file_name} -> {new_file_name}")

        print("Renaming completed!")
    except Exception as e:
        print(f"An error occurred: {e}")


def rename_files_in_folder2(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)

        for file_name in files:
            # Create the full path to the file
            old_file_path = os.path.join(folder_path, file_name)

            # Ensure we're only renaming files (not directories)
            if os.path.isfile(old_file_path):
                # Replace spaces with hyphens
                new_file_name = str(file_name.split("_jpg")[0]) + '.jpg'
                new_file_path = os.path.join(folder_path, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {file_name} -> {new_file_name}")

        print("Renaming completed!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the folder containing the images
folder_path = r"C:\Users\Pichau\Pesquisa\pesquisa\data\insplad-seg\insplad-seg-test-manual-annotation\glass-insulator\test\missing-cap"  # Replace with your folder path

rename_files_in_folder2(folder_path)
