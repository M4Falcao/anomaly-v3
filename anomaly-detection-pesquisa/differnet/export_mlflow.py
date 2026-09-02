import shutil
import os
import datetime
import config as c

def export_mlflow_data():
    # Configuration
    source_dir = "mlruns"
    export_dir = c.mlflow_export_dir
    
    # Ensure export directory exists
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"Created directory: {export_dir}")
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mlflow_export_{timestamp}"
    output_path = os.path.join(export_dir, filename)
    
    print(f"Zipping '{source_dir}' to '{output_path}.zip'...")
    
    try:
        # Create zip archive
        shutil.make_archive(output_path, 'zip', source_dir)
        print(f"Successfully created export: {output_path}.zip")
    except Exception as e:
        print(f"Error creating export: {e}")

if __name__ == "__main__":
    if os.path.exists("mlruns"):
        export_mlflow_data()
    else:
        print("Error: 'mlruns' directory not found in current location.")
