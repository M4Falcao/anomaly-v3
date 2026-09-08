"""Archive the local MLflow tracking store into a timestamped zip file.

The heavy lifting lives in :func:`core.mlflow_utils.export_mlflow_data`; this
script is only the command-line front-end for it.

Usage:
    python scripts/tools/export_mlflow.py

The archive is written to ``config.mlflow_export_dir``.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from core.mlflow_utils import export_mlflow_data
from core.paths import project_path

if __name__ == "__main__":
    if os.path.exists(project_path("mlruns")):
        export_mlflow_data()
    else:
        print("Error: 'mlruns' directory not found in the project root.")
