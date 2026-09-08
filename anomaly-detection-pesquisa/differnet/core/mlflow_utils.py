"""Checkpoint persistence and MLflow tracking helpers.

These helpers used to be copy-pasted between ``train.py``, ``train_backup.py``
and ``export_mlflow.py``. They are collected here so every entry point shares a
single implementation.

The module covers three concerns:

- Crash-safe checkpoint writing (:func:`safe_torch_save`).
- Serving and exposing the MLflow UI (:func:`run_mlflow_ui`, :func:`start_ngrok`,
  :func:`wait_for_server`).
- Archiving the local tracking store (:func:`export_mlflow_data`).
"""

import datetime
import os
import shutil
import socket
import time

import torch

import config as c
from core.paths import project_path


def safe_torch_save(obj, target_path, retries=2):
    """Serialize ``obj`` to ``target_path`` without risking a corrupt file.

    The object is first written to a ``.tmp`` sibling and only then moved into
    place with :func:`os.replace`, which is atomic on both POSIX and Windows.
    If the modern zipfile serializer fails (a known issue on some network and
    OneDrive-backed drives) the legacy pickle format is attempted before the
    next retry.

    Args:
        obj: Any object accepted by :func:`torch.save`, typically a checkpoint dict.
        target_path: Destination file path. Parent directories are created.
        retries: Number of additional attempts after the first one fails.

    Returns:
        ``True`` if the file was written, ``False`` if every attempt failed.
    """
    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    tmp_path = target_path + ".tmp"
    last_err = None

    for _attempt in range(retries + 1):
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, target_path)
            return True
        except Exception as e:
            last_err = e
            _remove_quietly(tmp_path)

            # Fall back to the legacy (non-zip) serializer before retrying.
            try:
                torch.save(obj, tmp_path, _use_new_zipfile_serialization=False)
                os.replace(tmp_path, target_path)
                return True
            except Exception as e_legacy:
                last_err = e_legacy
                _remove_quietly(tmp_path)

            time.sleep(0.5)

    print(f"Warning: failed to save checkpoint to {target_path}: {last_err}")
    return False


def _remove_quietly(path):
    """Delete ``path`` if it exists, ignoring OS-level failures."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def start_ngrok(port):
    """Expose a locally bound port through an ngrok tunnel.

    Args:
        port: TCP port the MLflow UI is listening on.
    """
    from pyngrok import ngrok
    ngrok.set_auth_token(c.ngrok_auth_token)
    public_url = ngrok.connect(port, host_header="rewrite").public_url
    print(f" * ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:{port}\"")


def run_mlflow_ui():
    """Run the MLflow UI in the foreground; intended to be called from a thread."""
    os.system(f"mlflow ui --backend-store-uri {c.mlflow_backend_store_uri} --port 5000 --host 0.0.0.0")


def wait_for_server(host, port, timeout=30):
    """Block until a TCP endpoint accepts connections.

    Args:
        host: Hostname or IP to probe.
        port: TCP port to probe.
        timeout: Maximum number of seconds to wait.

    Returns:
        ``True`` once the endpoint is reachable, ``False`` if ``timeout`` elapsed.
    """
    start_time = time.time()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)


def export_mlflow_data():
    """Archive the local ``mlruns`` tracking store into a timestamped zip file.

    The archive is written to ``config.mlflow_export_dir`` and named
    ``mlflow_export_<YYYYmmdd_HHMMSS>.zip``.

    Returns:
        The path of the created archive, or ``None`` if archiving failed.
    """
    source_dir = project_path("mlruns")
    export_dir = project_path(c.mlflow_export_dir)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"Created directory: {export_dir}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(export_dir, f"mlflow_export_{timestamp}")

    print(f"Zipping '{source_dir}' to '{output_path}.zip'...")

    try:
        shutil.make_archive(output_path, 'zip', source_dir)
        print(f"Successfully created export: {output_path}.zip")
        return output_path + ".zip"
    except Exception as e:
        print(f"Error creating export: {e}")
        return None
