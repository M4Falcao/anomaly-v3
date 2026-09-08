"""Filesystem anchors for the DifferNet project.

Every script in this repository historically resolved its output directories
relative to the process working directory, which meant they only worked when
launched from the project root. This module exposes an absolute ``PROJECT_ROOT``
so that moved scripts keep writing to the very same folders regardless of where
they are launched from.

Example:
    >>> from core.paths import PROJECT_ROOT, project_path
    >>> project_path("checkpoints", "run_01")
    '.../differnet/checkpoints/run_01'
"""

import os

# ``core/`` lives directly under the project root, hence a single ``os.pardir``.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def project_path(*parts: str) -> str:
    """Build an absolute path rooted at the project directory.

    Args:
        *parts: Path components appended to ``PROJECT_ROOT``.

    Returns:
        The absolute path formed by joining ``PROJECT_ROOT`` with ``parts``.
    """
    return os.path.join(PROJECT_ROOT, *parts)


def ensure_dir(*parts: str) -> str:
    """Build a project-relative path and create the directory if missing.

    Args:
        *parts: Path components appended to ``PROJECT_ROOT``.

    Returns:
        The absolute path of the (now guaranteed to exist) directory.
    """
    path = project_path(*parts)
    os.makedirs(path, exist_ok=True)
    return path
