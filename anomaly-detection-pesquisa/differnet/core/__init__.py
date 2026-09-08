"""Core library modules for the DifferNet / SEDifferNet anomaly detection project.

This package holds the reusable building blocks shared by every entry point:

- :mod:`core.freia_funcs`  - normalizing-flow primitives (FrEIA-derived).
- :mod:`core.model`        - DifferNet / SEDifferNet / CBAMDifferNet definitions.
- :mod:`core.utils`        - datasets, transforms, losses and score observers.
- :mod:`core.multi_transform_loader` - multi-transform ``ImageFolder`` variant.
- :mod:`core.localization` - gradient-map export for anomaly localization.
- :mod:`core.train`        - the image-level training loop.
- :mod:`core.paths`        - absolute project path anchors.

The module intentionally performs no imports at package level: several
submodules pull in heavy dependencies (torch, mlflow) and configure the CUDA
device on import, so they must only be loaded on demand.
"""
