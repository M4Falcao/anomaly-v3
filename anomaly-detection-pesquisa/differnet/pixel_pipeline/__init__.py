"""Pixel-level anomaly localization pipeline.

Modules:
    data       — limited-sample loader for lightning-rod-suspension
    backbone   — shared AlexNet multi-layer feature extractor
    methods    — D (PaDiM), E (PatchCore), F (FastFlow), G (CFLOW-AD)
    postproc   — A (Gaussian smoothing), B (input-gradient), C (ensemble)
    run        — main runner
"""
