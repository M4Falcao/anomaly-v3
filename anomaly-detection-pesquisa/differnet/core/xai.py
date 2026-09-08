"""Explainability (XAI) helpers shared by the GradCAM-based evaluation scripts.

Currently holds the GradCAM target used to attribute the normalizing-flow
anomaly score back to input pixels. It was previously duplicated verbatim
between ``evaluate_cam_xai_metrics.py`` and ``evaluate_cam_sanity_check.py``.

Note:
    ``scripts/eval/visualize_cam_metric_steps.py`` intentionally keeps its own,
    simplified target and is not wired to this module.
"""

import torch


class AnomalyScoreTarget:
    """GradCAM target for normalizing-flow models (DifferNet / SEDifferNet).

    The anomaly "score" is the squared norm of the latent vector ``z``, so
    maximizing it highlights the pixels that push a sample away from the
    learned normal distribution.
    """

    def __call__(self, model_output):
        """Reduce a model output to the scalar the attribution maximizes.

        Args:
            model_output: Latent tensor ``z``, or a tuple whose first element
                is that tensor. A 1-D tensor is treated as a single sample.

        Returns:
            Scalar tensor holding the mean squared latent norm of the batch.
        """
        if isinstance(model_output, tuple):
            model_output = model_output[0]

        # Ensure 2D [Batch, Feature]
        if model_output.dim() == 1:
            model_output = model_output.unsqueeze(0)

        return torch.mean(torch.sum(model_output ** 2, dim=1))
