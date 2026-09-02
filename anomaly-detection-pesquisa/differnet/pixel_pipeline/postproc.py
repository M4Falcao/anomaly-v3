"""Post-processing operations applied to every method.

A: Gaussian smoothing of the score map.
B: Input-pixel gradient saliency. Computes |d(score_sum)/d(input)|.
C: Ensemble — normalized sum of A and B.
"""
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_smooth(score_map, sigma=4.0):
    """Apply per-image Gaussian smoothing. score_map: (B,H,W) numpy."""
    out = np.empty_like(score_map)
    for b in range(score_map.shape[0]):
        out[b] = gaussian_filter(score_map[b], sigma=sigma)
    return out


def input_gradient_map(method, inputs, img_size):
    """Compute |d(sum(score_map))/d(input)| averaged over channels.

    `method` must expose `.score(inputs) -> (B,H,W)` returning a
    differentiable score map tensor (not numpy).
    """
    inputs = inputs.clone().detach().requires_grad_(True)
    score_map = method.score_tensor(inputs)  # (B, H, W)
    loss = score_map.sum()
    grad = torch.autograd.grad(loss, inputs, retain_graph=False)[0]
    pixel_map = grad.abs().mean(dim=1)  # (B, H, W_img)
    # Resize to img_size if needed
    if pixel_map.shape[-1] != img_size:
        pixel_map = F.interpolate(pixel_map.unsqueeze(1),
                                  size=(img_size, img_size),
                                  mode='bilinear', align_corners=False).squeeze(1)
    return pixel_map.detach().cpu().numpy()


def _minmax_norm(arr, eps=1e-8):
    """Per-image min-max normalization."""
    flat = arr.reshape(arr.shape[0], -1)
    mn = flat.min(axis=1, keepdims=True)
    mx = flat.max(axis=1, keepdims=True)
    return ((flat - mn) / (mx - mn + eps)).reshape(arr.shape)


def ensemble(map_a, map_b, w_a=0.5, w_b=0.5):
    """Weighted sum of normalized maps."""
    na = _minmax_norm(map_a)
    nb = _minmax_norm(map_b)
    return w_a * na + w_b * nb


def apply_postproc(method, inputs, raw_map, img_size, sigma=4.0):
    """Apply A, B, C to a raw score map.

    Args:
        method: anomaly method with `.score_tensor(inputs)` for gradients.
        inputs: input tensor (B,3,H,W) on device.
        raw_map: numpy (B, img_size, img_size) raw score map.
        img_size: spatial size.
    Returns:
        dict with keys 'raw', 'A', 'B', 'C'.
    """
    map_a = gaussian_smooth(raw_map, sigma=sigma)
    try:
        map_b = input_gradient_map(method, inputs, img_size)
        map_b = gaussian_smooth(map_b, sigma=sigma)
    except Exception as e:
        print(f"  [postproc] gradient map failed: {e}")
        map_b = np.zeros_like(raw_map)
    map_c = ensemble(map_a, map_b)
    return {'raw': raw_map, 'A': map_a, 'B': map_b, 'C': map_c}
