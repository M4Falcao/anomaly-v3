"""Shared CFLOW-AD pixel-level anomaly detection components.

This module is the single source of truth for the CFLOW pixel head. It replaces
the three divergent copies that previously lived in
``scripts/train/pixel_train_from_pretrained.py``, ``scripts/eval/evaluate_full_pipeline.py``
and ``scripts/analysis/analyze_class.py``.

Those copies had drifted apart in two ways that silently changed results:

1. The training script divided each level's NLL by its channel count and
   min-max normalized every level before summing; the evaluation script summed
   the raw NLLs. A checkpoint selected under one scoring rule was deployed
   under the other.
2. ``out_size`` (the spatial grid the conditional flow is conditioned on via the
   positional encoding) defaulted to 96 during training but was hardcoded to 56
   during evaluation. It does not affect ``state_dict`` shapes, so the mismatch
   raised no error.

Both are now explicit: :func:`CFlowPixelHead.score_map` takes a ``mode``, and the
hyperparameters travel inside the checkpoint (see :func:`save_cflow_checkpoint`).

Score modes
-----------
``raw``
    Sum of the un-normalized per-level NLL maps. Reproduces the historical
    ``evaluate_full_pipeline.py`` behaviour. The level with the largest magnitude
    dominates the sum.
``per_level_minmax``
    Divide each level's NLL by its channel count, min-max normalize it at native
    resolution, then sum. Reproduces the historical training behaviour.
``per_level_std``
    Standardize each level with scalar mean/std estimated on normal training
    data (:func:`compute_level_stats`), then sum. Puts every level on a common
    scale without relying on per-image statistics.
``per_level_prob``
    CFLOW-AD reference aggregation (Gudovskiy et al., WACV 2022): per-dimension
    log-likelihood ``logsigmoid(log p) / C`` is exponentiated into a
    pseudo-probability in ``[0, 1]`` per level, the levels are summed and the
    score is the negated sum. Bounded, so no level can dominate through raw
    magnitude and no per-image statistics are needed.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn
from tqdm import tqdm

SCORE_MODES = ('raw', 'per_level_minmax', 'per_level_std', 'per_level_prob')
PAD_MODES = ('reflect', 'replicate')

# The affine coupling clamp used while the released heads were trained. The
# historical evaluation code used 2.0, which makes the flow compute a different
# invertible transform than the one the weights were fitted for.
TRAIN_CLAMP_SCALE = 0.5
LEGACY_EVAL_CLAMP_SCALE = 2.0

# Reference configuration used to fit the released heads.
TRAIN_OUT_SIZE = 96
# Mode used to aggregate per-level NLL during in-training validation and checkpoint
# selection. Aligned with deployment inference ('raw' beat 'per_level_minmax' in
# 17/18 heads across all 5 benchmark classes).
TRAIN_SCORE_MODE = 'raw'
LEGACY_TRAIN_SCORE_MODE = 'per_level_minmax'

# The CFLOW pipeline is built on SEDifferNet: features must come from the SE
# (``simsa*``) blocks. The CBAM blocks an SE model also declares are never
# trained, so reading them yields features the flow was never meant to see.
DEFAULT_ATTENTION = 'se'

# Cumulative stride from the input to the deepest extracted level (conv1 stride 4
# followed by two stride-2 max pools). Reflect padding must be a multiple of it,
# otherwise the convolution sampling grid shifts by a fraction of a step and the
# cropped map no longer lines up with the ground-truth mask.
BACKBONE_STRIDE = 16

# Hyperparameters that must match between training and evaluation.
CFLOW_HPARAM_KEYS = (
    'out_size', 'cond_dim', 'n_blocks', 'hidden',
    'img_size', 'score_mode', 'reflect_pad',
    'clamp_scale', 'attention',
    # Added for the NF-head ablation; absent (None) in older checkpoints.
    'pad_mode', 'feat_pool', 'levels', 'feat_norm', 'feat_norm_levels', 'no_rotation',
)


# ═══════════════════════════════════════════════════════════════════════════════
# Backbone feature extractor
# ═══════════════════════════════════════════════════════════════════════════════

def reflect_pad_step(out_size, img_size, stride=BACKBONE_STRIDE):
    """Smallest reflect padding that keeps the cropped map registered.

    Two constraints must hold simultaneously:

    1. ``pad`` is a multiple of the cumulative backbone ``stride``, otherwise the
       strided convolutions resample on a shifted grid.
    2. ``pad`` maps to a whole number of feature cells
       (``pad * out_size / img_size`` integer), otherwise the crop removes a
       fractional cell and shifts the window.

    Args:
        out_size: Feature grid side length.
        img_size: Input side length in pixels.
        stride: Cumulative backbone stride.

    Returns:
        The step size; valid paddings are its positive multiples.
    """
    cell_step = img_size // math.gcd(out_size, img_size)
    return abs(stride * cell_step) // math.gcd(stride, cell_step)


def snap_reflect_pad(pad, out_size=TRAIN_OUT_SIZE, img_size=448, stride=BACKBONE_STRIDE):
    """Round a reflect-padding request up to the nearest valid value.

    Args:
        pad: Requested padding in input pixels.
        out_size: Feature grid side length.
        img_size: Input side length in pixels.
        stride: Cumulative backbone stride.

    Returns:
        The nearest valid padding, at least one step when ``pad > 0``.
    """
    if pad <= 0:
        return 0
    step = reflect_pad_step(out_size, img_size, stride)
    return max(step, int(round(pad / step)) * step)


def resolve_attention(model, attention=DEFAULT_ATTENTION, warn=True):
    """Decide which attention blocks a feature extractor should read.

    ``SEDifferNet`` and ``CBAMDifferNet`` both *declare* ``cbam*`` and ``simsa*``
    submodules; only some of them take part in each model's ``forward()``. A
    ``hasattr`` probe therefore cannot tell them apart, which is how the training
    script ended up extracting features through SEDifferNet's untrained CBAM
    blocks. Dispatching on the class name is unambiguous.

    Args:
        model: A ``SEDifferNet`` or ``CBAMDifferNet`` instance.
        attention: ``'se'`` (default), ``'cbam'``, ``'auto'`` (follow the model's
            own ``forward()``) or ``'legacy_train'`` (reproduce the old bug).
        warn: Print a warning when the chosen path is not the one the model's
            ``forward()`` uses, i.e. when the blocks being read are untrained.

    Returns:
        Either ``'se'`` or ``'cbam'``.
    """
    forward_path = 'cbam' if type(model).__name__ == 'CBAMDifferNet' else 'se'

    if attention == 'auto':
        return forward_path
    if attention == 'legacy_train':
        chosen = 'cbam' if hasattr(model, 'cbam1') else 'se'
    elif attention in ('se', 'cbam'):
        chosen = attention
    else:
        raise ValueError(f"unknown attention mode {attention!r}")

    if warn and chosen != forward_path:
        print(f"  WARNING: reading '{chosen}' attention blocks from "
              f"{type(model).__name__}, whose forward() uses '{forward_path}'. "
              f"Those blocks receive no gradient during backbone training. "
              f"Pass --attention auto to follow the model instead.")
    return chosen

class SEBackboneFeatureExtractor(nn.Module):
    """Extract multi-level spatial features from a frozen SE/CBAM DifferNet.

    Features are taken before global average pooling at three semantic levels:

    ===== ======== =====================================================
    Level Channels Source
    ===== ======== =====================================================
    L1    64       after attention block 1 (right after AlexNet conv1)
    L2    192      after attention block 2
    L3    256      after attention block 4
    ===== ======== =====================================================

    All maps are bilinearly resampled to ``out_size`` x ``out_size``.

    Args:
        model: A trained ``SEDifferNet`` or ``CBAMDifferNet``.
        out_size: Spatial resolution the feature maps are resampled to. This is
            also the grid the flow's positional encoding is built on, so it must
            match the value used at training time.
        reflect_pad: If > 0, reflect-pad the input by this many pixels before the
            backbone and crop the corresponding border off the feature maps. L1
            sits directly after an 11x11 stride-4 convolution with padding 2, so
            its border activations are otherwise computed largely over zeros.
            ``0`` reproduces the original behaviour exactly.
        attention: Which attention blocks to read features from.
            ``'se'`` (default) uses ``simsa*``, ``'cbam'`` uses ``cbam*``,
            ``'auto'`` picks whichever the model's own ``forward()`` uses.
            ``'legacy_train'`` reproduces the historical training script, which
            selected ``cbam*`` whenever they existed - including for SE models,
            where those blocks are never trained.
        pad_mode: Padding fill used when ``reflect_pad > 0``: ``'reflect'``
            (mirror) or ``'replicate'`` (edge value).
        feat_pool: Kernel of a stride-1 average pool applied to every level
            before resampling (PatchCore-style local neighbourhood
            aggregation). ``0``/``1`` disables it.
    """

    def __init__(self, model, out_size=TRAIN_OUT_SIZE, reflect_pad=0,
                 attention=DEFAULT_ATTENTION, pad_mode='reflect', feat_pool=0):
        super().__init__()
        if pad_mode not in PAD_MODES:
            raise ValueError(f"unknown pad_mode {pad_mode!r}; expected one of {PAD_MODES}")
        self.alexnet = model.alexnet
        self.attention = resolve_attention(model, attention)
        prefix = 'cbam' if self.attention == 'cbam' else 'simsa'
        self._use_cbam = self.attention == 'cbam'
        self.attn1 = getattr(model, f'{prefix}1')
        self.attn2 = getattr(model, f'{prefix}2')
        self.attn3 = getattr(model, f'{prefix}3')
        self.attn4 = getattr(model, f'{prefix}4')

        self.out_size = out_size
        self.reflect_pad = reflect_pad
        self.pad_mode = pad_mode
        self.feat_pool = int(feat_pool or 0)
        self.layer_channels = [64, 192, 256]

        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def concat_channels(self):
        """Total channel count of the concatenated feature tensor."""
        return sum(self.layer_channels)

    @torch.no_grad()
    def forward(self, x_input):
        """Extract features at three levels.

        Args:
            x_input: Image batch shaped ``(B, 3, H, W)``.

        Returns:
            Tuple ``(concat, levels)`` where ``concat`` is the channel-wise
            concatenation of the three resampled level maps and ``levels`` is
            the list of individual maps, each ``(B, out_size, out_size)``.
        """
        target, crop = self.out_size, 0
        if self.reflect_pad > 0:
            h = x_input.shape[-2]
            pad_eff = snap_reflect_pad(self.reflect_pad, self.out_size, h)
            crop = pad_eff * self.out_size // h
            target = self.out_size + 2 * crop
            x_input = F.pad(x_input, (pad_eff,) * 4, mode=self.pad_mode)

        feats = []
        x = self.alexnet.features[0](x_input)
        x = self.alexnet.features[1](x)
        x = self.attn1(x)
        feats.append(x)

        x = self.alexnet.features[2](x)
        x = self.alexnet.features[3](x)
        x = self.alexnet.features[4](x)
        x = self.attn2(x)
        feats.append(x)

        x = self.alexnet.features[5](x)
        x = self.alexnet.features[6](x)
        x = self.alexnet.features[7](x)
        x = self.attn3(x)
        x = self.alexnet.features[8](x)
        x = self.alexnet.features[9](x)
        x = self.alexnet.features[10](x)
        x = self.alexnet.features[11](x)
        x = self.attn4(x)
        feats.append(x)

        if self.feat_pool > 1:
            k = self.feat_pool
            feats = [F.avg_pool2d(f, k, stride=1, padding=k // 2) for f in feats]

        resized = [F.interpolate(f, size=(target, target),
                                 mode='bilinear', align_corners=False) for f in feats]
        if crop > 0:
            end = crop + self.out_size
            resized = [r[:, :, crop:end, crop:end] for r in resized]

        return torch.cat(resized, dim=1), resized


# ═══════════════════════════════════════════════════════════════════════════════
# Conditional normalizing flow
# ═══════════════════════════════════════════════════════════════════════════════

def pos_encoding(h, w, dim=64, device=None):
    """Build a 2D sinusoidal positional encoding.

    Args:
        h: Grid height.
        w: Grid width.
        dim: Encoding width; must be divisible by 4.
        device: Torch device for the returned tensor.

    Returns:
        Tensor of shape ``(h * w, dim)``.
    """
    assert dim % 4 == 0
    d_quarter = dim // 4
    y_pos = torch.arange(h, device=device).float()
    x_pos = torch.arange(w, device=device).float()
    div = torch.exp(torch.arange(0, d_quarter, device=device).float() *
                    (-math.log(10000.0) / d_quarter))
    yy = y_pos.unsqueeze(1) * div.unsqueeze(0)
    xx = x_pos.unsqueeze(1) * div.unsqueeze(0)
    pe_y = torch.cat([yy.sin(), yy.cos()], dim=1)
    pe_x = torch.cat([xx.sin(), xx.cos()], dim=1)
    pe_y = pe_y.unsqueeze(1).expand(h, w, dim // 2)
    pe_x = pe_x.unsqueeze(0).expand(h, w, dim // 2)
    return torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)


class CondCouplingBlock(nn.Module):
    """Affine coupling block conditioned on an external vector.

    Args:
        channels: Feature width being transformed.
        cond_dim: Width of the conditioning vector.
        hidden: Width of the subnetwork.
        clamp_scale: Bound on the log-scale, applied as ``tanh(s) * clamp_scale``.
            This defines the transform, so it must match the value used during
            training or the loaded weights compute a different function.
    """

    def __init__(self, channels, cond_dim, hidden=256, clamp_scale=TRAIN_CLAMP_SCALE):
        super().__init__()
        self.c_split = channels // 2
        self.clamp_scale = clamp_scale
        c_pass = channels - self.c_split
        self.net = nn.Sequential(
            nn.Linear(c_pass + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * self.c_split),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, cond):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        st = self.net(torch.cat([x2, cond], dim=1))
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * self.clamp_scale
        y1 = x1 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([y1, x2], dim=1), log_det


class CondFlow(nn.Module):
    """Stack of conditional coupling blocks with a fixed channel rotation."""

    def __init__(self, channels, cond_dim, n_blocks=4, hidden=256,
                 clamp_scale=TRAIN_CLAMP_SCALE):
        super().__init__()
        self.blocks = nn.ModuleList([
            CondCouplingBlock(channels, cond_dim, hidden, clamp_scale)
            for _ in range(n_blocks)
        ])
        self.register_buffer(
            'perm_idx',
            torch.tensor([(i + channels // 2) % channels for i in range(channels)])
        )

    def forward(self, x, cond):
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        for blk in self.blocks:
            x, ld = blk(x, cond)
            log_det_sum = log_det_sum + ld
            x = x.index_select(1, self.perm_idx)
        return x, log_det_sum


class CFlowPixelHead(nn.Module):
    """CFLOW-AD pixel head: one position-conditioned flow per feature level."""

    def __init__(self, layer_channels, cond_dim=64, n_blocks=4, hidden=256,
                 clamp_scale=TRAIN_CLAMP_SCALE):
        super().__init__()
        self.cond_dim = cond_dim
        self.clamp_scale = clamp_scale
        self.flows = nn.ModuleList([
            CondFlow(ch, cond_dim, n_blocks=n_blocks, hidden=hidden,
                     clamp_scale=clamp_scale)
            for ch in layer_channels
        ])

    def nll_per_level(self, feats_per_level, per_channel=False, chunk_size=4096):
        """Compute the negative log-likelihood map for each feature level.

        Args:
            feats_per_level: List of ``(B, C, H, W)`` feature maps.
            per_channel: Divide each level's NLL by its channel count, putting
                levels of different width on a comparable scale.
            chunk_size: Number of spatial positions per forward chunk. Chunking
                bounds peak VRAM; results are unaffected because every position
                is processed independently. ``None`` disables it.

        Returns:
            List of ``(B, H, W)`` NLL tensors, one per level.
        """
        maps = []
        for feat, flow in zip(feats_per_level, self.flows):
            B, C, H, W = feat.shape
            pe = pos_encoding(H, W, dim=self.cond_dim, device=feat.device)
            pe = pe.unsqueeze(0).expand(B, -1, -1)
            x = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            cond = pe.reshape(B * H * W, self.cond_dim)

            step = chunk_size or x.shape[0]
            nll_chunks = []
            for i in range(0, x.shape[0], step):
                z, log_det = flow(x[i:i + step], cond[i:i + step])
                log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * C * math.log(2 * math.pi)
                nll = -(log_prob + log_det)
                nll_chunks.append(nll / C if per_channel else nll)

            maps.append(torch.cat(nll_chunks, dim=0).view(B, H, W))
        return maps

    def score_map(self, feats_per_level, img_size, mode='raw',
                  level_stats=None, levels=None, chunk_size=4096, pos_stats=None):
        """Aggregate per-level NLL maps into a single anomaly map.

        Args:
            feats_per_level: List of ``(B, C, H, W)`` feature maps.
            img_size: Side length the final map is upsampled to.
            mode: One of :data:`SCORE_MODES`.
            level_stats: Required for ``per_level_std``; list of ``(mean, std)``
                scalars produced by :func:`compute_level_stats`.
            levels: Optional list of level indices to keep (e.g. ``[1, 2]`` to
                drop L1). Defaults to all levels.
            chunk_size: Forwarded to :meth:`nll_per_level`.
            pos_stats: Optional per-level positional calibration from
                :func:`compute_positional_stats`; see :meth:`aggregate_nll`.

        Returns:
            Tensor of shape ``(B, img_size, img_size)``.
        """
        if mode not in SCORE_MODES:
            raise ValueError(f"unknown score mode {mode!r}; expected one of {SCORE_MODES}")
        per_channel = mode == 'per_level_minmax'
        nll = self.nll_per_level(feats_per_level, per_channel=per_channel, chunk_size=chunk_size)
        channels = [f.shape[1] for f in feats_per_level]
        return self.aggregate_nll(nll, channels, img_size, mode=mode,
                                  level_stats=level_stats, levels=levels,
                                  pos_stats=pos_stats)

    @staticmethod
    def aggregate_nll(nll_per_level, channels, img_size, mode='raw',
                      level_stats=None, levels=None, pos_stats=None):
        """Turn already-computed per-level NLL maps into one anomaly map.

        Split out of :meth:`score_map` so that an ablation can run the flow
        once and re-aggregate the cached maps under every post-processing
        variant.

        Args:
            nll_per_level: List of ``(B, H, W)`` NLL maps. For
                ``per_level_minmax`` they must have been computed with
                ``per_channel=True`` (min-max is scale invariant, so this only
                matters for bit-exact reproduction of the historical numbers).
            channels: Channel count of each level (needed by ``per_level_prob``).
            img_size: Side length the final map is upsampled to.
            mode: One of :data:`SCORE_MODES`.
            level_stats: See :meth:`score_map`.
            levels: See :meth:`score_map`.
            pos_stats: Optional list, one entry per level, each either ``None``
                or a ``(mean_map, std_map)`` pair of ``(H, W)`` tensors. Where
                given, the level is standardized position-wise before
                aggregation. Must be in the same units as ``nll_per_level``.

        Returns:
            Tensor of shape ``(B, img_size, img_size)``.
        """
        if mode not in SCORE_MODES:
            raise ValueError(f"unknown score mode {mode!r}; expected one of {SCORE_MODES}")

        def _up(t):
            return F.interpolate(t.unsqueeze(1), size=(img_size, img_size),
                                 mode='bilinear', align_corners=False).squeeze(1)

        def _select(seq):
            return seq if levels is None else [seq[i] for i in levels]

        nll = list(nll_per_level)
        if pos_stats is not None:
            calibrated = []
            for m, st in zip(nll, pos_stats):
                if st is None:
                    calibrated.append(m)
                else:
                    mu, sd = st
                    calibrated.append((m - mu.to(m.device)) / (sd.to(m.device) + 1e-8))
            nll = calibrated

        if mode == 'raw':
            return sum(_up(m) for m in _select(nll))

        if mode == 'per_level_minmax':
            normalized = []
            for m in _select(nll):
                flat = m.view(m.shape[0], -1)
                mn = flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
                mx = flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
                normalized.append((m - mn) / (mx - mn + 1e-8))
            return _up(sum(normalized).cpu())

        if mode == 'per_level_prob':
            probs = []
            for m, C in zip(_select(nll), _select(list(channels))):
                probs.append(torch.exp(F.logsigmoid(-m) / C))
            return -sum(_up(p) for p in probs)

        if level_stats is None:
            raise ValueError("mode 'per_level_std' requires level_stats")
        standardized = [(m - mu) / (sd + 1e-8) for m, (mu, sd) in zip(nll, level_stats)]
        return sum(_up(m) for m in _select(standardized))


@torch.no_grad()
def compute_level_stats(backbone, cflow, loader, device, chunk_size=4096, desc='level stats',
                        levels=None, feat_stats=None):
    """Estimate scalar NLL mean/std per level over normal training images.

    Used by the ``per_level_std`` score mode. A single scalar pair per level is
    deliberate: per-position statistics were measured to generalize poorly for
    L1 (train/test profile correlation 0.70 against 0.99 for L2/L3), because its
    border bias is content-dependent rather than a fixed geometric offset.

    Args:
        backbone: Feature extractor.
        cflow: Trained pixel head.
        loader: Dataloader over defect-free training images.
        device: Device to run inference on.
        chunk_size: Forwarded to :meth:`CFlowPixelHead.nll_per_level`.
        desc: Progress bar label.
        levels: Feature levels the head was trained on (``None`` = all three).
        feat_stats: Optional per-channel standardization from
            :func:`compute_feature_stats`, applied before the flow.

    Returns:
        List of ``(mean, std)`` float pairs, one per level.
    """
    sums, sqs, counts = None, None, 0
    for data in tqdm(loader, desc=desc, leave=False):
        images = data[0] if isinstance(data, (list, tuple)) else data
        images = images.to(device)
        if images.dim() == 5:
            images = images[:, 0]
        _, feats = backbone(images)
        feats = apply_feature_stats(select_levels(feats, levels), feat_stats)
        nll = cflow.nll_per_level(feats, per_channel=False, chunk_size=chunk_size)
        if sums is None:
            sums = [0.0] * len(nll)
            sqs = [0.0] * len(nll)
        for i, m in enumerate(nll):
            sums[i] += m.sum().item()
            sqs[i] += (m ** 2).sum().item()
        counts += nll[0].numel()

    stats = []
    for s, q in zip(sums, sqs):
        mean = s / counts
        var = max(q / counts - mean ** 2, 0.0)
        stats.append((mean, math.sqrt(var)))
    return stats


def _unpack_images(data, device):
    images = data[0] if isinstance(data, (list, tuple)) else data
    images = images.to(device)
    if images.dim() == 5:
        images = images[:, 0]
    return images


@torch.no_grad()
def compute_positional_stats(backbone, cflow, loader, device, per_channel=False,
                             calibrate=None, feat_stats=None, levels=None,
                             chunk_size=4096, desc='positional stats'):
    """Estimate a per-position NLL mean/std map for each level on normal data.

    Position-wise calibration was measured to hurt L1 (its border bias is
    content dependent; train/test profile correlation 0.70) but L2/L3 profiles
    transfer well (0.99). ``calibrate`` selects which levels get a map; the
    others receive ``None`` so :meth:`CFlowPixelHead.aggregate_nll` leaves them
    untouched.

    Args:
        backbone: Feature extractor.
        cflow: Trained pixel head.
        loader: Dataloader over defect-free training images.
        device: Device to run inference on.
        per_channel: Must match the ``per_channel`` used when the maps being
            calibrated are computed.
        calibrate: Iterable of level indices (relative to ``cflow.flows``) to
            calibrate. ``None`` calibrates every level.
        feat_stats: Optional per-channel feature standardization from
            :func:`compute_feature_stats`, applied before the flow.
        levels: Feature levels the head was trained on (``None`` = all three);
            applied before ``feat_stats``.
        chunk_size: Forwarded to :meth:`CFlowPixelHead.nll_per_level`.
        desc: Progress bar label.

    Returns:
        List with one entry per level: ``(mean_map, std_map)`` CPU tensors of
        shape ``(H, W)`` or ``None`` for levels not calibrated.
    """
    n_levels = len(cflow.flows)
    calibrate = set(range(n_levels)) if calibrate is None else set(calibrate)
    sums, sqs, count = [None] * n_levels, [None] * n_levels, 0
    for data in tqdm(loader, desc=desc, leave=False):
        images = _unpack_images(data, device)
        _, feats = backbone(images)
        feats = apply_feature_stats(select_levels(feats, levels), feat_stats)
        nll = cflow.nll_per_level(feats, per_channel=per_channel, chunk_size=chunk_size)
        for i, m in enumerate(nll):
            if i not in calibrate:
                continue
            s = m.sum(dim=0)
            q = (m ** 2).sum(dim=0)
            sums[i] = s if sums[i] is None else sums[i] + s
            sqs[i] = q if sqs[i] is None else sqs[i] + q
        count += nll[0].shape[0]

    stats = []
    for i in range(n_levels):
        if sums[i] is None:
            stats.append(None)
            continue
        mean = sums[i] / count
        var = (sqs[i] / count - mean ** 2).clamp_min(0.0)
        stats.append((mean.cpu(), var.sqrt().cpu()))
    return stats


@torch.no_grad()
def compute_feature_stats(backbone, loader, device, desc='feature stats', norm_levels=None):
    """Per-channel mean/std of each feature level over normal training images.

    AlexNet ReLU features are unnormalized and heavy-tailed; standardizing them
    before the flow (PaDiM-style) puts every channel on the same scale and
    keeps the coupling subnetworks in their well-conditioned regime.

    Args:
        backbone: Feature extractor.
        loader: Dataloader over defect-free training images.
        device: Device to run inference on.
        desc: Progress bar label.
        norm_levels: Optional iterable of original level indices (0=L1, 1=L2, 2=L3)
            to standardize. Levels not in this set receive ``None`` in the returned
            list, so :func:`apply_feature_stats` leaves them unstandardized.
            Defaults to ``None`` (standardize all three levels).

    Returns:
        List of ``(mean, std)`` CPU tensors of shape ``(C,)``, or ``None``, one per level.
    """
    sums, sqs, count = None, None, 0
    norm_set = set(norm_levels) if norm_levels is not None else None
    for data in tqdm(loader, desc=desc, leave=False):
        images = _unpack_images(data, device)
        _, feats = backbone(images)
        if sums is None:
            sums = [torch.zeros(f.shape[1], device=device, dtype=torch.float64)
                    if (norm_set is None or i in norm_set) else None
                    for i, f in enumerate(feats)]
            sqs = [torch.zeros_like(s) if s is not None else None for s in sums]
        for i, f in enumerate(feats):
            if sums[i] is None:
                continue
            sums[i] += f.double().sum(dim=(0, 2, 3))
            sqs[i] += (f.double() ** 2).sum(dim=(0, 2, 3))
        count += feats[0].shape[0] * feats[0].shape[2] * feats[0].shape[3]

    stats = []
    for s, q in zip(sums, sqs):
        if s is None:
            stats.append(None)
        else:
            mean = s / count
            var = (q / count - mean ** 2).clamp_min(0.0)
            stats.append((mean.float().cpu(), var.sqrt().float().cpu()))
    return stats


def apply_feature_stats(feats, feat_stats, eps=1e-6):
    """Standardize each level per channel; ``feat_stats=None`` is a no-op.

    If an entry in ``feat_stats`` is ``None``, that level is passed through
    unmodified, supporting selective standardization (e.g. L1 only).
    """
    if feat_stats is None:
        return feats
    out = []
    for f, st in zip(feats, feat_stats):
        if st is None:
            out.append(f)
        else:
            mu, sd = st
            mu = mu.to(f.device).view(1, -1, 1, 1)
            sd = sd.to(f.device).view(1, -1, 1, 1)
            out.append((f - mu) / (sd + eps))
    return out


def select_levels(feats, levels):
    """Keep only the feature levels a head was trained on (``None`` = all)."""
    if levels is None:
        return list(feats)
    return [feats[i] for i in levels]


# ═════════════════════════════════════════════════════════════════════════════════
# Test-time augmentation
# ═════════════════════════════════════════════════════════════════════════════════

# Averaging the map over mirrored views symmetrizes the positional prior the
# flow learned, which is why it helps most on the levels that depend on context.
TTA_MODES = {
    'none': ('id',),
    'hflip': ('id', 'h'),
    'flips': ('id', 'h', 'v', 'hv'),
}


def flip_tensor(x, mode):
    """Apply one TTA flip to the last two dimensions; every mode is its own inverse."""
    if mode == 'h':
        return x.flip(-1)
    if mode == 'v':
        return x.flip(-2)
    if mode == 'hv':
        return x.flip(-1).flip(-2)
    return x


@torch.no_grad()
def nll_with_tta(backbone, cflow, images, levels=None, feat_stats=None, tta='none',
                 per_channel=False, chunk_size=4096):
    """Per-level NLL maps averaged over flip TTA.

    Each view is flipped back before averaging, so the result stays registered
    with the ground-truth mask.

    Args:
        backbone: Frozen feature extractor.
        cflow: Trained pixel head.
        images: Image batch shaped ``(B, 3, H, W)``.
        levels: Feature levels the head was trained on (``None`` = all three).
        feat_stats: Optional per-channel standardization applied before the flow.
        tta: Key of :data:`TTA_MODES`.
        per_channel: Divide each level's NLL by its channel count. Must be
            ``True`` to reproduce ``per_level_minmax`` exactly.
        chunk_size: Forwarded to :meth:`CFlowPixelHead.nll_per_level`.

    Returns:
        List of ``(B, H, W)`` NLL tensors, one per trained level.
    """
    if tta not in TTA_MODES:
        raise ValueError(f"unknown tta mode {tta!r}; expected one of {tuple(TTA_MODES)}")
    modes = TTA_MODES[tta]
    acc = None
    for mode in modes:
        _, feats = backbone(flip_tensor(images, mode))
        feats = apply_feature_stats(select_levels(feats, levels), feat_stats)
        nll = cflow.nll_per_level(feats, per_channel=per_channel, chunk_size=chunk_size)
        nll = [flip_tensor(m, mode) for m in nll]
        acc = nll if acc is None else [a + m for a, m in zip(acc, nll)]
    return [a / len(modes) for a in acc]


# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_smooth(score_map, sigma=4.0):
    """Blur each map in a batch. ``sigma <= 0`` returns the input unchanged."""
    if sigma <= 0:
        return score_map
    out = np.empty_like(score_map)
    for b in range(score_map.shape[0]):
        out[b] = gaussian_filter(score_map[b], sigma=sigma)
    return out


def minmax_norm(arr, eps=1e-8):
    """Scale each map in a batch to [0, 1] independently."""
    flat = arr.reshape(arr.shape[0], -1)
    mn = flat.min(axis=1, keepdims=True)
    mx = flat.max(axis=1, keepdims=True)
    return ((flat - mn) / (mx - mn + eps)).reshape(arr.shape)


def border_mask(size, margin):
    """Boolean mask that is ``False`` on a ``margin``-wide frame.

    Real defects sit near the image centre while the strongest false positives
    concentrate on the frame, so excluding it from pixel metrics removes a
    systematic bias rather than cherry-picking easy pixels. Report the margin
    whenever you report the metric.
    """
    keep = np.ones((size, size), dtype=bool)
    if margin > 0:
        keep[:margin] = keep[-margin:] = False
        keep[:, :margin] = keep[:, -margin:] = False
    return keep


def image_score_from_map(score_map, mode='max', top_percent=1.0):
    """Reduce a pixel anomaly map to one score per image.

    Args:
        score_map: Array shaped ``(B, H, W)``.
        mode: ``max`` (historical) or ``topk`` (mean of the highest scoring
            pixels). ``max`` is decided by a single pixel, so one border false
            positive can dominate an entire image.
        top_percent: Percentage of pixels averaged when ``mode='topk'``.

    Returns:
        Array of shape ``(B,)``.
    """
    flat = score_map.reshape(score_map.shape[0], -1)
    if mode == 'max':
        return flat.max(axis=1)
    if mode != 'topk':
        raise ValueError(f"unknown image score mode {mode!r}")
    k = max(1, int(flat.shape[1] * top_percent / 100.0))
    return -np.partition(-flat, k - 1, axis=1)[:, :k].mean(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint metadata
# ═══════════════════════════════════════════════════════════════════════════════

def infer_n_blocks(state_dict):
    """Recover the coupling-block count from a CFLOW state dict."""
    ids = {k.split('.')[3] for k in state_dict if k.startswith('flows.0.blocks.')}
    return len(ids)


def save_cflow_checkpoint(path, cflow, hparams, extra=None):
    """Persist a CFLOW head together with the hyperparameters needed to score it.

    Storing ``out_size`` and ``score_mode`` is what prevents the silent
    train/eval mismatch: neither is recoverable from the weights alone.
    """
    payload = {'cflow_state_dict': cflow.state_dict(),
               'cflow_hparams': {k: hparams.get(k) for k in CFLOW_HPARAM_KEYS}}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def read_cflow_hparams(ckpt):
    """Read stored CFLOW hyperparameters, returning ``{}`` for legacy files."""
    hp = ckpt.get('cflow_hparams') or {}
    return {k: v for k, v in hp.items() if v is not None}


def resolve_hparam(name, cli_value, ckpt_hparams, fallback, warn=True):
    """Resolve one hyperparameter as CLI > checkpoint > fallback.

    Legacy checkpoints carry no metadata, so falling back is unavoidable; the
    warning exists because that fallback is exactly the case that silently
    produced wrong scores before.

    Returns:
        Tuple ``(value, source)``.
    """
    if cli_value is not None:
        return cli_value, 'cli'
    if name in ckpt_hparams:
        return ckpt_hparams[name], 'checkpoint'
    if warn:
        print(f"  WARNING: '{name}' not stored in this checkpoint; falling back to "
              f"{fallback!r}. Pass --{name} explicitly if training used another value.")
    return fallback, 'fallback'
