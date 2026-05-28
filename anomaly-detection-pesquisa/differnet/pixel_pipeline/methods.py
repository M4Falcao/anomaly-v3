"""Four pixel-level anomaly detection methods sharing the same backbone.

D — PaDiM: per-position multivariate Gaussian, Mahalanobis distance.
E — PatchCore: memory bank of normal patch features + kNN distance.
F — FastFlow: convolutional normalizing flow per feature level.
G — CFLOW-AD: conditional normalizing flow with positional encoding.

Each method implements:
    fit(train_loader)         — build statistics / memory bank / train flow
    score(images) -> np ndarray (B, H, W) on cpu
    score_tensor(images) -> torch.Tensor (B, H, W) on device, differentiable
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from .backbone import MultiLayerAlexNet


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _to_device(x):
    return x.to(DEVICE, non_blocking=True)


# ═════════════════════════════════════════════════════════════════════════════
# D — PaDiM
# ═════════════════════════════════════════════════════════════════════════════
class PaDiM:
    """Per-position multivariate Gaussian over selected channels.

    Random channel sub-sampling reduces dimensionality (typical: 100 of 512).
    """

    def __init__(self, img_size=256, out_size=56, d_keep=100, seed=42):
        self.backbone = MultiLayerAlexNet(out_size=out_size).to(DEVICE).eval()
        self.img_size = img_size
        self.out_size = out_size
        self.d_keep = d_keep
        rng = np.random.RandomState(seed)
        self.channel_idx = torch.tensor(
            rng.choice(self.backbone.concat_channels, d_keep, replace=False),
            dtype=torch.long, device=DEVICE
        )
        self.mean = None       # (d, H*W)
        self.cov_inv = None    # (H*W, d, d)

    @torch.no_grad()
    def fit(self, train_loader):
        feats_all = []
        for batch in tqdm(train_loader, desc='[PaDiM] fit'):
            x = _to_device(batch)
            concat, _ = self.backbone(x)
            f = concat.index_select(1, self.channel_idx)  # (B, d, H, W)
            feats_all.append(f.cpu())
        feats = torch.cat(feats_all, dim=0)  # (N, d, H, W)
        N, d, H, W = feats.shape
        feats = feats.permute(2, 3, 1, 0).reshape(H * W, d, N)  # per-position

        mean = feats.mean(dim=-1)  # (H*W, d)
        cov = torch.empty(H * W, d, d)
        identity = torch.eye(d) * 0.01
        for i in range(H * W):
            centered = feats[i] - mean[i].unsqueeze(-1)
            cov[i] = (centered @ centered.T) / max(N - 1, 1) + identity
        self.mean = mean.to(DEVICE)
        self.cov_inv = torch.linalg.inv(cov).to(DEVICE)
        self.H, self.W = H, W

    def _maha_map(self, x):
        """Return (B, H, W) Mahalanobis distance map (differentiable)."""
        concat, _ = self.backbone(x)
        f = concat.index_select(1, self.channel_idx)  # (B, d, H, W)
        B, d, H, W = f.shape
        f = f.permute(2, 3, 0, 1).reshape(H * W, B, d)  # (HW, B, d)
        diff = f - self.mean.unsqueeze(1)  # (HW, B, d)
        # (HW, B, d) @ (HW, d, d) -> (HW, B, d), then dot with diff
        left = torch.einsum('hbd,hde->hbe', diff, self.cov_inv)
        dist = torch.sqrt(torch.clamp((left * diff).sum(-1), min=1e-12))  # (HW, B)
        dist = dist.T.reshape(B, H, W)
        return dist

    def score_tensor(self, x):
        dist = self._maha_map(x)  # (B, H, W)
        up = F.interpolate(dist.unsqueeze(1), size=(self.img_size, self.img_size),
                           mode='bilinear', align_corners=False).squeeze(1)
        return up

    @torch.no_grad()
    def score(self, x):
        return self.score_tensor(x).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# E — PatchCore
# ═════════════════════════════════════════════════════════════════════════════
class PatchCore:
    """Memory bank of normal patch features + kNN distance."""

    def __init__(self, img_size=256, out_size=56, coreset_ratio=0.1, k=3, seed=42):
        self.backbone = MultiLayerAlexNet(out_size=out_size).to(DEVICE).eval()
        self.img_size = img_size
        self.out_size = out_size
        self.coreset_ratio = coreset_ratio
        self.k = k
        self.seed = seed
        self.bank = None  # (M, d)

    @torch.no_grad()
    def fit(self, train_loader):
        feats_all = []
        for batch in tqdm(train_loader, desc='[PatchCore] fit'):
            x = _to_device(batch)
            concat, _ = self.backbone(x)
            # (B, C, H, W) -> (B*H*W, C)
            B, C, H, W = concat.shape
            f = concat.permute(0, 2, 3, 1).reshape(-1, C)
            feats_all.append(f.cpu())
        feats = torch.cat(feats_all, dim=0)  # (N*H*W, C)

        # Random subsample (lightweight coreset)
        n_select = max(int(len(feats) * self.coreset_ratio), 1024)
        n_select = min(n_select, len(feats))
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(feats), n_select, replace=False)
        self.bank = feats[idx].to(DEVICE)
        print(f"[PatchCore] memory bank: {self.bank.shape}")

    def _score_map(self, x):
        concat, _ = self.backbone(x)
        B, C, H, W = concat.shape
        f = concat.permute(0, 2, 3, 1).reshape(B * H * W, C)
        # Pairwise distances: chunked to fit memory
        chunk = 2048
        dists_min = torch.empty(B * H * W, device=DEVICE)
        for i in range(0, f.shape[0], chunk):
            d = torch.cdist(f[i:i+chunk], self.bank)  # (chunk, M)
            topk = torch.topk(d, k=self.k, dim=1, largest=False).values
            dists_min[i:i+chunk] = topk.mean(dim=1)
        score = dists_min.view(B, H, W)
        return score

    def score_tensor(self, x):
        s = self._score_map(x)
        up = F.interpolate(s.unsqueeze(1), size=(self.img_size, self.img_size),
                           mode='bilinear', align_corners=False).squeeze(1)
        return up

    @torch.no_grad()
    def score(self, x):
        return self.score_tensor(x).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# F — FastFlow (convolutional NF per feature level)
# ═════════════════════════════════════════════════════════════════════════════
class _Conv2dFlowBlock(nn.Module):
    """Simple 2D affine coupling block with conv subnets."""

    def __init__(self, channels, hidden=128, kernel_size=3):
        super().__init__()
        self.c_split = channels // 2
        c_pass = channels - self.c_split
        self.subnet = nn.Sequential(
            nn.Conv2d(c_pass, hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * self.c_split, kernel_size, padding=kernel_size // 2),
        )
        # Zero-init last conv for identity-init flow
        nn.init.zeros_(self.subnet[-1].weight)
        nn.init.zeros_(self.subnet[-1].bias)

    def forward(self, x):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        st = self.subnet(x2)
        s, t = st.chunk(2, dim=1)
        s = torch.tanh(s) * 2.0  # clamp scale
        y1 = x1 * torch.exp(s) + t
        log_det = s.flatten(1).sum(1)
        return torch.cat([y1, x2], dim=1), log_det


class _Conv2dFlow(nn.Module):
    def __init__(self, channels, n_blocks=4, hidden=128):
        super().__init__()
        # Alternate channel split direction via permutation
        self.blocks = nn.ModuleList([
            _Conv2dFlowBlock(channels, hidden) for _ in range(n_blocks)
        ])
        # Fixed channel permutations (alternate halves)
        self.register_buffer(
            'perm_idx',
            torch.tensor([(i + channels // 2) % channels for i in range(channels)])
        )

    def forward(self, x):
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        for i, blk in enumerate(self.blocks):
            x, ld = blk(x)
            log_det_sum = log_det_sum + ld
            x = x.index_select(1, self.perm_idx)
        return x, log_det_sum


class FastFlow:
    """Per-level 2D NF over backbone features (FastFlow-style)."""

    def __init__(self, img_size=256, out_size=56, n_blocks=4, hidden=128,
                 lr=1e-3, epochs=10):
        self.backbone = MultiLayerAlexNet(out_size=out_size).to(DEVICE).eval()
        self.img_size = img_size
        self.out_size = out_size
        self.epochs = epochs
        self.lr = lr
        self.flows = nn.ModuleList([
            _Conv2dFlow(c, n_blocks=n_blocks, hidden=hidden)
            for c in self.backbone.layer_channels
        ]).to(DEVICE)

    def _nll_per_level(self, feats_per_level):
        """Return list of (B, H, W) negative-log-prob maps per level."""
        maps = []
        for feat, flow in zip(feats_per_level, self.flows):
            z, log_det = flow(feat)
            # Standard normal log-prob
            log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * z.size(1) * math.log(2 * math.pi)
            # add log_det/HW broadcast (per-pixel approximation)
            log_det_per_pix = log_det / (z.size(2) * z.size(3))
            log_p_map = log_prob + log_det_per_pix.unsqueeze(-1).unsqueeze(-1)
            maps.append(-log_p_map)  # NLL map
        return maps

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.flows.parameters(), lr=self.lr)
        for ep in range(self.epochs):
            losses = []
            for batch in tqdm(train_loader, desc=f'[FastFlow] ep {ep+1}/{self.epochs}'):
                x = _to_device(batch)
                with torch.no_grad():
                    _, feats = self.backbone(x)
                opt.zero_grad()
                nll_maps = self._nll_per_level(feats)
                loss = sum(m.mean() for m in nll_maps)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"  loss: {np.mean(losses):.4f}")

    def score_tensor(self, x):
        _, feats = self.backbone(x)
        nll_maps = self._nll_per_level(feats)
        upsampled = [F.interpolate(m.unsqueeze(1),
                                   size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False).squeeze(1)
                     for m in nll_maps]
        return sum(upsampled)

    @torch.no_grad()
    def score(self, x):
        return self.score_tensor(x).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# G — CFLOW-AD (conditional NF with positional encoding)
# ═════════════════════════════════════════════════════════════════════════════
def _pos_encoding(h, w, dim=64, device=DEVICE):
    """2D sinusoidal positional encoding -> (h*w, dim)."""
    assert dim % 4 == 0
    d_quarter = dim // 4
    y_pos = torch.arange(h, device=device).float()
    x_pos = torch.arange(w, device=device).float()
    div = torch.exp(torch.arange(0, d_quarter, device=device).float() *
                    (-math.log(10000.0) / d_quarter))
    yy = y_pos.unsqueeze(1) * div.unsqueeze(0)  # (h, d_q)
    xx = x_pos.unsqueeze(1) * div.unsqueeze(0)  # (w, d_q)
    pe_y = torch.cat([yy.sin(), yy.cos()], dim=1)  # (h, 2 d_q)
    pe_x = torch.cat([xx.sin(), xx.cos()], dim=1)  # (w, 2 d_q)
    pe_y = pe_y.unsqueeze(1).expand(h, w, dim // 2)
    pe_x = pe_x.unsqueeze(0).expand(h, w, dim // 2)
    pe = torch.cat([pe_y, pe_x], dim=-1).reshape(h * w, dim)
    return pe


class _CondCouplingBlock(nn.Module):
    """Affine coupling conditioned on positional encoding."""

    def __init__(self, channels, cond_dim, hidden=256):
        super().__init__()
        self.c_split = channels // 2
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
        s = torch.tanh(s) * 2.0
        y1 = x1 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([y1, x2], dim=1), log_det


class _CondFlow(nn.Module):
    def __init__(self, channels, cond_dim, n_blocks=4, hidden=256):
        super().__init__()
        self.blocks = nn.ModuleList([
            _CondCouplingBlock(channels, cond_dim, hidden) for _ in range(n_blocks)
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


class CFLOW:
    """Position-conditional NF per feature level."""

    def __init__(self, img_size=256, out_size=56, cond_dim=64, n_blocks=4,
                 hidden=256, lr=1e-3, epochs=10):
        self.backbone = MultiLayerAlexNet(out_size=out_size).to(DEVICE).eval()
        self.img_size = img_size
        self.out_size = out_size
        self.cond_dim = cond_dim
        self.epochs = epochs
        self.lr = lr
        self.flows = nn.ModuleList([
            _CondFlow(c, cond_dim, n_blocks=n_blocks, hidden=hidden)
            for c in self.backbone.layer_channels
        ]).to(DEVICE)

    def _nll_per_level(self, feats_per_level):
        maps = []
        for feat, flow in zip(feats_per_level, self.flows):
            B, C, H, W = feat.shape
            pe = _pos_encoding(H, W, dim=self.cond_dim).unsqueeze(0).expand(B, -1, -1)  # (B, HW, cd)
            x = feat.permute(0, 2, 3, 1).reshape(B * H * W, C)
            cond = pe.reshape(B * H * W, self.cond_dim)
            z, log_det = flow(x, cond)
            log_prob = -0.5 * (z ** 2).sum(dim=1) - 0.5 * C * math.log(2 * math.pi)
            log_p = log_prob + log_det
            nll = (-log_p).view(B, H, W)
            maps.append(nll)
        return maps

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.flows.parameters(), lr=self.lr)
        for ep in range(self.epochs):
            losses = []
            for batch in tqdm(train_loader, desc=f'[CFLOW] ep {ep+1}/{self.epochs}'):
                x = _to_device(batch)
                with torch.no_grad():
                    _, feats = self.backbone(x)
                opt.zero_grad()
                nll_maps = self._nll_per_level(feats)
                loss = sum(m.mean() for m in nll_maps)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"  loss: {np.mean(losses):.4f}")

    def score_tensor(self, x):
        _, feats = self.backbone(x)
        nll_maps = self._nll_per_level(feats)
        upsampled = [F.interpolate(m.unsqueeze(1),
                                   size=(self.img_size, self.img_size),
                                   mode='bilinear', align_corners=False).squeeze(1)
                     for m in nll_maps]
        return sum(upsampled)

    @torch.no_grad()
    def score(self, x):
        return self.score_tensor(x).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# H — Standard DifferNet (GAP + NF, pixel via input-gradient)
# ═════════════════════════════════════════════════════════════════════════════
class _FCSubnet(nn.Module):
    """Fully connected subnet for coupling layers."""
    def __init__(self, size_in, size_out, internal_size=512, dropout=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size_in, internal_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(internal_size, internal_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(internal_size, size_out),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        return self.fc(x)


class _1DCouplingBlock(nn.Module):
    """1D affine coupling for vector NF."""
    def __init__(self, dim, internal=512, clamp=3.0):
        super().__init__()
        self.c_split = dim // 2
        c_pass = dim - self.c_split
        self.subnet = _FCSubnet(c_pass, 2 * self.c_split, internal)
        self.clamp = clamp

    def forward(self, x):
        x1, x2 = x[:, :self.c_split], x[:, self.c_split:]
        st = self.subnet(x2)
        s, t = st.chunk(2, dim=1)
        s = self.clamp * torch.tanh(s / self.clamp)
        y1 = x1 * torch.exp(s) + t
        log_det = s.sum(dim=1)
        return torch.cat([y1, x2], dim=1), log_det


class _VectorNF(nn.Module):
    """Simple 1D normalizing flow (permute + coupling)."""
    def __init__(self, dim, n_blocks=8, internal=512, clamp=3.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            _1DCouplingBlock(dim, internal, clamp) for _ in range(n_blocks)
        ])
        # Fixed permutations
        perms = []
        for i in range(n_blocks):
            perm = torch.randperm(dim)
            perms.append(perm)
        self.register_buffer('perms', torch.stack(perms))

    def forward(self, x):
        log_det_sum = torch.zeros(x.size(0), device=x.device)
        for i, blk in enumerate(self.blocks):
            x = x.index_select(1, self.perms[i])
            x, ld = blk(x)
            log_det_sum = log_det_sum + ld
        return x, log_det_sum


class DifferNetMethod:
    """Standard DifferNet: AlexNet features → GAP → NF.
    Pixel-level via input-gradient of NLL score.
    """

    def __init__(self, img_size=256, n_scales=3, n_coupling_blocks=8,
                 fc_internal=512, lr=2e-4, epochs=15, **kwargs):
        from torchvision.models import alexnet as _alexnet
        self.alexnet = _alexnet(pretrained=True).to(DEVICE).eval()
        for p in self.alexnet.parameters():
            p.requires_grad_(False)
        self.img_size = img_size
        self.n_scales = n_scales
        feat_dim = 256 * n_scales  # AlexNet features[-1] = 256 ch
        self.nf = _VectorNF(feat_dim, n_blocks=n_coupling_blocks,
                            internal=fc_internal).to(DEVICE)
        self.lr = lr
        self.epochs = epochs

    def _extract_features(self, x):
        """Multi-scale GAP features. x must have grad if needed."""
        y_cat = []
        for s in range(self.n_scales):
            x_s = torch.nn.functional.interpolate(
                x, size=self.img_size // (2 ** s)) if s > 0 else x
            feat = self.alexnet.features(x_s)
            y_cat.append(feat.mean(dim=(2, 3)))
        return torch.cat(y_cat, dim=1)

    def _nll(self, x):
        """Scalar NLL per sample (not reduced)."""
        y = self._extract_features(x)
        z, log_det = self.nf(y)
        nll = 0.5 * (z ** 2).sum(dim=1) - log_det
        return nll

    def fit(self, train_loader):
        opt = torch.optim.Adam(self.nf.parameters(), lr=self.lr,
                               betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-5)
        for ep in range(self.epochs):
            losses = []
            for batch in tqdm(train_loader, desc=f'[DifferNet] ep {ep+1}/{self.epochs}'):
                x = _to_device(batch)
                opt.zero_grad()
                nll = self._nll(x)
                loss = nll.mean()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"  loss: {np.mean(losses):.4f}")

    def score_tensor(self, x):
        """Pixel score via input-gradient of NLL."""
        x = x.clone().detach().requires_grad_(True)
        nll = self._nll(x)
        nll.sum().backward()
        grad = x.grad.abs().mean(dim=1)  # (B, H, W)
        return grad.detach()

    def score(self, x):
        return self.score_tensor(x).cpu().numpy()


# ═════════════════════════════════════════════════════════════════════════════
# I — SEDifferNet (SE attention blocks + NF, pixel via SE excitation maps)
# ═════════════════════════════════════════════════════════════════════════════
class _SEBlock(nn.Module):
    """Lightweight SE block (squeeze-excitation)."""
    def __init__(self, channels, reduction=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        squeeze = x.mean(dim=(2, 3))  # (B, C)
        excitation = self.fc(squeeze).view(B, C, 1, 1)
        return x * excitation


class SEDifferNetMethod:
    """SEDifferNet: AlexNet + SE blocks → GAP → NF.
    SE blocks are trained jointly (lower LR).
    Pixel map via SE excitation-weighted spatial features.
    """

    def __init__(self, img_size=256, n_scales=3, n_coupling_blocks=8,
                 fc_internal=512, lr=2e-4, epochs=15, **kwargs):
        from torchvision.models import alexnet as _alexnet
        self.alexnet = _alexnet(pretrained=True).to(DEVICE)
        for p in self.alexnet.parameters():
            p.requires_grad_(False)
        self.img_size = img_size
        self.n_scales = n_scales

        # SE blocks after specific layers (matching main model.py)
        self.se1 = _SEBlock(64, reduction=2).to(DEVICE)    # after features[1]
        self.se2 = _SEBlock(192, reduction=2).to(DEVICE)   # after features[5]
        self.se3 = _SEBlock(384, reduction=2).to(DEVICE)   # after features[7]
        self.se4 = _SEBlock(256, reduction=2).to(DEVICE)   # after features[11]

        feat_dim = 256 * n_scales
        self.nf = _VectorNF(feat_dim, n_blocks=n_coupling_blocks,
                            internal=fc_internal).to(DEVICE)
        self.lr = lr
        self.epochs = epochs

    def _forward_backbone(self, x_input):
        """Forward through AlexNet + SE, returns (feat_vector, se_info).
        se_info: list of (input_feat, excitation) per SE block per scale.
        """
        y_cat = []
        se_info = []  # [(input_feat, excitation), ...]

        for s in range(self.n_scales):
            x = torch.nn.functional.interpolate(
                x_input, size=self.img_size // (2 ** s)) if s > 0 else x_input

            # features[0] = Conv2d(3,64), features[1] = ReLU
            x = self.alexnet.features[0](x)
            x = self.alexnet.features[1](x)
            se_in1 = x
            x = self.se1(x)
            se_info.append((se_in1, self.se1.fc(se_in1.mean(dim=(2, 3)))))

            x = self.alexnet.features[2](x)  # MaxPool
            x = self.alexnet.features[3](x)  # Conv2d(64,192)
            x = self.alexnet.features[4](x)  # ReLU
            se_in2 = x
            x = self.se2(x)
            se_info.append((se_in2, self.se2.fc(se_in2.mean(dim=(2, 3)))))

            x = self.alexnet.features[5](x)  # MaxPool
            x = self.alexnet.features[6](x)  # Conv2d(192,384)
            x = self.alexnet.features[7](x)  # ReLU
            se_in3 = x
            x = self.se3(x)
            se_info.append((se_in3, self.se3.fc(se_in3.mean(dim=(2, 3)))))

            x = self.alexnet.features[8](x)  # Conv2d(384,256)
            x = self.alexnet.features[9](x)  # ReLU
            x = self.alexnet.features[10](x)  # Conv2d(256,256)
            x = self.alexnet.features[11](x)  # ReLU
            se_in4 = x
            x = self.se4(x)
            se_info.append((se_in4, self.se4.fc(se_in4.mean(dim=(2, 3)))))

            feat_s = self.alexnet.features[12](x)  # MaxPool
            y_cat.append(feat_s.mean(dim=(2, 3)))

        y = torch.cat(y_cat, dim=1)
        return y, se_info

    def _nll(self, x):
        y, _ = self._forward_backbone(x)
        z, log_det = self.nf(y)
        nll = 0.5 * (z ** 2).sum(dim=1) - log_det
        return nll

    def fit(self, train_loader):
        se_params = (list(self.se1.parameters()) + list(self.se2.parameters()) +
                     list(self.se3.parameters()) + list(self.se4.parameters()))
        opt = torch.optim.Adam([
            {'params': self.nf.parameters(), 'lr': self.lr},
            {'params': se_params, 'lr': self.lr * 0.5},
        ], betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-5)

        for ep in range(self.epochs):
            losses = []
            self.nf.train()
            self.se1.train(); self.se2.train()
            self.se3.train(); self.se4.train()
            for batch in tqdm(train_loader, desc=f'[SEDifferNet] ep {ep+1}/{self.epochs}'):
                x = _to_device(batch)
                opt.zero_grad()
                nll = self._nll(x)
                loss = nll.mean()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(f"  loss: {np.mean(losses):.4f}")
        self.nf.eval()
        self.se1.eval(); self.se2.eval()
        self.se3.eval(); self.se4.eval()

    def _se_anomaly_map(self, x):
        """SE excitation-weighted spatial anomaly map."""
        with torch.no_grad():
            _, se_info = self._forward_backbone(x)
        score_map = None
        for (feat, excit) in se_info:
            # feat: (B, C, H, W), excit: (B, C)
            weighted = feat * excit.unsqueeze(-1).unsqueeze(-1)
            spatial = weighted.sum(dim=1)  # (B, H, W)
            up = torch.nn.functional.interpolate(
                spatial.unsqueeze(1),
                size=(self.img_size, self.img_size),
                mode='bilinear', align_corners=False
            ).squeeze(1)
            if score_map is None:
                score_map = up
            else:
                score_map = score_map + up
        return score_map

    def score_tensor(self, x):
        return self._se_anomaly_map(x)

    @torch.no_grad()
    def score(self, x):
        return self.score_tensor(x).cpu().numpy()


# Registry
METHODS = {
    'D_PaDiM': PaDiM,
    'E_PatchCore': PatchCore,
    'F_FastFlow': FastFlow,
    'G_CFLOW': CFLOW,
    'H_DifferNet': DifferNetMethod,
    'I_SEDifferNet': SEDifferNetMethod,
}
