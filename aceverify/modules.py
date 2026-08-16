"""Building blocks for multi-domain deepfake detection.

Three families of module live here:

* Frequency domain (:class:`FrequencyStream`) -- DCT log-spectra plus fixed SRM
  high-pass residuals. Generator fingerprints (GAN upsampling checkerboards,
  diffusion denoiser spectra) sit in bands that an RGB-only backbone averages away.
* Localized attention (:class:`PatchArtifactAttention`) -- pools ViT patch tokens
  with a learned query and emits a per-patch artifact score, so blending seams and
  face-swap boundaries survive pooling instead of being diluted by 196 patches.
* Temporal coherence (:class:`TemporalCoherenceEncoder`,
  :class:`PatchTemporalIncoherence`) -- cross-frame attention over clip-level
  features and a patch-resolved frame-to-frame change map.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classic SRM high-pass residual kernels used in image forensics. They suppress
# scene content and leave the noise residual where synthesis artifacts live.
_SRM_KERNELS = [
    [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ],
    [
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ],
]
_SRM_NORMALIZERS = [2.0, 4.0, 12.0]


def dct_1d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II along the last dimension, evaluated with an FFT."""
    n = x.shape[-1]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    vc = torch.fft.fft(v, dim=-1)
    k = torch.arange(n, device=x.device, dtype=x.dtype) * (-math.pi / (2 * n))
    out = vc.real * torch.cos(k) - vc.imag * torch.sin(k)

    scale = torch.full((n,), 1.0 / math.sqrt(n / 2.0), device=x.device, dtype=x.dtype)
    scale[0] = 1.0 / math.sqrt(n)
    return out * scale


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal 2D DCT-II over the trailing (H, W) dimensions."""
    return dct_1d(dct_1d(x).transpose(-1, -2)).transpose(-1, -2)


def damp_residual_branches_(encoder: nn.TransformerEncoder, gamma: float = 0.1):
    """Shrink each residual branch's output projection so blocks start near identity.

    A fresh pre-norm block is still a large random perturbation at step 0, so
    stacking one in front of a pretrained backbone corrupts its features before
    they reach the classifier. Scaling rather than zeroing matters: at exactly zero
    the block's output stops depending on its input, which cuts gradient to
    everything feeding it.
    """
    with torch.no_grad():
        for layer in encoder.layers:
            layer.self_attn.out_proj.weight.mul_(gamma)
            layer.self_attn.out_proj.bias.mul_(gamma)
            layer.linear2.weight.mul_(gamma)
            layer.linear2.bias.mul_(gamma)
    return encoder


class SqueezeExcite(nn.Module):
    """Dynamic channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.sigmoid(self.fc(x.mean(dim=(-2, -1))))
        return x * weights[..., None, None]


class FrequencyBandGate(nn.Module):
    """Learnable soft mask over radial DCT frequency bands.

    Acts as the frequency-domain masking regularizer: the network chooses which
    bands to trust rather than inheriting whatever band a single training
    generator happens to corrupt.
    """

    def __init__(self, size: int, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands

        coords = torch.arange(size, dtype=torch.float32) / max(1, size - 1)
        radius = torch.sqrt(coords[:, None] ** 2 + coords[None, :] ** 2) / math.sqrt(2.0)
        band_idx = torch.clamp((radius * num_bands).long(), max=num_bands - 1)
        masks = F.one_hot(band_idx, num_bands).permute(2, 0, 1).float()

        self.register_buffer("band_masks", masks.unsqueeze(0), persistent=False)
        self.register_buffer("band_area", masks.sum(dim=(-2, -1)).clamp(min=1.0), persistent=False)
        self.gate = nn.Sequential(
            nn.Linear(num_bands, num_bands * 4),
            nn.GELU(),
            nn.Linear(num_bands * 4, num_bands),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        energy = (spectrum * self.band_masks).sum(dim=(-2, -1)) / self.band_area
        gates = 2.0 * torch.sigmoid(self.gate(energy))
        gate_map = (gates[..., None, None] * self.band_masks).sum(dim=1, keepdim=True)
        return spectrum * gate_map


class FrequencyStream(nn.Module):
    """Per-frame descriptor built from DCT log-spectra and SRM noise residuals.

    Input frames are expected already normalized with ``mean``/``std``; the module
    inverts that to recover [0, 1] pixels, because both the DCT magnitude scale and
    the SRM residuals are only meaningful on the original intensity range.
    """

    def __init__(
        self,
        out_dim: int = 256,
        img_size: int = 224,
        num_bands: int = 8,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1), persistent=False
        )

        srm = torch.tensor(_SRM_KERNELS, dtype=torch.float32)
        srm = srm / torch.tensor(_SRM_NORMALIZERS).view(-1, 1, 1)
        self.register_buffer("srm", srm.unsqueeze(1), persistent=False)

        self.band_gate = FrequencyBandGate(img_size, num_bands=num_bands)

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
            )

        self.encoder = nn.Sequential(
            block(4, 32),
            block(32, 64),
            SqueezeExcite(64),
            block(64, 128),
            block(128, 192),
            SqueezeExcite(192),
            nn.Conv2d(192, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.out_dim = out_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # FFT and the wide-dynamic-range log are done in fp32; under bf16/fp16
        # autocast the spectrum saturates and gradients vanish.
        with torch.amp.autocast(device_type=frames.device.type, enabled=False):
            pixels = frames.float() * self.std + self.mean
            gray = (pixels * self.luma).sum(dim=1, keepdim=True)

            spectrum = torch.log1p(dct_2d(gray).abs())
            spectrum = (spectrum - spectrum.mean(dim=(-2, -1), keepdim=True)) / (
                spectrum.std(dim=(-2, -1), keepdim=True) + 1e-5
            )
            spectrum = self.band_gate(spectrum)

            residual = F.conv2d(gray, self.srm, padding=2).clamp(-3.0, 3.0)
            stacked = torch.cat([spectrum, residual], dim=1)

        return self.encoder(stacked.to(dtype=next(self.encoder.parameters()).dtype))


class PatchArtifactAttention(nn.Module):
    """Pools ViT patch tokens toward locally anomalous regions.

    Returns the pooled descriptor plus a per-patch artifact logit map, which both
    localizes the manipulation and feeds the sparsity term in the loss.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1, num_prefix_tokens: int = 1):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )
        self.merge = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, tokens: torch.Tensor):
        normed = self.norm(tokens)
        query = self.query.expand(normed.shape[0], -1, -1).to(dtype=normed.dtype)
        pooled, _ = self.attention(query, normed, normed, need_weights=False)

        # Score only the spatial patches: the CLS token has no location, so keeping
        # it would break the map's correspondence to the 14x14 image grid.
        patches = normed[:, self.num_prefix_tokens :]
        patch_logits = self.score(patches).squeeze(-1)
        weights = torch.softmax(patch_logits, dim=1).unsqueeze(-1)
        anomaly = (patches * weights).sum(dim=1)

        return self.merge(torch.cat([pooled.squeeze(1), anomaly], dim=-1)), patch_logits


class PatchTemporalIncoherence(nn.Module):
    """Patch-resolved frame-to-frame change, summarized per frame.

    Global frame embeddings only expose slow drift. Face swaps instead produce
    jitter confined to a few patches around the blending boundary, which shows up
    as a high-max / low-mean temporal change distribution.
    """

    def __init__(self, in_dim: int, proj_dim: int = 64, out_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, proj_dim)
        self.temperature = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim + 3),
            nn.Linear(proj_dim + 3, out_dim),
            nn.GELU(),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.proj(patch_tokens), dim=-1)

        delta = (z[:, 1:] - z[:, :-1]).pow(2).sum(-1)
        delta = F.pad(delta, (0, 0, 1, 0))

        weights = torch.softmax(delta * self.temperature.exp().clamp(max=50.0), dim=-1)
        focused = (z * weights.unsqueeze(-1)).sum(dim=2)
        stats = torch.stack([delta.mean(-1), delta.amax(-1), delta.std(-1)], dim=-1)

        return self.head(torch.cat([focused, stats], dim=-1))


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attention_weights = torch.softmax(self.attention(sequence), dim=1)
        return torch.sum(sequence * attention_weights, dim=1)


class TemporalCoherenceEncoder(nn.Module):
    """Cross-frame self-attention with an explicit first-difference stream."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        depth: int = 2,
        max_frames: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, max_frames, dim))
        nn.init.trunc_normal_(self.position, std=0.02)

        self.difference = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        # Non-zero LayerScale: at exactly zero this gate would block all gradient
        # into the difference branch and it would never start learning.
        self.difference_scale = nn.Parameter(torch.full((dim,), 0.1))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = damp_residual_branches_(
            nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        )
        self.recurrent = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.pool = TemporalAttentionPooling(dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        frames = sequence.shape[1]
        if frames > self.position.shape[1]:
            raise ValueError(
                f"Clip has {frames} frames but TemporalCoherenceEncoder was built for at most "
                f"{self.position.shape[1]}; raise max_frames."
            )

        delta = F.pad(sequence[:, 1:] - sequence[:, :-1], (0, 0, 1, 0))
        x = sequence + self.difference_scale.to(dtype=sequence.dtype) * self.difference(delta)
        x = x + self.position[:, :frames].to(dtype=sequence.dtype)

        x = self.encoder(x)
        x, _ = self.recurrent(x)
        return self.pool(x)


class ModalityFusion(nn.Module):
    """Attention fusion over a small set of modality tokens."""

    def __init__(self, dim: int, num_tokens: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.type_embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.type_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = damp_residual_branches_(
            nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens + self.type_embedding.to(dtype=tokens.dtype)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1).to(dtype=x.dtype), x], dim=1)
        return self.norm(self.encoder(x)[:, 0])
