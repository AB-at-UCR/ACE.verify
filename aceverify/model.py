"""Multi-domain deepfake detector.

Four streams feed an attention fusion head:

* spatial   -- ViT-B/16 patch tokens pooled by :class:`PatchArtifactAttention`
* frequency -- DCT log-spectra + SRM noise residuals (:class:`FrequencyStream`)
* motion    -- patch-resolved frame-to-frame incoherence
* audio     -- EfficientNet-B0 over the log-mel spectrogram

Each visual stream produces a per-frame sequence that is collapsed by
:class:`TemporalCoherenceEncoder` (cross-frame attention + BiGRU).
"""

import torch
import torch.nn as nn
import timm

from .baseline_model import ACEVerifyBaseline
from .modules import (
    FrequencyStream,
    ModalityFusion,
    PatchArtifactAttention,
    PatchTemporalIncoherence,
    TemporalCoherenceEncoder,
)


class SpectrogramEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0_ns",
            pretrained=True,
            in_chans=1,
            num_classes=0,
        )
        feature_dim = self.backbone.num_features
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, audio_spec: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(audio_spec))


class ACEVerifyModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        unfrozen_blocks: int = 4,
        temporal_depth: int = 2,
        dropout: float = 0.2,
        use_frequency: bool = True,
        use_motion: bool = True,
        use_audio: bool = True,
    ):
        super().__init__()
        self.use_frequency = use_frequency
        self.use_motion = use_motion
        self.use_audio = use_audio
        self.embed_dim = embed_dim

        self.video_model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.1,
        )
        self.num_prefix_tokens = self.video_model.num_prefix_tokens
        vit_dim = self.video_model.embed_dim

        for param in self.video_model.parameters():
            param.requires_grad = False
        if unfrozen_blocks > 0:
            for param in self.video_model.blocks[-unfrozen_blocks:].parameters():
                param.requires_grad = True
            self.video_model.norm.requires_grad_(True)

        self.patch_attention = PatchArtifactAttention(
            vit_dim, num_heads=8, dropout=dropout / 2, num_prefix_tokens=self.num_prefix_tokens
        )
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(vit_dim), nn.Linear(vit_dim, embed_dim), nn.GELU()
        )
        self.spatial_temporal = TemporalCoherenceEncoder(
            embed_dim, depth=temporal_depth, dropout=dropout / 2
        )

        cfg = self.video_model.pretrained_cfg
        if use_frequency:
            self.frequency_stream = FrequencyStream(
                out_dim=embed_dim, mean=cfg["mean"], std=cfg["std"]
            )
            self.frequency_temporal = TemporalCoherenceEncoder(
                embed_dim, depth=1, dropout=dropout / 2
            )

        if use_motion:
            self.motion_stream = PatchTemporalIncoherence(vit_dim, proj_dim=64, out_dim=embed_dim)
            self.motion_temporal = TemporalCoherenceEncoder(embed_dim, depth=1, dropout=dropout / 2)

        if use_audio:
            self.audio_encoder = SpectrogramEncoder(embed_dim, dropout=dropout)

        num_tokens = 1 + int(use_frequency) + int(use_motion) + int(use_audio)
        self.fusion = ModalityFusion(embed_dim, num_tokens, dropout=dropout / 2)

        # Explicit spatial/frequency interaction terms; a pure sum inside the fusion
        # transformer cannot express the cross-domain products that separate a clean
        # face from one whose spectrum disagrees with its pixels.
        if use_frequency:
            self.interaction = nn.Sequential(
                nn.LayerNorm(2 * embed_dim), nn.Linear(2 * embed_dim, embed_dim), nn.GELU()
            )

        # The auxiliary streams are added to the spatial feature through a LayerScale
        # gate rather than replacing it, so the classifier always keeps a direct path
        # to the pretrained ViT representation; routing everything through freshly
        # initialized fusion layers measurably slowed convergence. The init must stay
        # non-zero: at exactly zero the gate blocks all gradient into the streams.
        self.stream_scale = nn.Parameter(torch.full((embed_dim,), 0.1))
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    @staticmethod
    def _prepare_audio(audio_spec, batch, reference):
        if audio_spec is None:
            return reference.new_zeros((batch, 1, 224, 224))
        if audio_spec.dim() == 3:
            audio_spec = audio_spec.unsqueeze(1)
        elif audio_spec.dim() == 5:
            audio_spec = audio_spec.squeeze(1)
        return audio_spec.to(dtype=reference.dtype)

    def forward(self, video: torch.Tensor, audio_spec=None, return_aux: bool = False):
        B, C, T, H, W = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        tokens = self.video_model.forward_features(frames)
        patch_tokens = tokens[:, self.num_prefix_tokens :]

        spatial, patch_logits = self.patch_attention(tokens)
        spatial = self.spatial_projection(spatial).reshape(B, T, self.embed_dim)
        spatial_vec = self.spatial_temporal(spatial)

        modality_tokens = [spatial_vec]
        frequency_vec = None

        if self.use_frequency:
            frequency = self.frequency_stream(frames).reshape(B, T, self.embed_dim)
            frequency_vec = self.frequency_temporal(frequency)
            modality_tokens.append(frequency_vec)

        if self.use_motion:
            grid = patch_tokens.reshape(B, T, patch_tokens.shape[1], patch_tokens.shape[2])
            motion = self.motion_stream(grid)
            modality_tokens.append(self.motion_temporal(motion))

        if self.use_audio:
            spec = self._prepare_audio(audio_spec, B, frames)
            modality_tokens.append(self.audio_encoder(spec))

        extra = self.fusion(torch.stack(modality_tokens, dim=1))
        if frequency_vec is not None:
            extra = extra + self.interaction(
                torch.cat([spatial_vec * frequency_vec, spatial_vec - frequency_vec], dim=-1)
            )

        embedding = self.embedding_head(
            spatial_vec + self.stream_scale.to(dtype=extra.dtype) * extra
        )
        logits = self.classifier(embedding)

        if return_aux:
            return logits, {
                "embedding": embedding,
                "patch_logits": patch_logits.reshape(B, T, -1),
            }
        return logits


def architecture_for_state_dict(state) -> str:
    """Identify which architecture a checkpoint was saved from.

    Checkpoints predating the multi-domain streams carry the old gated-concat
    fusion, so they have to be loaded into :class:`ACEVerifyBaseline`.
    """
    return "baseline" if any(k.startswith("fusion_gate.") for k in state) else "enhanced"


def load_from_checkpoint(path, map_location="cpu", **kwargs):
    """Load a checkpoint into whichever architecture produced it.

    Returns ``(model, architecture_name)``.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    name = architecture_for_state_dict(state)
    model = ACEVerifyBaseline() if name == "baseline" else ACEVerifyModel(**kwargs)
    model.load_state_dict(state)
    return model, name
