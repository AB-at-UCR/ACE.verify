import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attention_logits = self.attention(sequence)
        attention_weights = torch.softmax(attention_logits, dim=1)
        return torch.sum(sequence * attention_weights, dim=1)


class SpectrogramEncoder(nn.Module):
    def __init__(self, feature_dim: int = 1280):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b0_ns",
            pretrained=True,
            in_chans=1,
            num_classes=0,
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, audio_spec: torch.Tensor) -> torch.Tensor:
        audio_features = self.backbone(audio_spec)
        return self.projection(audio_features)


class ACEVerifyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.1,
        )

        for param in self.video_model.parameters():
            param.requires_grad = False
        for param in self.video_model.blocks[-4:].parameters():
            param.requires_grad = True

        self.temporal_layer = nn.GRU(768, 512, batch_first=True, bidirectional=True)
        self.temporal_pool = TemporalAttentionPooling(1024)

        self.video_projection = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.audio_encoder = SpectrogramEncoder()

        self.fusion_gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, video, audio_spec=None):
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        v_feat = self.video_model(video).reshape(B, T, 768)

        gru_out, _ = self.temporal_layer(v_feat)
        video_combined = self.temporal_pool(gru_out)
        video_combined = self.video_projection(video_combined)

        if audio_spec is None:
            audio_spec = video.new_zeros((B, 1, 224, 224))
        elif audio_spec.dim() == 3:
            audio_spec = audio_spec.unsqueeze(1)
        elif audio_spec.dim() == 5:
            audio_spec = audio_spec.squeeze(1)

        audio_spec = audio_spec.to(dtype=video.dtype)
        audio_features = self.audio_encoder(audio_spec)

        fusion_input = torch.cat([video_combined, audio_features], dim=-1)
        gate = self.fusion_gate(fusion_input)
        fused_features = torch.cat(
            [
                video_combined * gate,
                audio_features * (1.0 - gate),
                video_combined - audio_features,
                video_combined * audio_features,
            ],
            dim=-1,
        )

        return self.classifier(fused_features)