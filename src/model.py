import torch
import torch.nn as nn
import timm

class ACEVerifyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0, drop_path_rate=0.1)

        for param in self.video_model.parameters():
            param.requires_grad = False
        for param in self.video_model.blocks[-4:].parameters():
            param.requires_grad = True

        self.temporal_layer = nn.GRU(768, 512, batch_first=True, bidirectional=True)

        self.video_norm = nn.LayerNorm(1024)

        self.full_model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, video, audio_spec):
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        v_feat = self.video_model(video).reshape(B, T, 768)

        gru_out, _ = self.temporal_layer(v_feat)
        video_combined = self.video_norm(gru_out[:, -1, :])

        return self.full_model(video_combined)