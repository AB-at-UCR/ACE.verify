import torch
import torch.nn as nn
import timm

class ACEVerifyModel(nn.Module):
    def __init__(self, num_frames=16, num_classes=1):
        super(ACEVerifyModel, self).__init__()

        self.video_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.video_feature_dim = 768 
        
        self.temporal_layer = nn.GRU(input_size=self.video_feature_dim, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        self.audio_model = timm.create_model('resnet18', pretrained=True, in_chans=1, num_classes=0)
        self.audio_feature_dim = 512 
        
        self.full_model = nn.Sequential(nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, video, audio):
        batch_size, C, T, H, W = video.shape
        
        video = video.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        
        v_feat = self.video_model(video)
        v_feat = v_feat.view(batch_size, T, -1) 
        _, h_n = self.temporal_layer(v_feat)
        
        v_final = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        a_final = self.audio_model(audio) 
        
        combined = torch.cat((v_final, a_final), dim=1)
        return self.full_model(combined)