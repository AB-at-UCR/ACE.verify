import timm
import torch.nn as nn
from torchvision import models

class DeepfakeEfficientNet(nn.Module):
    def __init__(self):
        super(DeepfakeEfficientNet, self).__init__()
        # NOTE: We choose B4-variant as it's a great balance b/w speed & accuracy
        self.model = timm.create_model(
            'tf_efficientnet_b4', 
            pretrained=True, 
            num_classes=1
        )

    def forward(self, x):
        return self.model(x)