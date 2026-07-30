import timm
import torch.nn as nn

class DeepfakeXception(nn.Module):
    def __init__(self):
        super(DeepfakeXception, self).__init__()
        # Loading the SOTA Xception model from timm
        self.model = timm.create_model('xception', pretrained=True, num_classes=1)

    def forward(self, x):
        # Input shape: (Batch, 3, 224, 224)
        return self.model(x)