import os
import torch
import models

_ACEVERIFY_CHECKPOINT = "data/trained_model_paths/aceverify_model1.pth"

def load_model(model_name): # Cached Model Loading
    if model_name == "ACE.verify (Best)":
        # Loading optimized TorchScript version of model
        # model = torch.jit.load("weights/aceverify_opt.pt", map_location="cpu")
        if not os.path.exists(_ACEVERIFY_CHECKPOINT):
            raise FileNotFoundError(
                "ACE.verify checkpoint is not bundled with this Space. "
                f"Place a TorchScript file at {_ACEVERIFY_CHECKPOINT}, "
                "or choose EfficientNet-B4 / XceptionNet."
            )
        model = torch.jit.load(_ACEVERIFY_CHECKPOINT, map_location="cpu")
    elif model_name == "XceptionNet (Accurate)":
        model = models.DeepfakeXception()
        # model.load_state_dict(torch.load("weights/xception.pth", map_location="cpu"))
    else:  # EfficientNet-B4
        model = models.DeepfakeEfficientNet()
        # model.load_state_dict(torch.load("weights/efficient_b4.pth", map_location="cpu"))
    
    model.eval()
    return model

def get_fake_prob(output: torch.Tensor) -> float:
    """ Basic Sigmoid probability computation from the output """
    p = torch.sigmoid(output.float()).mean().item()
    return float(max(0.0, min(1.0, p)))