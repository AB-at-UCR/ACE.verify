import torch
import models

def load_model(model_name): # Cached Model Loading
    if model_name == "ACE.verify (Best)":
        # Loading optimized TorchScript version of model
        # model = torch.jit.load("weights/aceverify_opt.pt", map_location="cpu")
        model = torch.jit.load("data/trained_model_paths/aceverify_model1.pth", map_location="cpu")
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