import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

def generate_gradcam(model, input_tensor, target_layer=None, intensity=0.85):
    """
    Generate Grad-CAM heatmap for model prediction.
    
    Args:
        model: PyTorch model (in eval mode)
        input_tensor: torch.Tensor of shape [B, C, H, W] or [B, C, T, H, W]
        target_layer: name of layer to compute gradients for (e.g., 'model.blocks[-1]')
        intensity: float [0, 1] to scale heatmap opacity
    
    Returns:
        PIL.Image: RGB heatmap overlay
    """
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Handle both spatial [B, C, H, W] and temporal [B, C, T, H, W]
    if input_tensor.ndim == 5:
        # Temporal: average frames for single heatmap
        input_tensor = input_tensor.mean(dim=2, keepdim=True)  # [B, C, 1, H, W]
        input_tensor = input_tensor.squeeze(2)  # [B, C, H, W]
    
    b, c, h, w = input_tensor.shape
    
    # Get feature maps from penultimate layer
    input_tensor.requires_grad = True
    
    # Forward pass with hook to capture activations
    activations = {}
    gradients = {}
    
    def forward_hook(module, input, output):
        activations['features'] = output.detach()
    
    def backward_hook(module, grad_input, grad_output):
        gradients['features'] = grad_output[0].detach()
    
    # Register hooks on last conv layer
    if hasattr(model, 'model'):
        # For wrapped models like DeepfakeEfficientNet
        target = model.model
    else:
        target = model
    
    # Find last conv block
    last_conv = None
    for module in target.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    
    if last_conv is None:
        raise ValueError("Could not find Conv2d layer for Grad-CAM")
    
    fwd_handle = last_conv.register_forward_hook(forward_hook)
    bwd_handle = last_conv.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        output = model(input_tensor)
        
        # Backward pass
        model.zero_grad()
        if isinstance(output, torch.Tensor):
            if output.dim() > 0:
                loss = output.mean()
            else:
                loss = output
        else:
            loss = output[0].mean() if isinstance(output, tuple) else output.mean()
        
        loss.backward()
        
        # Compute Grad-CAM
        if 'features' not in activations or 'features' not in gradients:
            raise ValueError("Could not capture activations/gradients")
        
        acts = activations['features']  # [B, C_feat, H_feat, W_feat]
        grads = gradients['features']   # [B, C_feat, H_feat, W_feat]
        
        # Global average pooling on gradients
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C_feat, 1, 1]
        
        # Weighted sum of activations
        cam = (acts * weights).sum(dim=1, keepdim=True)  # [B, 1, H_feat, W_feat]
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        # Upsample to input size
        cam = F.interpolate(
            cam,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )
        
        # cam = cam.squeeze(0).squeeze(0).cpu().numpy()  # [H, W]
        cam = cam[0, 0].cpu().numpy()  # [H, W] NOTE: fix for dimension mismatch

        # Convert to heatmap
        heatmap = _create_heatmap(cam, input_tensor[0].cpu(), intensity)
        return heatmap
    
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def _create_heatmap(cam, input_img, intensity=0.85):
    """
    Overlay Grad-CAM on input image.
    
    Args:
        cam: numpy array [H, W] in [0, 1]
        input_img: torch tensor [C, H, W] in ImageNet-normalized range
        intensity: float to scale heatmap opacity
    
    Returns:
        PIL.Image RGB
    """

    # Ensure cam is 2D
    if cam.ndim != 2:
        raise ValueError(f"Expected cam to be 2D [H, W], got shape {cam.shape}")

    h, w = cam.shape
    
    # Denormalize input image
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = (input_img.detach().numpy() * std + mean).clip(0, 1)
    img_np = (img_np * 255).astype(np.uint8).transpose(1, 2, 0)  # [H, W, 3]
    
    # Create red heatmap
    heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap_rgb[:, :, 0] = (cam * 255).astype(np.uint8)  # Red channel
    
    # Blend with input
    alpha = intensity
    result = (1 - alpha) * img_np.astype(np.float32) + alpha * heatmap_rgb.astype(np.float32)
    result = result.astype(np.uint8)
    
    return Image.fromarray(result)

def region_scores_from_heatmap(heatmap_img):
    arr = np.array(heatmap_img.convert("RGB"))[:, :, 0].astype(np.float32) / 255.0
    h, w = arr.shape
    periocular = arr[int(0.15*h):int(0.45*h), int(0.2*w):int(0.8*w)].mean()
    mouth      = arr[int(0.55*h):int(0.9*h),  int(0.25*w):int(0.75*w)].mean()
    forehead   = arr[int(0.0*h):int(0.2*h),   int(0.2*w):int(0.8*w)].mean()
    chin       = arr[int(0.82*h):int(1.0*h),  int(0.25*w):int(0.75*w)].mean()
    return [
        ("Periocular", float(periocular)),
        ("Mouth", float(mouth)),
        ("Forehead", float(forehead)),
        ("Chin", float(chin)),
    ]

def evidence_from_regions(regions):
    d = {k: v for k, v in regions}
    return {
        "Eye-blink anomaly":     min(1.0, d["Periocular"] * 1.10),
        "Lip-sync mismatch":     min(1.0, d["Mouth"] * 1.10),
        "Texture inconsistency": min(1.0, (d["Periocular"] + d["Forehead"]) / 2),
        "Compression artifacts": min(1.0, (d["Forehead"] + d["Chin"]) / 2),
        "Head-pose jitter":      min(1.0, (d["Periocular"] + d["Chin"]) / 2),
        "Skin-tone boundary":    min(1.0, (d["Mouth"] + d["Chin"]) / 2),
    }