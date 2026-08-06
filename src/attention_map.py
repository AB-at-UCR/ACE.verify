import argparse
import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from aceverify.dataset import ACEDataset
from aceverify.model import ACEVerifyModel


def load_data(path, n, training):
    num_each = n // 2
    all_labels = []
    
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            all_labels.append(f[key].attrs['label'])

    all_labels = np.array(all_labels)
    real_indices = np.where(all_labels == 0)[0]
    fake_indices = np.where(all_labels == 1)[0]

    sel_real = np.random.choice(real_indices, min(len(real_indices), num_each), replace=False)
    sel_fake = np.random.choice(fake_indices, min(len(fake_indices), num_each), replace=False)

    sub_indices = np.concatenate([sel_real, sel_fake])
    np.random.shuffle(sub_indices)
    
    dataset = ACEDataset(h5_path=path, indices=sub_indices, is_training=training)
    return dataset

def attention_map(model, device, video_tensor, frame_idx=0):
    model.eval()
    attention_outputs = []

    def hook_fn(module, input, output):
        attention_outputs.append(output)

    try:
        target_layer = model.video_model.blocks[-1].attn.qkv
        handle = target_layer.register_forward_hook(hook_fn)
    except AttributeError:
        print("Error")
        return

    try:
        with torch.no_grad():
            input_video = video_tensor.permute(1, 0, 2, 3).to(device)
            _ = model.video_model(input_video)

        if not attention_outputs:
            print("Hook failed to capture data.")
            return

        qkv_out = attention_outputs[0]
        B, N, C3 = qkv_out.shape
        num_heads = 12  
        head_dim = (C3 // 3) // num_heads
    
        qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn_mean = attn.mean(dim=1)
        cls_attn = attn_mean[:, 0, 1:] 
        
        mask = cls_attn[frame_idx].reshape(14, 14).cpu().numpy()
        img_np = video_tensor[:, frame_idx, :, :].permute(1, 2, 0).cpu().numpy()
        mask_resized = cv2.resize(mask, (224, 224))
        mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)

        # Apply Heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        overlay = (0.4 * heatmap + 0.6 * img_np)
        overlay = np.clip(overlay, 0, 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Frame {frame_idx}")
        plt.imshow(img_np)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("ACEVerify Attention Map")
        plt.imshow(overlay)
        plt.axis('off')
        plt.show()

    finally:
        handle.remove()


def visualize_all_frames(model, device, video_tensor):
    model.eval()
    attention_outputs = []

    def hook_fn(module, input, output):
        attention_outputs.append(output)

    target_layer = model.video_model.blocks[-1].attn.qkv
    handle = target_layer.register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            input_video = video_tensor.permute(1, 0, 2, 3).to(device)
            _ = model.video_model(input_video)

        qkv_out = attention_outputs[0]
        B, N, C3 = qkv_out.shape
        num_heads = 12 
        head_dim = (C3 // 3) // num_heads
        qkv = qkv_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = attn.softmax(dim=-1).mean(dim=1) 
        cls_attn = attn[:, 0, 1:].reshape(-1, 14, 14)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle("ACEVerify Temporal Attention Tracking (All 16 Frames)", fontsize=20)

        for i in range(16):
            ax = axes[i // 4, i % 4]
            
            img_np = video_tensor[:, i, :, :].permute(1, 2, 0).cpu().numpy()
            mask = cls_attn[i].cpu().numpy()
            mask_resized = cv2.resize(mask, (224, 224))
            mask_norm = (mask_resized - mask_resized.min()) / (mask_resized.max() - mask_resized.min() + 1e-8)

            heatmap = cv2.applyColorMap(np.uint8(255 * mask_norm), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = np.clip((0.4 * heatmap + 0.6 * img_np), 0, 1)

            ax.imshow(overlay)
            ax.set_title(f"Frame {i}")
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    finally:
        handle.remove()


def predict_video(model, device, video_tensor, audio_tensor):
    model.eval()

    v = video_tensor.unsqueeze(0).to(device)
    a = audio_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(v, a)
        probability = torch.sigmoid(logits).item()
    
    label = "FAKE" if probability > 0.5 else "REAL"
    confidence = probability if probability > 0.5 else (1 - probability)
    
    return label, confidence * 100

def main():
    parser = argparse.ArgumentParser(description="Train ACEVerifyModel")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test HDF5 file")
    args = parser.parse_args()
    test_path = args.test_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ACEVerifyModel().to(device)
    checkpoint_path = "aceverify_final.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    test_dataset = load_data(test_path, 200, False)
    video, audio, label = test_dataset[0]

    pred_label, conf = predict_video(model, device, video, audio)
    true_label = "FAKE" if label == 1 else "REAL"

    print(f"Target: {true_label}")
    print(f"ACEVerify Prediction: {pred_label} ({conf:.2f}% confidence)")

    #attention_map(model, device, video, frame_idx=8)
    visualize_all_frames(model, device, video)

if __name__ == "__main__":
    main()