import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score
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

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

def evaluate_spatial_baseline(test_loader, model, device):
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (videos, audio, labels) in enumerate(test_loader):
            batch_size, c, f, h, w = videos.shape
            videos = videos.to(device)
            
            images = videos.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            
            if h != 380:
                images = F.interpolate(images, size=(380, 380), mode='bilinear', align_corners=False)
            
            images = (images - MEAN) / STD
            
            logits = baseline_model(images)
            probs = torch.sigmoid(logits)
            
            video_probs = probs.view(batch_size, f).mean(dim=1)
            
            preds = (video_probs > 0.5).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().numpy())

    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train ACEVerifyModel")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test HDF5 file")
    args = parser.parse_args()
    test_path = args.test_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking spatial2D models on: {device}")

    test_dataset = load_data(test_path, 200, training=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = timm.create_model('xception', pretrained=True, num_classes=1).to(device)
    model.eval()
    test_labels, test_preds = evaluate_spatial_baseline(test_loader, model, device)
    print("\n--- xception Final Report ---")
    print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))
    print(f"Total Accuracy: {accuracy_score(test_labels, test_preds):.2%}")

    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1).to(device)
    model.eval()
    test_labels, test_preds = evaluate_spatial_baseline(test_loader, model, device)
    print("\n--- efficientnet_b4 Final Report ---")
    print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))
    print(f"Total Accuracy: {accuracy_score(test_labels, test_preds):.2%}")

if __name__ == "__main__":
    main()