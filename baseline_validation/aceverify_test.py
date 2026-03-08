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
from src.dataset import ACEDataset
from src.model import ACEVerifyModel

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

def evaluate_veriface(dataloader, model, device):
    all_preds = []
    all_labels = []
    
    print("Starting ACEVerify AI Inference...")
    with torch.no_grad():
        for batch_idx, (videos, audio, labels) in enumerate(dataloader):
            videos = videos.to(device)
            audio = audio.to(device)
            
            logits = model(videos, audio)
            
            probs = torch.sigmoid(logits).squeeze()
            
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
                
            preds = (probs > 0.5).int().cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.int().cpu().numpy())

    return all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Train ACEVerifyModel")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test HDF5 file")
    args = parser.parse_args()
    test_path = args.test_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking ACEVerify AI on: {device}")

    model = ACEVerifyModel().to(device)

    checkpoint_path = 'aceverify_final.pth'
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded weights from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

    model.eval()

    test_dataset = load_data(test_path, 200, training=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_labels, test_preds = evaluate_veriface(test_loader, model, device)

    print("\n--- ACEVerify (ViT + 3-layer GRU) Final Report ---")
    print(classification_report(test_labels, test_preds, target_names=['Real', 'Fake']))
    print(f"Total Accuracy: {accuracy_score(test_labels, test_preds):.2%}")

if __name__ == "__main__":
    main()