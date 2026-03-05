import matplotlib.pyplot as plt
import torch
from dataset import *

def test_visualization(dataset, index=0):
    print(f"Full Dataset: Length: {len(dataset)}")
    video, spec, label = dataset[index]

    print(f"Video Shape: {video.shape}")
    print(f"Spectrogram Shape: {spec.shape}")
    print(f"Label: {label}")

    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    indices = [0, 5, 10, 15]
    for i, frame_idx in enumerate(indices):
        frame = video[:, frame_idx, :, :].permute(1, 2, 0).numpy()
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis('off')

    axes[4].imshow(spec.squeeze().numpy(), aspect='auto', origin='lower')
    axes[4].set_title("Mel-Spectrogram")
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()

def numRealAndFake(dataset):
    num_real = 0
    num_fake = 0
    for idx in range(len(dataset)):
        video, spec, label = dataset[idx]
        if label == 1:
            num_fake += 1
        else:
            num_real += 1
    
    print(f"Number of Real: {num_real}, Number of Fake: {num_fake}\n")


h5_files = ['data/processed_data_00.h5']
dataset = ACEDataset(h5_files)
#numRealAndFake(dataset)
test_visualization(dataset, 0)