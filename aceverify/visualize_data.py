import matplotlib.pyplot as plt
import numpy as np

from .dataset import ACEDataset, BACKBONE_MEAN, BACKBONE_STD, read_labels


def denormalize(video):
    """Undo the backbone normalization so frames are displayable in [0, 1]."""
    mean = np.array(BACKBONE_MEAN).reshape(3, 1, 1, 1)
    std = np.array(BACKBONE_STD).reshape(3, 1, 1, 1)
    return np.clip(video.numpy() * std + mean, 0.0, 1.0)


def test_visualization(dataset, index=0):
    print(f"Full Dataset: Length: {len(dataset)}")
    video, spec, label = dataset[index]

    print(f"Video Shape: {video.shape}")
    print(f"Spectrogram Shape: {spec.shape}")
    print(f"Label: {label}")

    frames = denormalize(video)
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))

    for i, frame_idx in enumerate([0, 5, 10, 15]):
        axes[i].imshow(frames[:, frame_idx].transpose(1, 2, 0))
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis('off')

    axes[4].imshow(spec.squeeze().numpy(), aspect='auto', origin='lower')
    axes[4].set_title("Log-Mel Spectrogram")
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()


def numRealAndFake(dataset):
    """Count labels from the cached attribute scan instead of decoding every clip."""
    labels = read_labels(dataset.h5_path)[dataset.indices]
    num_fake = int((labels == 1).sum())
    num_real = int((labels == 0).sum())
    print(f"Number of Real: {num_real}, Number of Fake: {num_fake}\n")
    return num_real, num_fake


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize ACE dataset samples")
    parser.add_argument("--h5", default="data/train_data.h5", help="Path to an HDF5 data file")
    parser.add_argument("--index", type=int, default=0, help="Dataset index to visualize")
    args = parser.parse_args()

    test_visualization(ACEDataset(args.h5), args.index)
