import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as transforms

class ACEDataset(Dataset):
    def __init__(self, h5_paths, transform=None):
        self.h5_paths = h5_paths
        self.transform = transform
        self.keys = []

        for path in self.h5_paths:
            with h5py.File(path, 'r') as f:
                for key in f.keys():
                    self.keys.append((path, key))

        self.spectogram_transform = transforms.MelSpectrogram(sample_rate=44000, n_mels=32, n_fft=400, hop_length=160)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        path, key = self.keys[idx]
        with h5py.File(path, 'r') as f:
            group = f[key]
            video = torch.from_numpy(group['video'][:]).float()
            audio = torch.from_numpy(group['audio'][:]).float()
            label = torch.tensor(group.attrs['label'], dtype=torch.long)

        video = video.permute(3, 0, 1, 2) / 255.0

        if audio.ndim > 1:
            audio = audio.mean(dim=0)
        spec = self.spectogram_transform(audio)
        spec = spec.unsqueeze(0)

        return video, spec, label