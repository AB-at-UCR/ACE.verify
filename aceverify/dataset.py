import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchaudio.transforms as transforms
import torch.nn.functional as F

class ACEDataset(Dataset):
    def __init__(self, h5_path, indices=None, is_training=False):
        self.h5_path = h5_path
        self.file = None
        self.keys = None
        self.is_training = is_training
        self.spectrogram_transform = transforms.MelSpectrogram(sample_rate=44100, n_mels=32, n_fft=400, hop_length=160)
        if indices is None:
            with h5py.File(self.h5_path, 'r') as f:
                self.indices = list(range(len(f.keys())))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            self.keys = list(self.file.keys())

        real_idx = self.indices[idx]
        key = self.keys[real_idx]
        group = self.file[key]

        full_video = group['video'][:]
        total_frames = full_video.shape[0]
        num_output_frames = 16
        stride = 2
        indices = np.arange(0, num_output_frames * stride, stride)

        if indices[-1] >= total_frames:
            indices = np.linspace(0, total_frames - 1, num_output_frames).astype(int)

        video = torch.from_numpy(full_video[indices]).float().permute(3, 0, 1, 2) / 255.0

        # Data augmentation
        aug = T.Compose([
            # T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
            # T.RandomErasing(p=0.7, scale=(0.1, 0.3), ratio=(0.3, 3.3), value=0),
        ])

        if self.is_training:
            video = torch.stack([aug(frame) for frame in video.permute(1, 0, 2, 3)]).permute(1, 0, 2, 3)

        audio = torch.from_numpy(group['audio'][:]).float()
        if audio.ndim > 1: audio = audio.mean(dim=0)
        spec = self.spectrogram_transform(audio).unsqueeze(0)

        spec = F.interpolate(spec.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        label = torch.tensor(group.attrs['label'], dtype=torch.long)

        return video, spec, label

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        if hasattr(self, 'file'):
            self.close()