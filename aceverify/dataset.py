import logging
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as audio_transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# vit_base_patch16_224 was pretrained with these; feeding it raw [0, 1] pixels puts
# the backbone off-distribution and costs several points of accuracy.
BACKBONE_MEAN = (0.5, 0.5, 0.5)
BACKBONE_STD = (0.5, 0.5, 0.5)

# ffmpeg extracts 0.5 s of audio at 48 kHz in preprocess.py, giving 24000 samples.
AUDIO_SAMPLE_RATE = 48000

_LABEL_CACHE = {}


def read_labels(h5_path: str) -> np.ndarray:
    """Return the label of every group, in ``f.keys()`` order.

    Scanning 4000 group attributes takes ~45 s, and the training loop resamples
    every epoch, so the result is memoized in-process and mirrored to a sidecar
    ``.labels.npy`` next to the archive.
    """
    key = os.path.abspath(h5_path)
    stamp = os.path.getmtime(h5_path)
    cached = _LABEL_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    sidecar = f"{key}.labels.npy"
    if os.path.exists(sidecar) and os.path.getmtime(sidecar) >= stamp:
        labels = np.load(sidecar)
        _LABEL_CACHE[key] = (stamp, labels)
        return labels

    logger.info("Scanning labels in %s (cached afterwards)", h5_path)
    with h5py.File(h5_path, "r") as f:
        labels = np.array([f[k].attrs["label"] for k in f.keys()], dtype=np.int64)

    try:
        np.save(sidecar, labels)
    except OSError:
        logger.warning("Could not write label cache to %s; keeping it in memory only", sidecar)

    _LABEL_CACHE[key] = (stamp, labels)
    return labels


class ACEDataset(Dataset):
    """Clips of ``num_frames`` RGB frames plus a log-mel spectrogram.

    Augmentation is sampled once per clip rather than per frame. Sampling it per
    frame injects synthetic flicker, which directly contradicts the temporal
    coherence signal the model is trained to detect.
    """

    def __init__(
        self,
        h5_path,
        indices=None,
        is_training=False,
        num_frames: int = 16,
        n_mels: int = 128,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ):
        self.h5_path = h5_path
        self.file = None
        self.keys = None
        self.is_training = is_training
        self.num_frames = num_frames

        # n_fft=2048 (23 Hz bins) keeps every mel filterbank non-empty at 128 mels;
        # at n_fft=1024 the lowest bands are narrower than one FFT bin and go to zero.
        self.spectrogram_transform = audio_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=n_mels, n_fft=2048, hop_length=190, f_min=20.0
        )
        self.to_db = audio_transforms.AmplitudeToDB(stype="power", top_db=80.0)

        self.register_normalization()

        if indices is None:
            with h5py.File(self.h5_path, "r") as f:
                self.indices = np.arange(len(f.keys()))
        else:
            self.indices = np.asarray(indices)

    def register_normalization(self):
        self.mean = torch.tensor(BACKBONE_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(BACKBONE_STD).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.indices)

    def _ensure_open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r")
            self.keys = list(self.file.keys())

    def _sample_frame_indices(self, total_frames: int) -> np.ndarray:
        if total_frames >= self.num_frames:
            return np.linspace(0, total_frames - 1, self.num_frames).astype(int)
        return np.clip(np.arange(self.num_frames), 0, total_frames - 1)

    def _augment(self, video: torch.Tensor) -> torch.Tensor:
        """Clip-consistent augmentation. Avoids blur/rescale, which would erase the
        high-frequency residuals the frequency stream relies on."""
        if torch.rand(()) < 0.5:
            video = video.flip(-1)

        brightness = 1.0 + (torch.rand(()) * 0.4 - 0.2)
        contrast = 1.0 + (torch.rand(()) * 0.4 - 0.2)
        video = video * brightness
        video = (video - video.mean()) * contrast + video.mean()

        if torch.rand(()) < 0.5:
            _, _, height, width = video.shape
            box_h = int(height * float(torch.empty(()).uniform_(0.1, 0.25)))
            box_w = int(width * float(torch.empty(()).uniform_(0.1, 0.25)))
            top = int(torch.randint(0, max(1, height - box_h), ()))
            left = int(torch.randint(0, max(1, width - box_w), ()))
            video[:, :, top : top + box_h, left : left + box_w] = 0.0

        return video.clamp_(0.0, 1.0)

    def _spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim > 1:
            audio = audio.mean(dim=0)

        # HDF5 stores raw PCM int16. Mel power of unscaled values overflows fp16 AMP,
        # while quiet clips (test set peaks at ~104) collapse to ~1e-6 without the
        # dB conversion below.
        if audio.abs().max() > 1.5:
            audio = audio / 32768.0

        spec = self.to_db(self.spectrogram_transform(audio)).unsqueeze(0)
        spec = (spec - spec.mean()) / (spec.std() + 1e-5)

        return F.interpolate(
            spec.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)

    def __getitem__(self, idx):
        self._ensure_open()

        group = self.file[self.keys[self.indices[idx]]]

        full_video = group["video"]
        frame_indices = self._sample_frame_indices(full_video.shape[0])
        video = torch.from_numpy(full_video[frame_indices]).float().permute(3, 0, 1, 2) / 255.0

        if self.is_training:
            video = self._augment(video)
        video = (video - self.mean) / self.std

        spec = self._spectrogram(torch.from_numpy(group["audio"][:]).float())
        label = torch.tensor(group.attrs["label"], dtype=torch.long)

        return video, spec, label

    def close(self):
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None

    def __del__(self):
        self.close()
