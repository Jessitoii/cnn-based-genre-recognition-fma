import librosa
import numpy as np
import torch
import yaml
from pathlib import Path


def load_config(config_path="../ai/config.yaml"):
    """Load config from the ai directory."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_audio(file_path_or_bytes, cfg):
    """
    Load audio and convert to normalized mel-spectrogram.
    Matches the preprocessing in ai/src/dataset.py.
    """
    sr = cfg["data"]["sample_rate"]
    duration = cfg["data"]["duration"]
    n_mels = cfg["data"]["n_mels"]
    n_fft = cfg["data"]["n_fft"]
    hop_length = cfg["data"]["hop_length"]

    # Load audio
    y, _ = librosa.load(
        file_path_or_bytes,
        sr=sr,
        duration=duration,
        mono=True,
    )

    # Pad if shorter than expected duration
    expected_len = sr * duration
    if len(y) < expected_len:
        y = np.pad(y, (0, int(expected_len - len(y))))

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Power to dB
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Normalize to [-1, 1] — following FMADataset.__getitem__ logic
    # Note: Using small epsilon to avoid div by zero
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    # Add batch and channel dims → (1, 1, n_mels, time)
    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor
