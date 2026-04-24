"""
Utility functions for the backend API.
Includes configuration loading and audio preprocessing logic that
aligns with the training pipeline.
"""

import librosa
import numpy as np
import torch
import yaml
from pathlib import Path


def load_config(config_path="../ai/config.yaml"):
    """
    Load the project configuration from a YAML file.

    Args:
        config_path (str): Relative or absolute path to the YAML config.
                          Defaults to "../ai/config.yaml".

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_audio(file_path_or_bytes, cfg):
    """
    Process an audio file into a normalized Mel-spectrogram tensor.

    This function duplicates the preprocessing steps used during training
    (resampling, padding, Mel conversion, dB scaling, and Z-score normalization)
    to ensure consistency between training and inference.

    Args:
        file_path_or_bytes (str or file-like): The audio data to process.
        cfg (dict): Project configuration parameters containing SR, n_mels, etc.

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 1, n_mels, time)
                      ready for model inference.
    """
    sr = cfg["data"]["sample_rate"]
    duration = cfg["data"]["duration"]
    n_mels = cfg["data"]["n_mels"]
    n_fft = cfg["data"]["n_fft"]
    hop_length = cfg["data"]["hop_length"]

    # Load audio with the target sample rate and duration
    y, _ = librosa.load(
        file_path_or_bytes,
        sr=sr,
        duration=duration,
        mono=True,
    )

    # Pad if shorter than expected duration to maintain consistent input width
    expected_len = sr * duration
    if len(y) < expected_len:
        y = np.pad(y, (0, int(expected_len - len(y))))

    # Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Convert power spectrogram to dB scale (logarithmic)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Apply Z-score normalization — mirrors FMADataset.__getitem__ implementation
    # Using small epsilon (1e-8) to avoid division by zero errors
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    # Convert to PyTorch tensor and add batch (B) and channel (C) dimensions
    # Results in shape: (1, 1, n_mels, time)
    tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor
