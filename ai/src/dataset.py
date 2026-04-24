"""
Data handling module for the FMA dataset.
Provides utilities for audio preprocessing (Mel-spectrogram conversion)
and PyTorch Dataset/DataLoader classes for training and evaluation.
"""

import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """
    Load project configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. Defaults to "config.yaml".

    Returns:
        dict: Parsed configuration parameters.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_track_path(track_id: int, data_root: str) -> Path:
    """
    Get the file path for a track in the FMA dataset structure.
    FMA uses a two-level directory structure: 000/000002.mp3.

    Args:
        track_id (int): Digit ID of the track.
        data_root (str): Root directory of the FMA dataset.

    Returns:
        Path: Full path to the MP3 file.
    """
    # Format track_id as 6-digit string and use first 3 digits as sub-folder
    tid = f"{track_id:06d}"
    return Path(data_root) / tid[:3] / f"{tid}.mp3"


def load_mel_spectrogram(path: str, cfg: dict) -> np.ndarray:
    """
    Load an audio file and convert it to a Mel-spectrogram.

    Args:
        path (str): Path to the audio file.
        cfg (dict): Configuration dictionary containing audio parameters.

    Returns:
        np.ndarray: Mel-spectrogram in dB scale as a (n_mels, time) array.
                   Returns None if loading fails.
    """
    try:
        # Load audio with fixed sample rate and duration
        y, sr = librosa.load(
            path,
            sr=cfg["data"]["sample_rate"],
            duration=cfg["data"]["duration"],
            mono=True,
        )
        # Pad short clips to ensure consistent temporal dimensions
        expected_len = cfg["data"]["sample_rate"] * cfg["data"]["duration"]
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))

        # Compute Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=cfg["data"]["n_mels"],
            n_fft=cfg["data"]["n_fft"],
            hop_length=cfg["data"]["hop_length"],
        )
        # Convert to log-scale (dB) for better neural network convergence
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def preprocess_and_save(cfg: dict, out_dir: str):
    """
    Perform one-time preprocessing by converting MP3 files to .npy Mel-spectrograms.

    This function filters the dataset to the 'small' subset and saves computed
    spectrograms to facilitate faster data loading during training.

    Args:
        cfg (dict): Configuration dictionary.
        out_dir (str): Directory where .npy files will be saved.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks_csv = cfg["paths"]["metadata"]
    df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Filter to fma_small subset for balanced and manageable training
    subset = df[df[("set", "subset")] == "small"]
    genre_col = ("track", "genre_top")
    subset = subset[subset[genre_col].notna()]

    genre_list = cfg["data"]["genres"]
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    skipped = 0
    saved = 0
    corrupted = 0

    for track_id, row in tqdm(
        subset.iterrows(), total=len(subset), desc="Preprocessing"
    ):
        genre = row[genre_col]
        if genre not in genre_to_idx:
            skipped += 1
            continue

        audio_path = get_track_path(track_id, cfg["paths"]["data_raw"])
        if not audio_path.exists():
            skipped += 1
            continue

        out_path = out_dir / f"{track_id}.npy"
        if out_path.exists():
            saved += 1
            continue

        mel = load_mel_spectrogram(str(audio_path), cfg)
        if mel is None:
            corrupted += 1
            continue

        np.save(out_path, mel)
        saved += 1

    logger.info(
        f"Preprocessing done. Saved: {saved}, Skipped: {skipped}, Corrupted: {corrupted}"
    )


class FMADataset(Dataset):
    """
    Custom PyTorch Dataset for the Free Music Archive (FMA) dataset.
    Loads pre-processed Mel-spectrograms and their corresponding genre labels.
    """

    def __init__(self, cfg: dict, split: str = "train", transform=None):
        """
        Initialize the dataset by loading metadata and filtering samples for the split.

        Args:
            cfg (dict): Loaded project configuration.
            split (str): Dataset split to load ('train', 'val', or 'test').
            transform (callable, optional): Torchvision-style transform to apply to samples.
        """
        self.cfg = cfg
        self.transform = transform
        self.spec_dir = Path(cfg["paths"]["data_processed"])

        tracks_csv = cfg["paths"]["metadata"]
        df = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

        # Filter to fma_small subset
        subset = df[df[("set", "subset")] == "small"]
        genre_col = ("track", "genre_top")
        subset = subset[subset[genre_col].notna()]

        genre_list = cfg["data"]["genres"]
        self.genre_to_idx = {g: i for i, g in enumerate(genre_list)}
        self.idx_to_genre = {i: g for g, i in self.genre_to_idx.items()}

        # Build samples list — only tracks with preprocessed .npy files available
        samples = []
        for track_id, row in subset.iterrows():
            genre = row[genre_col]
            if genre not in self.genre_to_idx:
                continue
            npy_path = self.spec_dir / f"{track_id}.npy"
            if npy_path.exists():
                samples.append((str(npy_path), self.genre_to_idx[genre]))

        # Deterministic splitting using a fixed random seed
        rng = np.random.default_rng(cfg["training"]["seed"])
        indices = rng.permutation(len(samples))

        n_test = int(len(samples) * cfg["training"]["test_split"])
        n_val = int(len(samples) * cfg["training"]["val_split"])

        test_idx = indices[:n_test]
        val_idx = indices[n_test : n_test + n_val]
        train_idx = indices[n_test + n_val :]

        split_map = {"train": train_idx, "val": val_idx, "test": test_idx}
        self.samples = [samples[i] for i in split_map[split]]
        logger.info(f"[{split}] {len(self.samples)} samples loaded.")

    def __len__(self):
        """
        Returns the total number of samples in this dataset split.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetch a sample and its label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (mel_spectrogram, label_idx) where mel_spectrogram is a
                   torch.Tensor of shape (1, n_mels, time).
        """
        path, label = self.samples[idx]
        mel = np.load(path)  # Shape: (n_mels, time)

        # Standard Z-score normalization to improve network stability
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # Add channel dimension (C, H, W) for Conv2D layers
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            mel = self.transform(mel)

        return mel, label


def get_dataloaders(cfg: dict):
    """
    Create PyTorch DataLoaders for train, validation, and test splits.

    Args:
        cfg (dict): Project configuration.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_ds = FMADataset(cfg, split="train")
    val_ds = FMADataset(cfg, split="val")
    test_ds = FMADataset(cfg, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
