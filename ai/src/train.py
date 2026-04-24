"""
Training script for the Genre Classification model.
Handles the training loop, validation, SpecAugment, checkpointing,
and early stopping. Supports resuming from existing checkpoints.
"""

import os
import time
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchaudio.transforms import FrequencyMasking, TimeMasking
from pathlib import Path
from tqdm import tqdm

from dataset import load_config, get_dataloaders
from model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
logger = logging.getLogger(__name__)


# ── SpecAugment ────────────────────────────────────────────────────────────────


class SpecAugment(nn.Module):
    """
    Frequency and time masking on mel-spectrograms.
    Applied during training only to improve model generalization by making
    it robust to missing spectral or temporal information.
    """

    def __init__(self, freq_mask=27, time_mask=100):
        """
        Initialize SpecAugment layers.

        Args:
            freq_mask (int): Maximum frequency width to mask.
            time_mask (int): Maximum time width to mask.
        """
        super().__init__()
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = TimeMasking(time_mask_param=time_mask)

    def forward(self, x):
        """
        Apply masking to the input tensor.

        Args:
            x (torch.Tensor): Input mel-spectrogram of shape (B, 1, F, T).

        Returns:
            torch.Tensor: Augmented mel-spectrogram.
        """
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x


# ── Metrics ────────────────────────────────────────────────────────────────────


def accuracy(logits, labels):
    """
    Calculate top-1 accuracy.

    Args:
        logits (torch.Tensor): Model output logits.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        float: Accuracy value in range [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── One epoch ──────────────────────────────────────────────────────────────────


def train_epoch(model, loader, optimizer, criterion, augment, device):
    """
    Run one training epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimization algorithm.
        criterion (nn.Module): Loss function.
        augment (nn.Module): Augmentation module.
        device (torch.device): Device to run computation on (e.g., 'cuda').

    Returns:
        tuple: (average_loss, average_accuracy) for the epoch.
    """
    model.train()
    total_loss, total_acc = 0.0, 0.0
    valid_batches = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        x = augment(x)  # Apply SpecAugment on the fly

        optimizer.zero_grad()
        try:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            valid_batches += 1
        except RuntimeError as e:
            # Handle sporadic OOM by skipping batch and clearing cache
            if "out of memory" in str(e):
                logger.warning("CUDA Out of Memory! Clearing cache and skipping batch.")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue
            else:
                raise e

    n = valid_batches if valid_batches > 0 else 1
    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    """
    Run one evaluation epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): Validation or test data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computation on.

    Returns:
        tuple: (average_loss, average_accuracy) for the data in loader.
    """
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy(logits, y)

    n = len(loader)
    return total_loss / n, total_acc / n


# ── Checkpoint ─────────────────────────────────────────────────────────────────


def save_checkpoint(state, path):
    """
    Save the model state and optimizer state to a file.

    Args:
        state (dict): Dictionary containing states and metadata.
        path (str): Destination path to save the checkpoint file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ── Main ───────────────────────────────────────────────────────────────────────


def train(config_path="config.yaml", colab=False):
    """
    Main training pipeline.
    Initializes model, data, and optimizer, then runs the epoch loop.

    Args:
        config_path (str): Path to the YAML configuration file.
        colab (bool): Whether running in Google Colab environment (adjusts paths).

    Returns:
        dict: Training history containing loss and accuracy per epoch.
    """
    cfg = load_config(config_path)

    # Colab: remap paths to Google Drive for persistent storage
    if colab:
        drive_root = cfg["colab"]["drive_root"]
        cfg["paths"][
            "data_processed"
        ] = f"{cfg['colab']['drive_data']}/processed/spectrograms"
        cfg["paths"][
            "metadata"
        ] = f"{cfg['colab']['drive_data']}/raw/fma_metadata/tracks.csv"
        cfg["paths"]["checkpoints"] = f"{cfg['colab']['drive_models']}/checkpoints"
        cfg["paths"]["best_model"] = f"{cfg['colab']['drive_models']}/best_model.pth"

    # Enforce reproducibility
    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Initialize data loaders
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Build model architecture
    model = build_model(cfg, device)

    # Loss — use label smoothing to improve generalization and handle noisy data
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Robust optimizer and learning rate schedule
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # Augmentation applied on the GPU during training
    augment = SpecAugment().to(device)

    # Early stopping and performance tracking setup
    patience = 7
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Resume from latest checkpoint if available
    start_epoch = 1
    checkpoint_dir = Path(cfg["paths"]["checkpoints"])
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pth"))
        if checkpoints:
            latest_ckpt = max(checkpoints, key=os.path.getmtime)
            logger.info(f"Resuming from checkpoint: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1

    # Main Training Loop
    for epoch in tqdm(range(start_epoch, cfg["training"]["epochs"] + 1), desc="Epochs"):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, augment, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Update per-epoch history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Early stopping and Persistence of the best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "cfg": cfg,
                },
                cfg["paths"]["best_model"],
            )
            logger.info(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"  Early stopping triggered after {epoch} epochs "
                    f"(val_loss did not improve for {patience} epochs)."
                )
                break

        # Periodic checkpointing for crash recovery
        if epoch % 10 == 0:
            ckpt_path = f"{cfg['paths']['checkpoints']}/epoch_{epoch:03d}.pth"
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict()}, ckpt_path
            )

    logger.info(
        f"Training complete. Best val_loss: {best_val_loss:.4f}, "
        f"Best val_acc: {best_val_acc:.4f}"
    )
    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Genre Recognition CNN.")
    parser.add_argument("--colab", action="store_true", help="Use Colab/Drive paths")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    train(config_path=args.config, colab=args.colab)
