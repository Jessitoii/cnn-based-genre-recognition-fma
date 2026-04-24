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
    """Frequency and time masking on mel-spectrograms (applied during training only)."""

    def __init__(self, freq_mask=27, time_mask=100):
        super().__init__()
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask)
        self.time_mask = TimeMasking(time_mask_param=time_mask)

    def forward(self, x):
        x = self.freq_mask(x)
        x = self.time_mask(x)
        return x


# ── Metrics ────────────────────────────────────────────────────────────────────


def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ── One epoch ──────────────────────────────────────────────────────────────────


def train_epoch(model, loader, optimizer, criterion, augment, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    valid_batches = 0

    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        x = augment(x)

        optimizer.zero_grad()
        try:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += accuracy(logits, y)
            valid_batches += 1
        except RuntimeError as e:
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ── Main ───────────────────────────────────────────────────────────────────────


def train(config_path="config.yaml", colab=False):
    cfg = load_config(config_path)

    # Colab: remap paths to Drive
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

    torch.manual_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # Model
    model = build_model(cfg, device)

    # Loss — label smoothing for overconfidence regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer + Scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # SpecAugment
    augment = SpecAugment().to(device)

    # Early stopping config
    patience = 7
    epochs_no_improve = 0
    best_val_loss = float("inf")

    # Training loop
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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

    for epoch in tqdm(range(start_epoch, cfg["training"]["epochs"] + 1), desc="Epochs"):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, augment, device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

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

        # Update best_val_acc for logging at the end
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Early stopping and Save best model based on val_loss
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

        # Periodic checkpoint every 10 epochs (Colab disconnect safety)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--colab", action="store_true", help="Use Colab/Drive paths")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(config_path=args.config, colab=args.colab)
