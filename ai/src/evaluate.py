import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from dataset import load_config, get_dataloaders, FMADataset
from model import build_model


@torch.no_grad()
def evaluate(config_path="config.yaml", colab=False):
    cfg = load_config(config_path)

    if colab:
        cfg["paths"]["data_processed"] = f"{cfg['colab']['drive_data']}/processed/spectrograms"
        cfg["paths"]["metadata"]       = f"{cfg['colab']['drive_data']}/raw/fma_metadata/tracks.csv"
        cfg["paths"]["best_model"]     = f"{cfg['colab']['drive_models']}/best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(cfg["paths"]["best_model"], map_location=device)
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Test loader
    _, _, test_loader = get_dataloaders(cfg)
    genre_names = cfg["data"]["genres"]

    all_preds, all_labels = [], []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print(classification_report(all_labels, all_preds, target_names=genre_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=genre_names, yticklabels=genre_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix — Test Acc: {acc*100:.2f}%")
    plt.tight_layout()

    out_path = "confusion_matrix.png"
    if colab:
        out_path = f"{cfg['colab']['drive_root']}/confusion_matrix.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")

    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate(config_path=args.config, colab=args.colab)
