# Model Checkpoints

This directory stores the trained weights, architecture snapshots, and training history for the Music Genre Classification CNN.

## Directory Structure

```text
models/
├── best_model.pth          # State dict of the best performing model
├── last_model.pth          # (Optional) Latest checkpoint from training
└── checkpoints/            # Intermediate epoch checkpoints for recovery
    ├── epoch_010.pth
    ├── epoch_020.pth
    └── ...
```

## Checkpoint Format

The model files (`.pth`) are PyTorch serialized dictionaries. 

-   **`best_model.pth`**: Saved whenever the validation accuracy improves. It contains the `model_state`, `optimizer_state`, and training metadata.
-   **`checkpoints/`**: Contains snapshots taken every 10 epochs. These are used to resume training in case of a crash or disconnect (e.g., in Google Colab).

### What's inside a checkpoint?
```python
checkpoint = {
    'epoch': int,
    'model_state': dict,         # model.state_dict()
    'optimizer_state': dict,     # optimizer.state_dict()
    'val_loss': float,
    'val_acc': float,
    'cfg': dict                  # Configuration used for training
}
```

## Loading for Inference

To load the model for prediction or evaluation, use the following snippet:

```python
import torch
from ai.src.model import GenreCNN

# Initialize model architecture
# Ensure num_classes matches the training configuration (default: 8)
model = GenreCNN(num_classes=8)

# Load state dictionary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("models/best_model.pth", map_location=device)

# Load weights (the key 'model_state' contains the actual state dict)
model.load_state_dict(checkpoint['model_state'])
model.eval()
```

## Maintenance

-   Checkpoints can consume significant disk space. While `best_model.pth` is critical, older files in the `checkpoints/` directory can be safely removed once training is finalized.
-   This directory is `.gitignored` to avoid pushing large binary files to the repository.

---
> [!TIP]
> Always use `map_location=device` when loading to ensure compatibility between GPU-trained models and CPU-only inference environments.
