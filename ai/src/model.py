"""
CNN Architecture for Music Genre Classification.
Defines the GenreCNN model using Mel-spectrogram inputs.
The architecture consists of several convolutional blocks followed by
Global Average Pooling and a dense classifier.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard convolutional block consisting of two Conv2D layers,
    BatchNorm, ReLU activation, and optional Max Pooling.
    """

    def __init__(self, in_channels, out_channels, pool=True):
        """
        Initialize the convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            pool (bool): Whether to include a MaxPool2D layer at the end.
        """
        super().__init__()
        layers = [
            # First Conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second Conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Processed tensor of shape (B, C_out, H', W').
        """
        return self.block(x)


class GenreCNN(nn.Module):
    """
    4-block CNN for mel-spectrogram genre classification.
    Input: (B, 1, 128, T) where T ~ 1292 for 30s clips.
    Output: (B, num_classes) logits.
    """

    def __init__(self, num_classes=8, dropout=0.3):
        """
        Initialize the GenreCNN architecture.

        Args:
            num_classes (int): Number of target genres.
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()

        # Feature extractor: hierarchical representation learning
        self.features = nn.Sequential(
            ConvBlock(1, 32, pool=True),  # Output: (B, 32,  64, T/2)
            ConvBlock(32, 64, pool=True),  # Output: (B, 64,  32, T/4)
            ConvBlock(64, 128, pool=True),  # Output: (B, 128, 16, T/8)
            ConvBlock(128, 256, pool=True),  # Output: (B, 256,  8, T/16)
        )

        # Global Average Pooling — collapses spatial dims, making the model
        # robust to variation in input temporal length.
        self.gap = nn.AdaptiveAvgPool2d(1)  # Output: (B, 256, 1, 1)

        # Dense classifier: Map extracted features to genre probabilities
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input mel-spectrogram tensor of shape (B, 1, 128, T).

        Returns:
            torch.Tensor: Genre logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_model(cfg: dict, device: torch.device) -> GenreCNN:
    """
    Initialize and return the GenreCNN model based on configuration.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.
        device (torch.device): Device to move the model to (CPU or CUDA).

    Returns:
        GenreCNN: Instantiated model.
    """
    model = GenreCNN(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    return model.to(device)


if __name__ == "__main__":
    # Sanity check for model architecture and parameter count
    model = GenreCNN()
    x = torch.randn(4, 1, 128, 1292)  # Simulate a batch of 4 spectrograms
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")
