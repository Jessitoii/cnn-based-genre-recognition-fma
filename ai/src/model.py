import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GenreCNN(nn.Module):
    """
    4-block CNN for mel-spectrogram genre classification.
    Input: (B, 1, 128, T) where T ~ 1292 for 30s clips
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1,   32,  pool=True),   # → (B, 32,  64, T/2)
            ConvBlock(32,  64,  pool=True),   # → (B, 64,  32, T/4)
            ConvBlock(64,  128, pool=True),   # → (B, 128, 16, T/8)
            ConvBlock(128, 256, pool=True),   # → (B, 256,  8, T/16)
        )

        # Global Average Pooling — collapses spatial dims, robust to input length variation
        self.gap = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_model(cfg: dict, device: torch.device) -> GenreCNN:
    model = GenreCNN(
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    return model.to(device)


if __name__ == "__main__":
    # Sanity check
    model = GenreCNN()
    x = torch.randn(4, 1, 128, 1292)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {params:,}")
